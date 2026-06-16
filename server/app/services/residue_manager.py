"""
Residue file manager service for decoupled two-stage ECM.

Handles:
- Parsing residue file metadata
- Storing/retrieving residue files
- Work assignment for stage 2
- Lifecycle management (claim, complete, expire)
"""

import logging
import hashlib
import re

import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from sqlalchemy.exc import IntegrityError

from ..models.residues import ECMResidue
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.projects import Project, ProjectComposite
from ..config import get_settings
from ..utils.file_cleanup import stage_residue_file_deletion
from ..utils.transactions import is_unique_violation
from .t_level_calculator import TLevelCalculator

logger = logging.getLogger(__name__)


def transition_residue_status(
    db: Session,
    residue_id: int,
    from_statuses: list,
    to_status: str,
    clear_claim: bool = False,
) -> bool:
    """
    Conditionally transition one residue's status, re-checking from_statuses
    at write time.

    Cleanup sweeps SELECT candidates and then mutate them one by one; a
    completion can land in between (status -> completed, file deleted). The
    WHERE re-checks status on the current row version, so a row that left the
    cleanable set is skipped instead of being overwritten. Returns True iff
    this statement won the transition, so the caller runs its side effect
    (file deletion, bookkeeping) exactly once.
    """
    values: dict = {ECMResidue.status: to_status}
    if clear_claim:
        values[ECMResidue.claimed_by] = None
        values[ECMResidue.claimed_at] = None
    return db.query(ECMResidue).filter(
        ECMResidue.id == residue_id,
        ECMResidue.status.in_(from_statuses),
    ).update(values, synchronize_session=False) > 0


class ResidueManager:
    """Manages ECM residue files for decoupled two-stage processing."""

    ALLOWED_EVAL_CHARS = "0123456789+-*/^()" # prevent abusing eval(…), DO NOT CHANGE!
    CHKCONST = 4294967291 # from GMP-ECM's CHKSUMMOD (ecm/ecm-ecm.h:170)
    EXPONENTATION_PATTERN = re.compile(r'(\d+)\*\*(\d+)')

    def __init__(self):
        """Initialize the residue manager."""
        settings = get_settings()
        self.storage_dir = Path(settings.residue_storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.t_level_calculator = TLevelCalculator()

    def parse_residue_file(self, file_content: bytes, b1_in: Optional[int]) -> Dict[str, Any]:
        """
        Parse metadata from a residue file.

        Args:
            file_content: Raw bytes of the residue file
            b1_in: B1 of a residue file that was created with Prime95/mprime

        Returns:
            Dict with keys: composite, b1, parametrization, curve_count

        Raises:
            ValueError: If file format is invalid or missing required fields
        """
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Residue file is not valid UTF-8: {e}")

        raw_lines = content.strip().replace('\r', '').split('\n')
        lines = [line.strip() for line in raw_lines if line.strip() and not line.lstrip().startswith(('[', '#'))]
        if not lines:
            raise ValueError("Residue file is empty")

        # Parse first line to get metadata
        first_line = lines[0]

        # There are multiple valid stage 1 save file formats. The most important ones are:
        # - GMP-ECM's current save format (has a checksum) that can be used with CPU and CUDA GPU
        #  …and…
        # - Prime95/mprime's format, accepted by GMP-ECM as input (from version 22 onward)
        # While using CGBN is useful for small- and medium-sized numbers, Prime95/mprime's fast FFTs shine when larger
        # numbers are being processed and especially when AVX512 is available. Though one should note that
        # Prime95/mprime limits bases to 32-bit numbers and are best when the base is as small as possible and there are
        # no non-trivial cofactors known. (Cofactors do not slow things down, but also do not speed things up unlike
        # with CGBN when going down to a smaller kernel.) Another disadvantage of Prime95/mprime is that it does not
        # include B1.
        # To use this feature, GmpEcmHook=1 needs to be added to prime.txt; output can be found in results.txt, but may
        # have to be stripped of timestamps.

        line_elements = [el.strip() for el in first_line.split(';') if el.strip()]
        split_elements = [el.split('=', 1) for el in line_elements]
        available_keys = [parts[0] for parts in split_elements if len(parts) > 1]

        param_of_residue: int
        param_of_residue = 0 # default for Prime95/mprime residues

        # Common parameters
        if "N" not in available_keys:
            raise ValueError("A residue was missing its N")
        n_str_index = available_keys.index("N")
        n_str: str
        n_str = split_elements[n_str_index][1] # hexadecimal number with prefix OR decimal expression
        curve_count = sum(1 for line in lines if "SIGMA=" in line) # count curves (each line with SIGMA= is a curve)

        # Check if the data comes from Prime95/mprime
        if "QX" in available_keys: # special key for identification
            if b1_in is None:
                raise ValueError("B1 was not set despite processing a Prime95/mprime file")

            b1 = b1_in
            composite = int(n_str, 0) # "base 0" allows for a 0x prefix

        else: # GMP-ECM format
            if "CHECKSUM" not in available_keys:
                raise ValueError("GMP-ECM residues must have a checksum")
            if "PARAM" not in available_keys:
                raise ValueError("A GMP-ECM residue was missing its parametrisation")
            if "B1" not in available_keys:
                raise ValueError("A GMP-ECM residue was missing its B1")

            param_of_residue_index = available_keys.index("PARAM")
            param_of_residue = int(split_elements[param_of_residue_index][1])
            b1_index = available_keys.index("B1")
            b1 = int(split_elements[b1_index][1])

            if n_str.isdecimal(): # pure decimal
                composite = int(n_str)

            elif n_str[:2].lower() == "0x" and n_str[2:].isdecimal(): # hexadecimal
                composite = int(n_str[2:], 16)

            elif not n_str.strip(self.ALLOWED_EVAL_CHARS): # only mathematical characters allowed
                python_expr: str
                python_expr = n_str.replace("^", "**").replace("/", "//")

                # only allow one exponentiation in expression (sufficient for OPN numbers)
                exponentiation_count = python_expr.count("**")
                if exponentiation_count > 1:
                    raise ValueError(f"Only one exponentation per expression allowed")

                elif exponentiation_count > 0:
                    match = self.EXPONENTATION_PATTERN.search(python_expr)
                    if not match:
                        raise ValueError(f"Exponentiations do not support brackets nor negative exponents")

                    # make sure the resulting number is reasonably small
                    base = int(match.group(1))
                    exponent = int(match.group(2))
                    if exponent * math.log10(base) > 100000: # limit of around 100K digits
                        raise ValueError(f"N is too big")

                composite = int(eval(python_expr))

            else:
                raise ValueError(f"N has an unknown format: {n_str}")

            # Check residues for validity
            for line in lines:
                line_elements = [el.strip() for el in line.split(';') if el.strip()]
                split_elements = [el.split('=', 1) for el in line_elements]
                available_keys = [parts[0] for parts in split_elements if len(parts) > 1]
                checksum_index = available_keys.index("CHECKSUM")
                checksum = int(split_elements[checksum_index][1])

                # Calculate checksum of line
                checksum_calc = b1
                sigma_index = available_keys.index("SIGMA")
                sigma = int(split_elements[sigma_index][1])
                # Note: differing from most checksum implementations, GMP-ECM always does a MOD operation first
                #       and a MUL operation second, which look like this:
                #       mpz_mul_ui(checksum_calc, checksum_calc, mpz_fdiv_ui(some_val, CHKCONST))
                checksum_calc *= sigma % self.CHKCONST
                checksum_calc *= composite % self.CHKCONST
                x_index = available_keys.index("X")
                x = int(split_elements[x_index][1], 0) # field X is always in hexadecimal format
                checksum_calc *= x % self.CHKCONST
                checksum_calc *= param_of_residue + 1 # does not need a MOD operation
                checksum_calc %= self.CHKCONST

                if checksum != checksum_calc:
                    raise ValueError("A line did not match its expected checksum")

        if not curve_count:
            raise ValueError("No curves found in residue file")

        return {
            'composite': str(composite),
            'b1': b1,
            'parametrization': param_of_residue,
            'curve_count': curve_count
        }

    def calculate_checksum(self, file_content: bytes) -> str:
        """Calculate SHA-256 checksum of file content."""
        return hashlib.sha256(file_content).hexdigest()

    def _find_duplicate_residue(self, db: Session, checksum: str) -> Optional[ECMResidue]:
        """Pre-insert duplicate lookup (separate method so tests can simulate
        the concurrent-upload race that slips past it)."""
        return db.query(ECMResidue).filter(
            ECMResidue.checksum == checksum
        ).first()

    def store_residue_file(
        self,
        db: Session,
        file_content: bytes,
        client_id: str,
        stage1_attempt_id: Optional[int] = None,
        b1: Optional[int] = None
    ) -> ECMResidue:
        """
        Store a residue file and create database record.

        Residues don't expire by time - they remain available until:
        - The composite is fully factored
        - A stage 2 worker completes processing them

        Args:
            db: Database session
            file_content: Raw residue file bytes
            client_id: ID of the uploading client
            stage1_attempt_id: Optional ID of the stage 1 attempt to link
            b1: B1 of a residue file that was created with Prime95/mprime

        Returns:
            ECMResidue database record

        Raises:
            ValueError: If file format is invalid or composite not found
        """
        # Parse metadata from file
        metadata = self.parse_residue_file(file_content, b1)
        composite_number = metadata['composite']

        # Look up composite in database
        composite = db.query(Composite).filter(
            Composite.current_composite == composite_number
        ).first()

        if not composite:
            # Try matching by number field as fallback
            composite = db.query(Composite).filter(
                Composite.number == composite_number
            ).first()

        if not composite:
            raise ValueError(f"Composite {composite_number[:50]}... not found in database")

        # Generate unique filename
        import uuid
        file_uuid = str(uuid.uuid4())
        composite_dir = self.storage_dir / str(composite.id)
        composite_dir.mkdir(parents=True, exist_ok=True)
        file_path = composite_dir / f"{file_uuid}.txt"

        # Calculate checksum
        checksum = self.calculate_checksum(file_content)

        # Check for duplicate by checksum. Fast path with a friendly error;
        # concurrent identical uploads can both pass it, so the unique
        # constraint on checksum is the authoritative guard below.
        existing = self._find_duplicate_residue(db, checksum)
        if existing:
            raise ValueError(f"Duplicate residue file (checksum matches residue ID {existing.id})")

        # Write file to storage
        file_path.write_bytes(file_content)
        logger.info(f"Stored residue file: {file_path} ({len(file_content)} bytes)")

        # Create database record
        # expires_at is None for available residues (no time-based expiration)
        # It will be set when claimed (claim timeout)
        residue = ECMResidue(
            composite_id=composite.id,
            client_id=client_id,
            stage1_attempt_id=stage1_attempt_id,
            b1=metadata['b1'],
            parametrization=metadata['parametrization'],
            curve_count=metadata['curve_count'],
            storage_path=str(file_path),
            file_size_bytes=len(file_content),
            checksum=checksum,
            status='available',
            expires_at=None
        )

        db.add(residue)
        try:
            db.flush()  # Get the ID
        except IntegrityError as e:
            # The insert failed; remove the file written for it (the
            # caller's rollback discards the row)
            try:
                file_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                logger.error(
                    f"Failed to remove residue file for failed insert "
                    f"{file_path}: {cleanup_err}"
                )
            # Only the checksum constraint means a concurrent duplicate
            # upload; anything else (FK violation, storage-path collision)
            # must surface as what it is, not be misreported as a duplicate.
            if is_unique_violation(
                    e, 'uq_ecm_residues_checksum', 'ecm_residues.checksum'):
                logger.info(
                    f"Concurrent duplicate residue upload from {client_id} "
                    f"(checksum {checksum[:16]}...) - rejecting the loser"
                )
                raise ValueError(
                    "Duplicate residue file (uploaded concurrently by another request)"
                ) from e
            raise

        logger.info(
            f"Created residue record ID {residue.id}: "
            f"composite={composite.id}, B1={metadata['b1']}, "
            f"curves={metadata['curve_count']}, param={metadata['parametrization']}"
        )

        return residue

    def get_available_work(
        self,
        db: Session,
        client_id: str,
        min_target_tlevel: Optional[float] = None,
        max_target_tlevel: Optional[float] = None,
        min_priority: Optional[int] = None,
        min_b1: Optional[int] = None,
        max_b1: Optional[int] = None,
        project: Optional[str] = None,
        exclude_ids: Optional[set] = None
    ) -> Optional[ECMResidue]:
        """
        Find an available residue for stage 2 processing.

        Args:
            db: Database session
            client_id: ID of requesting client
            min_target_tlevel: Minimum target t-level
            max_target_tlevel: Maximum target t-level
            min_priority: Minimum composite priority
            min_b1: Minimum B1 bound of residue
            max_b1: Maximum B1 bound of residue
            project: Optional project name filter (if not set, all projects)
            exclude_ids: Residue IDs to skip - the candidates this request
                already tried and lost (claimed by another consumer, or
                auto-completed). Without this, the selection is unlocked and
                returns the same top-priority row to every concurrent
                consumer, so the caller's retry loop would re-pick the
                contended row and exhaust its budget on it - returning a
                false "no work available" while other residues sit free.

        Returns:
            ECMResidue if found, None otherwise
        """
        # Available residues don't have time-based expiration
        # Only filter by status (and exclude factored composites)
        query = db.query(ECMResidue).join(
            Composite, ECMResidue.composite_id == Composite.id
        ).filter(
            ECMResidue.status == 'available',
            Composite.is_fully_factored == False  # noqa: E712
        )
        if exclude_ids:
            query = query.filter(ECMResidue.id.notin_(exclude_ids))

        # Apply filters
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)
        if min_priority is not None:
            query = query.filter(Composite.priority >= min_priority)
        if min_b1 is not None:
            query = query.filter(ECMResidue.b1 >= min_b1)
        if max_b1 is not None:
            query = query.filter(ECMResidue.b1 <= max_b1)

        # Apply project filter (join through project_composites)
        if project is not None:
            query = query.join(
                ProjectComposite, ProjectComposite.composite_id == Composite.id
            ).join(
                Project, Project.id == ProjectComposite.project_id
            ).filter(Project.name == project)

        # Prioritize by composite priority (descending), then by creation time (oldest first)
        query = query.order_by(
            Composite.priority.desc(),
            ECMResidue.created_at.asc()
        )

        # Unlocked read: claiming locks composite -> residue (the global
        # lock order) and revalidates the status. Locking the residue here
        # and the composite later (stale-attempt completion, t-level recalc)
        # would reverse the order and deadlock against submissions.
        residue = query.first()
        if residue:
            logger.info(f"Found available residue ID {residue.id} for client {client_id}")
        else:
            logger.info(f"No available residues for client {client_id}")

        return residue

    def claim_residue(
        self,
        db: Session,
        residue_id: int,
        client_id: str,
        claim_timeout_hours: int = 72  # 3 days default for large stage 2 work
    ) -> ECMResidue:
        """
        Claim a residue for stage 2 processing.

        Args:
            db: Database session
            residue_id: ID of residue to claim
            client_id: ID of claiming client
            claim_timeout_hours: Hours until claim expires (default 72h/3 days)

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found, not available, or its composite
                was fully factored after selection. Callers treat the
                ValueError as "lost the race, try the next candidate".
        """
        # Lock composite -> residue (global order) and refresh before the
        # availability check - the candidate was selected without a lock, so
        # another client may have claimed it, or a submission may have fully
        # factored its composite, in the meantime.
        residue = self.lock_residue(db, residue_id)
        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'available':
            raise ValueError(f"Residue {residue_id} is not available (status: {residue.status})")

        # Revalidate the composite under lock too: get_available_work filtered
        # out factored composites with an UNLOCKED read, so a concurrent
        # submission could have finished the composite since. Serving its
        # residue would re-run already-dead stage 2 work.
        # db.get() reads the identity map - lock_residue already locked and
        # refreshed this composite row, so no extra query is issued.
        composite = db.get(Composite, residue.composite_id)
        if composite is None or composite.is_fully_factored:
            raise ValueError(
                f"Residue {residue_id}'s composite is fully factored or gone"
            )

        residue.status = 'claimed'
        residue.claimed_at = datetime.utcnow()
        residue.claimed_by = client_id
        # Update expiration to claim timeout
        residue.expires_at = datetime.utcnow() + timedelta(hours=claim_timeout_hours)

        logger.info(f"Residue {residue_id} claimed by {client_id}")
        return residue

    def release_claim(self, db: Session, residue_id: int, client_id: str) -> ECMResidue:
        """
        Release a claimed residue back to available pool.

        Args:
            db: Database session
            residue_id: ID of residue to release
            client_id: ID of client releasing (must match claimer)

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found, not claimed, or wrong client
        """
        # Lock composite -> residue (global order) and refresh before the
        # status check: releasing based on a stale 'claimed' read would
        # overwrite a concurrent completion, recreating an available residue
        # whose file is gone and whose stage 1 is already superseded (the
        # poison-loop state). Ordered via lock_residue so this stays
        # deadlock-safe against submit/complete if it ever grows to touch the
        # composite (e.g. a t-level recalc on release).
        residue = self.lock_residue(db, residue_id)

        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'claimed':
            raise ValueError(f"Residue {residue_id} is not claimed (status: {residue.status})")

        if residue.claimed_by != client_id:
            raise ValueError(f"Residue {residue_id} is claimed by {residue.claimed_by}, not {client_id}")

        residue.status = 'available'
        residue.claimed_at = None
        residue.claimed_by = None
        # Clear expiration - available residues don't expire by time
        residue.expires_at = None

        logger.info(f"Residue {residue_id} released by {client_id}")
        return residue

    def lock_residue(self, db: Session, residue_id: int) -> Optional[ECMResidue]:
        """
        Lock a residue's rows for update in the global lock order and return
        the refreshed residue (or None if it - or its composite - no longer
        exists).

        Lock order is composite -> residue EVERYWHERE that mutates residue
        status or supersession, so concurrent submissions and completions
        can't deadlock by grabbing the pair in opposite orders.

        populate_existing() is load-bearing: with_for_update() acquires the
        row lock but does NOT overwrite attributes already loaded in this
        session, so a status/claim check made after waiting on the lock would
        otherwise still read the pre-lock value.

        Uses .first() (not .one()): a composite/residue deleted concurrently
        (admin delete) returns None for the caller to handle as "lost the
        race", rather than raising NoResultFound -> 500.
        """
        row = db.query(ECMResidue.composite_id).filter(
            ECMResidue.id == residue_id
        ).first()
        if row is None:
            return None
        composite = db.query(Composite).filter(
            Composite.id == row[0]
        ).populate_existing().with_for_update().first()
        if composite is None:
            return None
        return db.query(ECMResidue).filter(
            ECMResidue.id == residue_id
        ).populate_existing().with_for_update().first()

    def completion_rejection_reason(
        self,
        residue: ECMResidue,
        stage2_attempt: ECMAttempt
    ) -> Optional[str]:
        """
        Check whether a stage 2 attempt qualifies to complete a residue.

        Valid means: found a factor, or completed at least 75% of the assigned
        curves with a sufficient B2 (NULL/-1 = GMP-ECM default is accepted, an
        explicit B2 must be at least 100*B1 to be worth consuming the file).

        Returns:
            None if the completion is valid, otherwise a human-readable reason.
        """
        if stage2_attempt.factor_found is not None:
            return None

        min_curves_required = int(0.75 * residue.curve_count)
        if stage2_attempt.curves_completed < min_curves_required:
            return (
                f"no factor found and only {stage2_attempt.curves_completed} curves "
                f"completed out of {residue.curve_count} assigned "
                f"(minimum required: {min_curves_required}, 75%)"
            )

        stage2_b2 = stage2_attempt.b2
        if stage2_b2 is not None and stage2_b2 != -1:
            min_b2 = residue.b1 * 100
            if stage2_b2 < min_b2:
                return (
                    f"B2={stage2_b2} is less than the minimum "
                    f"required {min_b2} (100 * B1={residue.b1})"
                )

        return None

    def find_completing_attempt(
        self,
        db: Session,
        residue: ECMResidue
    ) -> Optional[ECMAttempt]:
        """
        Find an existing stage 2 attempt that already qualifies to complete
        this residue.

        Used to finalize residues whose completion call was lost (the result
        submission was accepted but the residue was never closed out) instead
        of re-serving curves that were already run.

        Returns:
            The best qualifying attempt (factor found preferred, then most
            curves completed, then oldest), or None.
        """
        candidates = db.query(ECMAttempt).filter(
            ECMAttempt.residue_checksum == residue.checksum,
            ECMAttempt.composite_id == residue.composite_id,
            ECMAttempt.superseded_by.is_(None),
            # Exclude stage 1-only attempts (b2=0); NULL b2 means GMP-ECM default
            or_(ECMAttempt.b2.is_(None), ECMAttempt.b2 != 0),
        ).order_by(
            ECMAttempt.factor_found.isnot(None).desc(),
            ECMAttempt.curves_completed.desc(),
            ECMAttempt.id.asc()
        ).all()

        for attempt in candidates:
            if self.completion_rejection_reason(residue, attempt) is None:
                return attempt
        return None

    def _resolve_terminal_attempt(
        self,
        db: Session,
        attempt_id: int,
        max_hops: int = 10
    ) -> Optional[int]:
        """
        Follow the superseded_by chain to the terminal (unsuperseded) attempt.

        The stage-1-designated winner may itself have been superseded later
        (e.g. by reconciliation), so callers must not point new duplicates at
        it directly - that's how supersession cycles form.

        Returns:
            The terminal attempt ID, or None if the chain is broken, cyclic,
            or too long (logged; callers should bail out rather than extend
            a bad chain).
        """
        seen = set()
        current_id = attempt_id
        for _ in range(max_hops):
            if current_id in seen:
                logger.warning(
                    f"Supersession cycle detected at attempt {current_id} "
                    f"(started from {attempt_id})"
                )
                return None
            seen.add(current_id)
            attempt = db.query(ECMAttempt).filter(ECMAttempt.id == current_id).first()
            if attempt is None:
                logger.warning(
                    f"Supersession chain from attempt {attempt_id} references "
                    f"missing attempt {current_id}"
                )
                return None
            if attempt.superseded_by is None:
                return current_id
            current_id = attempt.superseded_by

        logger.warning(
            f"Supersession chain from attempt {attempt_id} exceeded {max_hops} hops"
        )
        return None

    def _handle_already_completed(
        self,
        db: Session,
        residue: ECMResidue,
        stage2_attempt: ECMAttempt,
        recalculate_t_level: bool
    ) -> Tuple[ECMResidue, Optional[float]]:
        """
        Handle a completion request for a residue that is already completed.

        Happens when a client retries after a lost response, or when an older
        client calls /residues/{id}/complete after /submit_result already
        auto-completed the residue. If the retry carries a different
        (duplicate) attempt, supersede it by the original winner so the same
        curves aren't counted twice in the t-level.
        """
        winner_id = None
        if residue.stage1_attempt_id:
            stage1 = db.query(ECMAttempt).filter(
                ECMAttempt.id == residue.stage1_attempt_id
            ).first()
            if stage1 and stage1.superseded_by is not None:
                # Resolve to the terminal winner - the designated winner may
                # itself have been superseded since the residue completed
                winner_id = self._resolve_terminal_attempt(db, stage1.superseded_by)

        new_t_level = None
        if (winner_id is not None and winner_id != stage2_attempt.id
                and stage2_attempt.superseded_by is None):
            stage2_attempt.superseded_by = winner_id
            db.flush()
            logger.info(
                f"Residue {residue.id} already completed by attempt {winner_id}; "
                f"superseded duplicate attempt {stage2_attempt.id}"
            )
            if recalculate_t_level:
                new_t_level = self._recalculate_composite_t_level(db, residue.composite_id)
        else:
            logger.info(
                f"Residue {residue.id} already completed - treating completion "
                f"of attempt {stage2_attempt.id} as idempotent retry"
            )

        return residue, new_t_level

    def complete_residue(
        self,
        db: Session,
        residue_id: int,
        stage2_attempt_id: int,
        recalculate_t_level: bool = True
    ) -> Tuple[ECMResidue, Optional[float]]:
        """
        Mark residue as completed after stage 2 finishes.

        This supersedes the stage 1 attempt and deletes the residue file.
        Calling it again for an already-completed residue is an idempotent
        no-op (a duplicate attempt from a retry is superseded by the winner).

        Args:
            db: Database session
            residue_id: ID of the completed residue
            stage2_attempt_id: ID of the stage 2 ECM attempt
            recalculate_t_level: If False, skip the t-level recalculation;
                for callers that recalculate themselves in the same
                transaction (e.g. /submit_result auto-completion)

        Returns:
            Tuple of (residue, new_t_level)
            - new_t_level: Updated t-level after supersession (if applicable)

        Raises:
            ValueError: If residue or attempt not found, or completion invalid
        """
        # Serialize completions of this residue under the global lock order
        # (composite -> residue). Without this, two concurrent completions
        # can interleave the status check and the supersession updates,
        # forming A->B, B->A supersession cycles that exclude both attempts
        # from the t-level.
        residue = self.lock_residue(db, residue_id)
        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        # Get the stage 2 attempt
        stage2_attempt = db.query(ECMAttempt).filter(ECMAttempt.id == stage2_attempt_id).first()
        if not stage2_attempt:
            raise ValueError(f"Stage 2 attempt {stage2_attempt_id} not found")

        # An attempt for a different composite must never complete this
        # residue - superseding stage 1 with it corrupts both composites'
        # t-levels. Caller error: raise without releasing the residue.
        if stage2_attempt.composite_id != residue.composite_id:
            raise ValueError(
                f"Stage 2 attempt {stage2_attempt_id} belongs to composite "
                f"{stage2_attempt.composite_id}, not composite "
                f"{residue.composite_id} of residue {residue_id}"
            )

        # ...and it must be THIS residue's stage 2: a composite can have
        # several residues, and an attempt that consumed a sibling residue
        # must not supersede this one's stage 1 (each residue's curves are
        # distinct work). Same rule: raise without releasing the residue.
        if stage2_attempt.residue_checksum != residue.checksum:
            raise ValueError(
                f"Stage 2 attempt {stage2_attempt_id} does not carry the "
                f"checksum of residue {residue_id}; it cannot complete it"
            )

        # Idempotent retry: don't redo supersession/file deletion, and never
        # reject (releasing a completed residue would re-serve finished work)
        if residue.status == 'completed':
            return self._handle_already_completed(
                db, residue, stage2_attempt, recalculate_t_level
            )

        # Validate that this is a legitimate stage 2 completion:
        # must find a factor OR complete >= 75% of assigned curves with sane B2
        rejection = self.completion_rejection_reason(residue, stage2_attempt)
        if rejection:
            # Raise WITHOUT mutating the residue. The caller's transaction
            # rolls back on this ValueError, so any status change here would
            # be undone anyway - earlier code set status='available' and then
            # claimed it was released, but the rollback reverted it, leaving
            # the residue claimed while the response said "released". The
            # claim simply stays put and expires via cleanup_expired_claims.
            logger.warning(
                f"Rejected residue {residue_id} completion: stage2_attempt {stage2_attempt_id} "
                f"{rejection}. Claim left in place (will expire if not retried)."
            )
            raise ValueError(f"Invalid stage 2 completion: {rejection}")

        # Mark stage 1 attempt as superseded (if linked)
        if residue.stage1_attempt_id:
            stage1_attempt = db.query(ECMAttempt).filter(
                ECMAttempt.id == residue.stage1_attempt_id
            ).first()
            if stage1_attempt:
                stage1_attempt.superseded_by = stage2_attempt_id
                db.flush()  # Ensure supersession is visible to subsequent queries
                logger.info(
                    f"Marked stage 1 attempt {residue.stage1_attempt_id} as superseded by {stage2_attempt_id}"
                )

        # Update residue status
        residue.status = 'completed'
        residue.completed_at = datetime.utcnow()
        residue.expires_at = None

        # Delete the residue file only AFTER this transaction commits: an
        # inline unlink would leave the file gone if the transaction later
        # rolls back (reverting status to 'claimed' with no file behind it).
        stage_residue_file_deletion(db, residue.storage_path)

        # Mark orphaned attempts from the same residue as superseded
        # These are partial attempts that were submitted but failed to complete the residue
        # (e.g., client interrupted, then another client completed the same residue)
        orphaned_attempts = db.query(ECMAttempt).filter(
            ECMAttempt.residue_checksum == residue.checksum,
            # Same composite only: the checksum alone could match another
            # composite's attempts if the same number is registered twice
            ECMAttempt.composite_id == residue.composite_id,
            ECMAttempt.id != stage2_attempt_id,
            ECMAttempt.superseded_by.is_(None)  # Not already superseded
        ).all()

        for orphan in orphaned_attempts:
            orphan.superseded_by = stage2_attempt_id
            logger.info(
                f"Marked orphaned attempt {orphan.id} as superseded by {stage2_attempt_id} "
                f"(same residue checksum {residue.checksum[:16]}...)"
            )

        if orphaned_attempts:
            db.flush()  # Ensure supersession is visible to t-level calculation

        # Recalculate t-level for composite (excluding superseded attempts)
        new_t_level = None
        if recalculate_t_level:
            new_t_level = self._recalculate_composite_t_level(db, residue.composite_id)

        logger.info(
            f"Completed residue {residue_id}: stage2_attempt={stage2_attempt_id}, "
            f"new_t_level={new_t_level}"
        )

        return residue, new_t_level

    def _recalculate_composite_t_level(self, db: Session, composite_id: int) -> Optional[float]:
        """
        Recalculate t-level for a composite, excluding superseded attempts.

        Args:
            db: Database session
            composite_id: ID of composite to recalculate

        Returns:
            New t-level value, or None if calculation fails
        """
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return None

        # Use the t-level calculator's recalculate method which already
        # excludes superseded attempts
        old_t_level = composite.current_t_level or 0.0
        new_t_level = self.t_level_calculator.recalculate_composite_t_level(db, composite)

        logger.info(
            f"Recalculated t-level for composite {composite_id}: "
            f"{old_t_level:.2f} -> {new_t_level:.2f}"
        )

        return new_t_level

    def get_residue_file_path(self, db: Session, residue_id: int) -> Optional[Path]:
        """
        Get the filesystem path for a residue file.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Path to file, or None if not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        file_path = Path(residue.storage_path)
        if file_path.exists():
            return file_path

        logger.warning(f"Residue file not found: {file_path}")
        return None

    def suggest_b2_for_residue(self, db: Session, residue_id: int) -> Optional[int]:
        """
        Suggest an appropriate B2 value for stage 2 based on B1.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Suggested B2 value, or None if residue not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        # Standard B2 = 500 * B1 as default heuristic
        # For GPU work, even larger ratios can be beneficial
        suggested_b2 = residue.b1 * 500

        # Cap at a reasonable maximum (e.g., 10 trillion)
        max_b2 = 10_000_000_000_000
        suggested_b2 = min(suggested_b2, max_b2)

        return suggested_b2

    def cleanup_expired_claims(self, db: Session) -> int:
        """
        Release claims that have timed out (claimed but not completed in time).

        Only claimed residues have expiration times. Available residues don't expire.
        This releases the claim so another worker can pick up the work.

        Args:
            db: Database session

        Returns:
            Number of claims released
        """
        # Single conditional UPDATE: the status predicate is re-evaluated on
        # the current row version at write time, so a residue completed
        # between any earlier read and this statement is simply skipped. A
        # read-then-write here could overwrite 'completed' with 'available'
        # after the file was deleted and stage 1 superseded.
        count = db.query(ECMResidue).filter(
            ECMResidue.status == 'claimed',
            ECMResidue.expires_at < datetime.utcnow()
        ).update(
            {
                ECMResidue.status: 'available',
                ECMResidue.claimed_at: None,
                ECMResidue.claimed_by: None,
                ECMResidue.expires_at: None,
            },
            synchronize_session=False
        )

        if count > 0:
            logger.info(f"Released {count} expired claims")

        return count

    def cleanup_factored_composites(self, db: Session) -> int:
        """
        Clean up residues for composites that have been fully factored or completed.

        Residues are no longer useful once their composite is factored or marked complete.

        Args:
            db: Database session

        Returns:
            Number of residues cleaned up
        """
        # Find residues for fully factored or completed composites
        residues_to_cleanup = db.query(ECMResidue).join(
            Composite, ECMResidue.composite_id == Composite.id
        ).filter(
            or_(
                Composite.is_fully_factored == True,  # noqa: E712
                Composite.is_complete == True,  # noqa: E712
            ),
            ECMResidue.status.in_(['available', 'claimed'])
        ).all()

        count = 0
        for residue in residues_to_cleanup:
            try:
                if not transition_residue_status(
                        db, residue.id, ['available', 'claimed'], 'expired'):
                    continue

                # Defer deletion to after commit (consistent with
                # complete_residue): if this transaction rolls back, the
                # status reverts to available/claimed and the file must stay.
                stage_residue_file_deletion(db, residue.storage_path)
                count += 1
            except Exception as e:
                logger.error(f"Error cleaning up residue {residue.id}: {e}")

        if count > 0:
            logger.info(f"Cleaned up {count} residues for factored/completed composites")

        return count

    def get_stats(self, db: Session) -> Dict[str, int]:
        """
        Get statistics about residues in the system.

        Args:
            db: Database session

        Returns:
            Dict with counts by status and total pending curves
        """
        stats = {
            'total_available': 0,
            'total_claimed': 0,
            'total_completed': 0,
            'total_expired': 0,
            'total_curves_pending': 0
        }

        # Count by status
        status_counts = db.query(
            ECMResidue.status,
            func.count(ECMResidue.id)
        ).group_by(ECMResidue.status).all()

        for status, count in status_counts:
            if status == 'available':
                stats['total_available'] = count
            elif status == 'claimed':
                stats['total_claimed'] = count
            elif status == 'completed':
                stats['total_completed'] = count
            elif status == 'expired':
                stats['total_expired'] = count

        # Sum curves in available residues
        curves_sum = db.query(func.sum(ECMResidue.curve_count)).filter(
            ECMResidue.status == 'available'
        ).scalar()
        stats['total_curves_pending'] = curves_sum or 0

        return stats
