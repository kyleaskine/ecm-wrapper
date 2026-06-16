from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session, load_only
from sqlalchemy.exc import IntegrityError
from typing import Literal, Optional
import logging
from slowapi import Limiter

from ...rate_limit import get_real_client_ip
from ...database import get_db
from ...dependencies import get_composite_service, get_residue_manager
from ...schemas.submit import SubmitResultRequest, SubmitResultResponse
from ...models import Composite, ECMAttempt
from ...models.factors import Factor
from ...models.residues import ECMResidue
from ...models.work_assignments import WorkAssignment
from ...services.composites import CompositeService
from ...services.factors import FactorService
from ...services.residue_manager import ResidueManager
from ...utils.number_utils import (
    is_trivial_factor,
    verify_factor_divides,
    parse_sigma_with_parametrization,
    is_stale_ancestor_state,
    validate_integer,
)
from ...utils.transactions import transaction_scope, is_unique_violation

import math

logger = logging.getLogger(__name__)

router = APIRouter()


def _matches_composite_state(db: Session, composite: Composite, submitted: str) -> bool:
    """
    True if `submitted` is the composite's current cofactor or a genuine
    earlier state. Anything else - e.g. current * <unrelated number> - is a
    fabricated state and must not resolve.

    When the original number is numeric, this is exact arithmetic with no
    extra queries: a genuine prior state divides the original number and is
    divisible by the current cofactor (current | submitted | number). That
    handles repeated prime factors correctly - current * P^2 verifies iff
    P^2 really divides the original.

    Expression-form numbers (e.g. "2^1223-1") fall back to decomposing the
    quotient into recorded factors, one division per recorded row.
    """
    if submitted == composite.current_composite:
        return True

    if validate_integer(composite.number):
        if not validate_integer(submitted) or not validate_integer(composite.current_composite):
            return False
        number_int = int(composite.number)
        submitted_int = int(submitted)
        current_int = int(composite.current_composite)
        return (
            current_int > 0
            and submitted_int > 0  # "0" passes validate_integer; guard the modulo
            and submitted_int % current_int == 0
            and number_int % submitted_int == 0
        )

    known_factors = [
        row[0] for row in db.query(Factor.factor).filter(
            Factor.composite_id == composite.id
        ).all()
    ]
    return is_stale_ancestor_state(submitted, composite.current_composite, known_factors)


def _resolve_composite(db: Session, result_request: SubmitResultRequest, client_ip: str) -> Composite:
    """
    Resolve the submitted composite string to a Composite row.

    SECURITY: only composites already in the database can receive
    submissions - this prevents accidental pollution from local testing when
    users forget --no-submit.

    Resolution order:
    1. Residue checksum (AUTHORITATIVE): the checksum pins the composite the
       residue belongs to, whereas current_composite is not unique - a stale
       submitted string can collide with another composite's current value.
       A checksum whose composite doesn't match the submitted state is
       rejected outright rather than falling through to a string lookup that
       could attribute the work to the wrong row.
    2. Work assignment ID (AUTHORITATIVE): the server-issued UUID pins the
       composite the work was assigned for. A client/method mismatch on an
       existing assignment is rejected (403), and a submitted number that
       isn't a state of the assigned composite is rejected (400) - failing
       open to string resolution could attribute the work to a colliding
       composite. Only an unknown work_id (assignment already cleaned up)
       falls through.
    3. Direct indexed lookups, collected together for ambiguity detection:
       exact current_composite match (the normal case) and the original
       number, state-verified (stale work from before a factor was divided
       out mid-run). More than one distinct match is a 409 - silently
       picking one could attribute work to the wrong composite.
    4. Exact-ancestry scan over composites with recorded factors, for
       intermediate stale states (the worker was assigned an already-reduced
       cofactor and yet another factor landed). Expensive - O(composites
       with factors) big-int arithmetic in Python - so it runs ONLY when
       the direct lookups found nothing; clients that send a residue
       checksum or work_id never reach it.

    Raises:
        HTTPException: 400 (checksum/state mismatch), 409 (ambiguous
        intermediate state), 404 (no match)
    """
    if result_request.residue_checksum:
        checksum_residue = db.query(ECMResidue).filter(
            ECMResidue.checksum == result_request.residue_checksum
        ).first()
        if checksum_residue:
            candidate = db.query(Composite).filter(
                Composite.id == checksum_residue.composite_id
            ).first()
            if candidate is None or not _matches_composite_state(
                    db, candidate, result_request.composite):
                logger.warning(
                    f"Submission rejected from {client_ip}: residue checksum "
                    f"belongs to composite {checksum_residue.composite_id} but "
                    f"the submitted number is not a state of that composite"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Submitted number does not match the composite the residue checksum belongs to"
                )
            if result_request.composite != candidate.current_composite:
                logger.info(
                    f"Submission from {result_request.client_id} references "
                    f"a stale state of composite {candidate.id} (resolved "
                    f"via residue checksum) - accepting"
                )
            return candidate

    if result_request.work_id:
        assignment = db.query(WorkAssignment).filter(
            WorkAssignment.id == result_request.work_id
        ).first()
        # A 'p1' assignment is fulfilled by separate pm1 and pp1 submissions
        method_ok = assignment is not None and (
            assignment.method == result_request.method
            or (assignment.method == 'p1'
                and result_request.method in ('pm1', 'pp1'))
        )
        if assignment is not None and (
                assignment.client_id != result_request.client_id or not method_ok):
            # Fail closed: falling through to string resolution would let a
            # submission carrying a known-invalid assignment be attributed
            # to whatever composite its string happens to collide with
            logger.warning(
                f"Submission rejected from {client_ip}: work "
                f"{result_request.work_id} is assigned to "
                f"{assignment.client_id} for method {assignment.method}, but "
                f"{result_request.client_id} submitted method "
                f"{result_request.method}"
            )
            raise HTTPException(
                status_code=403,
                detail="work_id belongs to a different client or method"
            )
        if assignment is not None:
            candidate = db.query(Composite).filter(
                Composite.id == assignment.composite_id
            ).first()
            if candidate is None or not _matches_composite_state(
                    db, candidate, result_request.composite):
                logger.warning(
                    f"Submission rejected from {client_ip}: work "
                    f"{result_request.work_id} pins composite "
                    f"{assignment.composite_id} but the submitted number is "
                    f"not a state of that composite"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Submitted number does not match the composite the work_id was assigned for"
                )
            if result_request.composite != candidate.current_composite:
                logger.info(
                    f"Submission from {result_request.client_id} references "
                    f"a stale state of composite {candidate.id} (resolved "
                    f"via work_id) - accepting"
                )
            return candidate
        # Unknown work_id (assignment already cleaned up): fall through

    # Without a checksum, collect EVERY composite the submitted value could
    # belong to via the direct (indexed) lookups: exact current matches plus
    # a state-verified match on the original number. A single candidate
    # resolves; more than one is ambiguous and must be rejected - silently
    # picking one could attribute work (and divide factors out of) the wrong
    # composite, e.g. when a cofactor is also registered as its own composite.
    matches = {}
    for candidate in db.query(Composite).filter(
        Composite.current_composite == result_request.composite
    ).all():
        matches[candidate.id] = candidate

    # Original-number lookup, state-verified. This must run even when an
    # exact current match exists: a factorless imported composite (current
    # value set administratively, no Factor rows) is invisible to the
    # ancestry scan below, so without this lookup its original number
    # colliding with another composite's current value would silently
    # resolve to the other composite. Also covers expression-form numbers
    # (e.g. "2^1223-1") the scan can't verify arithmetically.
    for candidate in db.query(Composite).filter(
        Composite.number == result_request.composite
    ).all():
        if candidate.id not in matches and _matches_composite_state(
                db, candidate, result_request.composite):
            matches[candidate.id] = candidate

    if not matches and validate_integer(result_request.composite):
        # Prior-state ancestry scan, ONLY when every direct lookup missed:
        # per candidate it costs two big-int modulos in Python (plus a
        # factor query for expression-form numbers), so running it on every
        # submission would be an O(factored composites) tax on the hot path
        # and a DoS lever. digit_length prunes candidates SQL-side (a
        # cofactor can't have more digits than its ancestor).
        # Trade-off: a value that is one composite's exact CURRENT state and
        # another's verified PRIOR state resolves to the exact match instead
        # of 409ing - acceptable, because the factor still verifies against
        # the exact match's actual current value, so the division is
        # arithmetically sound; only stale-credit attribution is forgone.
        has_factors = db.query(Factor.id).filter(
            Factor.composite_id == Composite.id
        ).exists()
        scan_query = db.query(Composite).options(
            load_only(Composite.id, Composite.number, Composite.current_composite)
        ).filter(
            has_factors,
            Composite.digit_length <= len(result_request.composite)
        ).order_by(Composite.id)

        for candidate in scan_query:
            if _matches_composite_state(db, candidate, result_request.composite):
                matches[candidate.id] = candidate

    if len(matches) > 1:
        logger.warning(
            f"Submission rejected from {client_ip}: submitted number matches "
            f"multiple composites (current or prior states: "
            f"{sorted(matches.keys())}) - cannot attribute"
        )
        raise HTTPException(
            status_code=409,
            detail="Submitted number matches multiple composites (current or prior states); cannot attribute work unambiguously"
        )
    if matches:
        composite = next(iter(matches.values()))
        if result_request.composite != composite.current_composite:
            logger.info(
                f"Submission from {result_request.client_id} references a stale "
                f"state of composite {composite.id} (factor divided out mid-run) "
                f"- accepting"
            )
        return composite

    logger.warning(
        f"Submission rejected from {client_ip}: composite not in database "
        f"({result_request.composite[:20]}...)"
    )
    raise HTTPException(
        status_code=404,
        detail="Composite not found in database. Only registered composites can receive submissions. Use --no-submit for local testing."
    )


def _lock_composite_row(db: Session, composite_id: int) -> Composite:
    """
    Lock the composite row and refresh it in the identity map.

    populate_existing() is load-bearing: with_for_update() alone acquires
    the row lock but does NOT overwrite attributes already loaded in this
    session, so checks made after waiting on the lock would still see the
    pre-lock state.

    Lock order is composite -> residue everywhere (deadlock avoidance).

    Raises HTTPException(409) - not NoResultFound -> 500 - if the composite
    was deleted between resolution and this lock (admin delete races a
    submission in flight).
    """
    composite = db.query(Composite).filter(
        Composite.id == composite_id
    ).populate_existing().with_for_update().first()
    if composite is None:
        raise HTTPException(
            status_code=409,
            detail="Composite no longer exists (deleted concurrently); please resubmit"
        )
    return composite


def _effective_work_id(db: Session, work_id: Optional[str]) -> Optional[str]:
    """
    Return work_id only if it references a real assignment, else None.

    The work hash must not include an unknown (made-up or already cleaned-up)
    work_id: doing so let two identical submissions carrying different
    nonexistent IDs hash differently and each record a fresh attempt -
    double-counting the same ECM work in the t-level. An assignment that
    exists but belongs to another client/method is already rejected during
    composite resolution, so "exists" is the right gate here.
    """
    if not work_id:
        return None
    exists = db.query(WorkAssignment.id).filter(
        WorkAssignment.id == work_id
    ).first()
    return work_id if exists else None


def _find_duplicate_attempt(
    db: Session,
    work_hash: str,
    composite_id: int,
    residue_checksum: Optional[str],
) -> Optional[ECMAttempt]:
    """
    Pre-insert duplicate lookup (a separate function so tests can simulate
    the concurrent-submission race that slips past it - the same seam pattern
    as residue uploads).

    composite_id and residue_checksum are already folded into work_hash;
    filtering on them again guards against legacy rows hashed under the old
    identity-free scheme.
    """
    query = db.query(ECMAttempt).filter(
        ECMAttempt.work_hash == work_hash,
        ECMAttempt.composite_id == composite_id
    )
    if residue_checksum:
        query = query.filter(ECMAttempt.residue_checksum == residue_checksum)
    return query.first()


def _duplicate_response(
    db: Session,
    residue_manager: ResidueManager,
    composite_service: CompositeService,
    existing_attempt: ECMAttempt,
    composite: Composite,
    linked_residue: Optional[ECMResidue],
    method: str,
    client_id: str,
) -> SubmitResultResponse:
    """
    Build the response for a submission that duplicates an existing attempt.

    Used both by the pre-insert duplicate check and the insert-race recovery
    (two identical submissions racing on the unique work_hash). A retry whose
    first submission landed but whose bundled residue completion did not must
    still close out the residue here - returning a bare "duplicate" would
    leave it claimed until expiry.
    """
    residue_completed = False
    t_level_updated = False
    if linked_residue is not None:
        residue_completed = _try_complete_residue(
            db, residue_manager, linked_residue, existing_attempt, client_id
        )
        if residue_completed and method == 'ecm':
            try:
                # Composite lock already held (taken by _try_complete_residue)
                _lock_composite_row(db, composite.id)
                composite_service.update_t_level(db, composite.id)
                t_level_updated = True
            except Exception as e:
                logger.warning(
                    f"Failed to update t-level for composite {composite.id}: {e}"
                )
    return SubmitResultResponse(
        status="success",
        attempt_id=existing_attempt.id,
        composite_id=composite.id,
        message="Duplicate work detected - using existing attempt",
        factor_status="duplicate",
        residue_completed=residue_completed,
        # Only report a t-level if the recalc actually ran; reporting
        # current_t_level after a swallowed recalc error would be a stale value
        new_t_level=composite.current_t_level if t_level_updated else None
    )


def _try_complete_residue(
    db: Session,
    residue_manager: ResidueManager,
    linked_residue: ECMResidue,
    attempt: ECMAttempt,
    client_id: str,
) -> bool:
    """
    Complete the residue this attempt consumed, in the same transaction as
    the submission (supersedes stage 1 + any orphaned duplicate attempts).
    This closes the window where the result was accepted but the separate
    completion call never arrived. Old clients still call
    /residues/{id}/complete afterwards; that's now an idempotent retry.

    Returns True if the residue ended up completed; failure is non-fatal
    (the client's separate completion call remains the fallback).
    """
    try:
        # Re-read the residue under a row lock (composite first - the lock
        # order used everywhere) so the status/claim checks below and the
        # supersession updates in complete_residue can't interleave with a
        # concurrent completion of the same residue. populate_existing()
        # refreshes identity-map state that predates the lock.
        _lock_composite_row(db, linked_residue.composite_id)
        # .first() (not .one()): a concurrent delete returns None to handle
        # here, consistent with lock_residue / _lock_composite_row, rather
        # than relying on the broad except below to swallow a NoResultFound.
        locked = db.query(ECMResidue).filter(
            ECMResidue.id == linked_residue.id
        ).populate_existing().with_for_update().first()
        if locked is None:
            return False
        linked_residue = locked

        # The attempt must provably belong to this residue: same composite
        # and same file checksum. A duplicate-detection hit from before
        # identity was part of the work hash can hand us another composite's
        # attempt - that must never supersede this residue's stage 1.
        if (attempt.composite_id != linked_residue.composite_id
                or attempt.residue_checksum != linked_residue.checksum):
            logger.warning(
                f"Not auto-completing residue {linked_residue.id}: attempt "
                f"{attempt.id} (composite {attempt.composite_id}, checksum "
                f"{(attempt.residue_checksum or 'none')[:16]}...) does not match "
                f"the residue (composite {linked_residue.composite_id})"
            )
            return False

        # 'available' is completable too: a lapsed claim (released by expiry
        # cleanup) shouldn't force the work to be redone, and the checksum
        # match proves this client had the file. A residue claimed by a
        # DIFFERENT client is left alone.
        claim_ok = (
            linked_residue.status == 'available'
            or (linked_residue.status == 'claimed'
                and linked_residue.claimed_by == client_id)
        )
        if (claim_ok
                and residue_manager.completion_rejection_reason(linked_residue, attempt) is None):
            residue_manager.complete_residue(
                db, linked_residue.id, attempt.id, recalculate_t_level=False
            )
            logger.info(
                f"Auto-completed residue {linked_residue.id} with attempt "
                f"{attempt.id} for client {client_id}"
            )
            return True
        if linked_residue.status == 'completed':
            # Resubmission after a lost response: supersedes this duplicate
            # attempt so the curves aren't counted twice
            residue_manager.complete_residue(
                db, linked_residue.id, attempt.id, recalculate_t_level=False
            )
            return True
    except Exception as e:
        # The submission is still valid even if completion fails
        logger.warning(f"Failed to auto-complete residue {linked_residue.id}: {e}")
    return False


# Initialize rate limiter - 30 submissions per minute per IP
# Stage 2 consumers can submit rapidly when processing multiple residues
limiter = Limiter(key_func=get_real_client_ip)

@router.post("/submit_result", response_model=SubmitResultResponse)
@limiter.limit("30/minute")
def submit_result(
    result_request: SubmitResultRequest,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Submit factorization attempt result.

    This endpoint accepts results from distributed clients running ECM, P±1, and other
    factorization methods. It handles:
    - Validating composite exists in database (rejects unknown composites)
    - Recording the factorization attempt
    - Adding newly discovered factors
    - Idempotency for duplicate submissions
    - Factor validation

    Note: Only composites already registered in the database can receive submissions.
    Use --no-submit flag for local testing.
    """
    # Resolve real client IP (proxy-aware) up front so except blocks can log it
    client_ip = get_real_client_ip(request)

    with transaction_scope(db, "submit_result"):
        try:
            composite = _resolve_composite(db, result_request, client_ip)

            # Get parametrization from explicit parameter or parse from sigma string
            explicit_parametrization = result_request.parameters.parametrization
            sigma = None
            parametrization = explicit_parametrization

            # Parse sigma string (format: "3:123456" or just "123456")
            if result_request.parameters.sigma:
                sigma, parsed_param = parse_sigma_with_parametrization(
                    result_request.parameters.sigma,
                    default_parametrization=3
                )
                # Use explicit parametrization if provided, otherwise use parsed value
                if explicit_parametrization is None:
                    parametrization = parsed_param

            # Final validation of parametrization
            if parametrization is not None and parametrization not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid parametrization {parametrization}. Must be 0, 1, 2, or 3.")

            # Validate residue_checksum BEFORE it feeds the work hash: an
            # unknown checksum is normalized to None here and the normalized
            # value used for hashing, duplicate lookup, and storage. Hashing
            # the raw value while storing None made an identical retry hit a
            # unique-hash violation (500), and a different bogus checksum
            # produced a fresh hash - double t-level credit for the same work.
            residue_checksum = result_request.residue_checksum
            linked_residue = None
            if residue_checksum:
                linked_residue = db.query(ECMResidue).filter(
                    ECMResidue.checksum == residue_checksum,
                    ECMResidue.composite_id == composite.id
                ).first()
                if not linked_residue:
                    logger.warning(
                        f"Invalid residue_checksum {residue_checksum[:16]}... from {client_ip} "
                        f"for composite {composite.id} - not linking to residue"
                    )
                    # Don't fail submission - just don't link it (could be manual work)
                    residue_checksum = None

            # Generate work hash for duplicate detection
            # Convert sigma to int for hash generation (it can be a large number string)
            sigma_int = int(sigma) if sigma is not None else None
            # Normalize work_id the same way residue_checksum is: an unknown
            # ID is dropped from the hash so it can't fragment the identity.
            effective_work_id = _effective_work_id(db, result_request.work_id)
            work_hash = ECMAttempt.generate_work_hash(
                result_request.composite,
                result_request.method,
                result_request.parameters.b1,
                result_request.parameters.b2,
                parametrization,
                sigma_int,
                result_request.parameters.curves,
                # Identity, not just the submitted string: the same value can
                # be a state of two composites, and without these a duplicate
                # lookup can return another composite's attempt
                composite_id=composite.id,
                residue_checksum=residue_checksum,
                work_id=effective_work_id
            )

            # Check for existing work (the pre-insert fast path).
            existing_attempt = _find_duplicate_attempt(
                db, work_hash, composite.id, residue_checksum
            )
            if existing_attempt:
                return _duplicate_response(
                    db, residue_manager, composite_service, existing_attempt,
                    composite, linked_residue, result_request.method,
                    result_request.client_id
                )

            # Create attempt record with IP logging
            attempt = ECMAttempt(
                composite_id=composite.id,
                client_id=result_request.client_id,
                method=result_request.method,
                b1=result_request.parameters.b1,
                b2=result_request.parameters.b2,
                parametrization=parametrization,
                curves_requested=result_request.parameters.curves or 0,
                curves_completed=result_request.results.curves_completed,
                factor_found=result_request.results.factor_found,
                execution_time_seconds=result_request.results.execution_time,
                program=result_request.program,
                program_version=result_request.program_version,
                raw_output=result_request.raw_output,
                work_hash=work_hash,
                client_ip=client_ip,
                residue_checksum=residue_checksum
            )

            # Insert inside a savepoint so a lost race on the unique work_hash
            # (two identical submissions in flight at once) is recoverable:
            # roll back to the savepoint, load the winner, and take the
            # duplicate path instead of surfacing a 500.
            try:
                with db.begin_nested():
                    db.add(attempt)
                    db.flush()  # Get ID without committing the outer transaction
            except IntegrityError as e:
                if not is_unique_violation(e, 'work_hash'):
                    raise
                # work_hash is globally unique, so the winner is THE row with
                # this hash (no composite/checksum filter needed).
                existing_attempt = db.query(ECMAttempt).filter(
                    ECMAttempt.work_hash == work_hash
                ).first()
                if existing_attempt is None:
                    # Constraint fired but the row isn't visible - unexpected;
                    # re-raise rather than silently swallow.
                    raise
                logger.info(
                    f"Concurrent identical submission for composite {composite.id} "
                    f"(work_hash {work_hash[:16]}...) - using existing attempt "
                    f"{existing_attempt.id}"
                )
                return _duplicate_response(
                    db, residue_manager, composite_service, existing_attempt,
                    composite, linked_residue, result_request.method,
                    result_request.client_id
                )
            db.refresh(attempt)

            # Handle factor discovery - support both single and multiple factors
            factor_status: Literal["new_factor", "known_factor", "no_factor", "duplicate"] = "no_factor"
            factors_to_process = []

            # Collect factors from new or legacy format
            if result_request.results.factors_found:
                # New format: multiple factors with sigmas
                logger.info(f"Received {len(result_request.results.factors_found)} factors in batch submission")
                for factor_with_sigma in result_request.results.factors_found:
                    factors_to_process.append((factor_with_sigma.factor, factor_with_sigma.sigma))
                logger.info(f"Processing {len(factors_to_process)} factors: {[f[:20] + '...' for f, _ in factors_to_process]}")
            elif result_request.results.factor_found:
                # Legacy format: single factor with sigma from parameters
                logger.info(f"Received single factor (legacy format): {result_request.results.factor_found[:20]}...")
                # sigma is already a string (or None), no need to convert
                factors_to_process.append((result_request.results.factor_found, sigma))

            # Process all factors in batch
            if factors_to_process:
                # Serialize cofactor updates: without a row lock, two
                # concurrent submissions can both read P*Q*R, divide out P
                # and Q respectively, and the last write wins - leaving
                # recorded factors inconsistent with current_composite.
                composite = _lock_composite_row(db, composite.id)
                # Revalidate against the refreshed row: another request may
                # have changed the composite while we waited on the lock
                if not _matches_composite_state(db, composite, result_request.composite):
                    raise HTTPException(
                        status_code=409,
                        detail="Composite state changed concurrently; please resubmit"
                    )

                new_factors_count = 0
                known_factors_count = 0

                # Validate and add all factors BEFORE updating composite
                # First pass: calculate running cofactor to identify final prime.
                # Start from the composite's CURRENT state, not the submitted
                # string - a stale submission references an earlier state and
                # its already-divided factors must not be divided out again.
                running_cofactor = composite.current_composite
                factors_to_add = []  # Only factors that aren't the final prime
                stale_known_count = 0  # Re-found factors already divided out

                from ...utils.number_utils import divide_factor

                for factor_str, factor_sigma in factors_to_process:
                    # Check if it's a trivial factor
                    if is_trivial_factor(factor_str, result_request.composite):
                        continue  # Skip trivial factors

                    if not validate_integer(factor_str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid factor: {factor_str} is not a valid number"
                        )

                    # A factor must be > 1: gcd(0, n) == n, so "0" would
                    # otherwise claim the entire current cofactor
                    factor_int = int(factor_str)
                    if factor_int <= 1:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid factor: {factor_str}"
                        )

                    # SECURITY: every reported factor must be a proper divisor
                    # of the submitted state (which was itself verified as
                    # genuine at composite resolution). The one exception is a
                    # re-found already-recorded factor, which won't divide a
                    # post-division submitted string. Values like
                    # 2 * current_composite divide neither and are rejected.
                    if not verify_factor_divides(factor_str, result_request.composite):
                        already_known = db.query(Factor).filter(
                            Factor.composite_id == composite.id,
                            Factor.factor == factor_str
                        ).first() is not None
                        if not already_known:
                            logger.warning(
                                f"Invalid factor submitted by client {result_request.client_id} "
                                f"from IP {client_ip}: factor {factor_str[:20]}... does not divide "
                                f"composite {result_request.composite[:20]}..."
                            )
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid factor: {factor_str} does not divide the composite"
                            )
                        # Re-found known factor (stale work from before its
                        # discovery was divided out)
                        stale_known_count += 1
                        logger.info(
                            f"Factor {factor_str[:20]}... is already recorded for composite "
                            f"{composite.id} (re-found by stale work) - counting as known"
                        )
                        continue

                    # Split the reported factor against the current cofactor:
                    # gcd == factor: divides fully (the normal case).
                    # 1 < gcd < factor: partially stale composite factor - a
                    #   product of an already-divided factor and a new one
                    #   (e.g. stale P*Q reported after P was divided out);
                    #   the gcd is the surviving new component.
                    # gcd == 1: nothing new - the factor divides the verified
                    #   submitted state but not the current cofactor, so it is
                    #   composed entirely of already-divided primes.
                    shared = math.gcd(factor_int, int(running_cofactor))

                    if shared > 1:
                        surviving = str(shared)
                        if shared != factor_int:
                            stale_known_count += 1  # the already-divided part
                            logger.info(
                                f"Factor {factor_str[:20]}... is partially stale; "
                                f"processing surviving component {surviving[:20]}..."
                            )

                        new_cofactor = divide_factor(running_cofactor, surviving)

                        # If dividing would result in 1, this is the final prime - don't add it
                        if new_cofactor == "1":
                            logger.info(
                                f"Skipping final prime factor {surviving[:20]}{'...' if len(surviving) > 20 else ''} "
                                f"- not adding to factors table"
                            )
                            # Mark that we found the final prime (will set is_complete=True later)
                            running_cofactor = surviving  # The "cofactor" is now just this prime
                            continue

                        # Valid non-final factor - add to list and update running cofactor
                        factors_to_add.append((surviving, factor_sigma))
                        running_cofactor = new_cofactor
                        continue

                    stale_known_count += 1
                    logger.info(
                        f"Factor {factor_str[:20]}... no longer divides the current "
                        f"cofactor of composite {composite.id} (already divided out) "
                        f"- counting as known"
                    )

                # Second pass: add only the validated factors (excluding final prime)
                for factor_str, factor_sigma in factors_to_add:
                    # Parse sigma if it's a string (format: "3:12345")
                    parsed_sigma = None
                    if factor_sigma:
                        sigma_str = str(factor_sigma)
                        if ':' in sigma_str:
                            parsed_sigma = int(sigma_str.split(':', 1)[1])
                        else:
                            parsed_sigma = int(sigma_str)

                    # Add factor to database (with parametrization for group order calculation)
                    # Convert sigma to string for storage (supports large param 0 values)
                    sigma_for_storage = str(parsed_sigma) if parsed_sigma is not None else None
                    _, factor_created = FactorService.add_factor(
                        db, composite.id, factor_str, attempt.id, sigma_for_storage, parametrization,
                        method=result_request.method
                    )

                    if factor_created:
                        new_factors_count += 1
                        logger.info(f"  ✓ Added new factor {new_factors_count}: {factor_str[:20]}...")
                    else:
                        known_factors_count += 1
                        logger.info(f"  ○ Factor already known: {factor_str[:20]}...")

                # Re-found stale factors count as known for status reporting
                known_factors_count += stale_known_count

                # Completion must be based on VALIDATED factors, not the raw
                # request value: if nothing validated (only trivial values
                # like "1" or the composite itself), clear factor_found so a
                # junk value can't satisfy the residue completion's factor
                # check after a near-zero run.
                if new_factors_count == 0 and known_factors_count == 0:
                    if attempt.factor_found is not None:
                        logger.info(
                            f"No factor validated from submission by {result_request.client_id} "
                            f"- clearing factor_found on attempt {attempt.id}"
                        )
                        attempt.factor_found = None

                # Set factor status based on what was found
                if new_factors_count > 0:
                    factor_status = "new_factor"
                elif known_factors_count > 0:
                    factor_status = "known_factor"

                # Now update composite with the cofactor we calculated in the first pass
                if new_factors_count > 0 or known_factors_count > 0:
                    # Skip the no-op update when only stale/known factors were
                    # submitted (the cofactor didn't change)
                    if running_cofactor != composite.current_composite:
                        try:
                            composite_service.update_composite_to_cofactor(
                                db, composite.id, running_cofactor
                            )
                        except ValueError as e:
                            # Log but don't fail - the factors were still recorded
                            logger.warning(
                                f"Failed to update composite {composite.id} after factor division: {e}"
                            )

                    # Check if we now have complete factorization
                    if FactorService.verify_factorization(db, composite.id):
                        composite_service.mark_fully_factored(db, composite.id)

            # If this submission consumed a claimed residue, complete the
            # residue in the same transaction
            residue_completed = False
            if linked_residue is not None:
                residue_completed = _try_complete_residue(
                    db, residue_manager, linked_residue, attempt,
                    result_request.client_id
                )

            # Update t-level if this was an ECM attempt. Runs after residue
            # completion so the single recalculation excludes superseded attempts.
            t_level_updated = False
            if result_request.method == 'ecm':
                try:
                    # Serialize recalculations: two concurrent no-factor
                    # submissions would otherwise each recalculate without
                    # seeing the other's uncommitted attempt, and the last
                    # write would drop valid work until the next recalc.
                    # After the lock is acquired the other attempt is
                    # committed and visible. No-op if already held (factor
                    # processing / residue completion lock it earlier).
                    _lock_composite_row(db, composite.id)
                    composite_service.update_t_level(db, composite.id)
                    t_level_updated = True
                except Exception as e:
                    # Log the error but don't fail the whole submission
                    # The ECM result is still valid even if t-level update fails
                    logger.warning(f"Failed to update t-level for composite {composite.id}: {str(e)}")

            return SubmitResultResponse(
                status="success",
                attempt_id=attempt.id,
                composite_id=composite.id,
                message="Result logged successfully",
                factor_status=factor_status,
                residue_completed=residue_completed,
                # Report the t-level only when the recalc actually ran; a
                # swallowed recalc error must not surface a stale value
                new_t_level=composite.current_t_level if (residue_completed and t_level_updated) else None
            )

        except HTTPException:
            # Re-raise HTTPExceptions (like validation errors) without modification
            raise
        except ValueError as e:
            # Client-side validation errors (safe to expose)
            logger.warning(f"Validation error from {client_ip}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except (TypeError, AttributeError) as e:
            # Data structure errors (potentially from malformed requests)
            logger.error(f"Data structure error: {e}")
            raise HTTPException(status_code=400, detail="Invalid request format")
        except Exception as e:
            # Unexpected errors - log with full details but return generic message to client
            logger.exception(f"Unexpected error in submit_result: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error occurred while processing submission"
            )