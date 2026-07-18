from sqlalchemy.orm import Session, Query, defer
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.exc import IntegrityError
from typing import Optional, Dict, Any, List, Tuple, Sequence, Literal, cast
from datetime import datetime, timedelta
import uuid
import logging

from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.work_assignments import WorkAssignment
from ..models.residues import ECMResidue
from ..schemas.work import WorkRequest, WorkResponse
from .t_level_calculator import TLevelCalculator
from ..constants import ECM_BOUNDS, ACTIVE_WORK_STATUSES
from ..utils.transactions import is_unique_violation

logger = logging.getLogger(__name__)

class ECMParameterDecision:
    """ECM parameter decision engine based on t-level targeting and number analysis."""

    def __init__(self):
        self.t_level_calc = TLevelCalculator()

    # ECM parameter table imported from constants module
    ECM_BOUNDS = ECM_BOUNDS

    def get_ecm_parameters_with_t_level(self, composite: Composite, previous_attempts: List[ECMAttempt]) -> Tuple[int, int, int]:
        """
        Get optimal ECM parameters using t-level targeting.

        Args:
            composite: Composite object with t-level information
            previous_attempts: List of previous ECM attempts on this number

        Returns:
            Tuple of (b1, b2, suggested_curves)
        """
        # Calculate or update target t-level if not set
        if composite.target_t_level is None:
            target_t = self.t_level_calc.calculate_target_t_level(
                composite.digit_length,
                special_form=None,  # No auto-detection for simplified system
                snfs_difficulty=composite.snfs_difficulty  # Use SNFS difficulty if available
            )
            composite.target_t_level = target_t

        # Calculate current t-level from attempts, starting from prior_t_level if set
        # This gives us the true current position (prior work + work in this system)
        starting_t = composite.prior_t_level or 0.0
        current_t = self.t_level_calc.get_current_t_level_from_attempts(
            previous_attempts, starting_t_level=starting_t
        )
        composite.current_t_level = current_t

        # Get suggestions based on current t-level (which now includes prior)
        suggestion = self.t_level_calc.suggest_next_ecm_parameters(
            composite.target_t_level, current_t, composite.digit_length
        )

        if suggestion['status'] == 'target_reached':
            # Target reached, use escalated parameters
            logger.info(f"Target t-level reached for composite {composite.id}, escalating parameters")
            return self._get_escalated_parameters(composite.digit_length, previous_attempts)

        return suggestion['b1'], suggestion['b2'], suggestion['curves']

    @classmethod
    def get_ecm_parameters(cls, digit_length: int, previous_attempts: List[ECMAttempt]) -> Tuple[int, int, int]:
        """
        Legacy method - Get optimal ECM parameters for a number (fallback).

        Args:
            digit_length: Number of digits in the composite
            previous_attempts: List of previous ECM attempts on this number

        Returns:
            Tuple of (b1, b2, suggested_curves)
        """
        # Find appropriate bounds for this number size
        base_params = None
        for max_digits, b1, b2, curves in cls.ECM_BOUNDS:
            if digit_length <= max_digits:
                base_params = (b1, b2, curves)
                break

        if base_params is None:
            # For very large numbers, use the largest bounds
            base_params = cls.ECM_BOUNDS[-1][1:]

        b1, b2, curves = base_params

        # Analyze previous attempts to avoid duplication and escalate if needed
        if previous_attempts:
            max_b1_attempted = max(attempt.b1 for attempt in previous_attempts if attempt.method == 'ecm')

            # If we've already tried this B1 level extensively, escalate
            attempts_at_this_level = [a for a in previous_attempts
                                    if a.method == 'ecm' and a.b1 >= b1 * 0.8 and a.b1 <= b1 * 1.2]

            total_curves_attempted = sum(a.curves_completed for a in attempts_at_this_level)

            if total_curves_attempted >= curves * 0.8:  # 80% of recommended curves done
                # Escalate to next level
                next_level = None
                for max_digits, next_b1, next_b2, next_curves in cls.ECM_BOUNDS:
                    if next_b1 > b1:
                        next_level = (next_b1, next_b2, next_curves)
                        break

                if next_level:
                    b1, b2, curves = next_level
                    logger.info(f"Escalating ECM parameters to B1={b1} due to {total_curves_attempted} curves already attempted")

        # Adjust curve count for work assignment (smaller batches)
        suggested_curves = min(curves // 10, 100)  # Smaller work units
        suggested_curves = max(suggested_curves, 10)  # Minimum 10 curves

        return b1, b2, suggested_curves

    @classmethod
    def should_try_pm1(cls, digit_length: int, previous_attempts: List[ECMAttempt]) -> bool:
        """Determine if P-1 method should be tried."""
        # P-1 is good for numbers with factors having smooth p-1
        # Try P-1 before extensive ECM for smaller numbers
        if digit_length <= 50:
            pm1_attempts = [a for a in previous_attempts if a.method == 'pm1']
            return len(pm1_attempts) == 0  # Try P-1 once if not attempted
        return False

    @classmethod
    def get_pm1_parameters(cls, digit_length: int) -> Tuple[int, int]:
        """Get P-1 parameters based on number size."""
        if digit_length <= 40:
            return 1000000, 30000000
        elif digit_length <= 50:
            return 5000000, 150000000
        elif digit_length <= 60:
            return 25000000, 750000000
        else:
            return 100000000, 3000000000

    def _get_escalated_parameters(self, digit_length: int, previous_attempts: List[ECMAttempt]) -> Tuple[int, int, int]:
        """Get escalated parameters when target t-level is reached."""
        # Use higher bounds than standard
        max_b1_attempted = max((attempt.b1 for attempt in previous_attempts if attempt.method == 'ecm'), default=0)

        # Escalate to next level beyond what's been tried
        escalated_b1 = max_b1_attempted * 3

        # Use standard table as upper bound reference
        for max_digits, b1, b2, curves in self.ECM_BOUNDS:
            if digit_length <= max_digits and b1 > escalated_b1:
                return b1, b2, min(curves // 5, 200)  # Smaller batches for high bounds

        # Fallback to highest available
        return self.ECM_BOUNDS[-1][1], self.ECM_BOUNDS[-1][2], 100


# Markers identifying a violation of the one-active-assignment-per-composite
# partial unique index (PostgreSQL index name / SQLite message form)
UNIQUE_ACTIVE_WORK_MARKERS = (
    'uq_work_assignments_one_active_per_composite',
    'work_assignments.composite_id',
)


def pick_and_lock_composite(
    db: Session, ordered_query: Query, check_residues: bool, max_attempts: int = 5
) -> Optional[Composite]:
    """
    Lock a candidate composite and re-verify it is still free.

    FOR UPDATE SKIP LOCKED only prevents double assignment while the
    competing request still holds its row lock. Under READ COMMITTED the
    NOT EXISTS filters in `ordered_query` are evaluated against the
    statement snapshot: if a competing request commits while our SELECT is
    executing, its work_assignments INSERT is invisible to our snapshot and
    its row lock is already released, so the same composite is picked again
    (observed in production: two /ecm-work requests 16 ms apart were both
    assigned the same composite).

    Every assignment writer holds the composite row lock until commit, so
    once we hold the lock, fresh statements (new snapshot) are guaranteed
    to see any competing committed assignment. Re-check here and move to
    the next candidate on conflict. Rejected candidates stay locked until
    this transaction ends, which is harmless - they are busy anyway.

    The recheck costs up to two indexed point lookups on the no-contention
    path. That is deliberate: without it, losing the race means the client
    gets a spurious "no work" (via the unique-index backstop) instead of
    the next free composite.

    Args:
        db: Database session
        ordered_query: Filtered and ordered Composite query (must already
            exclude busy composites; the recheck mirrors those exclusions)
        check_residues: Also re-check for pending residues (stage 1 done,
            stage 2 pending) - used by /ecm-work but not /p1-work
        max_attempts: Candidates to try before giving up (bounds lock
            accumulation within one request)

    Returns:
        A locked, conflict-free Composite, or None if none available
    """
    tried_ids: list[int] = []
    for _ in range(max_attempts):
        candidate_query = ordered_query
        if tried_ids:
            candidate_query = candidate_query.filter(Composite.id.notin_(tried_ids))
        candidate = candidate_query.with_for_update(skip_locked=True, of=Composite).first()
        if candidate is None:
            return None

        conflict = db.query(WorkAssignment.id).filter(
            WorkAssignment.composite_id == candidate.id,
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        ).first() is not None
        if not conflict and check_residues:
            conflict = db.query(ECMResidue.id).filter(
                ECMResidue.composite_id == candidate.id,
                ECMResidue.status.in_(['available', 'claimed'])
            ).first() is not None

        if not conflict:
            return candidate

        logger.info(
            f"Composite {candidate.id} became busy after selection "
            f"(lost assignment race); trying next candidate"
        )
        tried_ids.append(candidate.id)
    return None


class WorkAssignmentService:
    """Service for managing work assignments and distribution."""

    def __init__(self, default_timeout_minutes: int = 60, max_work_per_client: int = 5):
        self.default_timeout_minutes = default_timeout_minutes
        self.max_work_per_client = max_work_per_client
        self.param_engine = ECMParameterDecision()

    def get_work_for_client(self, db: Session, work_request: WorkRequest) -> WorkResponse:
        """
        Assign work to a client based on their request.

        Args:
            db: Database session
            work_request: Client's work request with preferences

        Returns:
            WorkResponse with assigned work or explanation why no work available
        """
        # Check if client has too much active work
        active_work_count = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.client_id == work_request.client_id,
                WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
            )
        ).count()

        if active_work_count >= self.max_work_per_client:
            return WorkResponse(
                message=f"Client has {active_work_count} active work assignments (max: {self.max_work_per_client})"
            )

        # Clean up expired work assignments
        self._cleanup_expired_work(db)

        # Find a suitable composite: lock the row and re-verify it is still
        # free (the query's NOT EXISTS exclusion can be stale when a
        # concurrent request commits mid-query - see pick_and_lock_composite)
        composite = pick_and_lock_composite(
            db, self._suitable_composite_query(db, work_request), check_residues=False
        )
        if not composite:
            return WorkResponse(message="No suitable work available")

        # Get previous attempts for this composite
        # defer raw_output: large blob, not needed for parameter selection
        previous_attempts = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output)).filter(
            ECMAttempt.composite_id == composite.id
        ).all()

        # Determine best method and parameters
        method, parameters = self._determine_work_parameters(
            composite, previous_attempts, work_request.methods
        )

        if not method or parameters is None:
            return WorkResponse(message="No suitable method available for this composite")

        # Create work assignment. The recheck above should make a duplicate
        # unreachable, but the partial unique index is the final arbiter:
        # fail soft as "no work" (client re-requests) instead of a 500.
        try:
            with db.begin_nested():
                work_assignment = self._create_work_assignment(
                    db, composite, work_request.client_id, method, parameters
                )
        except IntegrityError as e:
            if not is_unique_violation(e, *UNIQUE_ACTIVE_WORK_MARKERS):
                raise
            logger.warning(
                f"Lost assignment race on composite {composite.id} at insert "
                f"(client {work_request.client_id}); returning no-work"
            )
            return WorkResponse(message="Lost assignment race, please request again")

        # Cast method to Literal type since it's already validated
        typed_method = cast(Literal["ecm", "pm1", "pp1", "qs", "nfs"], method)
        return WorkResponse(
            work_id=work_assignment.id,
            composite=composite.number,
            method=typed_method,
            parameters=parameters,
            estimated_time_minutes=work_assignment.estimated_time_minutes,
            expires_at=work_assignment.expires_at
        )

    def _cleanup_expired_work(self, db: Session):
        """Clean up expired work assignments."""
        expired_work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.expires_at < datetime.utcnow(),
                WorkAssignment.status.in_(['assigned', 'claimed'])
            )
        ).all()

        for work in expired_work:
            work.status = 'timeout'
            logger.info(f"Marked work assignment {work.id} as timeout")

        if expired_work:
            db.flush()  # Make changes visible within transaction

    def _suitable_composite_query(self, db: Session, work_request: WorkRequest) -> Query:
        """Build the filtered, ordered query for assignable composites."""
        query = db.query(Composite).filter(
            and_(
                Composite.is_active == True,  # Only assign active composites
                Composite.is_fully_factored == False,
                or_(Composite.is_complete.is_(None), Composite.is_complete == False)
            )
        )

        # Apply digit length filters
        if work_request.min_digits:
            query = query.filter(Composite.digit_length >= work_request.min_digits)
        if work_request.max_digits:
            query = query.filter(Composite.digit_length <= work_request.max_digits)

        # Exclude composites with active work assignments (NOT EXISTS is faster than NOT IN)
        query = query.filter(~db.query(WorkAssignment.id).filter(
            WorkAssignment.composite_id == Composite.id,
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        ).correlate(Composite).exists())

        # Order by priority: smaller numbers first, then by creation time
        return query.order_by(
            Composite.digit_length.asc(),
            Composite.created_at.asc()
        )

    def _determine_work_parameters(self, composite: Composite, previous_attempts: List[ECMAttempt],
                                 preferred_methods: Sequence[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Determine the best method and parameters for this composite using t-level targeting."""

        # No special form detection in simplified system - use standard ECM approach

        # Check if P-1 should be tried first
        if 'pm1' in preferred_methods and self.param_engine.should_try_pm1(composite.digit_length, previous_attempts):
            b1, b2 = self.param_engine.get_pm1_parameters(composite.digit_length)
            return 'pm1', {
                'b1': b1,
                'b2': b2,
                'curves': 1  # P-1 is typically single attempt
            }

        # Use t-level targeting for ECM
        if 'ecm' in preferred_methods:
            try:
                b1, b2, curves = self.param_engine.get_ecm_parameters_with_t_level(composite, previous_attempts)

                # Log t-level progress
                target_t = composite.target_t_level or 0.0
                current_t = composite.current_t_level or 0.0  # Now includes prior_t_level
                prior_t = composite.prior_t_level or 0.0

                if prior_t > 0:
                    logger.info(f"ECM work for composite {composite.id}: "
                               f"t{current_t:.1f} → t{target_t:.1f} "
                               f"(includes prior: t{prior_t:.1f}) "
                               f"(B1={b1:,}, {curves} curves)")
                else:
                    logger.info(f"ECM work for composite {composite.id}: "
                               f"t{current_t:.1f} → t{target_t:.1f} "
                               f"(B1={b1:,}, {curves} curves)")

                return 'ecm', {
                    'b1': b1,
                    'b2': b2,
                    'curves': curves,
                    'target_t_level': target_t,
                    'current_t_level': current_t,
                    'prior_t_level': prior_t
                }
            except Exception as e:
                logger.warning(f"T-level calculation failed for composite {composite.id}: {e}")
                # Fallback to legacy method
                b1, b2, curves = self.param_engine.get_ecm_parameters(composite.digit_length, previous_attempts)
                return 'ecm', {
                    'b1': b1,
                    'b2': b2,
                    'curves': curves
                }

        return None, None

    def _create_work_assignment(self, db: Session, composite: Composite, client_id: str,
                              method: str, parameters: Dict[str, Any]) -> WorkAssignment:
        """Create a new work assignment."""
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(minutes=self.default_timeout_minutes)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=composite.id,
            client_id=client_id,
            method=method,
            b1=parameters['b1'],
            b2=parameters.get('b2'),
            curves_requested=parameters['curves'],
            expires_at=expires_at,
            status='assigned'
        )

        db.add(work_assignment)
        db.flush()  # Get ID and make visible within transaction
        db.refresh(work_assignment)

        logger.info(f"Created work assignment {work_id} for client {client_id}: "
                   f"{method} on {composite.digit_length}-digit number")

        return work_assignment

    def claim_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Claim a work assignment for execution."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status == 'assigned'
            )
        ).first()

        if not work or work.is_expired:
            return False

        work.status = 'claimed'
        work.claimed_at = datetime.utcnow()
        db.flush()  # Make changes visible within transaction

        logger.info(f"Work assignment {work_id} claimed by client {client_id}")
        return True

    def start_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Mark work as started."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status == 'claimed'
            )
        ).first()

        if not work:
            return False

        work.status = 'running'
        db.flush()  # Make changes visible within transaction
        return True

    def update_progress(self, db: Session, work_id: str, client_id: str,
                       curves_completed: int, message: Optional[str] = None) -> bool:
        """Update work progress."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status == 'running'
            )
        ).first()

        if not work:
            return False

        # Extend deadline if making good progress
        if curves_completed > work.curves_completed:
            work.extend_deadline()

        work.curves_completed = curves_completed
        work.progress_message = message
        work.last_progress_at = datetime.utcnow()

        db.flush()  # Make changes visible within transaction
        return True

    def complete_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Mark work as completed."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
            )
        ).first()

        if not work:
            return False

        work.status = 'completed'
        work.completed_at = datetime.utcnow()
        db.flush()  # Make changes visible within transaction

        logger.info(f"Work assignment {work_id} completed by client {client_id}")
        return True

    def abandon_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Abandon/release a work assignment."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
            )
        ).first()

        if not work:
            return False

        # Mark as failed so it can be reassigned
        work.status = 'failed'
        db.flush()  # Make changes visible within transaction

        logger.info(f"Work assignment {work_id} abandoned by client {client_id}")
        return True