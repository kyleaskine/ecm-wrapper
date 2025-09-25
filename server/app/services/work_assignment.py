from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import uuid
import logging

from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.work_assignments import WorkAssignment
from ..models.clients import Client
from ..schemas.work import WorkRequest, WorkResponse
from .t_level_calculator import TLevelCalculator

logger = logging.getLogger(__name__)

class ECMParameterDecision:
    """ECM parameter decision engine based on t-level targeting and number analysis."""

    def __init__(self):
        self.t_level_calc = TLevelCalculator()

    # ECM parameter tables based on GMP-ECM documentation and best practices
    ECM_BOUNDS = [
        # (max_digits, b1, b2, typical_curves)
        (30, 2000, 147000, 25),
        (35, 11000, 1900000, 90),
        (40, 50000, 12500000, 300),
        (45, 250000, 128000000, 700),
        (50, 1000000, 1000000000, 1800),
        (55, 3000000, 5000000000, 5100),
        (60, 11000000, 35000000000, 10600),
        (65, 43000000, 240000000000, 19300),
        (70, 110000000, 873000000000, 49000),
        (75, 260000000, 2600000000000, 124000),
        (80, 850000000, 11700000000000, 210000),
        (85, 2900000000, 55300000000000, 340000),
    ]

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
                composite.digit_length, special_form=None  # No auto-detection for simplified system
            )
            composite.target_t_level = target_t

        # Calculate current t-level from attempts
        current_t = self.t_level_calc.get_current_t_level_from_attempts(previous_attempts)
        composite.current_t_level = current_t

        # Get suggestions based on t-level targeting
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
                WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
            )
        ).count()

        if active_work_count >= self.max_work_per_client:
            return WorkResponse(
                message=f"Client has {active_work_count} active work assignments (max: {self.max_work_per_client})"
            )

        # Clean up expired work assignments
        self._cleanup_expired_work(db)

        # Find suitable composite for work
        composite = self._find_suitable_composite(db, work_request)
        if not composite:
            return WorkResponse(message="No suitable work available")

        # Get previous attempts for this composite
        previous_attempts = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite.id
        ).all()

        # Determine best method and parameters
        method, parameters = self._determine_work_parameters(
            composite, previous_attempts, work_request.methods
        )

        if not method:
            return WorkResponse(message="No suitable method available for this composite")

        # Create work assignment
        work_assignment = self._create_work_assignment(
            db, composite, work_request.client_id, method, parameters
        )

        return WorkResponse(
            work_id=work_assignment.id,
            composite=composite.number,
            method=method,
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
            db.commit()

    def _find_suitable_composite(self, db: Session, work_request: WorkRequest) -> Optional[Composite]:
        """Find a composite suitable for the client's capabilities."""
        query = db.query(Composite).filter(
            and_(
                Composite.is_fully_factored == False,
                or_(Composite.is_prime.is_(None), Composite.is_prime == False)
            )
        )

        # Apply digit length filters
        if work_request.min_digits:
            query = query.filter(Composite.digit_length >= work_request.min_digits)
        if work_request.max_digits:
            query = query.filter(Composite.digit_length <= work_request.max_digits)

        # Exclude composites with active work assignments
        active_work_composites = db.query(WorkAssignment.composite_id).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).subquery()

        query = query.filter(~Composite.id.in_(active_work_composites))

        # Order by priority: smaller numbers first, then by creation time
        composite = query.order_by(
            Composite.digit_length.asc(),
            Composite.created_at.asc()
        ).first()

        return composite

    def _determine_work_parameters(self, composite: Composite, previous_attempts: List[ECMAttempt],
                                 preferred_methods: List[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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
                current_t = composite.current_t_level or 0.0
                logger.info(f"ECM work for composite {composite.id}: "
                           f"t{current_t:.1f} → t{target_t:.1f} "
                           f"(B1={b1:,}, {curves} curves)")

                return 'ecm', {
                    'b1': b1,
                    'b2': b2,
                    'curves': curves,
                    'target_t_level': target_t,
                    'current_t_level': current_t
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
        db.commit()
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
        db.commit()

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
        db.commit()
        return True

    def update_progress(self, db: Session, work_id: str, client_id: str,
                       curves_completed: int, message: str = None) -> bool:
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

        work.curves_completed = curves_completed
        work.progress_message = message
        work.last_progress_at = datetime.utcnow()

        # Extend deadline if making good progress
        if curves_completed > work.curves_completed:
            work.extend_deadline()

        db.commit()
        return True

    def complete_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Mark work as completed."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status == 'running'
            )
        ).first()

        if not work:
            return False

        work.status = 'completed'
        work.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Work assignment {work_id} completed by client {client_id}")
        return True

    def abandon_work(self, db: Session, work_id: str, client_id: str) -> bool:
        """Abandon/release a work assignment."""
        work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.id == work_id,
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
            )
        ).first()

        if not work:
            return False

        # Mark as failed so it can be reassigned
        work.status = 'failed'
        db.commit()

        logger.info(f"Work assignment {work_id} abandoned by client {client_id}")
        return True