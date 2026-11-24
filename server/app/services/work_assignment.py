from sqlalchemy.orm import Session, Query
from sqlalchemy import and_
from typing import Optional
from datetime import datetime
import logging

from ..models.composites import Composite
from ..models.work_assignments import WorkAssignment
from ..models.residues import ECMResidue
from ..constants import ACTIVE_WORK_STATUSES, PENDING_RESIDUE_STATUSES

logger = logging.getLogger(__name__)

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
            stage 2 pending). /ecm-work and /p1-work both pass True
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
                ECMResidue.status.in_(PENDING_RESIDUE_STATUSES)
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