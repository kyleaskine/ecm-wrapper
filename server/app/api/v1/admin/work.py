"""
Work assignment management routes for admin.
"""
from typing import Optional
from datetime import datetime, timedelta
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ....database import get_db
from ....dependencies import verify_admin_key, get_composite_service
from ....services.composites import CompositeService
from ....schemas.work import ManualReservationRequest, ManualReservationResponse
from ....utils.serializers import serialize_work_assignment
from ....utils.query_helpers import get_recent_work_assignments, get_expired_work_assignments
from ....services.work_assignment import UNIQUE_ACTIVE_WORK_MARKERS
from ....utils.transactions import transaction_scope, is_unique_violation
from ....utils.errors import get_or_404
from ....constants import ACTIVE_WORK_STATUSES

router = APIRouter()


@router.get("/work/assignments")
def get_work_assignments(
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Get work assignments with optional filtering."""
    # Use shared query helper
    assignments = get_recent_work_assignments(
        db,
        limit=limit,
        status_filter=status_filter,
        client_id=client_id
    )

    # Use shared serializer
    return {
        "assignments": [
            serialize_work_assignment(assignment, truncate_composite=True)
            for assignment in assignments
        ],
        "total_count": len(assignments),
        "filters_applied": {
            "status": status_filter,
            "client_id": client_id
        }
    }


@router.delete("/work/assignments/{work_id}")
def cancel_work_assignment(
    work_id: str,
    reason: str = "admin_cancel",
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Cancel a work assignment (admin override)."""
    from ....models.work_assignments import WorkAssignment

    assignment = db.query(WorkAssignment).filter(
        WorkAssignment.id == work_id
    ).first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found"
        )

    # Determine new status based on reason
    previous_status = assignment.status
    if "timeout" in reason.lower():
        new_status = 'timeout'
    else:
        new_status = 'failed'

    # Cancel the assignment
    with transaction_scope(db, "cancel_work"):
        assignment.status = new_status

    return {
        "work_id": work_id,
        "status": "cancelled",
        "reason": reason,
        "new_status": new_status,
        "previous_status": previous_status
    }


@router.post("/work/cleanup")
def cleanup_expired_work(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Manually trigger cleanup of expired work assignments."""
    # Use shared query helper
    expired_assignments = get_expired_work_assignments(db)

    # Mark them as timeout
    with transaction_scope(db, "cleanup_work"):
        for assignment in expired_assignments:
            assignment.status = 'timeout'

    return {
        "cleaned_up": len(expired_assignments),
        "status": "completed"
    }


@router.post("/work/reserve", response_model=ManualReservationResponse)
def reserve_composite(
    request: ManualReservationRequest,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Manually reserve a composite for external work (e.g., NFS at OPN).

    This creates a work assignment that blocks the composite from being assigned
    to ECM clients. Use this when running NFS or other external factorization work.

    Args:
        request: Reservation request with composite identifier and metadata
        db: Database session
        composite_service: Composite service for lookups

    Returns:
        ManualReservationResponse with reservation details
    """
    from ....models.work_assignments import WorkAssignment
    from ....models.composites import Composite

    # Look up the composite
    composite = get_or_404(
        composite_service.find_composite_by_identifier(db, request.composite_identifier),
        "Composite",
        request.composite_identifier
    )

    # Lock the composite row through commit, upholding the invariant the work
    # endpoints' recheck relies on: every assignment writer holds this lock,
    # so the reserved-check below and the INSERT are race-free. Blocks briefly
    # if a work endpoint is assigning this composite right now.
    composite = get_or_404(
        db.query(Composite).filter(Composite.id == composite.id)
        .with_for_update().first(),
        "Composite",
        request.composite_identifier
    )

    # Check if already reserved
    existing_assignment = db.query(WorkAssignment).filter(
        WorkAssignment.composite_id == composite.id,
        WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
    ).first()

    if existing_assignment:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Composite already has active work assignment (ID: {existing_assignment.id}, "
                   f"method: {existing_assignment.method}, client: {existing_assignment.client_id})"
        )

    # Create the reservation (work assignment)
    work_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=request.duration_hours)

    try:
        with transaction_scope(db, "reserve_composite"):
            assignment = WorkAssignment(
                id=work_id,
                composite_id=composite.id,
                client_id=request.client_id,
                method=request.method,
                b1=0,  # Not applicable for NFS
                b2=None,
                curves_requested=0,  # Not applicable for NFS
                expires_at=expires_at,
                status='running',  # Mark as running to indicate active work
                assigned_at=datetime.utcnow()
            )
            db.add(assignment)
    except IntegrityError as e:
        # Backstop only - the row lock above serializes writers, so this
        # should be unreachable; the partial unique index is the final arbiter
        if not is_unique_violation(e, *UNIQUE_ACTIVE_WORK_MARKERS):
            raise
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Composite was assigned to a client concurrently; reservation not created"
        )

    return ManualReservationResponse(
        work_id=work_id,
        composite_id=composite.id,
        composite_number=composite.number,
        client_id=request.client_id,
        method=request.method,
        expires_at=expires_at,
        status='running'
    )


@router.post("/work/release/{work_id}")
def release_reservation(
    work_id: str,
    reason: str = "completed",
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Release a manual reservation.

    Call this when NFS or other external work is complete (or cancelled).
    Marks the work assignment as completed, making the composite available
    for ECM work assignment again.

    Args:
        work_id: Work assignment ID to release
        reason: Reason for release (e.g., 'completed', 'cancelled', 'factored')
        db: Database session

    Returns:
        Status of the release operation
    """
    from ....models.work_assignments import WorkAssignment

    assignment = db.query(WorkAssignment).filter(
        WorkAssignment.id == work_id
    ).first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found"
        )

    previous_status = assignment.status

    # Determine final status based on reason
    if reason in ['completed', 'factored']:
        new_status = 'completed'
    elif reason in ['cancelled', 'timeout']:
        new_status = 'failed'
    else:
        new_status = 'completed'  # Default to completed

    with transaction_scope(db, "release_reservation"):
        assignment.status = new_status
        assignment.completed_at = datetime.utcnow()

    return {
        "work_id": work_id,
        "composite_id": assignment.composite_id,
        "previous_status": previous_status,
        "new_status": new_status,
        "reason": reason,
        "status": "released"
    }
