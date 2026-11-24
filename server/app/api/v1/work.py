from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from ...database import get_db
from ...dependencies import get_work_service
from ...services.work_assignment import WorkAssignmentService
from ...config import get_settings
from ...utils.transactions import transaction_scope

router = APIRouter()
settings = get_settings()

@router.post("/work/{work_id}/claim")
def claim_work(
    work_id: str,
    client_id: str,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Claim a specific work assignment.

    Clients claim an assignment to begin execution.

    Args:
        work_id: ID of the work assignment to claim
        client_id: Client claiming the work
        db: Database session

    Returns:
        Success status
    """
    with transaction_scope(db, "claim_work"):
        success = work_service.claim_work(db, work_id, client_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found, already claimed, or expired"
        )

    return {"status": "claimed", "work_id": work_id}


@router.post("/work/{work_id}/start")
def start_work(
    work_id: str,
    client_id: str,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Mark work as started.

    Clients call this when they begin executing the work.

    Args:
        work_id: ID of the work assignment
        client_id: Client starting the work
        db: Database session

    Returns:
        Success status
    """
    with transaction_scope(db, "start_work"):
        success = work_service.start_work(db, work_id, client_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found or not in claimable state"
        )

    return {"status": "started", "work_id": work_id}


@router.put("/work/{work_id}/progress")
def update_progress(
    work_id: str,
    client_id: str,
    curves_completed: int,
    message: Optional[str] = None,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Update work progress.

    Clients can periodically report their progress to extend deadlines
    and provide status updates.

    Args:
        work_id: ID of the work assignment
        client_id: Client updating progress
        curves_completed: Number of curves completed so far
        message: Optional progress message
        db: Database session

    Returns:
        Success status
    """
    with transaction_scope(db, "update_progress"):
        success = work_service.update_progress(
            db, work_id, client_id, curves_completed, message
        )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found or not in running state"
        )

    return {
        "status": "updated",
        "work_id": work_id,
        "curves_completed": curves_completed
    }


@router.post("/work/{work_id}/complete")
def complete_work(
    work_id: str,
    client_id: str,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Mark work as completed.

    Clients call this when they finish executing the work assignment.
    Results should be submitted separately via the existing submission endpoints.

    Args:
        work_id: ID of the work assignment
        client_id: Client completing the work
        db: Database session

    Returns:
        Success status
    """
    with transaction_scope(db, "complete_work"):
        success = work_service.complete_work(db, work_id, client_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found or not in active state (assigned/claimed/running)"
        )

    return {"status": "completed", "work_id": work_id}


@router.delete("/work/{work_id}")
def abandon_work(
    work_id: str,
    client_id: str,
    reason: Optional[str] = "client_request",
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Abandon/release a work assignment.

    Clients can abandon work if they encounter issues or need to stop.
    The work will be made available for other clients.

    Args:
        work_id: ID of the work assignment to abandon
        client_id: Client abandoning the work
        reason: Optional reason for abandoning
        db: Database session

    Returns:
        Success status
    """
    with transaction_scope(db, "abandon_work"):
        success = work_service.abandon_work(db, work_id, client_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found or not owned by client"
        )

    return {
        "status": "abandoned",
        "work_id": work_id,
        "reason": reason
    }


@router.get("/work/status/{client_id}")
def get_client_work_status(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    Get status of all work assignments for a specific client.

    Args:
        client_id: Client to get work status for
        db: Database session

    Returns:
        List of work assignments for the client
    """
    from ...models.work_assignments import WorkAssignment
    from ...models.composites import Composite
    from ...utils.serializers import serialize_work_assignment
    from sqlalchemy.orm import joinedload

    # Eagerly load composite relationship to avoid N+1 queries
    work_assignments = db.query(WorkAssignment).options(
        joinedload(WorkAssignment.composite)
    ).filter(
        WorkAssignment.client_id == client_id
    ).order_by(WorkAssignment.created_at.desc()).limit(50).all()

    return {
        "client_id": client_id,
        "work_assignments": [
            serialize_work_assignment(work, truncate_composite=True)
            for work in work_assignments
        ]
    }