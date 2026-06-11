from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Literal, cast

from ...database import get_db
from ...dependencies import get_work_service
from ...schemas.work import WorkRequest, WorkResponse
from ...services.work_assignment import WorkAssignmentService
from ...config import get_settings
from ...utils.transactions import transaction_scope

router = APIRouter()
settings = get_settings()

@router.get("/work", response_model=WorkResponse)
def get_work(
    client_id: str,
    methods: Optional[List[str]] = None,
    max_digits: Optional[int] = None,
    min_digits: Optional[int] = None,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Get work assignment for a client.

    This is the main endpoint clients use to poll for new work.

    Args:
        client_id: Unique identifier for the requesting client
        methods: Preferred factorization methods (ecm, pm1, pp1, etc.)
        max_digits: Maximum number of digits client can handle
        min_digits: Minimum number of digits client prefers
        db: Database session

    Returns:
        WorkResponse with assigned work or explanation if no work available
    """
    # Set default methods if not specified
    if methods is None:
        methods = ["ecm", "pm1"]

    # Validate methods
    valid_methods = ["ecm", "pm1", "pp1", "qs", "nfs"]
    methods = [m for m in methods if m in valid_methods]

    if not methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid methods specified"
        )

    # Create work request - cast methods since we've already validated them
    validated_methods = cast(List[Literal["ecm", "pm1", "pp1", "qs", "nfs"]], methods)
    work_request = WorkRequest(
        client_id=client_id,
        methods=validated_methods,
        max_digits=max_digits,
        min_digits=min_digits
    )

    # Get work assignment within transaction
    with transaction_scope(db, "get_work"):
        work_response = work_service.get_work_for_client(db, work_request)

    return work_response


@router.post("/work/{work_id}/claim")
def claim_work(
    work_id: str,
    client_id: str,
    db: Session = Depends(get_db),
    work_service: WorkAssignmentService = Depends(get_work_service)
):
    """
    Claim a specific work assignment.

    After getting work from GET /work, clients should claim it to begin execution.

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