"""
Work assignment management routes for admin.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key
from ....utils.serializers import serialize_work_assignment
from ....utils.query_helpers import get_recent_work_assignments, get_expired_work_assignments
from ....utils.transactions import transaction_scope

router = APIRouter()


@router.get("/work/assignments")
async def get_work_assignments(
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
async def cancel_work_assignment(
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

    # Cancel the assignment
    with transaction_scope(db, "cancel_work"):
        assignment.status = 'failed'

    return {
        "work_id": work_id,
        "status": "cancelled",
        "reason": reason,
        "previous_status": assignment.status
    }


@router.post("/work/cleanup")
async def cleanup_expired_work(
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
