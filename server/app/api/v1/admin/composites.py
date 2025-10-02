"""
Composite management routes for admin.
"""
import logging
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status
)
from fastapi.responses import HTMLResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ....config import get_settings
from ....database import get_db
from ....dependencies import verify_admin_key
from ....schemas.composites import BulkCompositeRequest
from ....services.composite_manager import CompositeManager
from ....templates import templates
from ....utils.html_helpers import get_unauthorized_redirect_html

router = APIRouter()
logger = logging.getLogger(__name__)
composite_manager = CompositeManager()


@router.post("/composites/upload")
async def upload_composites(
    file: UploadFile = File(...),
    source_type: str = Form("auto"),
    default_priority: int = Form(0),
    number_column: str = Form("number"),  # pylint: disable=unused-argument
    priority_column: Optional[str] = Form(None),  # pylint: disable=unused-argument
    project_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Upload composites from a file.

    Note: number_column and priority_column are reserved for future
    CSV column mapping functionality.
    """
    try:
        content = await file.read()
        content_str = content.decode('utf-8')

        # Auto-detect file type
        if source_type == "auto":
            source_type = "csv" if file.filename and file.filename.endswith('.csv') else "text"

        # Process based on file type
        if source_type == "csv":
            stats = composite_manager.bulk_load_composites(
                db, content_str, source_type="csv",
                default_priority=default_priority, project_name=project_name
            )
        else:
            lines = content_str.strip().split('\n')
            stats = composite_manager.bulk_load_composites(
                db, lines, source_type="list",
                default_priority=default_priority, project_name=project_name
            )

        return {
            "filename": file.filename,
            "file_size": len(content),
            "source_type": source_type,
            **stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing file: {str(e)}"
        ) from e


@router.post("/composites/bulk")
async def bulk_add_composites(
    numbers: List[str],
    default_priority: int = 0,
    project_name: Optional[str] = None,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Add a list of composite numbers."""
    try:
        stats = composite_manager.bulk_load_composites(
            db, numbers, source_type="list",
            default_priority=default_priority, project_name=project_name
        )
        return {"input_count": len(numbers), **stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing numbers: {str(e)}"
        ) from e


@router.post("/composites/bulk-structured")
async def bulk_add_composites_structured(
    request: BulkCompositeRequest,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Add composites with full metadata including SNFS fields."""
    try:
        logger.info("Bulk structured upload: %d composites", len(request.composites))

        composites_data = [
            {
                'number': c.number,
                'current_composite': c.current_composite,
                'has_snfs_form': c.has_snfs_form,
                'snfs_difficulty': c.snfs_difficulty,
                'priority': c.priority if c.priority is not None else request.default_priority
            }
            for c in request.composites
        ]

        stats = composite_manager.bulk_load_composites(
            db, composites_data, source_type="list",
            default_priority=request.default_priority,
            project_name=request.project_name
        )

        return {"input_count": len(request.composites), **stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing composites: {str(e)}"
        ) from e


@router.get("/composites/status")
async def get_queue_status(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Get comprehensive status of the work queue."""
    try:
        return composite_manager.get_work_queue_status(db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        ) from e


@router.get("/composites/{composite_id}")
async def get_composite_details(
    composite_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Get detailed information about a specific composite."""
    details = composite_manager.get_composite_details(db, composite_id)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )
    return details


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
async def get_composite_details_page(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(None)
):
    """Web page showing detailed information about a specific composite."""
    settings = get_settings()
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        return get_unauthorized_redirect_html()

    details = composite_manager.get_composite_details(db, composite_id)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    return templates.TemplateResponse("admin/composite_details.html", {
        "request": request,
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work']
    })


@router.put("/composites/{composite_id}/priority")
async def set_composite_priority(
    composite_id: int,
    priority: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Set priority for a composite."""
    success = composite_manager.set_composite_priority(db, composite_id, priority)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )
    return {
        "composite_id": composite_id,
        "priority": priority,
        "status": "updated"
    }


@router.post("/composites/{composite_id}/complete")
async def mark_composite_complete(
    composite_id: int,
    reason: str = "manual",
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Mark a composite as fully factored."""
    success = composite_manager.mark_composite_complete(db, composite_id, reason)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )
    return {
        "composite_id": composite_id,
        "status": "marked_complete",
        "reason": reason
    }


@router.delete("/composites/{composite_id}")
async def remove_composite(
    composite_id: int,
    reason: str = "admin_removal",
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Remove a composite from the queue entirely."""
    from ....models.composites import Composite
    from ....models.work_assignments import WorkAssignment
    from ....models.attempts import ECMAttempt
    from ....models.factors import Factor

    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    # Cancel active work assignments
    active_work = db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.composite_id == composite_id,
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        )
    ).all()

    for work in active_work:
        work.status = 'cancelled'

    # Delete related records
    db.query(ECMAttempt).filter(ECMAttempt.composite_id == composite_id).delete()
    db.query(Factor).filter(Factor.composite_id == composite_id).delete()
    db.query(WorkAssignment).filter(WorkAssignment.composite_id == composite_id).delete()

    # Delete the composite
    db.delete(composite)
    db.commit()

    return {
        "composite_id": composite_id,
        "status": "removed",
        "reason": reason,
        "cancelled_work_assignments": len(active_work)
    }
