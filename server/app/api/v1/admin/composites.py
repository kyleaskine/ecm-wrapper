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
    HTTPException,
    Request,
    UploadFile,
    status
)
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, verify_admin_key_html, get_composite_service
from ....schemas.composites import BulkCompositeRequest
from ....services.composites import CompositeService
from ....templates import templates
from ....utils.errors import get_or_404, not_found_error
from ....utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)

# Security: Maximum file upload size (10 MB)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@router.post("/composites/upload")
def upload_composites(
    file: UploadFile = File(...),
    source_type: str = Form("auto"),
    default_priority: int = Form(0),
    number_column: str = Form("number"),  # pylint: disable=unused-argument
    priority_column: Optional[str] = Form(None),  # pylint: disable=unused-argument
    project_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Upload composites from a file.

    Note: number_column and priority_column are reserved for future
    CSV column mapping functionality.
    """
    try:
        content = file.file.read()  # sync read; route runs in threadpool

        # Security: Check file size limit
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE / (1024 * 1024):.0f} MB"
            )

        # Security: Validate UTF-8 encoding
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be UTF-8 encoded text"
            )

        # Auto-detect file type
        if source_type == "auto":
            source_type = "csv" if file.filename and file.filename.endswith('.csv') else "text"

        # Process based on file type within a transaction
        with transaction_scope(db, "bulk_upload"):
            if source_type == "csv":
                stats = composite_service.bulk_load_composites(
                    db, content_str, source_type="csv",
                    default_priority=default_priority, project_name=project_name
                )
            else:
                lines = content_str.strip().split('\n')
                stats = composite_service.bulk_load_composites(
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
        logger.error("File upload error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing file. Please check the file format and try again."
        ) from e


@router.post("/composites/bulk")
def bulk_add_composites(
    numbers: List[str],
    default_priority: int = 0,
    project_name: Optional[str] = None,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Add a list of composite numbers."""
    try:
        with transaction_scope(db, "bulk_add"):
            stats = composite_service.bulk_load_composites(
                db, numbers, source_type="list",
                default_priority=default_priority, project_name=project_name
            )
        return {"input_count": len(numbers), **stats}
    except Exception as e:
        logger.error("Bulk add error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing numbers. Please check the input format."
        ) from e


@router.post("/composites/bulk-structured")
def bulk_add_composites_structured(
    request: BulkCompositeRequest,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
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
                'priority': c.priority if c.priority is not None else request.default_priority,
                'is_complete': c.is_complete,
                'is_fully_factored': c.is_fully_factored,
                'is_active': c.is_active,
                'prior_t_level': c.prior_t_level
            }
            for c in request.composites
        ]

        with transaction_scope(db, "bulk_structured"):
            stats = composite_service.bulk_load_composites(
                db, composites_data, source_type="list",
                default_priority=request.default_priority,
                project_name=request.project_name
            )

        return {"input_count": len(request.composites), **stats}
    except Exception as e:
        logger.error("Bulk structured upload error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error processing composites. Please check the data format."
        ) from e


@router.get("/composites/status")
def get_queue_status(
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Get comprehensive status of the work queue."""
    try:
        return composite_service.get_work_queue_status(db)
    except Exception as e:
        logger.error("Queue status error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving queue status"
        ) from e


@router.get("/composites/find")
def find_composite(
    q: str,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Find composite by ID, number (formula), or current_composite value.

    Args:
        q: Search query - can be composite ID, number (e.g., "2^1223-1"),
           or current_composite value

    Returns:
        Redirect to the composite's details page
    """
    from fastapi.responses import RedirectResponse

    composite = get_or_404(
        composite_service.find_composite_by_identifier(db, q),
        "Composite",
        q
    )

    # Redirect to the canonical details page URL
    return RedirectResponse(
        url=f"/api/v1/admin/composites/{composite.id}/details",
        status_code=status.HTTP_302_FOUND
    )


@router.get("/composites/{composite_id:int}")
def get_composite_details(
    composite_id: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Get detailed information about a specific composite."""
    details = get_or_404(
        composite_service.get_composite_details(db, composite_id),
        "Composite"
    )
    return details


@router.get("/composites/{composite_id:int}/details", response_class=HTMLResponse)
def get_composite_details_page(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key_html)
):
    """Web page showing detailed information about a specific composite."""

    details = get_or_404(
        composite_service.get_composite_details(db, composite_id),
        "Composite"
    )

    # Get method breakdown for tabbed interface (same as public page)
    method_breakdown = composite_service.get_method_breakdown(composite_id, db)

    return templates.TemplateResponse(request, "admin/composite_details.html", {
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work'],
        "all_factors": details['all_factors'],
        "factors_with_group_orders": details['factors_with_group_orders'],
        "method_breakdown": method_breakdown,
        "factor_counts": method_breakdown.get('_factor_counts', {}),
    })


@router.put("/composites/{composite_id:int}/priority")
def set_composite_priority(
    composite_id: int,
    priority: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Set priority for a composite."""
    success = composite_service.set_composite_priority(db, composite_id, priority)
    if not success:
        raise not_found_error("Composite")

    # Commit the transaction to persist the priority change
    db.commit()

    return {
        "composite_id": composite_id,
        "priority": priority,
        "status": "updated"
    }


@router.post("/composites/{composite_id:int}/complete")
def mark_composite_complete(
    composite_id: int,
    reason: str = "manual",
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Mark a composite as fully factored."""
    success = composite_service.mark_composite_complete(db, composite_id, reason)
    if not success:
        raise not_found_error("Composite")

    # Commit the transaction to persist the completion status
    db.commit()

    return {
        "composite_id": composite_id,
        "status": "marked_complete",
        "reason": reason
    }


@router.put("/composites/{composite_id:int}/activate")
def activate_composite(
    composite_id: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Activate a composite to make it available for work assignment."""
    composite = composite_service.get_composite_by_id(db, composite_id)
    if not composite:
        raise not_found_error("Composite")

    composite.is_active = True
    db.commit()

    return {
        "composite_id": composite_id,
        "is_active": True,
        "status": "activated"
    }


@router.put("/composites/{composite_id:int}/deactivate")
def deactivate_composite(
    composite_id: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Deactivate a composite to prevent it from being assigned as work."""
    composite = composite_service.get_composite_by_id(db, composite_id)
    if not composite:
        raise not_found_error("Composite")

    composite.is_active = False
    db.commit()

    return {
        "composite_id": composite_id,
        "is_active": False,
        "status": "deactivated"
    }


@router.post("/composites/bulk-activate")
def bulk_activate_composites(
    composite_ids: List[int],
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Activate multiple composites at once.

    Args:
        composite_ids: List of composite IDs to activate

    Returns:
        Summary of activation results
    """
    activated = []
    failed = []

    with transaction_scope(db, "bulk_activate"):
        for composite_id in composite_ids:
            composite = composite_service.get_composite_by_id(db, composite_id)
            if composite:
                composite.is_active = True
                activated.append(composite_id)
            else:
                failed.append(composite_id)

    return {
        "activated": activated,
        "failed": failed,
        "total_requested": len(composite_ids),
        "total_activated": len(activated)
    }


@router.delete("/composites/{composite_id:int}")
def remove_composite(
    composite_id: int,
    reason: str = "admin_removal",
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Remove a composite from the queue entirely."""
    result = composite_service.delete_composite(db, composite_id, reason)
    if not result:
        raise not_found_error("Composite")

    # Commit the transaction to persist the deletion
    db.commit()

    return result
