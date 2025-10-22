"""
Composite management routes for admin.
"""
import logging
import secrets
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
from ....dependencies import verify_admin_key, get_composite_service
from ....schemas.composites import BulkCompositeRequest
from ....services.composites import CompositeService
from ....templates import templates
from ....utils.html_helpers import get_unauthorized_redirect_html
from ....utils.errors import get_or_404, not_found_error
from ....utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)

# Security: Maximum file upload size (10 MB)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@router.post("/composites/upload")
async def upload_composites(
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
        content = await file.read()

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
async def bulk_add_composites(
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
async def bulk_add_composites_structured(
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
                'is_prime': c.is_prime,
                'is_fully_factored': c.is_fully_factored
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
async def get_queue_status(
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
async def find_composite(
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


@router.get("/composites/{composite_id}")
async def get_composite_details(
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


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
async def get_composite_details_page(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    x_admin_key: str = Header(None)
):
    """Web page showing detailed information about a specific composite."""
    settings = get_settings()
    # Security: Constant-time comparison to prevent timing attacks
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        return get_unauthorized_redirect_html()

    details = get_or_404(
        composite_service.get_composite_details(db, composite_id),
        "Composite"
    )

    return templates.TemplateResponse("admin/composite_details.html", {
        "request": request,
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work'],
        "factors_with_group_orders": details['factors_with_group_orders']
    })


@router.put("/composites/{composite_id}/priority")
async def set_composite_priority(
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
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Mark a composite as fully factored."""
    success = composite_service.mark_composite_complete(db, composite_id, reason)
    if not success:
        raise not_found_error("Composite")
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
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """Remove a composite from the queue entirely."""
    result = composite_service.delete_composite(db, composite_id, reason)
    if not result:
        raise not_found_error("Composite")
    return result
