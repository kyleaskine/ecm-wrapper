from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List

from ...database import get_db
from ...dependencies import get_composite_service
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...templates import templates

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """
    Simple dashboard showing all factorization results.
    """

    # Get recent composites with their attempts and factors
    composites = db.query(Composite).order_by(desc(Composite.created_at)).limit(50).all()

    # Get recent attempts
    attempts = db.query(ECMAttempt).order_by(desc(ECMAttempt.created_at)).limit(100).all()

    # Get all factors
    factors = db.query(Factor).order_by(desc(Factor.created_at)).all()

    # Build summary stats
    total_composites = db.query(Composite).count()
    total_attempts = db.query(ECMAttempt).count()
    total_factors = db.query(Factor).count()
    fully_factored = db.query(Composite).filter(Composite.is_fully_factored == True).count()

    return templates.TemplateResponse("public/dashboard.html", {
        "request": request,
        "composites": composites,
        "attempts": attempts,
        "factors": factors,
        "total_composites": total_composites,
        "total_attempts": total_attempts,
        "total_factors": total_factors,
        "fully_factored": fully_factored,
        "db": db,
        "Composite": Composite,
        "ECMAttempt": ECMAttempt,
        "Factor": Factor
    })


@router.get("/composites/find")
async def find_composite_public(
    q: str,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """Find composite by ID, number (formula), or current_composite value.

    Args:
        q: Search query - can be composite ID, number (e.g., "2^1223-1"),
           or current_composite value

    Returns:
        Redirect to the composite's details page
    """
    from fastapi.responses import RedirectResponse

    composite = composite_service.find_composite_by_identifier(db, q)
    if not composite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Composite not found: {q}"
        )

    # Redirect to the canonical details page URL
    return RedirectResponse(
        url=f"/api/v1/dashboard/composites/{composite.id}/details",
        status_code=status.HTTP_302_FOUND
    )


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
async def get_composite_details_public(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """
    Public web page showing detailed information about a specific composite.
    """
    details = composite_service.get_composite_details(db, composite_id)

    if not details:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Composite Not Found</title>
        </head>
        <body>
            <h1>Composite Not Found</h1>
            <p><a href="/api/v1/dashboard/">Back to Dashboard</a></p>
        </body>
        </html>
        """

    return templates.TemplateResponse("public/composite_details.html", {
        "request": request,
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work'],
        "factors_with_group_orders": details['factors_with_group_orders']
    })