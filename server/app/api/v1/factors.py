from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging

from ...database import get_db
from ...schemas.factors import FactorResponse, FactorWithComposite, FactorsListResponse
from ...models import Factor, ECMAttempt, Composite
from ...utils.errors import get_or_404

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/factors", response_model=FactorsListResponse)
def get_latest_factors(
    limit: int = Query(100, ge=1, le=1000, description="Number of factors to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    since: Optional[str] = Query(None, description="ISO timestamp - only return factors created after this time"),
    db: Session = Depends(get_db)
):
    """
    Get latest factors discovered by the system.

    This endpoint returns recently discovered factors with composite information,
    ordered by discovery time (newest first). Useful for external sites to poll
    for new factorization results.

    Parameters:
    - limit: Maximum number of factors to return (default 100, max 1000)
    - offset: Pagination offset (default 0)
    - since: ISO timestamp to filter factors created after this time
    """
    # Build query
    query = db.query(
        Factor.id,
        Factor.composite_id,
        Factor.factor,
        Factor.is_prime,
        Factor.found_by_attempt_id,
        Factor.created_at,
        Composite.number.label('number'),
        Composite.current_composite.label('composite_number'),
        ECMAttempt.client_id,
        ECMAttempt.method
    ).join(
        Composite, Factor.composite_id == Composite.id
    ).outerjoin(
        ECMAttempt, Factor.found_by_attempt_id == ECMAttempt.id
    )

    # Apply time filter if provided
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            query = query.filter(Factor.created_at > since_dt)
        except ValueError:
            logger.warning(f"Invalid since timestamp: {since}")

    # Get total count
    total = query.count()

    # Order by created_at descending (newest first) and apply pagination
    query = query.order_by(Factor.created_at.desc())
    query = query.offset(offset).limit(limit)

    # Execute query
    results = query.all()

    # Convert to response objects
    factors = [
        FactorWithComposite(
            id=r.id,
            composite_id=r.composite_id,
            number=r.number,
            composite_number=r.composite_number,
            factor=r.factor,
            is_prime=r.is_prime,
            found_by_attempt_id=r.found_by_attempt_id,
            created_at=r.created_at,
            client_id=r.client_id,
            method=r.method
        )
        for r in results
    ]

    return FactorsListResponse(
        factors=factors,
        total=total,
        page=offset // limit + 1 if limit > 0 else 1,
        page_size=limit
    )

@router.get("/factors/{factor_id}", response_model=FactorResponse)
def get_factor(
    factor_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific factor by ID"""
    factor = get_or_404(
        db.query(Factor).filter(Factor.id == factor_id).first(),
        "Factor",
        str(factor_id)
    )
    return factor
