from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional
from datetime import datetime, timedelta
import uuid
import logging
import json

from ...database import get_db
from ...schemas.ecm_work import ECMWorkResponse
from ...models.composites import Composite
from ...models.attempts import ECMAttempt
from ...models.work_assignments import WorkAssignment
from ...services.t_level_calculator import TLevelCalculator
from ...utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize t-level calculator
t_level_calc = TLevelCalculator()


@router.get("/ecm-work")
async def get_ecm_work(
    client_id: str,
    priority: Optional[int] = None,
    min_digits: Optional[int] = None,
    max_digits: Optional[int] = None,
    timeout_days: Optional[int] = 5,
    db: Session = Depends(get_db)
):
    """
    Get ECM work assignment with t-level targeting.

    This endpoint returns the smallest incomplete composite (current_t < target_t)
    that matches the filter criteria.

    Args:
        client_id: Unique identifier for the requesting client
        priority: Minimum priority level (filters for priority >= this value)
        min_digits: Minimum number of digits
        max_digits: Maximum number of digits
        timeout_days: Work assignment expiration in days (default: 5)
        db: Database session

    Returns:
        ECMWorkResponse with assigned work or explanation if no work available
    """
    with transaction_scope(db, "get_ecm_work"):
        # Build query for suitable composites
        query = db.query(Composite).filter(
            and_(
                Composite.is_fully_factored == False,
                or_(Composite.is_prime.is_(None), Composite.is_prime == False),
                Composite.current_t_level < Composite.target_t_level
            )
        )

        # Apply priority filter
        if priority is not None:
            query = query.filter(Composite.priority >= priority)

        # Apply digit length filters
        if min_digits is not None:
            query = query.filter(Composite.digit_length >= min_digits)
        if max_digits is not None:
            query = query.filter(Composite.digit_length <= max_digits)

        # Exclude composites with active work assignments
        active_work_composites = db.query(WorkAssignment.composite_id).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).subquery()

        query = query.filter(~Composite.id.in_(active_work_composites))

        # Order by smallest digit length first, then oldest submission
        composite = query.order_by(
            Composite.digit_length.asc(),
            Composite.created_at.asc()
        ).first()

        # No work available
        if not composite:
            response_data = {
                "work_id": None,
                "composite_id": None,
                "composite": None,
                "digit_length": None,
                "current_t_level": None,
                "target_t_level": None,
                "expires_at": None,
                "message": "No suitable work available matching criteria"
            }
            content = json.dumps(response_data, default=str) + "\n"
            return Response(content=content, media_type="application/json")

        # Get previous ECM attempts for parameter calculation
        previous_attempts = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite.id
        ).all()

        # Calculate suggested ECM parameters using t-level targeting
        try:
            suggestion = t_level_calc.suggest_next_ecm_parameters(
                composite.target_t_level,
                composite.current_t_level,
                composite.digit_length
            )

            if suggestion['status'] == 'target_reached':
                # Use escalated parameters
                b1, b2, curves = _get_escalated_parameters(composite.digit_length, previous_attempts)
            else:
                b1, b2, curves = suggestion['b1'], suggestion['b2'], suggestion['curves']

        except Exception as e:
            logger.warning(f"T-level calculation failed for composite {composite.id}: {e}")
            # Fallback to basic parameters
            b1, b2, curves = 50000, 12500000, 100

        # Create work assignment
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=timeout_days)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=composite.id,
            client_id=client_id,
            method='ecm',
            b1=b1,
            b2=b2,
            curves_requested=curves,
            expires_at=expires_at,
            status='assigned'
        )

        db.add(work_assignment)
        db.flush()

        logger.info(f"Created ECM work assignment {work_id} for client {client_id}: "
                   f"{composite.digit_length}-digit composite, "
                   f"t{composite.current_t_level:.1f} → t{composite.target_t_level:.1f}")

        response_data = {
            "work_id": work_id,
            "composite_id": composite.id,
            "composite": composite.current_composite,
            "digit_length": composite.digit_length,
            "current_t_level": composite.current_t_level,
            "target_t_level": composite.target_t_level,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "message": "Assigned smallest incomplete composite"
        }
        content = json.dumps(response_data, default=str) + "\n"
        return Response(content=content, media_type="application/json")


def _get_escalated_parameters(digit_length: int, previous_attempts: list) -> tuple:
    """Get escalated ECM parameters when target t-level is reached."""
    # Standard ECM bounds table
    ECM_BOUNDS = [
        (30, 2000, 147000, 25),
        (35, 11000, 1900000, 90),
        (40, 50000, 12500000, 300),
        (45, 250000, 128000000, 700),
        (50, 1000000, 1000000000, 1800),
        (55, 3000000, 5000000000, 5100),
        (60, 11000000, 35000000000, 10600),
        (65, 43000000, 240000000000, 19300),
        (70, 110000000, 873000000000, 49000),
        (75, 260000000, 2600000000000, 124000),
        (80, 850000000, 11700000000000, 210000),
        (85, 2900000000, 55300000000000, 340000),
    ]

    max_b1_attempted = max((attempt.b1 for attempt in previous_attempts if attempt.method == 'ecm'), default=0)
    escalated_b1 = max_b1_attempted * 3

    # Find next level beyond what's been tried
    for max_digits, b1, b2, curves in ECM_BOUNDS:
        if digit_length <= max_digits and b1 > escalated_b1:
            return b1, b2, min(curves // 5, 200)

    # Fallback to highest available
    return ECM_BOUNDS[-1][1], ECM_BOUNDS[-1][2], 100
