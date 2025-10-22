"""
Admin routes for managing ECM attempts (curves).
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, get_t_level_calculator
from ....models.attempts import ECMAttempt
from ....models.composites import Composite
from ....services.t_level_calculator import TLevelCalculator

router = APIRouter()


@router.delete("/attempts/{attempt_id}")
async def delete_attempt(
    attempt_id: int,
    composite_id: int = Query(..., description="Composite ID for t-level recalculation"),
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    t_level_calc: TLevelCalculator = Depends(get_t_level_calculator)
):
    """
    Delete an ECM attempt (curve) and recalculate the t-level for the composite.

    Args:
        attempt_id: ID of the ECM attempt to delete
        composite_id: ID of the composite (for t-level recalculation)

    Returns:
        Status and updated t-level information
    """
    # Get the attempt
    attempt = db.query(ECMAttempt).filter(ECMAttempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    # Get the composite
    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        raise HTTPException(status_code=404, detail="Composite not found")

    # Verify the attempt belongs to this composite
    if attempt.composite_id != composite_id:
        raise HTTPException(
            status_code=400,
            detail=f"Attempt {attempt_id} does not belong to composite {composite_id}"
        )

    # Store old t-level
    old_t_level = composite.current_t_level or 0.0

    # Delete the attempt
    db.delete(attempt)
    db.commit()

    # Recalculate t-level for the composite
    new_t_level = t_level_calc.recalculate_composite_t_level(db, composite)

    return {
        "status": "deleted",
        "attempt_id": attempt_id,
        "composite_id": composite_id,
        "old_t_level": round(old_t_level, 2),
        "new_t_level": round(new_t_level, 2),
        "message": f"Deleted attempt {attempt_id} and recalculated t-level for composite {composite_id}"
    }
