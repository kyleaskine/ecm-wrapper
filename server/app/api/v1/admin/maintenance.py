"""
Maintenance and system administration routes.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/composites/calculate-t-levels")
async def calculate_t_levels_for_all_composites(
    recalculate_all: bool = False,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Calculate and populate t-levels for all composites in the database.

    Uses the 4/13 * digits formula with SNFS discounts for special forms.
    Uses real t-level executable for current progress calculation.

    Args:
        recalculate_all: If True, recalculates current t-levels for ALL composites
                        If False, only calculates for composites missing target t-levels

    Returns:
        Statistics about t-level calculations performed
    """
    from ....models.composites import Composite
    from ....models.attempts import ECMAttempt
    from ....services.t_level_calculator import TLevelCalculator

    calculator = TLevelCalculator()

    # Get composites to update
    if recalculate_all:
        composites = db.query(Composite).all()
        operation_type = "Recalculated all"
    else:
        composites = db.query(Composite).filter(
            Composite.target_t_level.is_(None)
        ).all()
        operation_type = "Updated new"

    updated_count = 0
    current_t_updated = 0

    for composite in composites:
        try:
            # Calculate/update target t-level if not set or if recalculating all
            if composite.target_t_level is None or recalculate_all:
                target_t = calculator.calculate_target_t_level(
                    composite.digit_length,
                    special_form=None,
                    snfs_difficulty=composite.snfs_difficulty
                )
                composite.target_t_level = target_t

            # Recalculate current t-level from existing attempts
            previous_attempts = db.query(ECMAttempt).filter(
                ECMAttempt.composite_id == composite.id
            ).all()

            current_t = calculator.get_current_t_level_from_attempts(previous_attempts)
            if current_t != composite.current_t_level:
                composite.current_t_level = current_t
                current_t_updated += 1

            updated_count += 1

        except Exception as e:
            # Skip problematic composites but continue processing
            logger.warning("Failed to update composite %s: %s", composite.id, e)
            continue

    # Commit all changes
    db.commit()

    return {
        "status": "completed",
        "composites_updated": updated_count,
        "current_t_levels_updated": current_t_updated,
        "operation_type": operation_type,
        "message": (
            f"{operation_type} t-levels for {updated_count} composites. "
            f"Updated {current_t_updated} current t-level values using real executable."
        )
    }


@router.post("/composites/recalculate-all-t-levels")
async def recalculate_all_t_levels(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Force recalculation of ALL t-levels (both target and current) for all composites.

    This will replace any existing current t-level values with fresh calculations
    using the real t-level executable.

    Returns:
        Statistics about t-level recalculations performed
    """
    return await calculate_t_levels_for_all_composites(recalculate_all=True, db=db)
