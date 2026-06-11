"""
Factor management routes for admin.
"""
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key
from ....models.factors import Factor
from ....utils.errors import get_or_404

router = APIRouter()
logger = logging.getLogger(__name__)


@router.delete("/factors/{factor_id}")
def delete_factor(
    factor_id: int,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
) -> dict:
    """
    Delete a factor by ID.

    Args:
        factor_id: Factor ID to delete
        db: Database session
        _: Admin key verification

    Returns:
        Success message

    Raises:
        HTTPException: If factor not found
    """
    # Query the factor
    factor = get_or_404(
        db.query(Factor).filter(Factor.id == factor_id).first(),
        "Factor",
        str(factor_id)
    )

    # Store info for logging
    factor_value = factor.factor
    composite_id = factor.composite_id

    # Delete the factor
    db.delete(factor)
    db.commit()

    logger.info("Deleted factor %s (ID: %s) from composite %s",
                factor_value, factor_id, composite_id)

    return {
        "success": True,
        "message": f"Factor {factor_value} deleted successfully",
        "factor_id": factor_id
    }
