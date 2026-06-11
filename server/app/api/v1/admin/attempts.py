"""
Admin routes for managing ECM attempts (curves).
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, defer

from ....database import get_db
from ....dependencies import verify_admin_key, get_t_level_calculator
from ....models.attempts import ECMAttempt
from ....models.composites import Composite
from ....models.factors import Factor
from ....models.residues import ECMResidue
from ....services.t_level_calculator import TLevelCalculator
from ....templates import templates

router = APIRouter()


@router.get("/composites/{composite_id}/attempts-fragment", response_class=HTMLResponse)
def get_attempts_fragment(
    composite_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    limit: int = Query(25, ge=1, le=100)
):
    """
    Return HTML fragment of attempt detail rows for a composite.
    Used by the admin dashboard for lazy-loading on row expand.
    """
    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        raise HTTPException(status_code=404, detail="Composite not found")

    attempts = db.query(ECMAttempt).options(
        defer(ECMAttempt.raw_output)
    ).filter(
        ECMAttempt.composite_id == composite_id
    ).order_by(desc(ECMAttempt.created_at)).limit(limit).all()

    # Batch fetch factor counts for attempts that found factors
    attempt_ids = [a.id for a in attempts if a.factor_found]
    factor_counts: dict = {}
    if attempt_ids:
        rows = db.query(
            Factor.found_by_attempt_id,
            func.count(Factor.id).label('factor_count')
        ).filter(
            Factor.found_by_attempt_id.in_(attempt_ids)
        ).group_by(Factor.found_by_attempt_id).all()
        factor_counts = {row.found_by_attempt_id: row.factor_count for row in rows}

    html = templates.env.get_template("components/attempt_detail_rows.html").render(
        composite_id=composite_id,
        attempts=attempts,
        factor_counts=factor_counts
    )
    return HTMLResponse(content=html)


@router.delete("/attempts/{attempt_id}")
def delete_attempt(
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

    # Delete any residues that reference this attempt (foreign key constraint)
    db.query(ECMResidue).filter(ECMResidue.stage1_attempt_id == attempt_id).delete()

    # Delete the attempt
    db.delete(attempt)

    # Flush the deletion to ensure it's excluded from the recalculation query
    db.flush()

    # Recalculate t-level for the composite (both operations in one transaction)
    new_t_level = t_level_calc.recalculate_composite_t_level(db, composite)

    # Commit both operations atomically
    db.commit()

    return {
        "status": "deleted",
        "attempt_id": attempt_id,
        "composite_id": composite_id,
        "old_t_level": round(old_t_level, 2),
        "new_t_level": round(new_t_level, 2),
        "message": f"Deleted attempt {attempt_id} and recalculated t-level for composite {composite_id}"
    }
