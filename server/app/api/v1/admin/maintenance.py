"""
Maintenance and system administration routes.
"""
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy import update
from sqlalchemy.orm import Session, defer

from ....database import get_db
from ....dependencies import verify_admin_key, get_composite_service
from ....services.composites import CompositeService
from ....utils.transactions import transaction_scope
from ....utils.query_helpers import batch_fetch_attempts_by_composite

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state for background task
_recalculation_status: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "progress": 0,
    "total": 0,
    "completed": False,
    "result": None
}


@router.post("/composites/calculate-t-levels")
def calculate_t_levels_for_all_composites(
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
    from ....services.t_level_calculator import TLevelCalculator

    calculator = TLevelCalculator()

    # Get composites to update (defer heavy text columns not needed for t-level calc)
    base_query = db.query(Composite).options(
        defer(Composite.number), defer(Composite.current_composite)
    )
    if recalculate_all:
        composites = base_query.all()
        operation_type = "Recalculated all"
    else:
        composites = base_query.filter(
            Composite.target_t_level.is_(None)
        ).all()
        operation_type = "Updated new"

    updated_count = 0
    current_t_updated = 0

    # Batch fetch all attempts for these composites (eliminates N+1 queries)
    composite_ids = [c.id for c in composites]
    attempts_by_composite = batch_fetch_attempts_by_composite(
        db, composite_ids, exclude_superseded=True
    )

    with transaction_scope(db, "recalculate_t_levels"):
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

                # Recalculate current t-level from existing attempts (already fetched)
                # Use prior_t_level as starting point if set
                previous_attempts = attempts_by_composite.get(composite.id, [])

                starting_t = composite.prior_t_level or 0.0
                current_t = calculator.get_current_t_level_from_attempts(
                    previous_attempts, starting_t_level=starting_t
                )
                if current_t != composite.current_t_level:
                    composite.current_t_level = current_t
                    current_t_updated += 1

                updated_count += 1

            except Exception as e:
                # Skip problematic composites but continue processing
                logger.warning("Failed to update composite %s: %s", composite.id, e)
                continue

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


def _recalculate_all_t_levels_background():
    """
    Background task to recalculate all t-levels.

    Uses a new database session to avoid blocking the main request.
    """
    from ....models.composites import Composite
    from ....services.t_level_calculator import TLevelCalculator
    from ....database import SessionLocal
    from sqlalchemy.orm import defer

    global _recalculation_status

    # Create a new database session for this background task
    db = SessionLocal()

    try:
        calculator = TLevelCalculator()
        # Defer heavy text columns not needed for t-level calculation
        composites = db.query(Composite).options(
            defer(Composite.number), defer(Composite.current_composite)
        ).all()

        _recalculation_status["total"] = len(composites)
        _recalculation_status["progress"] = 0

        updated_count = 0
        current_t_updated = 0

        logger.info(f"Starting background t-level recalculation for {len(composites)} composites")

        # Batch fetch all attempts upfront (eliminates N+1 queries)
        composite_ids = [c.id for c in composites]
        attempts_by_composite = batch_fetch_attempts_by_composite(
            db, composite_ids, exclude_superseded=True
        )
        logger.info(f"Batch fetched attempts for {len(composite_ids)} composites")

        for idx, composite in enumerate(composites, 1):
            try:
                # Calculate/update target t-level
                target_t = calculator.calculate_target_t_level(
                    composite.digit_length,
                    special_form=None,
                    snfs_difficulty=composite.snfs_difficulty
                )
                composite.target_t_level = target_t

                # Recalculate current t-level from existing attempts (already fetched)
                # Use prior_t_level as starting point if set
                previous_attempts = attempts_by_composite.get(composite.id, [])

                starting_t = composite.prior_t_level or 0.0
                current_t = calculator.get_current_t_level_from_attempts(
                    previous_attempts, starting_t_level=starting_t
                )
                if current_t != composite.current_t_level:
                    composite.current_t_level = current_t
                    current_t_updated += 1

                updated_count += 1

                # Commit periodically to avoid huge transactions
                if idx % 100 == 0:
                    db.commit()
                    _recalculation_status["progress"] = idx
                    logger.info(f"T-level recalculation progress: {idx}/{len(composites)}")

            except Exception as e:
                logger.warning(f"Failed to update composite {composite.id}: {e}")
                db.rollback()  # Rollback on error to avoid blocking
                continue

        # Final commit
        db.commit()

        _recalculation_status["progress"] = len(composites)
        _recalculation_status["completed"] = True
        _recalculation_status["result"] = {
            "status": "completed",
            "composites_updated": updated_count,
            "current_t_levels_updated": current_t_updated,
            "message": f"Recalculated all t-levels for {updated_count} composites. Updated {current_t_updated} current t-level values."
        }

        logger.info(f"Background t-level recalculation completed: {updated_count} composites updated")

    except Exception as e:
        logger.error(f"Background t-level recalculation failed: {e}")
        _recalculation_status["completed"] = True
        _recalculation_status["result"] = {
            "status": "error",
            "message": f"T-level recalculation failed: {str(e)}"
        }
    finally:
        _recalculation_status["running"] = False
        db.close()


@router.post("/composites/recalculate-all-t-levels")
async def recalculate_all_t_levels(
    background_tasks: BackgroundTasks,
    _admin: bool = Depends(verify_admin_key)
):
    """
    Start background recalculation of ALL t-levels (both target and current) for all composites.

    This operation runs in the background to avoid blocking the server.
    Use GET /admin/composites/recalculate-status to check progress.

    Returns:
        Status indicating the background task has started
    """
    global _recalculation_status

    if _recalculation_status["running"]:
        return {
            "status": "already_running",
            "message": "T-level recalculation is already running",
            "progress": _recalculation_status["progress"],
            "total": _recalculation_status["total"]
        }

    # Reset status and start background task
    _recalculation_status = {
        "running": True,
        "started_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "total": 0,
        "completed": False,
        "result": None
    }

    # Start background task in a separate thread (FastAPI BackgroundTasks runs after response)
    thread = threading.Thread(target=_recalculate_all_t_levels_background, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "T-level recalculation started in background. Check /admin/composites/recalculate-status for progress.",
        "started_at": _recalculation_status["started_at"]
    }


@router.get("/composites/recalculate-status")
async def get_recalculation_status(
    _admin: bool = Depends(verify_admin_key)
):
    """
    Get the status of the background t-level recalculation task.

    Returns:
        Current status including progress and result
    """
    return {
        "running": _recalculation_status["running"],
        "started_at": _recalculation_status["started_at"],
        "progress": _recalculation_status["progress"],
        "total": _recalculation_status["total"],
        "completed": _recalculation_status["completed"],
        "result": _recalculation_status["result"]
    }


@router.post("/composites/{composite_id:int}/recalculate-t-level")
def recalculate_single_composite_t_level(
    composite_id: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Recalculate t-level for a single composite.

    This recalculates both current_t_level (from ECM attempts) and target_t_level
    (from composite size and SNFS difficulty). Useful after manual database changes
    or to verify calculations.

    Args:
        composite_id: ID of the composite to recalculate
        db: Database session
        composite_service: CompositeService instance
        _admin: Admin authentication

    Returns:
        JSON with old and new t-level values

    Raises:
        404: Composite not found
        500: Recalculation failed
    """
    from ....models import Composite
    from fastapi import HTTPException, status

    with transaction_scope(db, "recalculate_single_t_level"):
        # Get composite before recalculation
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Composite {composite_id} not found"
            )

        old_current_t = composite.current_t_level
        old_target_t = composite.target_t_level

        # Recalculate t-levels
        try:
            success = composite_service.update_t_level(db, composite_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to recalculate t-level for composite {composite_id}"
                )

            # Refresh to get updated values
            db.refresh(composite)
            new_current_t = composite.current_t_level
            new_target_t = composite.target_t_level

            logger.info(
                f"Recalculated t-level for composite {composite_id}: "
                f"current {old_current_t:.2f} → {new_current_t:.2f}, "
                f"target {old_target_t:.2f} → {new_target_t:.2f}"
            )

            return {
                "status": "success",
                "composite_id": composite_id,
                "old_current_t_level": round(old_current_t, 2) if old_current_t is not None else None,
                "new_current_t_level": round(new_current_t, 2) if new_current_t is not None else None,
                "old_target_t_level": round(old_target_t, 2) if old_target_t is not None else None,
                "new_target_t_level": round(new_target_t, 2) if new_target_t is not None else None,
                "message": f"T-level recalculated: current t{new_current_t:.2f}, target t{new_target_t:.2f}"
            }

        except ValueError as e:
            logger.error(f"Error recalculating t-level for composite {composite_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            ) from e


@router.post("/ecm-attempts/cleanup-logs")
def cleanup_old_attempt_logs(
    days: int = Query(30, ge=1, description="Clear raw_output for attempts older than this many days"),
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Null out raw_output on ECM attempts older than `days` days.

    The raw_output column stores full GMP-ECM/YAFU stdout for debugging. It
    grows unbounded and reaches multi-GB sizes; old logs are rarely useful, so
    we clear them but keep the attempt rows (they're referenced by foreign
    keys and drive t-level calculations).

    Note: PostgreSQL stores large TEXT in TOAST. Clearing the column won't
    immediately shrink on-disk size; autovacuum reclaims the space over time.
    Run VACUUM FULL or pg_repack manually if you need immediate reclaim.
    """
    from ....models.attempts import ECMAttempt

    cutoff = datetime.utcnow() - timedelta(days=days)

    with transaction_scope(db, "cleanup_old_attempt_logs"):
        result = db.execute(
            update(ECMAttempt)
            .where(ECMAttempt.created_at < cutoff)
            .where(ECMAttempt.raw_output.isnot(None))
            .values(raw_output=None)
        )
        rows_cleared = result.rowcount

    logger.info(
        "Cleared raw_output on %s ECM attempts older than %s days (cutoff: %s)",
        rows_cleared, days, cutoff.isoformat()
    )

    return {
        "status": "success",
        "rows_cleared": rows_cleared,
        "days": days,
        "cutoff": cutoff.isoformat(),
        "message": f"Cleared raw_output on {rows_cleared} ECM attempts older than {days} days"
    }


@router.post("/residues/cleanup-orphaned")
def cleanup_orphaned_residues(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Find and cleanup orphaned residue records (database entries without files).

    This scans all residue records in the database and checks if the
    corresponding file exists. If the file is missing, the residue is
    marked as 'expired' and any claims are released.

    Useful after deployments that may have lost residue files.

    Args:
        db: Database session
        _admin: Admin authentication

    Returns:
        JSON with count and list of cleaned up residue IDs
    """
    from ....models.residues import ECMResidue
    from fastapi import HTTPException, status

    with transaction_scope(db, "cleanup_orphaned_residues"):
        try:
            # Get residues that should have files (not completed or expired)
            # Completed residues have their files deleted intentionally
            active_residues = db.query(ECMResidue).filter(
                ECMResidue.status.in_(['available', 'claimed'])
            ).all()

            orphaned: List[Dict[str, Any]] = []

            for residue in active_residues:
                # Check if file exists
                file_path = Path(residue.storage_path)

                if not file_path.exists():
                    # File is missing - mark as orphaned
                    old_status = residue.status

                    # Mark as expired and release claim
                    residue.status = 'expired'
                    residue.claimed_by = None
                    residue.claimed_at = None

                    orphaned.append({
                        'id': residue.id,
                        'composite_id': residue.composite_id,
                        'old_status': old_status,
                        'storage_path': str(residue.storage_path),
                        'curve_count': residue.curve_count,
                        'b1': residue.b1
                    })

                    logger.info(
                        f"Marked orphaned residue {residue.id} as expired "
                        f"(was {old_status}, file missing: {residue.storage_path})"
                    )

            if orphaned:
                logger.info(f"Cleaned up {len(orphaned)} orphaned residues")
            else:
                logger.info("No orphaned residues found")

            return {
                "status": "success",
                "cleaned_up": len(orphaned),
                "total_checked": len(active_residues),
                "orphaned_residues": orphaned,
                "message": f"Cleaned up {len(orphaned)} orphaned residue(s) out of {len(active_residues)} active" if orphaned else f"No orphaned residues found (checked {len(active_residues)} active)"
            }

        except Exception as e:
            logger.error(f"Error cleaning up orphaned residues: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to cleanup orphaned residues: {str(e)}"
            ) from e
