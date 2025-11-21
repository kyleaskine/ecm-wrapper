"""
Maintenance and system administration routes.
"""
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, get_composite_service
from ....services.composites import CompositeService
from ....utils.transactions import transaction_scope

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
    from ....models.attempts import ECMAttempt
    from ....services.t_level_calculator import TLevelCalculator
    from ....database import SessionLocal

    global _recalculation_status

    # Create a new database session for this background task
    db = SessionLocal()

    try:
        calculator = TLevelCalculator()
        composites = db.query(Composite).all()

        _recalculation_status["total"] = len(composites)
        _recalculation_status["progress"] = 0

        updated_count = 0
        current_t_updated = 0

        logger.info(f"Starting background t-level recalculation for {len(composites)} composites")

        for idx, composite in enumerate(composites, 1):
            try:
                # Calculate/update target t-level
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

                # Commit very frequently and yield to allow other requests
                if idx % 5 == 0:
                    db.commit()
                    _recalculation_status["progress"] = idx
                    logger.info(f"T-level recalculation progress: {idx}/{len(composites)}")

                    # Sleep briefly to yield control and allow other requests to process
                    time.sleep(0.1)

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


@router.post("/composites/{composite_id}/recalculate-t-level")
async def recalculate_single_composite_t_level(
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


@router.post("/residues/cleanup-orphaned")
async def cleanup_orphaned_residues(
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
            # Get all residues
            all_residues = db.query(ECMResidue).all()

            orphaned: List[Dict[str, Any]] = []

            for residue in all_residues:
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
                        'claimed_by': residue.claimed_by if old_status == 'claimed' else None,
                        'storage_path': str(residue.storage_path),
                        'curves': residue.curves,
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
                "orphaned_residues": orphaned,
                "message": f"Cleaned up {len(orphaned)} orphaned residue(s)" if orphaned else "No orphaned residues found"
            }

        except Exception as e:
            logger.error(f"Error cleaning up orphaned residues: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to cleanup orphaned residues: {str(e)}"
            ) from e
