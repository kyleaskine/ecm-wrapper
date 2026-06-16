"""
Admin-specific residue management endpoints.
Provides functionality to delete residues and trigger cleanup of expired entries.
"""
import logging
from fastapi import APIRouter, Depends, Header
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, get_residue_manager, get_composite_service
from ....models.attempts import ECMAttempt
from ....models.composites import Composite
from ....models.residues import ECMResidue
from ....services.composites import CompositeService
from ....services.residue_manager import ResidueManager
from ....utils.errors import get_or_404
from ....utils.file_cleanup import stage_residue_file_deletion
from ....utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)


@router.delete("/residues/{residue_id}")
def delete_residue(
    residue_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Admin endpoint to delete a residue (even if claimed/completed).

    This forcibly removes a residue record and its associated file
    regardless of status. Use with caution.

    Args:
        residue_id: ID of the residue to delete
        db: Database session
        _admin: Admin authentication check
        residue_manager: Residue manager service

    Returns:
        Success message with deleted residue info

    Raises:
        HTTPException: If residue not found
    """
    # Get residue
    residue = get_or_404(
        db.query(ECMResidue).filter(ECMResidue.id == residue_id).first(),
        "Residue",
        str(residue_id)
    )

    # Store info for response
    composite_id = residue.composite_id
    status = residue.status
    storage_path = residue.storage_path

    # Delete database record within transaction
    with transaction_scope(db, "delete_residue"):
        # Stage the file deletion so it runs only after the row delete
        # commits; an inline remove before commit would orphan the file if
        # the delete rolls back (consistent with complete_residue/cleanup_*).
        if storage_path:
            stage_residue_file_deletion(db, storage_path)
        db.delete(residue)

    return {
        "success": True,
        "message": f"Residue {residue_id} deleted",
        "residue_id": residue_id,
        "composite_id": composite_id,
        "previous_status": status
    }


@router.delete("/residues/{residue_id}/release")
def admin_release_residue_claim(
    residue_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Admin endpoint to release a claimed residue back to the available pool.

    Unlike the client endpoint, this doesn't require the X-Client-ID header
    and can release any claimed residue regardless of who claimed it.

    Args:
        residue_id: ID of the residue to release
        db: Database session
        _admin: Admin authentication check
        residue_manager: Residue manager service

    Returns:
        Success message with released residue info

    Raises:
        HTTPException: If residue not found or not claimed
    """
    with transaction_scope(db, "admin_release_residue_claim"):
        # Lock composite -> residue (the global order) and refresh before
        # the status check: a stale 'claimed' read could otherwise overwrite
        # a concurrent completion, recreating an available residue whose file
        # is already gone and whose stage 1 is already superseded.
        residue = get_or_404(
            residue_manager.lock_residue(db, residue_id),
            "Residue",
            str(residue_id)
        )

        if residue.status != 'claimed':
            return {
                "success": False,
                "message": f"Residue {residue_id} is not claimed (status: {residue.status})",
                "residue_id": residue_id,
                "status": residue.status
            }

        previous_claimer = residue.claimed_by
        residue.status = 'available'
        residue.claimed_by = None
        residue.claimed_at = None
        residue.expires_at = None

    logger.info(f"Admin released claim on residue {residue_id} (was claimed by {previous_claimer})")

    return {
        "success": True,
        "message": f"Residue {residue_id} released back to available pool",
        "residue_id": residue_id,
        "previous_claimer": previous_claimer,
        "status": "available"
    }


@router.post("/residues/cleanup")
def cleanup_residues(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Manually trigger cleanup of residues.

    This performs two types of cleanup:
    1. Releases expired claims (claims that timed out without completion)
    2. Deletes residues for fully factored composites

    Available residues don't expire by time - only claims have timeouts.

    Args:
        db: Database session
        _admin: Admin authentication check
        residue_manager: Residue manager service

    Returns:
        Cleanup summary with counts
    """
    # Release expired claims (claimed residues that timed out)
    claims_released = residue_manager.cleanup_expired_claims(db)

    # Delete residues for fully factored or completed composites
    factored_cleaned = residue_manager.cleanup_factored_composites(db)

    total_cleaned = claims_released + factored_cleaned

    # Commit all changes
    db.commit()

    return {
        "success": True,
        "message": f"Released {claims_released} expired claim(s), cleaned {factored_cleaned} factored composite residue(s)",
        "claims_released": claims_released,
        "factored_composites_cleaned": factored_cleaned,
        "total_cleaned": total_cleaned
    }


@router.post("/residues/reconcile")
def reconcile_residues(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """
    One-time/periodic repair for residues whose completion call was lost.

    Two passes:
    1. Available/claimed residues that already have a qualifying stage 2
       attempt (factor found, or >=75% curves with sane B2) are completed
       with that attempt - superseding stage 1 and any duplicate attempts.
    2. Leftover unsuperseded stage 2 attempts sharing a residue_checksum
       (e.g. on residues completed before orphan handling existed) are
       superseded by the best attempt in the group.

    T-levels are recalculated ONLY for the composites actually touched -
    never the recalc-all path, which takes hours on this server.

    Returns:
        Report of residues completed, attempts superseded, and per-composite
        t-level changes
    """
    completed_residues = []
    superseded_attempts = []
    affected_composite_ids = set()

    with transaction_scope(db, "reconcile_residues"):
        # Pass 1: finalize stuck residues that already earned completion
        has_linked_attempt = (
            db.query(ECMAttempt.id)
            .filter(ECMAttempt.residue_checksum == ECMResidue.checksum)
            .exists()
        )
        stuck_residues = db.query(ECMResidue).filter(
            ECMResidue.status.in_(['available', 'claimed']),
            has_linked_attempt
        ).all()

        for residue in stuck_residues:
            attempt = residue_manager.find_completing_attempt(db, residue)
            if attempt is None:
                continue
            residue_manager.complete_residue(
                db, residue.id, attempt.id, recalculate_t_level=False
            )
            completed_residues.append({
                "residue_id": residue.id,
                "attempt_id": attempt.id,
                "composite_id": residue.composite_id,
                "factor_found": attempt.factor_found is not None
            })
            affected_composite_ids.add(residue.composite_id)

        # Pass 2: supersede leftover duplicate attempts sharing a checksum.
        # Pass 1's completions already superseded their checksum-mates, so
        # the superseded_by IS NULL filter only picks up older leftovers.
        stage2_only = or_(ECMAttempt.b2.is_(None), ECMAttempt.b2 != 0)
        dup_checksums = (
            db.query(ECMAttempt.residue_checksum)
            .filter(
                ECMAttempt.residue_checksum.isnot(None),
                ECMAttempt.superseded_by.is_(None),
                stage2_only
            )
            .group_by(ECMAttempt.residue_checksum)
            .having(func.count(ECMAttempt.id) > 1)
            .all()
        )

        for (checksum,) in dup_checksums:
            attempts = db.query(ECMAttempt).filter(
                ECMAttempt.residue_checksum == checksum,
                ECMAttempt.superseded_by.is_(None),
                stage2_only
            ).order_by(
                ECMAttempt.factor_found.isnot(None).desc(),
                ECMAttempt.curves_completed.desc(),
                ECMAttempt.id.asc()
            ).all()

            # The completion path resolves the winner via stage1.superseded_by,
            # so the attempt stage 1 points at must stay unsuperseded - dethroning
            # it here would let a delayed old-client completion call create a
            # supersession cycle that excludes both attempts from the t-level.
            designated = (
                db.query(ECMAttempt.superseded_by)
                .join(ECMResidue, ECMResidue.stage1_attempt_id == ECMAttempt.id)
                .filter(
                    ECMResidue.checksum == checksum,
                    ECMAttempt.superseded_by.isnot(None)
                )
                .first()
            )
            designated_id = designated[0] if designated else None
            winner = next((a for a in attempts if a.id == designated_id), attempts[0])

            for loser in attempts:
                if loser.id == winner.id:
                    continue
                loser.superseded_by = winner.id
                superseded_attempts.append({
                    "attempt_id": loser.id,
                    "superseded_by": winner.id,
                    "composite_id": loser.composite_id
                })
                affected_composite_ids.add(loser.composite_id)

        db.flush()

        # Targeted recalculation: only the composites touched above
        composites_recalculated = []
        if affected_composite_ids:
            composites = db.query(Composite).filter(
                Composite.id.in_(affected_composite_ids)
            ).all()
            t_levels_before = {c.id: c.current_t_level for c in composites}
            for composite in composites:
                composite_service.update_t_level(db, composite.id)
            composites_recalculated = [
                {
                    "composite_id": c.id,
                    "t_level_before": t_levels_before[c.id],
                    "t_level_after": c.current_t_level
                }
                for c in composites
            ]

    return {
        "success": True,
        "message": (
            f"Completed {len(completed_residues)} stale residue(s), "
            f"superseded {len(superseded_attempts)} duplicate attempt(s), "
            f"recalculated {len(composites_recalculated)} composite t-level(s)"
        ),
        "residues_completed": completed_residues,
        "attempts_superseded": superseded_attempts,
        "composites_recalculated": composites_recalculated
    }
