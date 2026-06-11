"""
Admin-specific residue management endpoints.
Provides functionality to delete residues and trigger cleanup of expired entries.
"""
import logging
import os
from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, get_residue_manager
from ....models.residues import ECMResidue
from ....services.residue_manager import ResidueManager
from ....utils.errors import get_or_404
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

    # Delete file if exists
    if residue.storage_path and os.path.exists(residue.storage_path):
        try:
            os.remove(residue.storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete residue file {residue.storage_path}: {e}")

    # Store info for response
    composite_id = residue.composite_id
    status = residue.status

    # Delete database record within transaction
    with transaction_scope(db, "delete_residue"):
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
    residue = get_or_404(
        db.query(ECMResidue).filter(ECMResidue.id == residue_id).first(),
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

    # Store info for response
    previous_claimer = residue.claimed_by

    # Release the claim
    with transaction_scope(db, "admin_release_residue_claim"):
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
