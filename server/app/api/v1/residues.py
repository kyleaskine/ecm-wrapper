"""
API endpoints for ECM residue management (decoupled two-stage ECM).

Provides endpoints for:
- Uploading stage 1 residue files
- Requesting stage 2 work
- Downloading residue files
- Completing stage 2 work
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Header, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
import logging

from ...database import get_db, SessionLocal
from ...dependencies import get_residue_manager
from ...models.attempts import ECMAttempt
from ...models.residues import ECMResidue
from ...models.composites import Composite
from ...models.projects import Project, ProjectComposite
from ...services.residue_manager import ResidueManager
from ...schemas.residues import (
    ResidueUploadResponse,
    ResidueWorkResponse,
    ResidueCompleteRequest,
    ResidueCompleteResponse,
    ResidueInfoResponse,
    ResidueStatsResponse
)
from ...utils.transactions import transaction_scope
from ...utils.errors import get_or_404

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=ResidueUploadResponse)
def upload_residue(
    file: UploadFile = File(..., description="ECM residue file from stage 1"),
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    stage1_attempt_id: Optional[int] = Query(None, description="ID of stage 1 ECM attempt to link"),
    b1: Optional[int] = Query(None, ge=250000, description="B1 if residues are from Prime95/mprime"),
    db: Session = Depends(get_db),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Upload a residue file after completing stage 1 ECM.

    Residues don't expire by time - they remain available until:
    - The composite is fully factored
    - A stage 2 worker completes processing them

    The server parses the file to extract:
    - Composite number (N=)
    - B1 parameter (for residues directly from GMP-ECM)
    - Parametrization (PARAM=)
    - Curve count

    Args:
        file: The residue file content
        client_id: ID of the uploading client (header)
        stage1_attempt_id: Optional ID of the stage 1 attempt for supersession tracking
        b1: B1 of a residue file that was created with Prime95/mprime
        db: Database session

    Returns:
        Parsed metadata and residue ID
    """
    with transaction_scope(db, "upload_residue"):
        # Read file content
        content = file.file.read()  # sync read; route runs in threadpool

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Limit file size (50 MB max)
        max_size = 50 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)} MB"
            )

        try:
            # Store residue and create database record
            residue = residue_manager.store_residue_file(
                db=db,
                file_content=content,
                client_id=client_id,
                stage1_attempt_id=stage1_attempt_id,
                b1=b1
            )

            # Get composite for response
            composite = db.query(Composite).filter(
                Composite.id == residue.composite_id
            ).first()

            logger.info(
                f"Client {client_id} uploaded residue ID {residue.id} "
                f"for composite {residue.composite_id}"
            )

            return ResidueUploadResponse(
                residue_id=residue.id,
                composite_id=residue.composite_id,
                composite=composite.current_composite if composite else "",
                b1=residue.b1,
                parametrization=residue.parametrization,
                curve_count=residue.curve_count,
                file_size_bytes=residue.file_size_bytes,
                message=f"Residue uploaded successfully. {residue.curve_count} curves ready for stage 2."
            )

        except ValueError as e:
            logger.warning(f"Failed to process residue upload from {client_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error uploading residue from {client_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store residue file"
            )


@router.get("/work", response_model=ResidueWorkResponse)
def get_residue_work(
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    min_target_tlevel: Optional[float] = Query(None, description="Minimum target t-level"),
    max_target_tlevel: Optional[float] = Query(None, description="Maximum target t-level"),
    min_priority: Optional[int] = Query(None, description="Minimum composite priority"),
    min_b1: Optional[int] = Query(None, ge=1, description="Minimum B1 bound of residue"),
    max_b1: Optional[int] = Query(None, ge=1, description="Maximum B1 bound of residue"),
    project: Optional[str] = Query(None, description="Project name filter (if not set, all projects)"),
    claim_timeout_hours: int = Query(72, ge=1, le=336, description="Hours until claim expires (default 72h/3 days)"),
    db: Session = Depends(get_db),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Request stage 2 work (an available residue file).

    Finds an available residue, claims it for this client, and returns
    the information needed to process stage 2.

    Args:
        client_id: ID of the requesting client
        min_target_tlevel: Minimum target t-level filter
        max_target_tlevel: Maximum target t-level filter
        min_priority: Minimum composite priority filter
        min_b1: Minimum B1 bound of residue
        max_b1: Maximum B1 bound of residue
        claim_timeout_hours: Hours until claim expires (default 72h/3 days, max 14 days)
        db: Database session

    Returns:
        Residue details and download URL, or message if none available
    """
    with transaction_scope(db, "get_residue_work"):
        # Find available residue. If a candidate already has a qualifying
        # stage 2 attempt (its completion call was lost), finalize it instead
        # of re-serving curves that were already run, and look for another.
        # Bounded so one request doesn't drain a large backlog of stale
        # residues; subsequent requests pick up where this one left off.
        residue = None
        for _ in range(5):
            candidate = residue_manager.get_available_work(
                db=db,
                client_id=client_id,
                min_target_tlevel=min_target_tlevel,
                max_target_tlevel=max_target_tlevel,
                min_priority=min_priority,
                min_b1=min_b1,
                max_b1=max_b1,
                project=project
            )

            if not candidate:
                break

            stale_attempt = residue_manager.find_completing_attempt(db, candidate)
            if stale_attempt is None:
                residue = candidate
                break

            logger.info(
                f"Residue {candidate.id} already has qualifying stage 2 attempt "
                f"{stale_attempt.id}; auto-completing instead of re-serving"
            )
            residue_manager.complete_residue(db, candidate.id, stale_attempt.id)

        if not residue:
            return ResidueWorkResponse(
                message="No residues available for stage 2 processing"
            )

        # Claim the residue
        try:
            residue = residue_manager.claim_residue(
                db=db,
                residue_id=residue.id,
                client_id=client_id,
                claim_timeout_hours=claim_timeout_hours
            )
        except ValueError as e:
            logger.warning(f"Failed to claim residue {residue.id}: {e}")
            return ResidueWorkResponse(
                message=f"Failed to claim residue: {str(e)}"
            )

        # Get composite details
        composite = db.query(Composite).filter(
            Composite.id == residue.composite_id
        ).first()

        # Suggest B2
        suggested_b2 = residue_manager.suggest_b2_for_residue(db, residue.id)

        logger.info(
            f"Client {client_id} claimed residue ID {residue.id} "
            f"for composite {residue.composite_id}"
        )

        return ResidueWorkResponse(
            residue_id=residue.id,
            composite_id=residue.composite_id,
            composite=composite.current_composite if composite else "",
            digit_length=composite.digit_length if composite else 0,
            b1=residue.b1,
            parametrization=residue.parametrization,
            curve_count=residue.curve_count,
            stage1_attempt_id=residue.stage1_attempt_id,
            download_url=f"/api/v1/residues/{residue.id}/download",
            suggested_b2=suggested_b2,
            expires_at=residue.expires_at,
            message=f"Claimed {residue.curve_count} curves for stage 2 (B1={residue.b1})"
        )


@router.get("/{residue_id}/download")
async def download_residue(
    residue_id: int,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
):
    """
    Download a residue file for stage 2 processing.

    Only the client who claimed the residue can download it.
    Uses a manual DB session so the connection is released before
    file streaming begins (avoids holding a connection during transfer).
    """
    from pathlib import Path

    # Use a manual session so we release the DB connection before streaming
    db = SessionLocal()
    try:
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Residue {residue_id} not found"
            )

        if residue.claimed_by != client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Residue {residue_id} is not claimed by client {client_id}"
            )

        storage_path = residue.storage_path
    finally:
        db.close()

    # DB connection is now released — file transfer won't hold it open
    file_path = Path(storage_path) if storage_path else None
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Residue file not found on disk"
        )

    logger.info(f"Client {client_id} downloading residue {residue_id}")

    response = FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename=f"residue_{residue_id}.txt"
    )
    # Prevent Cloudflare from caching/intercepting file downloads
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/{residue_id}/complete", response_model=ResidueCompleteResponse)
def complete_residue(
    residue_id: int,
    request: ResidueCompleteRequest,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    db: Session = Depends(get_db),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Mark a residue as completed after stage 2 finishes.

    This:
    1. Links the stage 2 attempt to supersede the stage 1 attempt
    2. Recalculates the composite's t-level (excluding superseded S1)
    3. Deletes the residue file
    4. Updates residue status to 'completed'

    Args:
        residue_id: ID of the completed residue
        request: Contains the stage 2 attempt ID
        client_id: ID of the client completing the work
        db: Database session

    Returns:
        Completion confirmation with updated t-level
    """
    with transaction_scope(db, "complete_residue"):
        residue = get_or_404(
            db.query(ECMResidue).filter(ECMResidue.id == residue_id).first(),
            "Residue",
            str(residue_id)
        )

        # Authorization: the claim holder may complete. If the claim lapsed
        # (released by expiry cleanup) or the residue was finalized by another
        # path (submit-time auto-complete, reconcile), accept a client whose
        # own attempt provably consumed this residue file - the checksum and
        # composite must match and the attempt must belong to this client.
        if residue.claimed_by != client_id:
            attempt = db.query(ECMAttempt).filter(
                ECMAttempt.id == request.stage2_attempt_id
            ).first()
            attempt_authorizes = (
                attempt is not None
                and attempt.client_id == client_id
                and attempt.residue_checksum == residue.checksum
                and attempt.composite_id == residue.composite_id
            )
            if not attempt_authorizes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"Residue {residue_id} is not claimed by client {client_id} "
                        f"and attempt {request.stage2_attempt_id} does not match "
                        f"this residue and client"
                    )
                )

        try:
            completed_residue, new_t_level = residue_manager.complete_residue(
                db=db,
                residue_id=residue_id,
                stage2_attempt_id=request.stage2_attempt_id
            )

            logger.info(
                f"Client {client_id} completed residue {residue_id} "
                f"with stage2_attempt {request.stage2_attempt_id}"
            )

            return ResidueCompleteResponse(
                residue_id=completed_residue.id,
                stage1_attempt_id=completed_residue.stage1_attempt_id,
                stage2_attempt_id=request.stage2_attempt_id,
                composite_id=completed_residue.composite_id,
                new_t_level=new_t_level,
                message=f"Stage 2 complete. Residue file deleted. T-level updated to {new_t_level:.2f}" if new_t_level else "Stage 2 complete. Residue file deleted."
            )

        except ValueError as e:
            logger.warning(f"Failed to complete residue {residue_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.delete("/{residue_id}/claim")
def abandon_residue_claim(
    residue_id: int,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    db: Session = Depends(get_db),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Release a claimed residue back to the available pool.

    Args:
        residue_id: ID of the residue to release
        client_id: ID of the client releasing (must match claimer)
        db: Database session

    Returns:
        Confirmation message
    """
    with transaction_scope(db, "abandon_residue_claim"):
        try:
            residue = residue_manager.release_claim(db, residue_id, client_id)

            logger.info(f"Client {client_id} released claim on residue {residue_id}")

            return {
                "residue_id": residue_id,
                "status": "available",
                "message": "Residue claim released successfully"
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.get("/{residue_id}", response_model=ResidueInfoResponse)
def get_residue_info(
    residue_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a residue.

    Args:
        residue_id: ID of the residue
        db: Database session

    Returns:
        Residue details
    """
    residue = get_or_404(
        db.query(ECMResidue).filter(ECMResidue.id == residue_id).first(),
        "Residue",
        str(residue_id)
    )

    composite = db.query(Composite).filter(
        Composite.id == residue.composite_id
    ).first()

    return ResidueInfoResponse(
        residue_id=residue.id,
        composite_id=residue.composite_id,
        composite=composite.current_composite if composite else "",
        client_id=residue.client_id,
        stage1_attempt_id=residue.stage1_attempt_id,
        b1=residue.b1,
        parametrization=residue.parametrization,
        curve_count=residue.curve_count,
        file_size_bytes=residue.file_size_bytes,
        status=residue.status,
        created_at=residue.created_at,
        expires_at=residue.expires_at,
        claimed_at=residue.claimed_at,
        claimed_by=residue.claimed_by,
        completed_at=residue.completed_at
    )


@router.get("/stats/summary", response_model=ResidueStatsResponse)
def get_residue_stats(
    db: Session = Depends(get_db),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Get statistics about residues in the system.

    Returns:
        Counts by status and total pending curves
    """
    stats = residue_manager.get_stats(db)
    return ResidueStatsResponse(**stats)
