from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional

from ...database import get_db
from ...schemas.composites import (
    CompositeStats, EffortLevel, ECMWorkSummary,
    BatchStatusRequest, BatchStatusResponse, CompositeBatchStatus,
    CompositeProgressItem, TopCompositesResponse
)
from ...models import Composite, ECMAttempt, Factor, ProjectComposite, Project
from ...services.composites import CompositeService

router = APIRouter()

@router.get("/stats/{composite}", response_model=CompositeStats)
async def get_composite_stats(
    composite: str = Path(..., description="The composite number to get stats for"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive statistics for a composite number.
    
    Returns information about:
    - Composite properties (bit/digit length, factorization status)
    - All known factors
    - Summary of factorization work performed
    - Associated projects
    """
    # Get composite from database
    comp = CompositeService.get_composite_by_number(db, composite)
    if not comp:
        raise HTTPException(status_code=404, detail="Composite not found in database")
    
    # Get all factors
    factors = db.query(Factor).filter(Factor.composite_id == comp.id).all()
    factors_list = [f.factor for f in factors]
    
    # Determine status
    if comp.is_prime:
        status = "prime"
    elif comp.is_fully_factored:
        status = "fully_factored"
    else:
        status = "composite"
    
    # Get ECM work summary
    attempts = db.query(ECMAttempt).filter(ECMAttempt.composite_id == comp.id).all()
    
    total_attempts = len(attempts)
    total_curves = sum(attempt.curves_completed for attempt in attempts)
    last_attempt = max((attempt.created_at for attempt in attempts), default=None)
    
    # Group efforts by B1 level
    effort_groups = {}
    for attempt in attempts:
        b1 = attempt.b1
        if b1 not in effort_groups:
            effort_groups[b1] = 0
        effort_groups[b1] += attempt.curves_completed
    
    effort_by_level = [
        EffortLevel(b1=b1, curves=curves) 
        for b1, curves in sorted(effort_groups.items())
    ]
    
    ecm_work = ECMWorkSummary(
        total_attempts=total_attempts,
        total_curves=total_curves,
        effort_by_level=effort_by_level,
        last_attempt=last_attempt
    )
    
    # Get associated projects
    project_links = db.query(ProjectComposite).filter(
        ProjectComposite.composite_id == comp.id
    ).all()
    
    project_names = []
    for link in project_links:
        project = db.query(Project).filter(Project.id == link.project_id).first()
        if project:
            project_names.append(project.name)
    
    return CompositeStats(
        composite=comp.number,
        current_composite=comp.current_composite,
        digit_length=comp.digit_length,
        has_snfs_form=comp.has_snfs_form,
        snfs_difficulty=comp.snfs_difficulty,
        target_t_level=comp.target_t_level,
        current_t_level=comp.current_t_level,
        priority=comp.priority,
        status=status,
        factors_found=factors_list,
        ecm_work=ecm_work,
        projects=project_names
    )


@router.post("/composites/batch-status", response_model=BatchStatusResponse)
async def get_batch_composite_status(
    request: BatchStatusRequest,
    db: Session = Depends(get_db)
):
    """
    Get t-level status for multiple composites in a single request.

    Returns current and target t-levels for each composite number.
    If a composite is not found in the database, returns found=False.
    """
    results = []

    for number in request.numbers:
        # Try to find composite by number
        comp = db.query(Composite).filter(Composite.number == number).first()

        if comp:
            results.append(CompositeBatchStatus(
                number=number,
                target_t_level=comp.target_t_level,
                current_t_level=comp.current_t_level,
                digit_length=comp.digit_length,
                has_snfs_form=comp.has_snfs_form,
                snfs_difficulty=comp.snfs_difficulty,
                found=True
            ))
        else:
            results.append(CompositeBatchStatus(
                number=number,
                target_t_level=None,
                current_t_level=None,
                digit_length=None,
                has_snfs_form=None,
                snfs_difficulty=None,
                found=False
            ))

    return BatchStatusResponse(composites=results)


@router.get("/composites/top-progress", response_model=TopCompositesResponse)
async def get_top_composites_by_progress(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of composites to return"),
    project_name: Optional[str] = Query(None, description="Filter by project name"),
    min_priority: Optional[int] = Query(None, description="Minimum priority level"),
    include_factored: bool = Query(False, description="Include fully factored composites"),
    db: Session = Depends(get_db)
):
    """
    Get top composites ranked by ECM progress (current_t_level/target_t_level).

    Returns composites sorted by completion percentage (highest first),
    with optional filtering by project and priority.

    Args:
        limit: Maximum number of composites to return (default 50, max 500)
        project_name: Optional project name filter
        min_priority: Optional minimum priority filter
        include_factored: Include fully factored composites (default False)
        db: Database session

    Returns:
        TopCompositesResponse with composites sorted by progress
    """
    # Build base query
    query = db.query(Composite)

    # Base filters
    filters = [Composite.target_t_level.isnot(None)]

    if not include_factored:
        filters.append(~Composite.is_fully_factored)

    if min_priority is not None:
        filters.append(Composite.priority >= min_priority)

    # Project filter
    if project_name:
        # Find project
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_name}' not found"
            )

        # Join with ProjectComposite to filter by project
        query = query.join(
            ProjectComposite,
            ProjectComposite.composite_id == Composite.id
        ).filter(ProjectComposite.project_id == project.id)

    # Apply filters
    composites = query.filter(and_(*filters)).all()

    # Calculate completion percentage and sort
    def get_completion_pct(comp):
        if comp.target_t_level and comp.target_t_level > 0:
            current_t = comp.current_t_level or 0.0
            return (current_t / comp.target_t_level) * 100
        return 0.0

    composites.sort(key=get_completion_pct, reverse=True)
    total = len(composites)
    composites = composites[:limit]

    # Get project associations for each composite
    result_items = []
    for comp in composites:
        # Get associated projects
        project_links = db.query(ProjectComposite).filter(
            ProjectComposite.composite_id == comp.id
        ).all()

        project_names = []
        for link in project_links:
            proj = db.query(Project).filter(Project.id == link.project_id).first()
            if proj:
                project_names.append(proj.name)

        result_items.append(CompositeProgressItem(
            id=comp.id,
            number=comp.number,
            current_composite=comp.current_composite,
            digit_length=comp.digit_length,
            has_snfs_form=comp.has_snfs_form,
            snfs_difficulty=comp.snfs_difficulty,
            target_t_level=comp.target_t_level,
            current_t_level=comp.current_t_level,
            completion_pct=get_completion_pct(comp),
            priority=comp.priority,
            is_fully_factored=comp.is_fully_factored,
            projects=project_names
        ))

    return TopCompositesResponse(
        composites=result_items,
        total=total,
        limit=limit
    )