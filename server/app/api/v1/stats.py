from typing import List, Optional, Literal, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session, defer
from sqlalchemy import and_, case, func

from ...database import get_db
from ...dependencies import get_composite_service
from ...schemas.composites import (
    CompositeStats, EffortLevel, ECMWorkSummary,
    BatchStatusRequest, BatchStatusResponse, CompositeBatchStatus,
    CompositeProgressItem, TopCompositesRequest, TopCompositesResponse
)
from ...models import Composite, ECMAttempt, Factor, ProjectComposite, Project
from ...services.composites import CompositeService
from ...utils.errors import get_or_404, not_found_error
from ...utils.calculations import ECMCalculations

router = APIRouter()

@router.get("/stats/{composite:path}", response_model=CompositeStats)
def get_composite_stats(
    composite: str = Path(..., description="The composite number to get stats for"),
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
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
    comp = get_or_404(
        composite_service.get_composite_by_number(db, composite),
        "Composite",
        composite
    )

    # Get all factors
    factors = db.query(Factor).filter(Factor.composite_id == comp.id).all()
    factors_list = [f.factor for f in factors]

    # Determine status
    status: Literal["composite", "sufficient", "fully_factored", "complete"]
    if comp.is_complete and comp.is_fully_factored:
        status = "complete"
    elif comp.is_complete:
        status = "sufficient"
    elif comp.is_fully_factored:
        status = "fully_factored"
    else:
        status = "composite"

    # Get ECM work summary (exclude superseded stage 1 attempts)
    # defer raw_output: large blob, only aggregate stats are needed
    attempts = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output)).filter(
        ECMAttempt.composite_id == comp.id,
        ECMAttempt.superseded_by.is_(None)
    ).all()

    total_attempts = len(attempts)
    total_curves = sum(attempt.curves_completed for attempt in attempts)
    last_attempt = max((attempt.created_at for attempt in attempts), default=None)

    # Group efforts by B1 level
    effort_data = ECMCalculations.group_attempts_by_b1_sorted(attempts)
    effort_by_level = [
        EffortLevel(b1=item['b1'], curves=item['curves'])
        for item in effort_data
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
        is_active=comp.is_active,
        status=status,
        factors_found=factors_list,
        ecm_work=ecm_work,
        projects=project_names
    )


@router.post("/composites/batch-status", response_model=BatchStatusResponse)
def get_batch_composite_status(
    request: BatchStatusRequest,
    db: Session = Depends(get_db)
):
    """
    Get t-level status for multiple composites in a single request.

    Returns current and target t-levels for each composite number.
    If a composite is not found in the database, returns found=False.
    """
    # Fetch all matching composites in a single query
    composites = db.query(Composite).filter(
        Composite.number.in_(request.numbers)
    ).all()

    # Build lookup dict by number
    comp_by_number = {c.number: c for c in composites}

    # Build results preserving original order
    results = []
    for number in request.numbers:
        comp = comp_by_number.get(number)
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


@router.post("/composites/top-progress", response_model=TopCompositesResponse)
def get_top_composites_by_progress(
    request: TopCompositesRequest,
    db: Session = Depends(get_db)
):
    """
    Get top composites ranked by ECM progress (current_t_level/target_t_level).

    Returns composites sorted by completion percentage (highest first),
    with optional filtering by project, priority, difficulty, and specific formulas.

    This is a POST endpoint to support large formula lists that would exceed URL length limits.

    Args:
        request: Request body with filtering and pagination options
                 - Difficulty filtering uses effective_difficulty = min(digit_length, snfs_difficulty)
                   if snfs_difficulty exists, otherwise digit_length
        db: Database session

    Returns:
        TopCompositesResponse with composites sorted by progress
    """
    # Extract values from request
    limit = request.limit
    project_name = request.project_name
    min_priority = request.min_priority
    include_factored = request.include_factored
    formulas = request.formulas
    min_difficulty = request.min_difficulty
    max_difficulty = request.max_difficulty

    # Build base query
    query = db.query(Composite)

    # Base filters - use Any to allow mixed SQLAlchemy expression types
    filters: List[Any] = [Composite.target_t_level.isnot(None)]

    if not include_factored:
        filters.append(Composite.is_fully_factored == False)

    if min_priority is not None:
        filters.append(Composite.priority >= min_priority)

    if formulas is not None and len(formulas) > 0:
        filters.append(Composite.number.in_(formulas))

    # Difficulty filters (effective difficulty = min of digit_length and snfs_difficulty)
    # Use coalesce to handle NULL: least(digit_length, snfs_difficulty) OR digit_length if NULL
    if min_difficulty is not None or max_difficulty is not None:
        effective_difficulty = func.coalesce(
            func.least(Composite.digit_length, Composite.snfs_difficulty),
            Composite.digit_length
        )
        if min_difficulty is not None:
            filters.append(effective_difficulty >= min_difficulty)
        if max_difficulty is not None:
            filters.append(effective_difficulty <= max_difficulty)

    # Project filter
    if project_name:
        # Find project
        project: Project = get_or_404(
            db.query(Project).filter(Project.name == project_name).first(),
            "Project",
            project_name
        )

        # Join with ProjectComposite to filter by project
        query = query.join(
            ProjectComposite,
            ProjectComposite.composite_id == Composite.id
        ).filter(ProjectComposite.project_id == project.id)

    # Apply filters
    query = query.filter(and_(*filters))

    # Get total count before pagination
    total = query.count()

    # Sort by ecm_progress (pre-computed column) and apply limit at DB level
    # This is much faster than loading all composites and sorting in Python
    composites = query.order_by(
        Composite.ecm_progress.desc().nulls_last()
    ).limit(limit).all()

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
            completion_pct=(comp.ecm_progress or 0) * 100,  # Use pre-computed column
            priority=comp.priority,
            is_fully_factored=comp.is_fully_factored,
            is_active=comp.is_active,
            projects=project_names
        ))

    return TopCompositesResponse(
        composites=result_items,
        total=total,
        limit=limit
    )
