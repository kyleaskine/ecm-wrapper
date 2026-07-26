from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from slowapi import Limiter
from sqlalchemy.orm import Session, defer
from sqlalchemy import desc, func, case, distinct
from typing import Any, List, Optional
from datetime import datetime, timedelta

from ...database import get_db
from ...dependencies import get_composite_service
from ...models import Composite, ECMAttempt, Factor
from ...models.projects import Project, ProjectComposite
from ...services.composites import CompositeService
from ...templates import templates
from ...utils.query_helpers import get_aggregated_attempts, prefetch_factor_counts_for_attempts, get_residues_filtered, calculate_pagination
from ...constants import get_b1_above_tlevel
from ...rate_limit import get_real_client_ip

router = APIRouter()
limiter = Limiter(key_func=get_real_client_ip)

@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    priority: Optional[int] = Query(None, ge=1, description="Filter by priority level"),
    project: Optional[int] = Query(None, ge=1, description="Filter by project ID")
):
    """
    Simple dashboard showing all factorization results.
    Optionally filter by priority level and/or project.
    """
    # Fetch all projects for dropdown
    all_projects = db.query(Project).order_by(Project.name).all()

    # Build query for composites with optional priority/project filter
    composites_query = db.query(Composite)
    if priority is not None:
        composites_query = composites_query.filter(Composite.priority == priority)
    if project is not None:
        composites_query = composites_query.join(
            ProjectComposite, ProjectComposite.composite_id == Composite.id
        ).filter(ProjectComposite.project_id == project)
    composites = composites_query.order_by(desc(Composite.created_at)).limit(50).all()

    # Get recent attempts (aggregated by composite), filtered by priority/project
    # attempts_per_composite=0 skips loading individual rows — those are lazy-loaded on expand
    attempts, _ = get_aggregated_attempts(db, limit=50, priority=priority, project_id=project, attempts_per_composite=0)

    # Get recent factors (limited for main page), filtered by project
    factors_query = db.query(Factor)
    if project is not None:
        factors_query = factors_query.join(
            ProjectComposite, ProjectComposite.composite_id == Factor.composite_id
        ).filter(ProjectComposite.project_id == project)
    factors = factors_query.order_by(desc(Factor.created_at)).limit(25).all()

    # Pre-fetch composites and attempts for factor rows (eliminates N+1 template queries)
    factor_composite_ids_set = list({f.composite_id for f in factors if f.composite_id})
    factor_attempt_ids = list({f.found_by_attempt_id for f in factors if f.found_by_attempt_id})

    factor_composites_by_id = {}
    if factor_composite_ids_set:
        factor_composites = db.query(Composite).filter(Composite.id.in_(factor_composite_ids_set)).all()
        factor_composites_by_id = {c.id: c for c in factor_composites}

    factor_attempts_by_id = {}
    if factor_attempt_ids:
        factor_attempts = db.query(ECMAttempt).filter(ECMAttempt.id.in_(factor_attempt_ids)).all()
        factor_attempts_by_id = {a.id: a for a in factor_attempts}

    # Build summary stats (filtered by priority/project if specified)
    stats_query = db.query(Composite)
    if priority is not None:
        stats_query = stats_query.filter(Composite.priority == priority)
    if project is not None:
        stats_query = stats_query.join(
            ProjectComposite, ProjectComposite.composite_id == Composite.id
        ).filter(ProjectComposite.project_id == project)

    total_composites = stats_query.count()
    fully_factored = stats_query.filter(Composite.is_fully_factored == True).count()

    # Total attempts - need to join through composites if filtering by priority/project
    attempts_query = db.query(ECMAttempt).filter(ECMAttempt.superseded_by.is_(None))
    if priority is not None or project is not None:
        attempts_query = attempts_query.join(Composite)
        if priority is not None:
            attempts_query = attempts_query.filter(Composite.priority == priority)
        if project is not None:
            attempts_query = attempts_query.join(
                ProjectComposite, ProjectComposite.composite_id == Composite.id
            ).filter(ProjectComposite.project_id == project)
    total_attempts = attempts_query.count()

    # Total factors count (filtered by project if specified)
    factors_count_query = db.query(Factor)
    if project is not None:
        factors_count_query = factors_count_query.join(
            ProjectComposite, ProjectComposite.composite_id == Factor.composite_id
        ).filter(ProjectComposite.project_id == project)
    total_factors = factors_count_query.count()

    return templates.TemplateResponse(request, "public/dashboard.html", {
        "composites": composites,
        "attempts": attempts,
        "factors": factors,
        "factor_composites": factor_composites_by_id,
        "factor_attempts": factor_attempts_by_id,
        "total_composites": total_composites,
        "total_attempts": total_attempts,
        "total_factors": total_factors,
        "fully_factored": fully_factored,
        "priority": priority,
        "project": project,
        "all_projects": all_projects,
    })


@router.get("/testing-status", response_class=HTMLResponse)
def testing_status(
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    limit: int = Query(200, ge=1, le=200, description="Composites per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    min_t_level: Optional[float] = Query(None, ge=0, description="Minimum current t-level"),
    max_t_level: Optional[float] = Query(None, ge=0, description="Maximum current t-level"),
    min_priority: Optional[int] = Query(None, ge=0, description="Minimum priority"),
    snfs_difficulty: Optional[int] = Query(None, ge=0, description="SNFS difficulty filter")
):
    """
    Public testing status dashboard showing t-level progress and method coverage.

    Now supports pagination and filtering for efficient handling of large datasets.
    """
    # Get milestone groups (not paginated, shows overall progress)
    milestone_groups = composite_service.get_milestone_groups(db)

    # Build base filter for unfactored composites
    base_filter = [Composite.is_fully_factored == False]
    if min_t_level is not None:
        base_filter.append(Composite.current_t_level >= min_t_level)
    if max_t_level is not None:
        base_filter.append(Composite.current_t_level <= max_t_level)
    if min_priority is not None:
        base_filter.append(Composite.priority >= min_priority)
    if snfs_difficulty is not None:
        base_filter.append(Composite.snfs_difficulty == snfs_difficulty)

    # Count total before pagination
    total = db.query(func.count(Composite.id)).filter(*base_filter).scalar() or 0

    # Select only the columns needed (avoids loading huge number/current_composite text)
    composites = db.query(
        Composite.id,
        Composite.digit_length,
        Composite.priority,
        Composite.current_t_level,
        Composite.prior_t_level,
        Composite.target_t_level,
        Composite.ecm_progress,
    ).filter(
        *base_filter
    ).order_by(
        Composite.priority.desc(), Composite.digit_length.asc()
    ).offset(offset).limit(limit).all()

    # Get method counts for all composites in this page using aggregation (fixes N+1 query problem)
    composite_ids = [c.id for c in composites]

    method_counts = {}
    if composite_ids:
        # Aggregate method counts in a single query
        counts_query = db.query(
            ECMAttempt.composite_id,
            ECMAttempt.method,
            func.count(ECMAttempt.id).label('attempt_count'),
            func.sum(
                case(
                    (ECMAttempt.method == 'ecm', ECMAttempt.curves_completed),
                    else_=0
                )
            ).label('ecm_curves')
        ).filter(
            ECMAttempt.composite_id.in_(composite_ids)
        ).group_by(
            ECMAttempt.composite_id,
            ECMAttempt.method
        ).all()

        # Build method counts dictionary
        for composite_id, method, attempt_count, ecm_curves in counts_query:
            if composite_id not in method_counts:
                method_counts[composite_id] = {
                    'ecm_curves': 0,
                    'pm1_count': 0,
                    'pp1_count': 0
                }

            if method == 'ecm':
                method_counts[composite_id]['ecm_curves'] = int(ecm_curves or 0)
            elif method == 'pm1':
                method_counts[composite_id]['pm1_count'] = attempt_count
            elif method == 'pp1':
                method_counts[composite_id]['pp1_count'] = attempt_count

    # Calculate status for each composite
    composite_data = []
    for comp in composites:
        # current_t_level now includes prior_t_level
        current_t = comp.current_t_level
        if current_t == 0:
            status = "not_started"
            status_label = "Not Started"
            status_color = "red"
        elif current_t < 30:
            status = "initial"
            status_label = "Initial"
            status_color = "yellow"
        elif current_t < 40:
            status = "standard"
            status_label = "Standard"
            status_color = "orange"
        elif current_t < 50:
            status = "advanced"
            status_label = "Advanced"
            status_color = "blue"
        else:
            status = "deep"
            status_label = "Deep"
            status_color = "green"

        # Get method counts from aggregated results
        counts = method_counts.get(comp.id, {'ecm_curves': 0, 'pm1_count': 0, 'pp1_count': 0})
        ecm_curves = counts['ecm_curves']
        pm1_count = counts['pm1_count']
        pp1_count = counts['pp1_count']

        # Use pre-computed ecm_progress column (indexed, auto-updated)
        progress_pct = (comp.ecm_progress or 0) * 100

        composite_data.append({
            'id': comp.id,
            'digit_length': comp.digit_length,
            'priority': comp.priority,
            'current_t_level': current_t,
            'prior_t_level': comp.prior_t_level,
            'target_t_level': comp.target_t_level,
            'progress_pct': progress_pct,
            'status': status,
            'status_label': status_label,
            'status_color': status_color,
            'ecm_curves': ecm_curves,
            'pm1_count': pm1_count,
            'pp1_count': pp1_count,
            'methods_used': sum([ecm_curves > 0, pm1_count > 0, pp1_count > 0])
        })

    # Calculate pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    showing_from = offset + 1 if total > 0 else 0
    showing_to = min(offset + limit, total)

    return templates.TemplateResponse(request, "public/testing_status.html", {
        "milestone_groups": milestone_groups,
        "composites": composite_data,
        # Pagination metadata
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "showing_from": showing_from,
        "showing_to": showing_to,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        # Filter values (for form state)
        "min_t_level": min_t_level,
        "max_t_level": max_t_level,
        "min_priority": min_priority,
        "snfs_difficulty": snfs_difficulty
    })


@router.get("/p1-testing-status", response_class=HTMLResponse)
def p1_testing_status(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=500, description="Composites per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    min_target_tlevel: Optional[float] = Query(None, ge=0, description="Minimum target t-level"),
    max_target_tlevel: Optional[float] = Query(None, ge=0, description="Maximum target t-level"),
    status_filter: Optional[str] = Query(None, description="Filter by P1 status: none, pm1_only, pp1_only, complete")
):
    """
    P-1/P+1 testing status dashboard showing coverage across composites.

    Shows which composites have had P-1 and/or P+1 run at the required B1 level.
    """
    # Build base filter for active, unfactored composites with target t-levels
    base_filter = [
        Composite.is_active == True,
        Composite.is_fully_factored == False,
        Composite.target_t_level.isnot(None),
    ]
    if min_target_tlevel is not None:
        base_filter.append(Composite.target_t_level >= min_target_tlevel)
    if max_target_tlevel is not None:
        base_filter.append(Composite.target_t_level <= max_target_tlevel)

    # Select only the columns needed (avoids loading huge number/current_composite text)
    all_composites = db.query(
        Composite.id,
        Composite.digit_length,
        Composite.target_t_level,
    ).filter(
        *base_filter
    ).order_by(Composite.target_t_level.asc()).all()

    # Pre-compute PM1/PP1 coverage for all composites
    composite_ids = [c.id for c in all_composites]

    # Get max B1 for PM1/PP1 attempts per composite
    pm1_max_b1 = {}
    pp1_max_b1 = {}
    pm1_curves = {}
    pp1_curves = {}

    if composite_ids:
        # PM1 stats
        pm1_stats = db.query(
            ECMAttempt.composite_id,
            func.max(ECMAttempt.b1).label('max_b1'),
            func.sum(ECMAttempt.curves_completed).label('total_curves')
        ).filter(
            ECMAttempt.composite_id.in_(composite_ids),
            ECMAttempt.method == 'pm1'
        ).group_by(ECMAttempt.composite_id).all()

        for composite_id, max_b1, total_curves in pm1_stats:
            pm1_max_b1[composite_id] = max_b1 or 0
            pm1_curves[composite_id] = int(total_curves or 0)

        # PP1 stats
        pp1_stats = db.query(
            ECMAttempt.composite_id,
            func.max(ECMAttempt.b1).label('max_b1'),
            func.sum(ECMAttempt.curves_completed).label('total_curves')
        ).filter(
            ECMAttempt.composite_id.in_(composite_ids),
            ECMAttempt.method == 'pp1'
        ).group_by(ECMAttempt.composite_id).all()

        for composite_id, max_b1, total_curves in pp1_stats:
            pp1_max_b1[composite_id] = max_b1 or 0
            pp1_curves[composite_id] = int(total_curves or 0)

    # Build composite data with P1 coverage info
    composite_data = []
    pm1_complete_count = 0
    pp1_complete_count = 0
    both_complete_count = 0
    neither_count = 0
    total_pm1_curves = 0
    total_pp1_curves = 0

    for comp in all_composites:
        required_b1 = get_b1_above_tlevel(comp.target_t_level or 35.0)
        comp_pm1_max = pm1_max_b1.get(comp.id, 0)
        comp_pp1_max = pp1_max_b1.get(comp.id, 0)
        comp_pm1_curves = pm1_curves.get(comp.id, 0)
        comp_pp1_curves = pp1_curves.get(comp.id, 0)

        pm1_done = comp_pm1_max >= required_b1
        pp1_done = comp_pp1_max >= required_b1

        # Update summary stats
        total_pm1_curves += comp_pm1_curves
        total_pp1_curves += comp_pp1_curves

        if pm1_done:
            pm1_complete_count += 1
        if pp1_done:
            pp1_complete_count += 1
        if pm1_done and pp1_done:
            both_complete_count += 1
        if not pm1_done and not pp1_done:
            neither_count += 1

        # Determine coverage score for sorting (0=none, 1=one, 2=both)
        coverage_score = (1 if pm1_done else 0) + (1 if pp1_done else 0)

        # Apply status filter
        if status_filter:
            if status_filter == 'none' and (pm1_done or pp1_done):
                continue
            if status_filter == 'pm1_only' and not (pm1_done and not pp1_done):
                continue
            if status_filter == 'pp1_only' and not (pp1_done and not pm1_done):
                continue
            if status_filter == 'complete' and not (pm1_done and pp1_done):
                continue

        composite_data.append({
            'id': comp.id,
            'digit_length': comp.digit_length,
            'target_t_level': comp.target_t_level,
            'required_b1': required_b1,
            'pm1_done': pm1_done,
            'pp1_done': pp1_done,
            'pm1_max_b1': comp_pm1_max,
            'pp1_max_b1': comp_pp1_max,
            'pm1_curves': comp_pm1_curves,
            'pp1_curves': comp_pp1_curves,
            'coverage_score': coverage_score,
        })

    # Apply pagination to filtered results
    total = len(composite_data)
    paginated_composites = composite_data[offset:offset + limit]

    # Calculate pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    showing_from = offset + 1 if total > 0 else 0
    showing_to = min(offset + limit, total)

    # Summary stats
    total_composites = len(all_composites)
    summary = {
        'total': total_composites,
        'pm1_complete': pm1_complete_count,
        'pp1_complete': pp1_complete_count,
        'both_complete': both_complete_count,
        'neither': neither_count,
        'pm1_pct': (pm1_complete_count / total_composites * 100) if total_composites > 0 else 0,
        'pp1_pct': (pp1_complete_count / total_composites * 100) if total_composites > 0 else 0,
        'both_pct': (both_complete_count / total_composites * 100) if total_composites > 0 else 0,
        'neither_pct': (neither_count / total_composites * 100) if total_composites > 0 else 0,
        'total_pm1_curves': total_pm1_curves,
        'total_pp1_curves': total_pp1_curves,
    }

    return templates.TemplateResponse(request, "public/p1_testing_status.html", {
        "summary": summary,
        "composites": paginated_composites,
        # Pagination metadata
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "showing_from": showing_from,
        "showing_to": showing_to,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        # Filter values
        "min_target_tlevel": min_target_tlevel,
        "max_target_tlevel": max_target_tlevel,
        "status_filter": status_filter,
    })


@router.get("/residue-status", response_class=HTMLResponse)
def residue_status_public(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Filter by status"),
    composite_id: Optional[str] = Query(None, description="Filter by composite ID"),
):
    """
    Public read-only dashboard showing residue pool status.
    No authentication required.
    """
    from ...services.residue_manager import ResidueManager
    residue_manager = ResidueManager()

    # Parse composite_id (empty string from form submission → None)
    parsed_composite_id = int(composite_id) if composite_id and composite_id.strip() else None

    # Get filtered residues
    residues, total = get_residues_filtered(
        db,
        limit=limit,
        offset=offset,
        status_filter=status,
        composite_id=parsed_composite_id,
    )

    # Get summary statistics
    stats = residue_manager.get_stats(db)
    pagination = calculate_pagination(offset, limit, total)

    return templates.TemplateResponse(request, "public/residue_status.html", {
        "residues": residues,
        **pagination.to_dict(),
        # Filter values for form state
        "filter_status": status,
        "filter_composite_id": parsed_composite_id,
        # Statistics
        "stats": stats,
        "now": datetime.utcnow(),
    })


@router.get("/composites/{composite_id}/attempts-fragment", response_class=HTMLResponse)
@limiter.limit("60/minute")
def get_attempts_fragment_public(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(25, ge=1, le=100)
):
    """
    Return HTML fragment of attempt detail rows for a composite.
    Used by the public dashboard for lazy-loading on row expand.
    """
    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        raise HTTPException(status_code=404, detail="Composite not found")

    attempts = db.query(ECMAttempt).options(
        defer(ECMAttempt.raw_output)
    ).filter(
        ECMAttempt.composite_id == composite_id
    ).order_by(desc(ECMAttempt.created_at)).limit(limit).all()

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

    html = templates.env.get_template("components/attempt_detail_rows_public.html").render(
        composite_id=composite_id,
        attempts=attempts,
        factor_counts=factor_counts
    )
    return HTMLResponse(content=html)


@router.get("/composites/find")
def find_composite_public(
    q: str,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """Find composite by ID, number (formula), or current_composite value.

    Args:
        q: Search query - can be composite ID, number (e.g., "2^1223-1"),
           or current_composite value

    Returns:
        Redirect to the composite's details page
    """
    composite = composite_service.find_composite_by_identifier(db, q)
    if not composite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Composite not found: {q}"
        )

    # Redirect to the canonical details page URL
    return RedirectResponse(
        url=f"/api/v1/dashboard/composites/{composite.id}/details",
        status_code=status.HTTP_302_FOUND
    )


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
@limiter.limit("30/minute")
def get_composite_details_public(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """
    Public web page showing detailed information about a specific composite.
    """
    details = composite_service.get_composite_details(db, composite_id)

    if not details:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Composite Not Found</title>
        </head>
        <body>
            <h1>Composite Not Found</h1>
            <p><a href="/api/v1/dashboard/">Back to Dashboard</a></p>
        </body>
        </html>
        """

    # Get method breakdown for tabbed interface
    method_breakdown = composite_service.get_method_breakdown(composite_id, db)

    return templates.TemplateResponse(request, "public/composite_details.html", {
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work'],
        "all_factors": details['all_factors'],
        "factors_with_group_orders": details['factors_with_group_orders'],
        "method_breakdown": method_breakdown,
        "factor_counts": method_breakdown.get('_factor_counts', {}),
    })


@router.get("/curves", response_class=HTMLResponse)
def recent_curves(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    group_by_composite: bool = Query(True, description="Group curves by composite")
):
    """
    Public page showing recent ECM curves with filtering and pagination.
    """
    if group_by_composite:
        # SQL-level pagination and filtering - no large in-memory loads
        attempts, total = get_aggregated_attempts(
            db, limit=limit, offset=offset, client_id=client_id
        )

        # Pre-fetch factor counts for attempt detail rows (eliminates N+1 queries in template)
        factor_counts_by_attempt = prefetch_factor_counts_for_attempts(db, attempts)
    else:
        # Get individual attempts
        query = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output)).filter(ECMAttempt.superseded_by.is_(None))

        if client_id:
            query = query.filter(ECMAttempt.client_id == client_id)

        total = query.count()
        attempts = query.order_by(desc(ECMAttempt.created_at)).offset(offset).limit(limit).all()
        factor_counts_by_attempt = {}

    # Pre-fetch composites for non-grouped mode (eliminates N+1 template queries)
    attempt_composites_by_id = {}
    if not group_by_composite and attempts:
        comp_ids = list({a.composite_id for a in attempts if a.composite_id})
        if comp_ids:
            comps = db.query(Composite).filter(Composite.id.in_(comp_ids)).all()
            attempt_composites_by_id = {c.id: c for c in comps}

    # Get list of active clients for filter dropdown
    clients_rows: List[Any] = db.query(distinct(ECMAttempt.client_id)).filter(
        ECMAttempt.client_id.isnot(None)
    ).order_by(ECMAttempt.client_id).limit(100).all()
    clients: List[str] = [c[0] for c in clients_rows if c[0]]

    # Pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    return templates.TemplateResponse(request, "public/recent_curves.html", {
        "attempts": attempts,
        "group_by_composite": group_by_composite,
        "factor_counts_by_attempt": factor_counts_by_attempt,
        "attempt_composites": attempt_composites_by_id,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        "client_id": client_id,
        "clients": clients,
    })


@router.get("/factors", response_class=HTMLResponse)
def recent_factors(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    client_id: Optional[str] = Query(None, description="Filter by client who found factor"),
    min_digits: Optional[int] = Query(None, ge=1, description="Minimum factor digits"),
    max_digits: Optional[int] = Query(None, ge=1, description="Maximum factor digits")
):
    """
    Public page showing discovered factors with filtering and pagination.
    """
    # Build query
    query = db.query(Factor)

    # Filter by client who found the factor (via the attempt)
    if client_id:
        query = query.join(ECMAttempt, Factor.found_by_attempt_id == ECMAttempt.id).filter(
            ECMAttempt.client_id == client_id
        )

    # Apply digit filters using length of factor string
    if min_digits:
        query = query.filter(func.length(Factor.factor) >= min_digits)
    if max_digits:
        query = query.filter(func.length(Factor.factor) <= max_digits)

    total = query.count()
    factors = query.order_by(desc(Factor.created_at)).offset(offset).limit(limit).all()

    # Pre-fetch composites and attempts for factor rows (eliminates N+1 template queries)
    factor_composite_ids = list({f.composite_id for f in factors if f.composite_id})
    factor_attempt_ids = list({f.found_by_attempt_id for f in factors if f.found_by_attempt_id})

    factor_composites_by_id = {}
    if factor_composite_ids:
        fc = db.query(Composite).filter(Composite.id.in_(factor_composite_ids)).all()
        factor_composites_by_id = {c.id: c for c in fc}

    factor_attempts_by_id = {}
    if factor_attempt_ids:
        fa = db.query(ECMAttempt).filter(ECMAttempt.id.in_(factor_attempt_ids)).all()
        factor_attempts_by_id = {a.id: a for a in fa}

    # Get list of clients who have found factors for filter dropdown
    clients_rows: List[Any] = db.query(distinct(ECMAttempt.client_id)).join(
        Factor, Factor.found_by_attempt_id == ECMAttempt.id
    ).filter(ECMAttempt.client_id.isnot(None)).order_by(ECMAttempt.client_id).limit(100).all()
    clients: List[str] = [c[0] for c in clients_rows if c[0]]

    # Pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    return templates.TemplateResponse(request, "public/recent_factors.html", {
        "factors": factors,
        "factor_composites": factor_composites_by_id,
        "factor_attempts": factor_attempts_by_id,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        "client_id": client_id,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "clients": clients,
    })


@router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(
    request: Request,
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Days to look back")
):
    """
    Public leaderboard showing top contributors by curves and factors.
    """
    since = datetime.utcnow() - timedelta(days=days)

    # Top contributors by curves
    top_by_curves = db.query(
        ECMAttempt.client_id,
        func.sum(ECMAttempt.curves_completed).label('total_curves'),
        func.count(ECMAttempt.id).label('attempt_count'),
        func.max(ECMAttempt.created_at).label('last_active')
    ).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.client_id.isnot(None),
        ECMAttempt.superseded_by.is_(None)
    ).group_by(ECMAttempt.client_id).order_by(
        desc('total_curves')
    ).limit(25).all()

    # Top contributors by factors found
    top_by_factors = db.query(
        ECMAttempt.client_id,
        func.count(Factor.id).label('factor_count'),
        func.max(Factor.created_at).label('last_factor')
    ).join(
        Factor, Factor.found_by_attempt_id == ECMAttempt.id
    ).filter(
        Factor.created_at >= since,
        ECMAttempt.client_id.isnot(None)
    ).group_by(ECMAttempt.client_id).order_by(
        desc('factor_count')
    ).limit(25).all()

    # Recent activity stats
    total_curves_period = db.query(func.sum(ECMAttempt.curves_completed)).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.superseded_by.is_(None)
    ).scalar() or 0

    total_factors_period = db.query(func.count(Factor.id)).filter(
        Factor.created_at >= since
    ).scalar() or 0

    active_clients = db.query(func.count(distinct(ECMAttempt.client_id))).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.superseded_by.is_(None)
    ).scalar() or 0

    # Activity by day (last 14 days for chart) - 2 queries instead of 28
    lookback_days = min(14, days)
    chart_since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=lookback_days - 1)

    # Get curves by day in single query
    curves_by_day = db.query(
        func.date(ECMAttempt.created_at).label('day'),
        func.sum(ECMAttempt.curves_completed).label('curves')
    ).filter(
        ECMAttempt.created_at >= chart_since,
        ECMAttempt.superseded_by.is_(None)
    ).group_by(func.date(ECMAttempt.created_at)).all()

    # Get factors by day in single query
    factors_by_day = db.query(
        func.date(Factor.created_at).label('day'),
        func.count(Factor.id).label('factors')
    ).filter(
        Factor.created_at >= chart_since
    ).group_by(func.date(Factor.created_at)).all()

    # Build lookup dicts
    curves_lookup = {str(row.day): int(row.curves or 0) for row in curves_by_day}
    factors_lookup = {str(row.day): int(row.factors or 0) for row in factors_by_day}

    # Build daily activity list (oldest first)
    daily_activity = []
    for i in range(lookback_days - 1, -1, -1):
        day = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)).strftime('%Y-%m-%d')
        daily_activity.append({
            'date': day,
            'curves': curves_lookup.get(day, 0),
            'factors': factors_lookup.get(day, 0)
        })

    return templates.TemplateResponse(request, "public/leaderboard.html", {
        "top_by_curves": top_by_curves,
        "top_by_factors": top_by_factors,
        "total_curves_period": total_curves_period,
        "total_factors_period": total_factors_period,
        "active_clients": active_clients,
        "daily_activity": daily_activity,
        "days": days
    })