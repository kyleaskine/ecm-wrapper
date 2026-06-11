"""
Common database query patterns to reduce duplication across routes.
Centralizes frequently-used queries for composites, work assignments, and related entities.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any, List, Dict

from sqlalchemy import and_, desc, func, distinct, ColumnElement
from sqlalchemy.orm import Session, defer, joinedload

from ..constants import ACTIVE_WORK_STATUSES


@dataclass
class PaginationMetadata:
    """Pagination metadata for paginated responses."""
    page: int
    total_pages: int
    showing_from: int
    showing_to: int
    has_prev: bool
    has_next: bool
    limit: int
    offset: int
    total: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template context."""
        return {
            "page": self.page,
            "total_pages": self.total_pages,
            "showing_from": self.showing_from,
            "showing_to": self.showing_to,
            "has_prev": self.has_prev,
            "has_next": self.has_next,
            "limit": self.limit,
            "offset": self.offset,
            "total": self.total
        }


def calculate_pagination(offset: int, limit: int, total: int) -> PaginationMetadata:
    """
    Calculate pagination metadata for paginated responses.

    This centralizes the pagination logic used across dashboard endpoints
    to ensure consistency and reduce code duplication.

    Args:
        offset: Current offset into results
        limit: Number of items per page
        total: Total number of items available

    Returns:
        PaginationMetadata with all calculated values

    Example:
        >>> pagination = calculate_pagination(offset=20, limit=10, total=55)
        >>> pagination.page
        3
        >>> pagination.total_pages
        6
        >>> pagination.has_next
        True
    """
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    showing_from = offset + 1 if total > 0 else 0
    showing_to = min(offset + limit, total)
    has_prev = offset > 0
    has_next = offset + limit < total

    return PaginationMetadata(
        page=page,
        total_pages=total_pages,
        showing_from=showing_from,
        showing_to=showing_to,
        has_prev=has_prev,
        has_next=has_next,
        limit=limit,
        offset=offset,
        total=total
    )


def get_recent_work_assignments(
    db: Session,
    limit: int = 20,
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None
):
    """
    Get recent work assignments with optional filtering.

    Args:
        db: Database session
        limit: Maximum number of assignments to return
        status_filter: Filter by status (assigned, claimed, running, etc.)
        client_id: Filter by specific client

    Returns:
        List of WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    # Eager-load composite — admin dashboard template accesses work.composite.* per row.
    query = db.query(WorkAssignment).options(joinedload(WorkAssignment.composite))

    filters = []
    if status_filter:
        filters.append(WorkAssignment.status == status_filter)
    if client_id:
        filters.append(WorkAssignment.client_id == client_id)

    if filters:
        query = query.filter(and_(*filters))

    return query.order_by(desc(WorkAssignment.created_at)).limit(limit).all()


def get_active_work_assignments(db: Session, limit: int = 100):
    """
    Get currently active work assignments.

    Args:
        db: Database session
        limit: Maximum number to return

    Returns:
        List of active WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    return db.query(WorkAssignment).filter(
        WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
    ).order_by(desc(WorkAssignment.created_at)).limit(limit).all()


def get_composites_by_completion(
    db: Session,
    limit: int = 50,
    include_factored: bool = False
):
    """
    Get composites sorted by ECM completion percentage.

    Args:
        db: Database session
        limit: Maximum number to return
        include_factored: If True, includes fully factored composites

    Returns:
        List of Composite models sorted by completion (high to low)
    """
    from ..models.composites import Composite

    query = db.query(Composite)

    filters: List[Any] = [Composite.target_t_level.isnot(None)]
    if not include_factored:
        filters.append(Composite.is_fully_factored == False)

    # Use indexed ecm_progress column for efficient sorting in SQL
    # ecm_progress is a generated column: current_t_level / target_t_level
    return query.filter(and_(*filters)).order_by(
        Composite.ecm_progress.desc().nulls_last()
    ).limit(limit).all()


def get_recent_clients(db: Session, limit: int = 10, days: int = 7):
    """
    Get recently active clients with work statistics.

    Args:
        db: Database session
        limit: Maximum number of clients to return
        days: Look back this many days for activity

    Returns:
        List of tuples (client_id, work_count, last_seen)
    """
    from ..models.work_assignments import WorkAssignment

    since = datetime.utcnow() - timedelta(days=days)

    return db.query(
        WorkAssignment.client_id,
        func.count(WorkAssignment.id).label('work_count'),
        func.max(WorkAssignment.created_at).label('last_seen')
    ).filter(
        WorkAssignment.created_at >= since
    ).group_by(
        WorkAssignment.client_id
    ).order_by(
        desc('last_seen')
    ).limit(limit).all()


def get_recent_factors(db: Session, limit: int = 10):
    """
    Get recently discovered factors.

    Args:
        db: Database session
        limit: Maximum number to return

    Returns:
        List of Factor models with composite relationship loaded
    """
    from ..models.factors import Factor

    # Eager-load composite — admin dashboard template accesses factor.composite.* per row.
    return (
        db.query(Factor)
        .options(joinedload(Factor.composite))
        .order_by(desc(Factor.created_at))
        .limit(limit)
        .all()
    )


def get_recent_attempts(db: Session, limit: int = 100, method: Optional[str] = None):
    """
    Get recent ECM attempts with optional method filtering.

    Args:
        db: Database session
        limit: Maximum number to return
        method: Filter by method (ecm, pm1, pp1, etc.)

    Returns:
        List of ECMAttempt models
    """
    from ..models.attempts import ECMAttempt

    query = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output))

    if method:
        query = query.filter(ECMAttempt.method == method)

    return query.order_by(desc(ECMAttempt.created_at)).limit(limit).all()


def get_aggregated_attempts(
    db: Session, limit: int = 50, offset: int = 0,
    method: Optional[str] = None, priority: Optional[int] = None,
    client_id: Optional[str] = None, attempts_per_composite: int = 25,
    project_id: Optional[int] = None
):
    """
    Get recent ECM attempts aggregated by composite.

    For each composite with recent work, returns:
    - Composite information
    - Aggregate stats (computed in SQL, not by loading all attempts)
    - Factor count
    - List of recent individual attempts for expansion (limited per composite)

    Args:
        db: Database session
        limit: Maximum number of composites to return
        offset: Number of composites to skip (for pagination)
        method: Filter by method (ecm, pm1, pp1, etc.)
        priority: Filter by composite priority level
        client_id: Filter to composites where this client has attempts
        attempts_per_composite: Max detail attempts to load per composite
        project_id: Filter to composites belonging to this project

    Returns:
        Tuple of (list of dicts with aggregated attempt data, total composite count)
    """
    from ..models.attempts import ECMAttempt
    from ..models.composites import Composite
    from ..models.factors import Factor
    from ..models.projects import ProjectComposite

    # Query 1: Get composite IDs with recent attempts, ordered by most recent activity
    query = db.query(
        ECMAttempt.composite_id,
        func.max(ECMAttempt.created_at).label('latest_attempt')
    )

    # Join with Composite table if filtering by priority or project
    if priority is not None or project_id is not None:
        query = query.join(Composite)
        if priority is not None:
            query = query.filter(Composite.priority == priority)
        if project_id is not None:
            query = query.join(
                ProjectComposite,
                and_(ProjectComposite.composite_id == Composite.id,
                     ProjectComposite.project_id == project_id)
            )

    if client_id:
        query = query.filter(ECMAttempt.client_id == client_id)

    query = query.group_by(ECMAttempt.composite_id)

    if method:
        query = query.filter(ECMAttempt.method == method)

    ordered_query = query.order_by(desc('latest_attempt'))

    # Get total count for pagination
    total = ordered_query.count()

    # Apply pagination
    composite_id_rows = ordered_query.offset(offset).limit(limit).all()
    composite_ids = [row[0] for row in composite_id_rows]

    if not composite_ids:
        return [], total

    # Query 2: Batch fetch composites
    composites = db.query(Composite).filter(Composite.id.in_(composite_ids)).all()
    composites_by_id = {c.id: c for c in composites}

    # Query 3: Aggregate stats in SQL (no ORM object loading needed)
    agg_query = db.query(
        ECMAttempt.composite_id,
        func.count(ECMAttempt.id).label('attempt_count'),
        func.coalesce(func.sum(ECMAttempt.curves_completed), 0).label('total_curves'),
        func.coalesce(func.sum(ECMAttempt.execution_time_seconds), 0.0).label('total_time'),
        func.min(ECMAttempt.created_at).label('earliest_attempt'),
        func.max(ECMAttempt.created_at).label('latest_attempt'),
    ).filter(
        ECMAttempt.composite_id.in_(composite_ids)
    )
    if method:
        agg_query = agg_query.filter(ECMAttempt.method == method)
    agg_rows = agg_query.group_by(ECMAttempt.composite_id).all()
    stats_by_composite = {row.composite_id: row for row in agg_rows}

    # Query 4: Recent attempts for expandable detail rows (limited)
    # Cap total loaded attempts to avoid memory issues on large datasets.
    # Most recent attempts across all composites - simple and fast.
    max_detail_attempts = limit * attempts_per_composite
    attempts_query = db.query(ECMAttempt).options(
        defer(ECMAttempt.raw_output)
    ).filter(
        ECMAttempt.composite_id.in_(composite_ids)
    )
    if method:
        attempts_query = attempts_query.filter(ECMAttempt.method == method)
    detail_attempts = attempts_query.order_by(
        desc(ECMAttempt.created_at)
    ).limit(max_detail_attempts).all()

    # Group detail attempts by composite_id
    attempts_by_composite: dict[int, list] = {}
    for attempt in detail_attempts:
        if attempt.composite_id not in attempts_by_composite:
            attempts_by_composite[attempt.composite_id] = []
        attempts_by_composite[attempt.composite_id].append(attempt)

    # Query 5: Factor counts per composite (templates only need the count)
    factor_count_rows = db.query(
        Factor.composite_id,
        func.count(Factor.id).label('factor_count')
    ).filter(
        Factor.composite_id.in_(composite_ids)
    ).group_by(Factor.composite_id).all()
    factor_count_by_composite = {row.composite_id: row.factor_count for row in factor_count_rows}

    # Build aggregated results preserving order from original query
    aggregated = []
    for composite_id in composite_ids:
        composite = composites_by_id.get(composite_id)
        if not composite:
            continue

        stats = stats_by_composite.get(composite_id)
        if not stats:
            continue

        aggregated.append({
            'composite': composite,
            'attempt_count': stats.attempt_count,
            'total_curves': stats.total_curves,
            'total_time': stats.total_time,
            'earliest_attempt': stats.earliest_attempt,
            'latest_attempt': stats.latest_attempt,
            'factor_count': factor_count_by_composite.get(composite_id, 0),
            'attempts': attempts_by_composite.get(composite_id, [])
        })

    return aggregated, total


def prefetch_factor_counts_for_attempts(db: Session, aggregated_results: list) -> Dict[int, int]:
    """
    Pre-fetch factor counts per attempt to avoid N+1 queries in templates.

    Templates need to know how many factors each attempt found (for [+N more] badges).
    Instead of querying per-attempt inside Jinja2, batch fetch all counts upfront.

    Args:
        db: Database session
        aggregated_results: List of dicts from get_aggregated_attempts()

    Returns:
        Dict mapping attempt_id to factor count
    """
    from ..models.factors import Factor

    # Collect attempt IDs that found factors (only those need counts)
    attempt_ids = []
    for agg in aggregated_results:
        for attempt in agg['attempts']:
            if attempt.factor_found:
                attempt_ids.append(attempt.id)

    if not attempt_ids:
        return {}

    rows = db.query(
        Factor.found_by_attempt_id,
        func.count(Factor.id).label('factor_count')
    ).filter(
        Factor.found_by_attempt_id.in_(attempt_ids)
    ).group_by(Factor.found_by_attempt_id).all()

    return {row.found_by_attempt_id: row.factor_count for row in rows}


def get_expired_work_assignments(db: Session):
    """
    Get work assignments that have expired but are still marked as active.

    Args:
        db: Database session

    Returns:
        List of expired WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    return db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.expires_at < datetime.utcnow(),
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        )
    ).all()


def get_composite_with_details(db: Session, composite_id: int):
    """
    Get a composite with all related data (attempts, factors, work).

    Args:
        db: Database session
        composite_id: ID of the composite

    Returns:
        Tuple of (composite, attempts, factors, active_work) or None if not found
    """
    from ..models.composites import Composite
    from ..models.attempts import ECMAttempt
    from ..models.factors import Factor
    from ..models.work_assignments import WorkAssignment

    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        return None

    attempts = db.query(ECMAttempt).options(
        defer(ECMAttempt.raw_output)
    ).filter(
        ECMAttempt.composite_id == composite_id
    ).order_by(desc(ECMAttempt.created_at)).limit(50).all()

    factors = db.query(Factor).filter(
        Factor.composite_id == composite_id
    ).all()

    active_work = db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.composite_id == composite_id,
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        )
    ).all()

    return composite, attempts, factors, active_work


def count_active_clients(db: Session, days: int = 7) -> int:
    """
    Count distinct clients active in the last N days.

    Args:
        db: Database session
        days: Look back this many days

    Returns:
        Count of unique active clients
    """
    from ..models.work_assignments import WorkAssignment

    since = datetime.utcnow() - timedelta(days=days)

    return db.query(distinct(WorkAssignment.client_id)).filter(
        and_(
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES),
            WorkAssignment.created_at >= since
        )
    ).count()


def get_summary_statistics(db: Session) -> dict:
    """
    Get high-level summary statistics for the system.

    Args:
        db: Database session

    Returns:
        Dictionary with various counts and metrics
    """
    from ..models.composites import Composite
    from ..models.work_assignments import WorkAssignment
    from ..models.attempts import ECMAttempt
    from ..models.factors import Factor

    last_24h = datetime.utcnow() - timedelta(hours=24)

    return {
        "total_composites": db.query(Composite).count(),
        "factored_composites": db.query(Composite).filter(
            Composite.is_fully_factored
        ).count(),
        "active_work": db.query(WorkAssignment).filter(
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        ).count(),
        "recent_attempts_24h": db.query(ECMAttempt).filter(
            ECMAttempt.created_at >= last_24h
        ).count(),
        "recent_factors_24h": db.query(Factor).filter(
            Factor.created_at >= last_24h
        ).count(),
        "active_clients": count_active_clients(db, days=7)
    }


def get_inactive_composites(db: Session, limit: int = 100, offset: int = 0):
    """
    Get inactive composites that are not fully factored.

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset

    Returns:
        Tuple of (composites list, total count)
    """
    from ..models.composites import Composite

    query = db.query(Composite).filter(
        and_(
            Composite.is_active == False,
            Composite.is_fully_factored == False
        )
    )

    total = query.count()
    composites = query.order_by(
        desc(Composite.priority),
        desc(Composite.created_at)
    ).offset(offset).limit(limit).all()

    return composites, total


def get_outstanding_work_assignments(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    client_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    method_filter: Optional[str] = None
):
    """
    Get work assignments with status in ['assigned', 'claimed', 'running'].

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset
        client_id: Optional filter by client ID
        status_filter: Optional filter by status
        method_filter: Optional filter by method

    Returns:
        Tuple of (assignments list, total count)
    """
    from ..models.work_assignments import WorkAssignment

    # Eager-load composite — outstanding-work template accesses work.composite.* per row.
    query = db.query(WorkAssignment).options(joinedload(WorkAssignment.composite)).filter(
        WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
    )

    # Apply optional filters
    filters = []
    if client_id:
        filters.append(WorkAssignment.client_id == client_id)
    if status_filter:
        filters.append(WorkAssignment.status == status_filter)
    if method_filter:
        filters.append(WorkAssignment.method == method_filter)

    if filters:
        query = query.filter(and_(*filters))

    total = query.count()
    assignments = query.order_by(
        desc(WorkAssignment.created_at)
    ).offset(offset).limit(limit).all()

    return assignments, total


def get_recently_added_composites(
    db: Session,
    days: int = 30,
    limit: int = 100,
    offset: int = 0
):
    """
    Get composites created in the last N days.

    Args:
        db: Database session
        days: Number of days to look back
        limit: Maximum number to return
        offset: Pagination offset

    Returns:
        Tuple of (composites list, total count)
    """
    from ..models.composites import Composite

    since = datetime.utcnow() - timedelta(days=days)

    query = db.query(Composite).filter(
        Composite.created_at >= since
    )

    total = query.count()
    composites = query.order_by(
        desc(Composite.created_at)
    ).offset(offset).limit(limit).all()

    return composites, total


def batch_fetch_attempts_by_composite(
    db: Session,
    composite_ids: List[int],
    exclude_superseded: bool = True
) -> dict[int, list]:
    """
    Batch fetch ECM attempts for multiple composites in a single query.

    This eliminates N+1 query patterns when processing multiple composites.

    Args:
        db: Database session
        composite_ids: List of composite IDs to fetch attempts for
        exclude_superseded: If True, exclude attempts that have been superseded
                           (for accurate t-level calculations)

    Returns:
        Dictionary mapping composite_id to list of ECMAttempt objects
    """
    from ..models.attempts import ECMAttempt

    if not composite_ids:
        return {}

    query = db.query(ECMAttempt).filter(
        ECMAttempt.composite_id.in_(composite_ids)
    )

    # Exclude superseded attempts for t-level calculations
    if exclude_superseded:
        query = query.filter(ECMAttempt.superseded_by.is_(None))

    all_attempts = query.all()

    # Group by composite_id
    attempts_by_composite: dict[int, list] = {}
    for attempt in all_attempts:
        if attempt.composite_id not in attempts_by_composite:
            attempts_by_composite[attempt.composite_id] = []
        attempts_by_composite[attempt.composite_id].append(attempt)

    return attempts_by_composite


def get_residues_filtered(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None,
    composite_id: Optional[int] = None,
    expiring_soon: bool = False
):
    """
    Get residues with comprehensive filtering.

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset
        status_filter: Filter by status (available/claimed/completed/expired)
        client_id: Filter by client that uploaded
        composite_id: Filter by specific composite
        expiring_soon: If True, show only residues expiring in < 24 hours

    Returns:
        Tuple of (residues list, total count)
    """
    from ..models.residues import ECMResidue

    # Eager-load composite — templates access residue.composite for every row,
    # which would otherwise fire one query per residue.
    query = db.query(ECMResidue).options(joinedload(ECMResidue.composite))

    # Apply filters
    filters: List[ColumnElement[bool]] = []
    if status_filter == 'available+claimed':
        filters.append(ECMResidue.status.in_(['available', 'claimed']))
    elif status_filter:
        filters.append(ECMResidue.status == status_filter)
    if client_id:
        filters.append(ECMResidue.client_id == client_id)
    if composite_id:
        filters.append(ECMResidue.composite_id == composite_id)
    if expiring_soon:
        # Only claimed residues have expires_at set (claim timeout)
        # Available residues don't expire by time
        expiry_threshold = datetime.utcnow() + timedelta(hours=24)
        filters.append(ECMResidue.expires_at < expiry_threshold)
        filters.append(ECMResidue.status == 'claimed')

    if filters:
        query = query.filter(and_(*filters))

    total = query.count()
    residues = query.order_by(
        desc(ECMResidue.created_at)
    ).offset(offset).limit(limit).all()

    return residues, total
