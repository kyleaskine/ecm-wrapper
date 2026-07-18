from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session, defer
from sqlalchemy import and_, or_, case, func
from sqlalchemy.exc import IntegrityError
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import logging
import json

from ...database import get_db
from ...dependencies import get_t_level_calculator
from ...models.composites import Composite
from ...models.attempts import ECMAttempt
from ...models.work_assignments import WorkAssignment
from ...models.residues import ECMResidue
from ...models.projects import Project, ProjectComposite
from ...services.t_level_calculator import TLevelCalculator
from ...services.work_assignment import pick_and_lock_composite, UNIQUE_ACTIVE_WORK_MARKERS
from ...utils.transactions import transaction_scope, is_unique_violation
from ...config import get_settings
from ...constants import ECM_BOUNDS, OPTIMAL_B1_TABLE, get_b1_above_tlevel, ACTIVE_WORK_STATUSES

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


def _json_response(data: Dict[str, Any]) -> Response:
    """Return a JSON response with consistent formatting."""
    content = json.dumps(data, default=str) + "\n"
    return Response(content=content, media_type="application/json")


def _no_work_response(message: str, include_p1_fields: bool = False) -> Response:
    """Uniform empty-work payload for /ecm-work and /p1-work."""
    data: Dict[str, Any] = {
        "work_id": None,
        "composite_id": None,
        "composite": None,
        "digit_length": None,
        "current_t_level": None,
        "target_t_level": None,
        "expires_at": None,
        "message": message,
    }
    if include_p1_fields:
        data["pm1_b1"] = None
        data["pp1_b1"] = None
    return _json_response(data)


@router.get("/ecm-work")
def get_ecm_work(
    client_id: str,
    priority: Optional[int] = None,
    min_target_tlevel: Optional[float] = None,
    max_target_tlevel: Optional[float] = None,
    max_current_tlevel: Optional[float] = None,
    min_digits: Optional[int] = None,
    max_digits: Optional[int] = None,
    timeout_days: int = 1,
    work_type: str = "standard",
    project: Optional[str] = None,
    db: Session = Depends(get_db),
    t_level_calc: TLevelCalculator = Depends(get_t_level_calculator)
):
    """
    Get ECM work assignment with t-level targeting.

    This endpoint returns an incomplete composite (current_t < target_t)
    that matches the filter criteria, sorted by work_type strategy.

    Exclusions:
    - Composites with active work assignments
    - Composites with pending residues (status='available' or 'claimed')
      to prevent duplicate stage 1 work while waiting for stage 2 processing

    Args:
        client_id: Unique identifier for the requesting client
        priority: Minimum priority level (filters for priority >= this value)
        min_target_tlevel: Minimum target t-level (filters for target_t_level >= this value)
        max_target_tlevel: Maximum target t-level (filters for target_t_level <= this value)
        timeout_days: Work assignment expiration in days (default: 1)
        work_type: Work assignment strategy - "standard" (easiest/lowest target t-level first) or "progressive" (least ECM done first)
        project: Optional project name to filter composites by (if not set, all projects)
        db: Database session

    Returns:
        JSON response with work assignment details or explanation if no work available
    """
    with transaction_scope(db, "get_ecm_work"):
        # Validate work_type parameter
        if work_type not in ["standard", "progressive"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid work_type: {work_type}. Must be 'standard' or 'progressive'"
            )

        # Check if client has too much active work (exclude expired assignments)
        active_work_count = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(ACTIVE_WORK_STATUSES),
                WorkAssignment.expires_at > datetime.utcnow()
            )
        ).count()

        if active_work_count >= settings.max_work_items_per_client:
            return _no_work_response(
                f"Client has {active_work_count} active work assignments "
                f"(max: {settings.max_work_items_per_client})"
            )

        # Build query for suitable composites
        # current_t_level now includes prior_t_level (calculated using -w flag)
        # Use ecm_progress < 1.0 which leverages the indexed generated column
        # Defer the large `number` column - only `current_composite` is needed for the response
        query = db.query(Composite).options(defer(Composite.number)).filter(
            and_(
                Composite.is_active == True,  # Only assign active composites
                Composite.is_fully_factored == False,
                or_(Composite.is_complete.is_(None), Composite.is_complete == False),
                Composite.ecm_progress.isnot(None),  # Has a target set
                Composite.ecm_progress < 1.0  # Not yet complete (indexed)
            )
        )

        # Apply priority filter
        if priority is not None:
            query = query.filter(Composite.priority >= priority)

        # Apply target t-level filters
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)

        # Filter by current t-level (e.g. client with --tlevel 35 shouldn't get composites already at t39)
        if max_current_tlevel is not None:
            query = query.filter(Composite.current_t_level < max_current_tlevel)

        # Apply digit length filters
        if min_digits is not None:
            query = query.filter(Composite.digit_length >= min_digits)
        if max_digits is not None:
            query = query.filter(Composite.digit_length <= max_digits)

        # Apply project filter (join through project_composites)
        if project is not None:
            query = query.join(
                ProjectComposite, ProjectComposite.composite_id == Composite.id
            ).join(
                Project, Project.id == ProjectComposite.project_id
            ).filter(Project.name == project)

        # Exclude composites with active work assignments (NOT EXISTS is faster than NOT IN)
        query = query.filter(~db.query(WorkAssignment.id).filter(
            WorkAssignment.composite_id == Composite.id,
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        ).correlate(Composite).exists())

        # Exclude composites with pending residues (stage 1 done, stage 2 not yet completed)
        # This prevents duplicate stage 1 work when residues are waiting to be processed
        query = query.filter(~db.query(ECMResidue.id).filter(
            ECMResidue.composite_id == Composite.id,
            ECMResidue.status.in_(['available', 'claimed'])
        ).correlate(Composite).exists())

        # Apply sorting strategy based on work_type
        if work_type == "progressive":
            # Progressive: prioritize composites with least ECM work done
            query = query.order_by(
                Composite.current_t_level.asc(),
                Composite.target_t_level.asc(),
                Composite.digit_length.asc()
            )
        else:  # "standard"
            # Standard: prioritize easiest composites first (by target t-level, which accounts for SNFS)
            query = query.order_by(
                Composite.target_t_level.asc(),
                Composite.created_at.asc()
            )

        # Lock the chosen row and re-verify it is still free (the NOT EXISTS
        # filters above can be stale when a concurrent request commits
        # mid-query - see pick_and_lock_composite)
        composite = pick_and_lock_composite(db, query, check_residues=True)

        # No work available
        if not composite:
            return _no_work_response("No suitable work available matching criteria")

        # Calculate suggested ECM parameters using t-level targeting
        # Note: target_t_level is guaranteed non-None by the filter above
        # current_t_level now includes prior_t_level
        try:
            suggestion = t_level_calc.suggest_next_ecm_parameters(
                composite.target_t_level or 0.0,  # Default to 0 if None (shouldn't happen due to filter)
                composite.current_t_level,  # Includes prior_t_level
                composite.digit_length
            )

            if suggestion['status'] == 'target_reached':
                # Only need max B1 for escalation — use aggregate instead of loading all attempts
                max_b1 = db.query(func.max(ECMAttempt.b1)).filter(
                    ECMAttempt.composite_id == composite.id,
                    ECMAttempt.method == 'ecm'
                ).scalar() or 0
                b1, b2, curves = _get_escalated_parameters(composite.digit_length, max_b1)
            else:
                b1, b2, curves = suggestion['b1'], suggestion['b2'], suggestion['curves']

        except Exception as e:
            logger.warning(f"T-level calculation failed for composite {composite.id}: {e}")
            # Fallback to basic parameters
            b1, b2, curves = 50000, 12500000, 100

        # Create work assignment
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=timeout_days)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=composite.id,
            client_id=client_id,
            method='ecm',
            b1=b1,
            b2=b2,
            curves_requested=curves,
            expires_at=expires_at,
            status='assigned'
        )

        # The recheck under lock should make a duplicate unreachable, but the
        # partial unique index is the final arbiter: fail soft as "no work"
        # (client re-requests) instead of a 500. Savepoint keeps the session
        # usable after a rejected flush.
        try:
            with db.begin_nested():
                db.add(work_assignment)
                db.flush()
        except IntegrityError as e:
            if not is_unique_violation(e, *UNIQUE_ACTIVE_WORK_MARKERS):
                raise
            logger.warning(
                f"Lost assignment race on composite {composite.id} at insert "
                f"(client {client_id}); returning no-work"
            )
            return _no_work_response("Lost assignment race, please request again")

        prior_t = composite.prior_t_level
        current_t = composite.current_t_level

        if prior_t:
            logger.info(f"Created ECM work assignment {work_id} for client {client_id}: "
                       f"{composite.digit_length}-digit composite, "
                       f"t{current_t:.1f} → t{composite.target_t_level:.1f} "
                       f"(includes prior: t{prior_t:.1f})")
        else:
            logger.info(f"Created ECM work assignment {work_id} for client {client_id}: "
                       f"{composite.digit_length}-digit composite, "
                       f"t{current_t:.1f} → t{composite.target_t_level:.1f}")

        # Build message based on work type strategy
        if work_type == "progressive":
            message = f"Assigned composite with least ECM work (t{current_t:.1f})"
        else:
            message = f"Assigned easiest incomplete composite (target: t{composite.target_t_level:.1f})"

        response_data = {
            "work_id": work_id,
            "composite_id": composite.id,
            "composite": composite.current_composite,
            "digit_length": composite.digit_length,
            "current_t_level": current_t,
            "prior_t_level": prior_t,
            "target_t_level": composite.target_t_level,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "message": message
        }
        return _json_response(response_data)


def _get_escalated_parameters(digit_length: int, max_b1_attempted: int) -> tuple:
    """Get escalated ECM parameters when target t-level is reached."""
    escalated_b1 = max_b1_attempted * 3

    # Find next level beyond what's been tried
    for max_digits, b1, b2, curves in ECM_BOUNDS:
        if digit_length <= max_digits and b1 > escalated_b1:
            return b1, b2, min(curves // 5, 200)

    # Fallback to highest available
    return ECM_BOUNDS[-1][1], ECM_BOUNDS[-1][2], 100


def _build_required_b1_case():
    """
    Build a SQL CASE expression mapping target_t_level to required B1
    for PM1/PP1 sweeps (one step above target in the optimal B1 table).

    Returns a SQLAlchemy case() expression usable in queries/filters.
    """
    whens = []
    for i, (t_level, _b1) in enumerate(OPTIMAL_B1_TABLE):
        if i + 1 < len(OPTIMAL_B1_TABLE):
            next_b1 = OPTIMAL_B1_TABLE[i + 1][1]
        else:
            next_b1 = OPTIMAL_B1_TABLE[-1][1]
        whens.append((Composite.target_t_level <= t_level, next_b1))
    return case(*whens, else_=OPTIMAL_B1_TABLE[-1][1])


def _pm1pp1_exists_subquery(db: Session, method_name: str, required_b1_expr):
    """
    Build a correlated EXISTS subquery checking if a composite already has
    a PM1 or PP1 attempt at the required B1 level.

    Args:
        db: Database session
        method_name: 'pm1' or 'pp1'
        required_b1_expr: SQL CASE expression for the required B1

    Returns:
        SQLAlchemy exists() clause usable in .filter()
    """
    return (
        db.query(ECMAttempt.id)
        .filter(
            ECMAttempt.composite_id == Composite.id,
            ECMAttempt.method == method_name,
            ECMAttempt.b1 >= required_b1_expr,
        )
        .correlate(Composite)
        .exists()
    )


@router.get("/p1-work")
def get_p1_work(
    client_id: str,
    method: str = "p1",
    priority: Optional[int] = None,
    min_target_tlevel: Optional[float] = None,
    max_target_tlevel: Optional[float] = None,
    min_digits: Optional[int] = None,
    max_digits: Optional[int] = None,
    timeout_days: int = 1,
    work_type: str = "standard",
    project: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get P-1/P+1 work assignment.

    Assigns composites that haven't had PM1/PP1 done at the required B1 level.
    B1 is calculated as one step above the composite's target t-level in the
    optimal B1 table.

    Unlike /ecm-work, this endpoint does NOT filter by ecm_progress < 1.0,
    since PM1/PP1 sweeps are valuable even on composites that have reached
    their ECM target.

    Args:
        client_id: Unique identifier for the requesting client
        method: Which methods to check - "pm1" (P-1 only), "pp1" (P+1 only),
                or "p1" (both P-1 and P+1, default)
        priority: Minimum priority level
        min_target_tlevel: Minimum target t-level filter
        max_target_tlevel: Maximum target t-level filter
        timeout_days: Work assignment expiration in days (default: 1)
        work_type: "standard" (easiest first) or "progressive" (least work first)
        project: Optional project name to filter composites by (if not set, all projects)

    Returns:
        JSON response with work assignment details including pm1_b1/pp1_b1
    """
    with transaction_scope(db, "get_p1_work"):
        # Validate parameters
        if method not in ("pm1", "pp1", "p1"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid method: {method}. Must be 'pm1', 'pp1', or 'p1'"
            )
        if work_type not in ("standard", "progressive"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid work_type: {work_type}. Must be 'standard' or 'progressive'"
            )

        check_pm1 = method in ("pm1", "p1")
        check_pp1 = method in ("pp1", "p1")

        # Check active work limit
        active_work_count = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(ACTIVE_WORK_STATUSES),
                WorkAssignment.expires_at > datetime.utcnow()
            )
        ).count()

        if active_work_count >= settings.max_work_items_per_client:
            return _no_work_response(
                f"Client has {active_work_count} active work assignments "
                f"(max: {settings.max_work_items_per_client})",
                include_p1_fields=True
            )

        # Build candidate query - no ecm_progress filter (PM1/PP1 valuable regardless)
        query = db.query(Composite).filter(
            and_(
                Composite.is_active == True,
                Composite.is_fully_factored == False,
                or_(Composite.is_complete.is_(None), Composite.is_complete == False),
                Composite.target_t_level.isnot(None),
            )
        )

        if priority is not None:
            query = query.filter(Composite.priority >= priority)
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)
        if min_digits is not None:
            query = query.filter(Composite.digit_length >= min_digits)
        if max_digits is not None:
            query = query.filter(Composite.digit_length <= max_digits)

        # Apply project filter (join through project_composites)
        if project is not None:
            query = query.join(
                ProjectComposite, ProjectComposite.composite_id == Composite.id
            ).join(
                Project, Project.id == ProjectComposite.project_id
            ).filter(Project.name == project)

        # Exclude composites with active work assignments (NOT EXISTS is faster than NOT IN)
        query = query.filter(~db.query(WorkAssignment.id).filter(
            WorkAssignment.composite_id == Composite.id,
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
        ).correlate(Composite).exists())

        # Build SQL CASE expression: map target_t_level -> required B1
        # (one step above target in the optimal B1 table)
        required_b1 = _build_required_b1_case()

        # Filter to composites that still need PM1/PP1 at the required B1
        # Uses correlated NOT EXISTS against ecm_attempts (hits composite_id,method index)
        if check_pm1 and check_pp1:
            # method='p1': needs work only if BOTH pm1 AND pp1 are missing
            pm1_covered = _pm1pp1_exists_subquery(db, 'pm1', required_b1)
            pp1_covered = _pm1pp1_exists_subquery(db, 'pp1', required_b1)
            query = query.filter(and_(~pm1_covered, ~pp1_covered))
        elif check_pm1:
            pm1_covered = _pm1pp1_exists_subquery(db, 'pm1', required_b1)
            query = query.filter(~pm1_covered)
        else:
            pp1_covered = _pm1pp1_exists_subquery(db, 'pp1', required_b1)
            query = query.filter(~pp1_covered)

        # Apply sorting strategy
        if work_type == "progressive":
            query = query.order_by(
                Composite.current_t_level.asc(),
                Composite.target_t_level.asc(),
                Composite.digit_length.asc()
            )
        else:
            query = query.order_by(
                Composite.target_t_level.asc(),
                Composite.created_at.asc()
            )

        # Lock the chosen row and re-verify it is still free (the NOT EXISTS
        # filter above can be stale when a concurrent request commits
        # mid-query - see pick_and_lock_composite)
        assigned_composite = pick_and_lock_composite(db, query, check_residues=False)

        if not assigned_composite:
            return _no_work_response(
                "No composites need P-1/P+1 work matching criteria",
                include_p1_fields=True
            )

        # Compute B1 for the assigned composite
        computed_b1 = get_b1_above_tlevel(assigned_composite.target_t_level or 35.0)

        # Create work assignment
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=timeout_days)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=assigned_composite.id,
            client_id=client_id,
            method=method,
            b1=computed_b1,
            b2=0,  # PM1/PP1 uses GMP-ECM default B2
            curves_requested=1,  # Client decides actual curve count
            expires_at=expires_at,
            status='assigned'
        )

        # See /ecm-work: unique index is the final arbiter; fail soft
        try:
            with db.begin_nested():
                db.add(work_assignment)
                db.flush()
        except IntegrityError as e:
            if not is_unique_violation(e, *UNIQUE_ACTIVE_WORK_MARKERS):
                raise
            logger.warning(
                f"Lost assignment race on composite {assigned_composite.id} at insert "
                f"(client {client_id}); returning no-work"
            )
            return _no_work_response(
                "Lost assignment race, please request again",
                include_p1_fields=True
            )

        logger.info(
            f"Created P1 work assignment {work_id} for client {client_id}: "
            f"{assigned_composite.digit_length}-digit composite, "
            f"method={method}, B1={computed_b1}, "
            f"target_t={assigned_composite.target_t_level:.1f}"
        )

        response_data = {
            "work_id": work_id,
            "composite_id": assigned_composite.id,
            "composite": assigned_composite.current_composite,
            "digit_length": assigned_composite.digit_length,
            "current_t_level": assigned_composite.current_t_level,
            "target_t_level": assigned_composite.target_t_level,
            "pm1_b1": computed_b1 if check_pm1 else None,
            "pp1_b1": computed_b1 if check_pp1 else None,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "message": f"Assigned {assigned_composite.digit_length}-digit composite for {method.upper()} sweep (B1={computed_b1})"
        }
        return _json_response(response_data)
