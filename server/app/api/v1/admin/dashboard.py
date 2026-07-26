"""
Admin dashboard and authentication routes.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import and_, distinct
from sqlalchemy.orm import Session

from ....constants import ACTIVE_WORK_STATUSES
from ....database import get_db
from ....dependencies import verify_admin_key, verify_admin_key_html
from ....templates import templates
from ....utils.query_helpers import (
    get_composites_by_completion,
    get_recent_clients,
    get_recent_factors,
    get_recent_work_assignments,
    get_aggregated_attempts
)

router = APIRouter()


@router.get("/stats/summary")
def get_admin_summary(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Get high-level summary statistics for admin dashboard.

    Returns:
        Summary statistics
    """
    from ....models.composites import Composite
    from ....models.work_assignments import WorkAssignment
    from ....models.attempts import ECMAttempt
    from ....models.factors import Factor

    # Time range for recent activity
    last_24h = datetime.utcnow() - timedelta(hours=24)
    last_week = datetime.utcnow() - timedelta(days=7)

    # Basic counts
    total_composites = db.query(Composite).count()
    factored_composites = db.query(Composite).filter(
        Composite.is_fully_factored
    ).count()
    active_work = db.query(WorkAssignment).filter(
        WorkAssignment.status.in_(ACTIVE_WORK_STATUSES)
    ).count()

    # Recent activity
    recent_attempts = db.query(ECMAttempt).filter(
        ECMAttempt.created_at >= last_24h
    ).count()
    recent_factors = db.query(Factor).filter(Factor.created_at >= last_24h).count()

    # Active clients
    active_clients = db.query(distinct(WorkAssignment.client_id)).filter(
        and_(
            WorkAssignment.status.in_(ACTIVE_WORK_STATUSES),
            WorkAssignment.created_at >= last_week
        )
    ).count()

    # T-level targeting statistics
    composites_with_target = db.query(Composite).filter(
        and_(
            Composite.target_t_level.isnot(None),
            Composite.is_fully_factored == False  # SQLAlchemy comparison, not Python 'not'
        )
    ).count()

    return {
        "overview": {
            "total_composites": total_composites,
            "factored_composites": factored_composites,
            "unfactored_composites": total_composites - factored_composites,
            "active_work_assignments": active_work,
            "active_clients": active_clients
        },
        "recent_activity_24h": {
            "new_attempts": recent_attempts,
            "factors_found": recent_factors
        },
        "t_level_targeting": {
            "composites_with_targets": composites_with_target
        },
        "health": {
            "factorization_rate": round(
                (factored_composites / max(total_composites, 1)) * 100, 1
            ),
            "work_queue_utilization": round(
                (active_work / max(total_composites - factored_composites, 1)) * 100, 1
            ) if total_composites > factored_composites else 0,
            "t_level_coverage": round(
                (composites_with_target / max(total_composites - factored_composites, 1)) * 100, 1
            ) if total_composites > factored_composites else 0
        }
    }


@router.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """
    Admin login page - prompts for API key and redirects to dashboard.
    """
    return templates.TemplateResponse(request, "admin/login.html")


@router.get("/dashboard", response_class=HTMLResponse)
def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key_html)
):
    """
    Admin dashboard showing system status, work assignments, and management tools.
    This endpoint allows initial page load without auth, but JavaScript checks the key.
    """

    # Get summary statistics
    summary_stats = get_admin_summary(db)

    # Use shared query helpers
    recent_work = get_recent_work_assignments(db, limit=20)
    composites = get_composites_by_completion(db, limit=50, include_factored=False)
    recent_clients = get_recent_clients(db, limit=10, days=7)
    recent_factors = get_recent_factors(db, limit=10)
    recent_attempts, _ = get_aggregated_attempts(db, limit=50, attempts_per_composite=0)

    # Return template response
    return templates.TemplateResponse(request, "admin/dashboard.html", {
        "summary_stats": summary_stats,
        "recent_work": recent_work,
        "composites": composites,
        "recent_clients": recent_clients,
        "recent_factors": recent_factors,
        "recent_attempts": recent_attempts,
        "now": datetime.utcnow(),
    })
