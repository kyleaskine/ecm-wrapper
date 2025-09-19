from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, distinct, func, desc
from typing import List, Optional, Dict, Any
import io
import logging
from datetime import datetime, timedelta

from ...database import get_db
from ...services.composite_manager import CompositeManager
from ...services.work_assignment import WorkAssignmentService
# Removed family-related services for minimal ECM middleware
from ...config import get_settings

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize services
composite_manager = CompositeManager()
work_service = WorkAssignmentService(
    default_timeout_minutes=settings.default_work_timeout_minutes,
    max_work_per_client=settings.max_work_items_per_client
)

@router.post("/admin/composites/upload")
async def upload_composites(
    file: UploadFile = File(...),
    source_type: str = Form("auto"),
    default_priority: int = Form(0),
    number_column: str = Form("number"),
    priority_column: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload composites from a file.

    Supports text files (one number per line) and CSV files with headers.

    Args:
        file: Uploaded file containing numbers
        source_type: Type of file ('text', 'csv', or 'auto' to detect)
        default_priority: Default priority for new composites
        number_column: Column name for numbers in CSV files
        priority_column: Optional column name for priorities in CSV
        db: Database session

    Returns:
        Upload statistics
    """
    try:
        content = await file.read()
        content_str = content.decode('utf-8')

        # Auto-detect file type if needed
        if source_type == "auto":
            if file.filename and file.filename.endswith('.csv'):
                source_type = "csv"
            else:
                source_type = "text"

        # Process based on file type
        if source_type == "csv":
            stats = composite_manager.bulk_load_composites(
                db, content_str, source_type="csv", default_priority=default_priority
            )
        else:
            # Treat as text file - extract numbers from content
            lines = content_str.strip().split('\n')
            stats = composite_manager.bulk_load_composites(
                db, lines, source_type="list", default_priority=default_priority
            )

        return {
            "filename": file.filename,
            "file_size": len(content),
            "source_type": source_type,
            **stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing file: {str(e)}"
        )


@router.post("/admin/composites/bulk")
async def bulk_add_composites(
    numbers: List[str],
    default_priority: int = 0,
    db: Session = Depends(get_db)
):
    """
    Add a list of composite numbers.

    Args:
        numbers: List of number strings
        default_priority: Default priority for new composites
        db: Database session

    Returns:
        Addition statistics
    """
    try:
        stats = composite_manager.bulk_load_composites(
            db, numbers, source_type="list", default_priority=default_priority
        )

        return {
            "input_count": len(numbers),
            **stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing numbers: {str(e)}"
        )


@router.get("/admin/composites/status")
async def get_queue_status(db: Session = Depends(get_db)):
    """
    Get comprehensive status of the work queue.

    Returns:
        Detailed statistics about composites and work assignments
    """
    try:
        status_info = composite_manager.get_work_queue_status(db)
        return status_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/admin/composites/{composite_id}")
async def get_composite_details(
    composite_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific composite.

    Args:
        composite_id: ID of the composite
        db: Database session

    Returns:
        Detailed composite information including attempts and active work
    """
    details = composite_manager.get_composite_details(db, composite_id)

    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    return details


@router.put("/admin/composites/{composite_id}/priority")
async def set_composite_priority(
    composite_id: int,
    priority: int,
    db: Session = Depends(get_db)
):
    """
    Set priority for a composite.

    Args:
        composite_id: ID of the composite
        priority: New priority value (higher = more priority)
        db: Database session

    Returns:
        Success status
    """
    success = composite_manager.set_composite_priority(db, composite_id, priority)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    return {
        "composite_id": composite_id,
        "priority": priority,
        "status": "updated"
    }


@router.post("/admin/composites/{composite_id}/complete")
async def mark_composite_complete(
    composite_id: int,
    reason: str = "manual",
    db: Session = Depends(get_db)
):
    """
    Mark a composite as fully factored.

    This will cancel any active work assignments for this composite.

    Args:
        composite_id: ID of the composite
        reason: Reason for marking complete
        db: Database session

    Returns:
        Success status
    """
    success = composite_manager.mark_composite_complete(db, composite_id, reason)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    return {
        "composite_id": composite_id,
        "status": "marked_complete",
        "reason": reason
    }


@router.delete("/admin/composites/{composite_id}")
async def remove_composite(
    composite_id: int,
    reason: str = "admin_removal",
    db: Session = Depends(get_db)
):
    """
    Remove a composite from the queue entirely.

    This will cancel any active work assignments and delete the composite.

    Args:
        composite_id: ID of the composite to remove
        reason: Reason for removal
        db: Database session

    Returns:
        Success status
    """
    from ...models.composites import Composite
    from ...models.work_assignments import WorkAssignment
    from ...models.attempts import ECMAttempt
    from ...models.factors import Factor

    composite = db.query(Composite).filter(Composite.id == composite_id).first()

    if not composite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Composite not found"
        )

    # Cancel any active work assignments
    active_work = db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.composite_id == composite_id,
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        )
    ).all()

    for work in active_work:
        work.status = 'cancelled'

    # Delete related records
    db.query(ECMAttempt).filter(ECMAttempt.composite_id == composite_id).delete()
    db.query(Factor).filter(Factor.composite_id == composite_id).delete()
    db.query(WorkAssignment).filter(WorkAssignment.composite_id == composite_id).delete()

    # Delete the composite
    db.delete(composite)
    db.commit()

    return {
        "composite_id": composite_id,
        "status": "removed",
        "reason": reason,
        "cancelled_work_assignments": len(active_work)
    }


@router.get("/admin/work/assignments")
async def get_work_assignments(
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get work assignments with optional filtering.

    Args:
        status_filter: Filter by status (assigned, claimed, running, completed, etc.)
        client_id: Filter by client ID
        limit: Maximum number of assignments to return
        db: Database session

    Returns:
        List of work assignments
    """
    from ...models.work_assignments import WorkAssignment
    from sqlalchemy import and_

    query = db.query(WorkAssignment)

    # Apply filters
    filters = []
    if status_filter:
        filters.append(WorkAssignment.status == status_filter)
    if client_id:
        filters.append(WorkAssignment.client_id == client_id)

    if filters:
        query = query.filter(and_(*filters))

    # Get assignments with composite info
    assignments = query.order_by(
        WorkAssignment.created_at.desc()
    ).limit(limit).all()

    return {
        "assignments": [
            {
                "work_id": assignment.id,
                "composite_id": assignment.composite_id,
                "composite_number": assignment.composite.number[:20] + "..." if len(assignment.composite.number) > 20 else assignment.composite.number,
                "composite_digits": assignment.composite.digit_length,
                "client_id": assignment.client_id,
                "method": assignment.method,
                "b1": assignment.b1,
                "b2": assignment.b2,
                "curves_requested": assignment.curves_requested,
                "curves_completed": assignment.curves_completed,
                "status": assignment.status,
                "priority": assignment.priority,
                "assigned_at": assignment.assigned_at,
                "claimed_at": assignment.claimed_at,
                "expires_at": assignment.expires_at,
                "completed_at": assignment.completed_at,
                "estimated_time_minutes": assignment.estimated_time_minutes,
                "is_expired": assignment.is_expired
            }
            for assignment in assignments
        ],
        "total_count": len(assignments),
        "filters_applied": {
            "status": status_filter,
            "client_id": client_id
        }
    }


@router.delete("/admin/work/assignments/{work_id}")
async def cancel_work_assignment(
    work_id: str,
    reason: str = "admin_cancel",
    db: Session = Depends(get_db)
):
    """
    Cancel a work assignment (admin override).

    Args:
        work_id: ID of the work assignment to cancel
        reason: Reason for cancellation
        db: Database session

    Returns:
        Success status
    """
    from ...models.work_assignments import WorkAssignment

    assignment = db.query(WorkAssignment).filter(
        WorkAssignment.id == work_id
    ).first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work assignment not found"
        )

    # Cancel the assignment
    assignment.status = 'failed'
    db.commit()

    return {
        "work_id": work_id,
        "status": "cancelled",
        "reason": reason,
        "previous_status": assignment.status
    }


@router.post("/admin/work/cleanup")
async def cleanup_expired_work(db: Session = Depends(get_db)):
    """
    Manually trigger cleanup of expired work assignments.

    Returns:
        Number of assignments cleaned up
    """
    from ...models.work_assignments import WorkAssignment
    from sqlalchemy import and_
    from datetime import datetime

    # Find expired assignments
    expired_assignments = db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.expires_at < datetime.utcnow(),
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        )
    ).all()

    # Mark them as timeout
    for assignment in expired_assignments:
        assignment.status = 'timeout'

    db.commit()

    return {
        "cleaned_up": len(expired_assignments),
        "status": "completed"
    }


@router.post("/admin/composites/calculate-t-levels")
async def calculate_t_levels_for_all_composites(
    recalculate_all: bool = False,
    db: Session = Depends(get_db)
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
    from ...models.composites import Composite
    from ...models.attempts import ECMAttempt
    from ...services.t_level_calculator import TLevelCalculator

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

    for composite in composites:
        try:
            # Calculate/update target t-level if not set or if recalculating all
            if composite.target_t_level is None or recalculate_all:
                target_t = calculator.calculate_target_t_level(
                    composite.digit_length,
                    special_form=None  # No auto-detection, projects can specify if needed
                )
                composite.target_t_level = target_t

            # Always recalculate current t-level from existing attempts using real executable
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
            logger.warning(f"Failed to update composite {composite.id}: {e}")
            continue

    # Commit all changes
    db.commit()

    return {
        "status": "completed",
        "composites_updated": updated_count,
        "current_t_levels_updated": current_t_updated,
        "operation_type": operation_type,
        "message": f"{operation_type} t-levels for {updated_count} composites. Updated {current_t_updated} current t-level values using real executable."
    }


@router.post("/admin/composites/recalculate-all-t-levels")
async def recalculate_all_t_levels(db: Session = Depends(get_db)):
    """
    Force recalculation of ALL t-levels (both target and current) for all composites.

    This will replace any existing current t-level values with fresh calculations
    using the real t-level executable.

    Returns:
        Statistics about t-level recalculations performed
    """
    return await calculate_t_levels_for_all_composites(recalculate_all=True, db=db)


@router.get("/admin/stats/summary")
async def get_admin_summary(db: Session = Depends(get_db)):
    """
    Get high-level summary statistics for admin dashboard.

    Returns:
        Summary statistics
    """
    from ...models.composites import Composite
    from ...models.work_assignments import WorkAssignment
    from ...models.attempts import ECMAttempt
    from ...models.factors import Factor
    from sqlalchemy import func, distinct
    from datetime import datetime, timedelta

    # Time range for recent activity
    last_24h = datetime.utcnow() - timedelta(hours=24)
    last_week = datetime.utcnow() - timedelta(days=7)

    # Basic counts
    total_composites = db.query(Composite).count()
    factored_composites = db.query(Composite).filter(Composite.is_fully_factored == True).count()
    active_work = db.query(WorkAssignment).filter(
        WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
    ).count()

    # Recent activity
    recent_attempts = db.query(ECMAttempt).filter(ECMAttempt.created_at >= last_24h).count()
    recent_factors = db.query(Factor).filter(Factor.created_at >= last_24h).count()

    # Active clients
    active_clients = db.query(distinct(WorkAssignment.client_id)).filter(
        and_(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running']),
            WorkAssignment.created_at >= last_week
        )
    ).count()

    # T-level targeting statistics
    composites_with_target = db.query(Composite).filter(
        and_(
            Composite.target_t_level.isnot(None),
            Composite.is_fully_factored == False
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
            "factorization_rate": round((factored_composites / max(total_composites, 1)) * 100, 1),
            "work_queue_utilization": round((active_work / max(total_composites - factored_composites, 1)) * 100, 1) if total_composites > factored_composites else 0,
            "t_level_coverage": round((composites_with_target / max(total_composites - factored_composites, 1)) * 100, 1) if total_composites > factored_composites else 0
        }
    }


# Family management endpoints removed for minimal ECM middleware


@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(db: Session = Depends(get_db)):
    """
    Admin dashboard showing system status, work assignments, and management tools.
    """
    from ...models.composites import Composite
    from ...models.work_assignments import WorkAssignment
    from ...models.attempts import ECMAttempt
    from ...models.factors import Factor
    from ...models.clients import Client

    # Get summary statistics
    summary_stats = await get_admin_summary(db)

    # Get recent work assignments
    recent_work = db.query(WorkAssignment).order_by(
        desc(WorkAssignment.created_at)
    ).limit(20).all()

    # Get composites sorted by ECM completion percentage (how close to target t-level)
    # Only show unfactored composites with target t-levels set
    composites = db.query(Composite).filter(
        and_(
            Composite.is_fully_factored == False,
            Composite.target_t_level.isnot(None)
        )
    ).all()

    # Sort by completion percentage (current_t / target_t)
    def get_completion_pct(comp):
        if comp.target_t_level and comp.target_t_level > 0:
            current_t = comp.current_t_level or 0.0
            return (current_t / comp.target_t_level) * 100
        return 0.0

    composites.sort(key=get_completion_pct, reverse=True)
    composites = composites[:50]  # Limit to 50

    # Get recent clients
    recent_clients = db.query(
        WorkAssignment.client_id,
        func.count(WorkAssignment.id).label('work_count'),
        func.max(WorkAssignment.created_at).label('last_seen')
    ).group_by(WorkAssignment.client_id).order_by(
        desc('last_seen')
    ).limit(10).all()

    # Get recent factors
    recent_factors = db.query(Factor).order_by(desc(Factor.created_at)).limit(10).all()

# Simplified ECM middleware - no family tracking needed

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECM Admin Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            .section {{ background: white; margin-bottom: 20px; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .section-header {{ background: #667eea; color: white; padding: 15px 20px; font-weight: bold; }}
            .section-content {{ padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            .status-assigned {{ color: #fd7e14; font-weight: bold; }}
            .status-claimed {{ color: #0d6efd; font-weight: bold; }}
            .status-running {{ color: #198754; font-weight: bold; }}
            .status-completed {{ color: #6c757d; }}
            .status-timeout {{ color: #dc3545; }}
            .number {{ font-family: monospace; word-break: break-all; background: #f8f9fa; padding: 2px 6px; border-radius: 3px; }}
            .progress {{ background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; }}
            .progress-bar {{ background: #198754; height: 100%; transition: width 0.3s; }}
            .tools {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 15px; }}
            .btn {{ background: #667eea; color: white; padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px; }}
            .btn:hover {{ background: #5a67d8; }}
            .health-good {{ color: #198754; }}
            .health-warning {{ color: #fd7e14; }}
            .health-critical {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔧 ECM Coordination Dashboard</h1>
            <p>Distributed ECM coordination middleware - work assignment and t-level progress tracking</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{summary_stats['overview']['total_composites']}</div>
                <div class="stat-label">Total Composites</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary_stats['overview']['active_work_assignments']}</div>
                <div class="stat-label">Active Work</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary_stats['overview']['active_clients']}</div>
                <div class="stat-label">Active Clients</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary_stats['health']['work_queue_utilization']}%</div>
                <div class="stat-label">Queue Utilization</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary_stats['t_level_targeting']['composites_with_targets']}</div>
                <div class="stat-label">T-Level Targets Set</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary_stats['health']['t_level_coverage']}%</div>
                <div class="stat-label">T-Level Coverage</div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">📊 ECM Progress Overview</div>
            <div class="section-content">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div>
                        <strong>Work Queue Status:</strong>
                        <span class="{'health-good' if summary_stats['health']['work_queue_utilization'] > 50 else 'health-warning' if summary_stats['health']['work_queue_utilization'] > 0 else 'health-critical'}">{summary_stats['health']['work_queue_utilization']}% utilized</span>
                    </div>
                    <div>
                        <strong>T-Level Coverage:</strong>
                        <span class="{'health-good' if summary_stats['health']['t_level_coverage'] > 80 else 'health-warning' if summary_stats['health']['t_level_coverage'] > 50 else 'health-critical'}">{summary_stats['health']['t_level_coverage']}%</span>
                    </div>
                    <div>
                        <strong>Recent Activity (24h):</strong>
                        {summary_stats['recent_activity_24h']['new_attempts']} attempts, {summary_stats['recent_activity_24h']['factors_found']} factors
                    </div>
                </div>

                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4 style="margin-top: 0;">⚡ ECM Coordination Status</h4>
                    <p style="margin: 0; color: #666;">
                        This system coordinates distributed ECM work across clients, tracking t-level progress and managing the work queue.
                        Projects can submit numbers with target t-levels and receive factorization results.
                    </p>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">⚡ Active Work Assignments</div>
            <div class="section-content">
    """

    if recent_work:
        html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Work ID</th>
                            <th>Client</th>
                            <th>Composite (digits)</th>
                            <th>T-Level Target</th>
                            <th>Method</th>
                            <th>Parameters</th>
                            <th>Progress</th>
                            <th>Status</th>
                            <th>Expires</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for work in recent_work:
            composite = work.composite
            progress_pct = (work.curves_completed / work.curves_requested * 100) if work.curves_requested > 0 else 0
            status_class = f"status-{work.status.replace('_', '-')}"

            composite_display = f"{composite.number[:20]}..." if len(composite.number) > 20 else composite.number
            time_remaining = work.expires_at - datetime.utcnow() if work.expires_at > datetime.utcnow() else timedelta(0)
            time_str = f"{int(time_remaining.total_seconds() / 60)}m" if time_remaining.total_seconds() > 0 else "Expired"

            # T-level display for work
            current_t = composite.current_t_level or 0.0
            target_t = composite.target_t_level or 0.0

            html_content += f"""
                        <tr>
                            <td><span class="number">{work.id[:8]}...</span></td>
                            <td>{work.client_id}</td>
                            <td><span class="number">{composite_display}</span> ({composite.digit_length})</td>
                            <td>
                                <span style="font-size: 0.9em;">📊 t{current_t:.1f}→t{target_t:.1f}</span>
                            </td>
                            <td>{work.method.upper()}</td>
                            <td>B1={work.b1:,}{f', B2={work.b2:,}' if work.b2 else ''}</td>
                            <td>
                                <div class="progress">
                                    <div class="progress-bar" style="width: {progress_pct}%;"></div>
                                </div>
                                {work.curves_completed}/{work.curves_requested}
                            </td>
                            <td><span class="{status_class}">{work.status}</span></td>
                            <td>{time_str}</td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
        """
    else:
        html_content += "<p>No active work assignments</p>"

    html_content += """
            </div>
        </div>

        <div class="section">
            <div class="section-header">📋 Composite Queue - Sorted by ECM Completion</div>
            <div class="section-content">
                <p style="margin-bottom: 15px; color: #666; font-style: italic;">
                    Showing unfactored composites sorted by completion percentage (closest to target t-level first)
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Number</th>
                            <th>Digits</th>
                            <th>T-Level Progress</th>
                            <th>Priority</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for composite in composites:
        composite_display = f"{composite.number[:30]}..." if len(composite.number) > 30 else composite.number
        created_str = composite.created_at.strftime("%Y-%m-%d %H:%M") if composite.created_at else "Unknown"

        # T-level progress display
        current_t = composite.current_t_level or 0.0
        target_t = composite.target_t_level or 0.0
        t_progress_pct = (current_t / target_t * 100) if target_t > 0 else 0
        t_progress_color = "#198754" if t_progress_pct >= 100 else "#fd7e14" if t_progress_pct >= 50 else "#dc3545"

        html_content += f"""
                        <tr>
                            <td>{composite.id}</td>
                            <td><span class="number">{composite_display}</span></td>
                            <td>{composite.digit_length}</td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div class="progress" style="width: 100px;">
                                        <div class="progress-bar" style="width: {min(t_progress_pct, 100)}%; background-color: {t_progress_color};"></div>
                                    </div>
                                    <span style="font-size: 0.85em; font-weight: bold; color: {t_progress_color};">
                                        t{current_t:.1f} / t{target_t:.1f} ({t_progress_pct:.1f}%)
                                    </span>
                                </div>
                            </td>
                            <td>
                                <span style="font-weight: bold; color: {'#198754' if composite.priority > 0 else '#666'};">{composite.priority}</span>
                            </td>
                            <td>{created_str}</td>
                            <td>
                                <button class="btn" onclick="viewDetails({composite.id})" style="margin-right: 5px;">Details</button>
                                <button class="btn" onclick="removeComposite({composite.id})" style="background: #dc3545;">Remove</button>
                            </td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-header">👥 Recent Clients</div>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Client ID</th>
                            <th>Work Count</th>
                            <th>Last Seen</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for client_info in recent_clients:
        last_seen_str = client_info.last_seen.strftime("%Y-%m-%d %H:%M") if client_info.last_seen else "Unknown"
        html_content += f"""
                        <tr>
                            <td>{client_info.client_id}</td>
                            <td>{client_info.work_count}</td>
                            <td>{last_seen_str}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-header">🎯 Recent Factors Found</div>
            <div class="section-content">
    """

    if recent_factors:
        html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Factor</th>
                            <th>Composite</th>
                            <th>Found</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for factor in recent_factors:
            composite = factor.composite
            composite_display = f"{composite.number[:30]}..." if len(composite.number) > 30 else composite.number
            found_str = factor.created_at.strftime("%Y-%m-%d %H:%M") if factor.created_at else "Unknown"

            html_content += f"""
                        <tr>
                            <td><span class="number">{factor.factor}</span></td>
                            <td><span class="number">{composite_display}</span></td>
                            <td>{found_str}</td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
        """
    else:
        html_content += "<p>No factors found recently</p>"

    html_content += """
            </div>
        </div>

        <div class="section">
            <div class="section-header">🛠️ Admin Tools</div>
            <div class="section-content">
                <div class="tools">
                    <h4>Quick Actions:</h4>
                    <button class="btn" onclick="location.href='/docs#/admin'">API Documentation</button>
                    <button class="btn" onclick="location.href='/api/v1/dashboard/'">User Dashboard</button>
                    <button class="btn" onclick="cleanupExpired()">Cleanup Expired Work</button>
                    <button class="btn" onclick="calculateTLevels()">Calculate T-Levels</button>
                    <button class="btn" onclick="recalculateAllTLevels()" style="background: #dc3545;">Fix Random T-Levels</button>
                    <button class="btn" onclick="refreshPage()">Refresh</button>
                </div>

                <div style="margin-top: 20px;">
                    <h4>Bulk Operations:</h4>
                    <p>Use the API endpoints to:</p>
                    <ul>
                        <li><strong>POST /admin/composites/upload</strong> - Upload composite files</li>
                        <li><strong>POST /admin/composites/bulk</strong> - Add numbers via JSON</li>
                        <li><strong>GET /admin/work/assignments</strong> - View all work assignments</li>
                        <li><strong>POST /admin/work/cleanup</strong> - Clean up expired work</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            function viewDetails(compositeId) {
                location.href = '/docs#/admin/get_composite_details_admin_composites__composite_id__get';
            }

            function removeComposite(compositeId) {
                if (confirm('PERMANENTLY REMOVE this composite from the queue?\\n\\nThis will:\\n- Cancel any active work assignments\\n- Delete all ECM attempts and factors\\n- Remove the composite entirely\\n\\nThis action cannot be undone!')) {
                    fetch(`/api/v1/admin/composites/${compositeId}`, {method: 'DELETE'})
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'removed') {
                                alert(`Composite ${compositeId} removed successfully.\\nCancelled ${data.cancelled_work_assignments} work assignments.`);
                                location.reload();
                            } else {
                                alert('Error removing composite: ' + JSON.stringify(data));
                            }
                        })
                        .catch(error => {
                            alert('Error removing composite: ' + error);
                        });
                }
            }

            function cleanupExpired() {
                if (confirm('Clean up expired work assignments?')) {
                    fetch('/api/v1/admin/work/cleanup', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            alert('Cleaned up ' + data.cleaned_up + ' expired assignments');
                            location.reload();
                        });
                }
            }

            function calculateTLevels() {
                if (confirm('Calculate t-levels for all composites using 4/13 * digits formula?')) {
                    fetch('/api/v1/admin/composites/calculate-t-levels', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message || 'T-levels calculated successfully');
                            location.reload();
                        })
                        .catch(error => {
                            alert('Error calculating t-levels: ' + error);
                        });
                }
            }

            function recalculateAllTLevels() {
                if (confirm('RECALCULATE ALL t-levels? This will replace current random t-level values with real calculations using the t-level executable.')) {
                    fetch('/api/v1/admin/composites/recalculate-all-t-levels', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message || 'All t-levels recalculated successfully');
                            location.reload();
                        })
                        .catch(error => {
                            alert('Error recalculating t-levels: ' + error);
                        });
                }
            }

            function refreshPage() {
                location.reload();
            }

            // Simplified ECM coordination - focus on core functionality

            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """

    return html_content