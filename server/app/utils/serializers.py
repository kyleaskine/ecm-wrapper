"""
Serializers for converting database models to API-friendly dictionaries.
Reduces duplication in route handlers by centralizing data transformation logic.
"""
from typing import Dict, Any, Optional
from datetime import datetime


def serialize_composite(composite, include_full_number: bool = False) -> Dict[str, Any]:
    """
    Serialize a Composite model to a dictionary.

    Args:
        composite: Composite database model
        include_full_number: If False, truncates long numbers for display

    Returns:
        Dictionary representation of the composite
    """
    number_display = composite.number
    if not include_full_number and len(composite.number) > 50:
        number_display = composite.number[:50] + "..."

    return {
        "id": composite.id,
        "number": number_display,
        "full_number": composite.number if include_full_number else None,
        "current_composite": composite.current_composite,
        "digit_length": composite.digit_length,
        "priority": composite.priority,
        "is_prime": composite.is_prime,
        "is_fully_factored": composite.is_fully_factored,
        "current_t_level": composite.current_t_level,
        "target_t_level": composite.target_t_level,
        "has_snfs_form": composite.has_snfs_form,
        "snfs_difficulty": composite.snfs_difficulty,
        "created_at": composite.created_at,
        "updated_at": composite.updated_at
    }


def serialize_work_assignment(assignment, truncate_composite: bool = True) -> Dict[str, Any]:
    """
    Serialize a WorkAssignment model to a dictionary.

    Args:
        assignment: WorkAssignment database model
        truncate_composite: If True, truncates composite number to 20 chars

    Returns:
        Dictionary representation of the work assignment
    """
    composite_number = assignment.composite.number
    if truncate_composite and len(composite_number) > 20:
        composite_number = composite_number[:20] + "..."

    return {
        "work_id": assignment.id,
        "composite_id": assignment.composite_id,
        "composite_number": composite_number,
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


def serialize_ecm_attempt(attempt, _truncate_composite: bool = True) -> Dict[str, Any]:
    """
    Serialize an ECMAttempt model to a dictionary.

    Args:
        attempt: ECMAttempt database model
        _truncate_composite: Reserved for future use (currently unused)

    Returns:
        Dictionary representation of the ECM attempt
    """
    return {
        "id": attempt.id,
        "composite_id": attempt.composite_id,
        "client_id": attempt.client_id,
        "method": attempt.method,
        "b1": attempt.b1,
        "b2": attempt.b2,
        "curves_completed": attempt.curves_completed,
        "curves_requested": attempt.curves_requested,
        "factor_found": attempt.factor_found,
        "sigma": attempt.sigma,
        "execution_time_seconds": attempt.execution_time_seconds,
        "created_at": attempt.created_at
    }


def serialize_factor(factor, include_composite_details: bool = False) -> Dict[str, Any]:
    """
    Serialize a Factor model to a dictionary.

    Args:
        factor: Factor database model
        include_composite_details: If True, includes composite info

    Returns:
        Dictionary representation of the factor
    """
    result = {
        "id": factor.id,
        "composite_id": factor.composite_id,
        "factor": factor.factor,
        "discovery_method": factor.discovery_method,
        "found_by_attempt_id": factor.found_by_attempt_id,
        "created_at": factor.created_at
    }

    if include_composite_details and hasattr(factor, 'composite') and factor.composite:
        result["composite_number"] = (
            factor.composite.number[:40] + "..."
            if len(factor.composite.number) > 40
            else factor.composite.number
        )
        result["composite_digits"] = factor.composite.digit_length

    return result


def serialize_project(project) -> Dict[str, Any]:
    """
    Serialize a Project model to a dictionary.

    Args:
        project: Project database model

    Returns:
        Dictionary representation of the project
    """
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
        "updated_at": project.updated_at
    }


def serialize_client_info(client_id: str, work_count: int, last_seen: Optional[datetime]) -> Dict[str, Any]:
    """
    Serialize client information from aggregated query results.

    Args:
        client_id: Client identifier
        work_count: Number of work assignments
        last_seen: Last activity timestamp

    Returns:
        Dictionary representation of client info
    """
    return {
        "client_id": client_id,
        "work_count": work_count,
        "last_seen": last_seen,
        "last_seen_str": last_seen.strftime('%Y-%m-%d %H:%M') if last_seen else 'Unknown'
    }


def serialize_bulk_response(stats: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized bulk operation response.

    Args:
        stats: Statistics from the bulk operation
        additional_info: Optional additional information to include

    Returns:
        Standardized response dictionary
    """
    response = {
        "status": "completed",
        **stats
    }
    if additional_info:
        response.update(additional_info)
    return response
