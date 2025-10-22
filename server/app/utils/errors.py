"""
Error handling utilities for consistent HTTP exceptions across the API.

Provides reusable helpers for common error patterns like 404 Not Found,
400 Bad Request for duplicates, and resource lookup with automatic 404.
"""

from fastapi import HTTPException, status
from typing import Optional, TypeVar
from sqlalchemy.orm import Session

T = TypeVar('T')


def not_found_error(resource_type: str, identifier: str = None) -> HTTPException:
    """
    Create a consistent 404 Not Found error.

    Args:
        resource_type: Type of resource (e.g., "Composite", "Project", "Work Assignment")
        identifier: Optional identifier to include in error message

    Returns:
        HTTPException with 404 status code

    Example:
        raise not_found_error("Composite", "123456789")
        # HTTPException(status_code=404, detail="Composite not found: 123456789")
    """
    msg = f"{resource_type} not found"
    if identifier:
        msg += f": {identifier}"
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=msg
    )


def already_exists_error(resource_type: str, name: str) -> HTTPException:
    """
    Create a consistent 400 Bad Request error for duplicate resources.

    Args:
        resource_type: Type of resource (e.g., "Project", "Client")
        name: Name/identifier of the duplicate resource

    Returns:
        HTTPException with 400 status code

    Example:
        raise already_exists_error("Project", "aliquot_sequences")
        # HTTPException(status_code=400, detail="Project 'aliquot_sequences' already exists")
    """
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"{resource_type} '{name}' already exists"
    )


def get_or_404(query_result: Optional[T], resource_type: str, identifier: str = None) -> T:
    """
    Helper to get a resource or raise 404 if not found.

    Args:
        query_result: Result from database query (or None)
        resource_type: Type of resource for error message
        identifier: Optional identifier for error message

    Returns:
        The query result if it exists

    Raises:
        HTTPException: 404 if query_result is None

    Example:
        composite = get_or_404(
            db.query(Composite).filter(Composite.id == comp_id).first(),
            "Composite",
            str(comp_id)
        )
    """
    if query_result is None:
        raise not_found_error(resource_type, identifier)
    return query_result


def ensure_not_exists(db: Session, model_class, error_name: str = None, **filters) -> None:
    """
    Check that a resource doesn't exist, raise error if it does.

    Args:
        db: Database session
        model_class: SQLAlchemy model class to query
        error_name: Name to use in error message (defaults to filter values)
        **filters: Filter conditions for the query

    Raises:
        HTTPException: 400 if resource already exists

    Example:
        ensure_not_exists(db, Project, error_name="my_project", name="my_project")
        # Raises 400 if project with that name exists
    """
    existing = db.query(model_class).filter_by(**filters).first()
    if existing:
        name = error_name or str(filters)
        raise already_exists_error(model_class.__name__, name)


def bad_request_error(detail: str) -> HTTPException:
    """
    Create a consistent 400 Bad Request error.

    Args:
        detail: Error message detail

    Returns:
        HTTPException with 400 status code

    Example:
        raise bad_request_error("Invalid factorization method specified")
    """
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail
    )
