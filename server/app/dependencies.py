"""
Dependency injection for FastAPI routes.

Provides reusable dependencies for:
- Admin authentication
- Service injection (composite service, project service, etc.)
- Configuration access
"""

import secrets
from typing import Optional
from fastapi import Header, HTTPException, status
from .config import get_settings

settings = get_settings()


class AdminAuthRedirect(Exception):
    """Raised when an HTML admin page fails auth - triggers a redirect to login."""
    pass


async def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """
    Dependency to verify admin API key from header.

    Requires X-Admin-Key header to match ADMIN_API_KEY environment variable.
    Uses constant-time comparison to prevent timing attacks.

    Raises:
        HTTPException: 401 if key missing or invalid
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required. Provide X-Admin-Key header."
        )

    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )

    return True


async def verify_admin_key_html(x_admin_key: Optional[str] = Header(None)):
    """
    Dependency to verify admin API key for HTML dashboard routes.

    Like verify_admin_key, but raises AdminAuthRedirect (which triggers a
    redirect to the login page) instead of returning a JSON 401 error.
    The exception handler is registered in main.py.

    Raises:
        AdminAuthRedirect: if key is missing or invalid
    """
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise AdminAuthRedirect()

    return True


# ==================== Service Dependencies ====================

def get_composite_service():
    """
    Dependency for composite service injection.

    Returns:
        CompositeService instance

    Example:
        @router.get("/composites/{composite_id}")
        async def get_composite(
            composite_id: int,
            composite_service: CompositeService = Depends(get_composite_service)
        ):
            return composite_service.get_composite_by_id(db, composite_id)
    """
    from .services.composites import CompositeService
    return CompositeService()


def get_project_service():
    """
    Dependency for project service injection.

    Returns:
        ProjectService instance

    Example:
        @router.post("/projects")
        async def create_project(
            name: str,
            project_service: ProjectService = Depends(get_project_service)
        ):
            return project_service.create_project(db, name)
    """
    from .services.project_service import ProjectService
    return ProjectService()


def get_work_service():
    """
    Dependency for work assignment service injection (claim, start,
    progress, complete and abandon of existing assignments).

    Returns:
        WorkAssignmentService instance
    """
    from .services.work_assignment import WorkAssignmentService
    return WorkAssignmentService()


def get_t_level_calculator():
    """
    Dependency for t-level calculator service injection.

    Returns:
        TLevelCalculator instance

    Example:
        @router.post("/calculate")
        async def calculate_t_level(
            t_level_calc: TLevelCalculator = Depends(get_t_level_calculator)
        ):
            return t_level_calc.calculate_target_t_level(digit_length)
    """
    from .services.t_level_calculator import TLevelCalculator
    return TLevelCalculator()


def get_residue_manager():
    """
    Dependency for residue manager service injection.

    Returns:
        ResidueManager instance

    Example:
        @router.post("/residues/upload")
        async def upload_residue(
            residue_manager: ResidueManager = Depends(get_residue_manager)
        ):
            return residue_manager.store_residue_file(...)
    """
    from .services.residue_manager import ResidueManager
    return ResidueManager()
