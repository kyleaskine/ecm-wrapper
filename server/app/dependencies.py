"""
Dependency injection for FastAPI routes.

Provides reusable dependencies for:
- Admin authentication
- Service injection (composite service, project service, etc.)
- Configuration access
"""

from fastapi import Header, HTTPException, status, Depends
from .config import get_settings

settings = get_settings()

async def verify_admin_key(x_admin_key: str = Header(None)):
    """
    Dependency to verify admin API key from header.

    Requires X-Admin-Key header to match ADMIN_API_KEY environment variable.

    Raises:
        HTTPException: 401 if key missing or invalid
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required. Provide X-Admin-Key header."
        )

    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )

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


def get_work_service(settings_dep=Depends(get_settings)):
    """
    Dependency for work assignment service injection.

    Args:
        settings_dep: Settings dependency (automatically injected)

    Returns:
        WorkAssignmentService instance configured with settings

    Example:
        @router.get("/work")
        async def get_work(
            client_id: str,
            work_service: WorkAssignmentService = Depends(get_work_service)
        ):
            return work_service.get_work_for_client(db, client_id)
    """
    from .services.work_assignment import WorkAssignmentService
    return WorkAssignmentService(
        default_timeout_minutes=settings_dep.default_work_timeout_minutes,
        max_work_per_client=settings_dep.max_work_items_per_client
    )


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
