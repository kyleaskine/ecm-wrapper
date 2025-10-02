"""
Admin API routes - modular organization.
"""
from fastapi import APIRouter

from .dashboard import router as dashboard_router
from .composites import router as composites_router
from .work import router as work_router
from .projects import router as projects_router
from .maintenance import router as maintenance_router

# Main admin router that aggregates all admin sub-routers
router = APIRouter()
router.include_router(dashboard_router, prefix="/admin", tags=["admin-dashboard"])
router.include_router(composites_router, prefix="/admin", tags=["admin-composites"])
router.include_router(work_router, prefix="/admin", tags=["admin-work"])
router.include_router(projects_router, prefix="/admin", tags=["admin-projects"])
router.include_router(maintenance_router, prefix="/admin", tags=["admin-maintenance"])

__all__ = [
    'router',
    'dashboard_router',
    'composites_router',
    'work_router',
    'projects_router',
    'maintenance_router'
]
