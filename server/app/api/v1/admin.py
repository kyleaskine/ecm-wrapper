"""
Admin API router - aggregates all admin sub-modules.

This file acts as a clean aggregator, routing requests to specialized modules:
- dashboard: Login, dashboard, and stats
- composites: Composite management operations
- work: Work assignment management
- projects: Project management
- maintenance: System maintenance and t-level calculations
"""
from fastapi import APIRouter

from .admin import (
    dashboard_router,
    composites_router,
    work_router,
    projects_router,
    maintenance_router
)

router = APIRouter()

# Include all sub-routers with /admin prefix
router.include_router(dashboard_router, prefix="/admin", tags=["admin-dashboard"])
router.include_router(composites_router, prefix="/admin", tags=["admin-composites"])
router.include_router(work_router, prefix="/admin", tags=["admin-work"])
router.include_router(projects_router, prefix="/admin", tags=["admin-projects"])
router.include_router(maintenance_router, prefix="/admin", tags=["admin-maintenance"])
