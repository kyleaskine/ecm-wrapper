from fastapi import APIRouter
from .submit import router as submit_router
from .stats import router as stats_router
from .web import router as web_router
from .work import router as work_router
from .ecm_work import router as ecm_work_router
from .projects import router as projects_router
from .factors import router as factors_router
from .admin import router as admin_router

# Main v1 API router
v1_router = APIRouter(prefix="/v1")

# Include all v1 endpoints
v1_router.include_router(submit_router, tags=["submit"])
v1_router.include_router(stats_router, tags=["stats"])
v1_router.include_router(web_router, prefix="/dashboard", tags=["web"])
v1_router.include_router(work_router, tags=["work"])
v1_router.include_router(ecm_work_router, tags=["ecm-work"])
v1_router.include_router(projects_router, tags=["projects"])
v1_router.include_router(factors_router, tags=["factors"])
v1_router.include_router(admin_router, tags=["admin"])