"""
Project management routes for admin.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key
from ....schemas.composites import ProjectCreate, ProjectResponse
from ....services.project_service import ProjectService
from ....utils.transactions import transaction_scope

router = APIRouter()


@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Create a new project."""
    with transaction_scope(db, "create_project"):
        return ProjectService.create_project(db, project.name, project.description)


@router.delete("/projects/by-name/{project_name}")
async def delete_project_by_name(
    project_name: str,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Delete a project by name and its composite associations."""
    with transaction_scope(db, "delete_project"):
        return ProjectService.delete_project(db, project_name)


@router.delete("/projects/{project_id}")
async def delete_project_by_id(
    project_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Delete a project by ID and its composite associations."""
    with transaction_scope(db, "delete_project"):
        return ProjectService.delete_project(db, project_id)
