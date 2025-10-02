"""
Project management routes for admin.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key
from ....schemas.composites import ProjectCreate, ProjectResponse

router = APIRouter()


@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Create a new project."""
    from ....models.projects import Project

    # Check if project already exists
    existing = db.query(Project).filter(Project.name == project.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project '{project.name}' already exists"
        )

    new_project = Project(
        name=project.name,
        description=project.description
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return new_project


@router.delete("/projects/by-name/{project_name}")
async def delete_project_by_name(
    project_name: str,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Delete a project by name and its composite associations."""
    from ....models.projects import Project, ProjectComposite

    project = db.query(Project).filter(Project.name == project_name).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_name}' not found"
        )

    # Delete project-composite associations
    db.query(ProjectComposite).filter(
        ProjectComposite.project_id == project.id
    ).delete()

    # Delete project
    db.delete(project)
    db.commit()

    return {
        "message": f"Project '{project.name}' deleted successfully",
        "project_id": project.id,
        "project_name": project.name
    }


@router.delete("/projects/{project_id}")
async def delete_project_by_id(
    project_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """Delete a project by ID and its composite associations."""
    from ....models.projects import Project, ProjectComposite

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    # Delete project-composite associations
    db.query(ProjectComposite).filter(
        ProjectComposite.project_id == project_id
    ).delete()

    # Delete project
    db.delete(project)
    db.commit()

    return {
        "message": f"Project '{project.name}' deleted successfully",
        "project_id": project_id,
        "project_name": project.name
    }
