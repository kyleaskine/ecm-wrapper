from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List

from ...database import get_db
from ...schemas.composites import ProjectResponse, ProjectStats
from ...models.projects import Project, ProjectComposite
from ...models.composites import Composite

router = APIRouter()


def _calculate_project_stats(db: Session, project: Project) -> ProjectStats:
    """
    Calculate statistics for a project.

    Args:
        db: Database session
        project: Project model instance

    Returns:
        ProjectStats with composite counts
    """
    total_composites = db.query(ProjectComposite).filter(
        ProjectComposite.project_id == project.id
    ).count()

    unfactored = db.query(ProjectComposite).join(
        Composite, ProjectComposite.composite_id == Composite.id
    ).filter(
        and_(
            ProjectComposite.project_id == project.id,
            Composite.is_fully_factored == False  # noqa: E712 - SQLAlchemy comparison
        )
    ).count()

    factored = total_composites - unfactored

    return ProjectStats(
        project=ProjectResponse.model_validate(project),
        total_composites=total_composites,
        unfactored_composites=unfactored,
        factored_composites=factored
    )


@router.get("/projects", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    """
    List all projects (PUBLIC).

    Returns:
        List of all projects
    """
    projects = db.query(Project).order_by(Project.name).all()
    return projects


@router.get("/projects/by-name/{project_name}", response_model=ProjectStats)
def get_project_by_name(
    project_name: str,
    db: Session = Depends(get_db)
):
    """
    Get project statistics by name (PUBLIC).

    Args:
        project_name: Project name
        db: Database session

    Returns:
        Project with statistics
    """
    project = db.query(Project).filter(Project.name == project_name).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_name}' not found"
        )

    return _calculate_project_stats(db, project)


@router.get("/projects/{project_id}", response_model=ProjectStats)
def get_project_by_id(
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Get project statistics by ID (PUBLIC).

    Args:
        project_id: Project ID
        db: Database session

    Returns:
        Project with statistics
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    return _calculate_project_stats(db, project)
