"""
Project service for managing project CRUD operations.

Consolidates business logic for project management to reduce duplication
in admin route handlers.
"""

from typing import Dict, Union
from sqlalchemy.orm import Session

from ..models.projects import Project, ProjectComposite
from ..utils.errors import not_found_error, ensure_not_exists


class ProjectService:
    """Service for managing project operations."""

    @staticmethod
    def create_project(db: Session, name: str, description: str = None) -> Project:
        """
        Create a new project.

        Args:
            db: Database session
            name: Project name (must be unique)
            description: Optional project description

        Returns:
            Created Project instance

        Raises:
            HTTPException: 400 if project with that name already exists
        """
        # Check if project already exists
        ensure_not_exists(db, Project, error_name=name, name=name)

        new_project = Project(name=name, description=description)
        db.add(new_project)
        db.flush()  # Make visible within transaction
        db.refresh(new_project)

        return new_project

    @staticmethod
    def get_project_by_identifier(
        db: Session,
        identifier: Union[int, str]
    ) -> Project:
        """
        Get project by ID or name.

        Args:
            db: Database session
            identifier: Project ID (int) or name (str)

        Returns:
            Project instance if found, None otherwise
        """
        if isinstance(identifier, int):
            return db.query(Project).filter(Project.id == identifier).first()
        else:
            return db.query(Project).filter(Project.name == identifier).first()

    @staticmethod
    def delete_project(
        db: Session,
        identifier: Union[int, str]
    ) -> Dict:
        """
        Delete project by ID or name and all associated composite relationships.

        Args:
            db: Database session
            identifier: Project ID (int) or name (str)

        Returns:
            Dictionary with deletion details

        Raises:
            HTTPException: 404 if project not found
        """
        # Get project by ID or name
        project = ProjectService.get_project_by_identifier(db, identifier)

        if not project:
            raise not_found_error("Project", str(identifier))

        # Delete project-composite associations
        associations_deleted = db.query(ProjectComposite).filter(
            ProjectComposite.project_id == project.id
        ).delete()

        # Store project info before deletion
        project_info = {
            "message": f"Project '{project.name}' deleted successfully",
            "project_id": project.id,
            "project_name": project.name,
            "associations_deleted": associations_deleted
        }

        # Delete project
        db.delete(project)
        db.flush()  # Make visible within transaction

        return project_info

    @staticmethod
    def list_projects(db: Session, limit: int = 100) -> list[Project]:
        """
        List all projects.

        Args:
            db: Database session
            limit: Maximum number of projects to return

        Returns:
            List of Project instances
        """
        return db.query(Project).limit(limit).all()

    @staticmethod
    def get_project_composites(db: Session, project_id: int) -> list:
        """
        Get all composites associated with a project.

        Args:
            db: Database session
            project_id: Project ID

        Returns:
            List of composite IDs associated with the project
        """
        associations = db.query(ProjectComposite).filter(
            ProjectComposite.project_id == project_id
        ).all()

        return [assoc.composite_id for assoc in associations]
