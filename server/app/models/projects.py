from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .composites import Composite


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    composites: Mapped[List["ProjectComposite"]] = relationship("ProjectComposite", back_populates="project")


class ProjectComposite(Base):
    __tablename__ = "project_composites"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    composite_id: Mapped[int] = mapped_column(ForeignKey("composites.id"), primary_key=True)
    priority: Mapped[int] = mapped_column(default=0, nullable=False)  # 0-10 scale
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="composites")
    composite: Mapped["Composite"] = relationship("Composite")
