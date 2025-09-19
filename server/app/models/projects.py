from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin

class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    composites = relationship("ProjectComposite", back_populates="project")

class ProjectComposite(Base):
    __tablename__ = "project_composites"

    project_id = Column(Integer, ForeignKey("projects.id"), primary_key=True)
    composite_id = Column(Integer, ForeignKey("composites.id"), primary_key=True)
    priority = Column(Integer, default=0, nullable=False)  # 0-10 scale
    added_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="composites")
    composite = relationship("Composite")