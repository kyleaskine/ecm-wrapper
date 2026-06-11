from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict

class EffortLevel(BaseModel):
    b1: int
    curves: int

class ECMWorkSummary(BaseModel):
    total_attempts: int
    total_curves: int
    effort_by_level: List[EffortLevel]
    last_attempt: Optional[datetime]

class CompositeStats(BaseModel):
    composite: str = Field(..., description="The composite number (original form)")
    current_composite: str = Field(
        ..., description="Current composite being factored"
    )
    digit_length: int = Field(..., description="Decimal digit length")
    has_snfs_form: bool = Field(
        ..., description="Whether number has SNFS polynomial form"
    )
    snfs_difficulty: Optional[int] = Field(
        default=None, description="GNFS-equivalent digit count for SNFS numbers"
    )
    target_t_level: Optional[float] = Field(..., description="Target t-level")
    current_t_level: Optional[float] = Field(
        ..., description="Current t-level achieved (includes prior_t_level if set)"
    )
    prior_t_level: Optional[float] = Field(
        default=None, description="T-level from work done before import"
    )
    priority: int = Field(..., description="Priority level")
    is_active: bool = Field(..., description="Whether composite is available for work assignment")
    status: Literal["composite", "sufficient", "fully_factored", "complete"] = Field(
        ..., description="Current status"
    )
    factors_found: List[str] = Field(
        default_factory=list, description="Known factors"
    )
    ecm_work: ECMWorkSummary = Field(..., description="Summary of ECM work done")
    projects: List[str] = Field(
        default_factory=list, description="Associated projects"
    )

class CompositeResponse(BaseModel):
    id: int
    number: str
    current_composite: str
    digit_length: int
    has_snfs_form: bool
    snfs_difficulty: Optional[int]
    target_t_level: Optional[float]
    current_t_level: Optional[float]  # Includes prior_t_level if set
    prior_t_level: Optional[float]
    priority: int
    is_complete: Optional[bool]  # Marks composite as sufficiently complete for OPN purposes
    is_fully_factored: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

class CompositeInput(BaseModel):
    """Schema for bulk composite input with optional SNFS fields"""
    number: str = Field(
        ..., description="Original number or mathematical form (e.g., '2^1223-1')"
    )
    current_composite: Optional[str] = Field(
        default=None,
        description="Current composite being factored (if different from number)"
    )
    has_snfs_form: bool = Field(
        default=False, description="Whether number has SNFS polynomial form"
    )
    snfs_difficulty: Optional[int] = Field(
        default=None, description="GNFS-equivalent digit count for SNFS numbers"
    )
    priority: int = Field(default=0, description="Priority level for work assignment")
    is_complete: Optional[bool] = Field(
        default=None, description="Whether the composite is sufficiently complete for OPN purposes"
    )
    is_fully_factored: Optional[bool] = Field(
        default=None, description="Whether the composite is fully factored"
    )
    is_active: Optional[bool] = Field(
        default=None, description="Whether composite is available for work assignment (None = preserve existing, defaults to False for new composites)"
    )
    prior_t_level: Optional[float] = Field(
        default=None, description="T-level from work done before import (e.g., from factordb or previous campaigns)"
    )

class BulkCompositeRequest(BaseModel):
    """Schema for bulk composite upload"""
    composites: List[CompositeInput] = Field(
        ..., description="List of composites to add"
    )
    default_priority: int = Field(
        default=0, description="Default priority for composites without specified priority"
    )
    project_name: Optional[str] = Field(
        default=None, description="Optional project name to associate composites with"
    )

class ProjectCreate(BaseModel):
    """Schema for creating a project"""
    name: str = Field(..., description="Unique project name")
    description: Optional[str] = Field(default=None, description="Project description")

class ProjectResponse(BaseModel):
    """Schema for project response"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

class ProjectStats(BaseModel):
    """Schema for project statistics"""
    project: ProjectResponse
    total_composites: int
    unfactored_composites: int
    factored_composites: int

class BatchStatusRequest(BaseModel):
    """Schema for batch status request"""
    numbers: List[str] = Field(..., description="List of composite numbers to check")

class CompositeBatchStatus(BaseModel):
    """Schema for individual composite status in batch response"""
    number: str = Field(..., description="The composite number")
    target_t_level: Optional[float] = Field(default=None, description="Target t-level")
    current_t_level: Optional[float] = Field(default=None, description="Current t-level achieved (includes prior_t_level)")
    prior_t_level: Optional[float] = Field(default=None, description="T-level from work done before import")
    digit_length: Optional[int] = Field(default=None, description="Decimal digit length")
    has_snfs_form: Optional[bool] = Field(default=None, description="Whether number has SNFS form")
    snfs_difficulty: Optional[int] = Field(default=None, description="GNFS-equivalent digit count for SNFS")
    found: bool = Field(..., description="Whether the composite exists in database")

class BatchStatusResponse(BaseModel):
    """Schema for batch status response"""
    composites: List[CompositeBatchStatus] = Field(..., description="Status for each composite")

class CompositeProgressItem(BaseModel):
    """Schema for composite with progress information"""
    id: int = Field(..., description="Composite ID")
    number: str = Field(..., description="Original composite number")
    current_composite: str = Field(..., description="Current composite being factored")
    digit_length: int = Field(..., description="Decimal digit length")
    has_snfs_form: bool = Field(..., description="Whether number has SNFS form")
    snfs_difficulty: Optional[int] = Field(
        default=None, description="GNFS-equivalent digit count for SNFS"
    )
    target_t_level: Optional[float] = Field(default=None, description="Target t-level")
    current_t_level: Optional[float] = Field(default=None, description="Current t-level achieved (includes prior_t_level)")
    prior_t_level: Optional[float] = Field(default=None, description="T-level from work done before import")
    completion_pct: float = Field(
        ..., description="ECM completion percentage (current / target * 100)"
    )
    priority: int = Field(..., description="Priority level")
    is_fully_factored: bool = Field(..., description="Whether fully factored")
    is_active: bool = Field(..., description="Whether composite is available for work assignment")
    projects: List[str] = Field(
        default_factory=list, description="Associated projects"
    )

class TopCompositesRequest(BaseModel):
    """Schema for top composites by progress request"""
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum number of composites to return")
    project_name: Optional[str] = Field(default=None, description="Filter by project name")
    min_priority: Optional[int] = Field(default=None, description="Minimum priority level")
    include_factored: bool = Field(default=False, description="Include fully factored composites")
    formulas: Optional[List[str]] = Field(default=None, description="Filter to only these composite formulas")
    min_difficulty: Optional[float] = Field(default=None, description="Minimum effective difficulty (min of digit_length and snfs_difficulty)")
    max_difficulty: Optional[float] = Field(default=None, description="Maximum effective difficulty (min of digit_length and snfs_difficulty)")

class TopCompositesResponse(BaseModel):
    """Schema for top composites by progress response"""
    composites: List[CompositeProgressItem] = Field(
        ..., description="Composites sorted by progress"
    )
    total: int = Field(..., description="Total matching composites")
    limit: int = Field(..., description="Requested limit")
