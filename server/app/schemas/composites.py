from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Literal
from datetime import datetime

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
        None, description="GNFS-equivalent digit count for SNFS numbers"
    )
    target_t_level: Optional[float] = Field(..., description="Target t-level")
    current_t_level: Optional[float] = Field(
        ..., description="Current t-level achieved"
    )
    priority: int = Field(..., description="Priority level")
    status: Literal["composite", "prime", "fully_factored"] = Field(
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
    current_t_level: Optional[float]
    priority: int
    is_prime: Optional[bool]
    is_fully_factored: bool
    created_at: datetime
    updated_at: datetime

class CompositeInput(BaseModel):
    """Schema for bulk composite input with optional SNFS fields"""
    number: str = Field(
        ..., description="Original number or mathematical form (e.g., '2^1223-1')"
    )
    current_composite: Optional[str] = Field(
        None,
        description="Current composite being factored (if different from number)"
    )
    has_snfs_form: bool = Field(
        False, description="Whether number has SNFS polynomial form"
    )
    snfs_difficulty: Optional[int] = Field(
        None, description="GNFS-equivalent digit count for SNFS numbers"
    )
    priority: int = Field(0, description="Priority level for work assignment")
    is_prime: Optional[bool] = Field(
        None, description="Whether the composite is actually prime"
    )
    is_fully_factored: Optional[bool] = Field(
        None, description="Whether the composite is fully factored"
    )

class BulkCompositeRequest(BaseModel):
    """Schema for bulk composite upload"""
    composites: List[CompositeInput] = Field(
        ..., description="List of composites to add"
    )
    default_priority: int = Field(
        0, description="Default priority for composites without specified priority"
    )
    project_name: Optional[str] = Field(
        None, description="Optional project name to associate composites with"
    )

class ProjectCreate(BaseModel):
    """Schema for creating a project"""
    name: str = Field(..., description="Unique project name")
    description: Optional[str] = Field(None, description="Project description")

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
    target_t_level: Optional[float] = Field(None, description="Target t-level")
    current_t_level: Optional[float] = Field(None, description="Current t-level achieved")
    digit_length: Optional[int] = Field(None, description="Decimal digit length")
    has_snfs_form: Optional[bool] = Field(None, description="Whether number has SNFS form")
    snfs_difficulty: Optional[int] = Field(None, description="GNFS-equivalent digit count for SNFS")
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
    snfs_difficulty: Optional[int] = Field(None, description="GNFS-equivalent digit count for SNFS")
    target_t_level: Optional[float] = Field(None, description="Target t-level")
    current_t_level: Optional[float] = Field(None, description="Current t-level achieved")
    completion_pct: float = Field(..., description="ECM completion percentage (current_t_level / target_t_level * 100)")
    priority: int = Field(..., description="Priority level")
    is_fully_factored: bool = Field(..., description="Whether fully factored")
    projects: List[str] = Field(default_factory=list, description="Associated projects")

class TopCompositesResponse(BaseModel):
    """Schema for top composites by progress response"""
    composites: List[CompositeProgressItem] = Field(..., description="Composites sorted by progress")
    total: int = Field(..., description="Total matching composites")
    limit: int = Field(..., description="Requested limit")