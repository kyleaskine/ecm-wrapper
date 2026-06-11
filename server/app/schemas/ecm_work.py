from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ECMWorkResponse(BaseModel):
    """Response schema for ECM-specific work assignment."""
    work_id: Optional[str] = Field(default=None, description="Unique work identifier")
    composite_id: Optional[int] = Field(default=None, description="Database ID of the composite")
    composite: Optional[str] = Field(default=None, description="Number to factor")
    digit_length: Optional[int] = Field(default=None, description="Number of digits in composite")
    current_t_level: Optional[float] = Field(default=None, description="Current t-level progress")
    target_t_level: Optional[float] = Field(default=None, description="Target t-level to reach")
    expires_at: Optional[datetime] = Field(default=None, description="Work assignment expiration")
    message: Optional[str] = Field(default=None, description="Status message or reason for no work")
