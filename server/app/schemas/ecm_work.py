from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ECMWorkResponse(BaseModel):
    """Response schema for ECM-specific work assignment."""
    work_id: Optional[str] = Field(None, description="Unique work identifier")
    composite_id: Optional[int] = Field(None, description="Database ID of the composite")
    composite: Optional[str] = Field(None, description="Number to factor")
    digit_length: Optional[int] = Field(None, description="Number of digits in composite")
    current_t_level: Optional[float] = Field(None, description="Current t-level progress")
    target_t_level: Optional[float] = Field(None, description="Target t-level to reach")
    expires_at: Optional[datetime] = Field(None, description="Work assignment expiration")
    message: Optional[str] = Field(None, description="Status message or reason for no work")
