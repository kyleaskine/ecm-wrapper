from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class WorkRequest(BaseModel):
    client_id: str = Field(..., description="Client requesting work")
    methods: List[Literal["ecm", "pm1", "pp1", "qs", "nfs"]] = Field(
        default=["ecm", "pm1"], description="Preferred methods"
    )
    max_digits: Optional[int] = Field(None, description="Maximum digits to handle")
    min_digits: Optional[int] = Field(None, description="Minimum digits to handle")

class WorkResponse(BaseModel):
    work_id: Optional[str] = Field(None, description="Unique work identifier")
    composite: Optional[str] = Field(None, description="Number to factor")
    method: Optional[Literal["ecm", "pm1", "pp1", "qs", "nfs"]] = Field(None, description="Assigned method")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated completion time")
    expires_at: Optional[datetime] = Field(None, description="Work assignment expiration")
    message: Optional[str] = Field(None, description="Status message or reason for no work")