from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class WorkRequest(BaseModel):
    client_id: str = Field(..., description="Client requesting work")
    methods: List[Literal["ecm", "pm1", "pp1", "qs", "nfs"]] = Field(
        default=["ecm", "pm1"], description="Preferred methods"
    )
    max_digits: Optional[int] = Field(default=None, description="Maximum digits to handle")
    min_digits: Optional[int] = Field(default=None, description="Minimum digits to handle")

class WorkResponse(BaseModel):
    work_id: Optional[str] = Field(default=None, description="Unique work identifier")
    composite: Optional[str] = Field(default=None, description="Number to factor")
    method: Optional[Literal["ecm", "pm1", "pp1", "qs", "nfs"]] = Field(default=None, description="Assigned method")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    estimated_time_minutes: Optional[int] = Field(default=None, description="Estimated completion time")
    expires_at: Optional[datetime] = Field(default=None, description="Work assignment expiration")
    message: Optional[str] = Field(default=None, description="Status message or reason for no work")

class ManualReservationRequest(BaseModel):
    """Request schema for manually reserving composites for external work (e.g., NFS)"""
    composite_identifier: str = Field(..., description="Composite number or ID to reserve")
    client_id: str = Field(..., description="Client/system reserving the composite (e.g., 'opn-nfs')")
    method: Literal["nfs", "snfs", "gnfs", "other"] = Field(..., description="Type of work being reserved for")
    duration_hours: int = Field(default=168, description="How long to reserve (hours), default 7 days")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the reservation")

class ManualReservationResponse(BaseModel):
    """Response schema for manual reservation"""
    work_id: str = Field(..., description="Work assignment ID for this reservation")
    composite_id: int = Field(..., description="Database ID of the composite")
    composite_number: str = Field(..., description="The composite number")
    client_id: str = Field(..., description="Client/system that reserved it")
    method: str = Field(..., description="Type of work")
    expires_at: datetime = Field(..., description="When this reservation expires")
    status: str = Field(..., description="Status of the reservation")