from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from datetime import datetime

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