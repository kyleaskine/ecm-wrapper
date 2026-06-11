from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ResidueUploadResponse(BaseModel):
    """Response after uploading a residue file."""
    residue_id: int = Field(..., description="Unique ID of the residue record")
    composite_id: int = Field(..., description="Database ID of the composite")
    composite: str = Field(..., description="The composite number (parsed from file)")
    b1: int = Field(..., description="B1 parameter (parsed from file)")
    parametrization: int = Field(..., description="ECM parametrization (parsed from file)")
    curve_count: int = Field(..., description="Number of curves in residue file")
    file_size_bytes: int = Field(..., description="Size of uploaded file")
    message: str = Field(..., description="Status message")


class ResidueWorkResponse(BaseModel):
    """Response for stage 2 work request."""
    residue_id: Optional[int] = Field(default=None, description="Unique ID of the residue")
    composite_id: Optional[int] = Field(default=None, description="Database ID of the composite")
    composite: Optional[str] = Field(default=None, description="Number to factor")
    digit_length: Optional[int] = Field(default=None, description="Number of digits in composite")
    b1: Optional[int] = Field(default=None, description="B1 used in stage 1")
    parametrization: Optional[int] = Field(default=None, description="ECM parametrization")
    curve_count: Optional[int] = Field(default=None, description="Number of curves to process")
    stage1_attempt_id: Optional[int] = Field(default=None, description="ID of the stage 1 attempt to supersede")
    download_url: Optional[str] = Field(default=None, description="URL to download residue file")
    suggested_b2: Optional[int] = Field(default=None, description="Recommended B2 for stage 2")
    expires_at: Optional[datetime] = Field(default=None, description="Work assignment expiration")
    message: Optional[str] = Field(default=None, description="Status message or reason for no work")


class ResidueCompleteRequest(BaseModel):
    """Request to mark residue as completed after stage 2."""
    stage2_attempt_id: int = Field(..., description="ID of the stage 2 ECM attempt that was submitted")


class ResidueCompleteResponse(BaseModel):
    """Response after completing stage 2 work."""
    residue_id: int = Field(..., description="ID of the completed residue")
    stage1_attempt_id: Optional[int] = Field(default=None, description="ID of the superseded stage 1 attempt")
    stage2_attempt_id: int = Field(..., description="ID of the stage 2 attempt")
    composite_id: int = Field(..., description="Database ID of the composite")
    new_t_level: Optional[float] = Field(default=None, description="Updated t-level after supersession")
    message: str = Field(..., description="Status message")


class ResidueInfoResponse(BaseModel):
    """Detailed information about a residue."""
    residue_id: int
    composite_id: int
    composite: str
    client_id: str
    stage1_attempt_id: Optional[int]
    b1: int
    parametrization: int
    curve_count: int
    file_size_bytes: int
    status: str
    created_at: datetime
    expires_at: Optional[datetime]  # Only set when claimed (claim timeout)
    claimed_at: Optional[datetime]
    claimed_by: Optional[str]
    completed_at: Optional[datetime]


class ResidueStatsResponse(BaseModel):
    """Statistics about residues in the system."""
    total_available: int = Field(..., description="Residues waiting for stage 2")
    total_claimed: int = Field(..., description="Residues currently being processed")
    total_completed: int = Field(..., description="Residues fully processed")
    total_expired: int = Field(..., description="Residues that expired without processing")
    total_curves_pending: int = Field(..., description="Sum of curves in available residues")
