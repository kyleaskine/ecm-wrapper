from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

class FactorResponse(BaseModel):
    """Schema for factor information"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    composite_id: int
    factor: str = Field(..., description="The factor value")
    is_prime: Optional[bool] = Field(None, description="Whether factor is prime (if tested)")
    found_by_attempt_id: Optional[int] = Field(None, description="ECM attempt that found this factor")
    sigma: Optional[int] = Field(None, description="Sigma value that found this factor (ECM only)")
    created_at: datetime = Field(..., description="When factor was discovered")

class FactorWithComposite(BaseModel):
    """Schema for factor with composite information"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    composite_id: int
    number: str = Field(..., description="Original number or mathematical form (e.g., '2^1223-1')")
    composite_number: str = Field(..., description="The composite number that was factored")
    factor: str = Field(..., description="The factor value")
    is_prime: Optional[bool] = Field(None, description="Whether factor is prime (if tested)")
    found_by_attempt_id: Optional[int] = Field(None, description="ECM attempt that found this factor")
    sigma: Optional[int] = Field(None, description="Sigma value that found this factor (ECM only)")
    created_at: datetime = Field(..., description="When factor was discovered")
    client_id: Optional[str] = Field(None, description="Client that found the factor")
    method: Optional[str] = Field(None, description="Method used to find factor (ecm, pm1, etc)")

class FactorsListResponse(BaseModel):
    """Schema for paginated factors list"""
    factors: List[FactorWithComposite]
    total: int = Field(..., description="Total number of factors")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of factors per page")
