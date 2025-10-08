from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime

class ParametersSchema(BaseModel):
    b1: int = Field(..., description="Stage 1 bound")
    b2: Optional[int] = Field(None, description="Stage 2 bound (optional)")
    curves: Optional[int] = Field(None, description="Number of curves requested")
    parametrization: Optional[int] = Field(None, ge=0, le=3, description="ECM parametrization type (0, 1, 2, or 3)")
    sigma: Optional[int] = Field(None, description="ECM curve parameter (can include parametrization like '3:12345')")
    a: Optional[int] = Field(None, description="PP1 base parameter")

class ResultsSchema(BaseModel):
    factor_found: Optional[str] = Field(None, description="Factor found (if any)")
    curves_completed: int = Field(..., description="Actual curves completed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")

class SubmitResultRequest(BaseModel):
    composite: str = Field(..., description="The number being factored")
    project: Optional[str] = Field(None, description="Project name (optional)")
    client_id: str = Field(..., description="Client identifier")
    method: Literal["ecm", "pm1", "pp1", "qs", "nfs"] = Field(..., description="Factorization method")
    program: str = Field(..., description="Program used (e.g., 'gmp-ecm', 'yafu')")
    program_version: Optional[str] = Field(None, description="Program version")
    parameters: ParametersSchema
    results: ResultsSchema
    raw_output: Optional[str] = Field(None, description="Full program output")

class ErrorDetail(BaseModel):
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused error (if applicable)")

class SubmitResultResponse(BaseModel):
    status: Literal["success", "error"] = Field(..., description="Request status")
    attempt_id: Optional[int] = Field(None, description="Created attempt ID")
    composite_id: Optional[int] = Field(None, description="Composite ID")
    message: str = Field(..., description="Status message")
    factor_status: Optional[Literal["new_factor", "known_factor", "no_factor", "duplicate"]] = Field(
        None, description="Factor discovery status"
    )
    errors: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")