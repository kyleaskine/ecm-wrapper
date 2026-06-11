from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List

# Length caps below protect against unbounded payloads. Values are
# generous ceilings, not validation — they should never reject legitimate
# input from any client wrapper in this repo.
MAX_NUMBER_LEN = 10_000          # composites/factors: ECM range tops out far below this
MAX_RAW_OUTPUT_LEN = 1_048_576   # 1 MB; GMP-ECM/YAFU output is typically <100 KB
MAX_ID_LEN = 256                 # client_id, project, sigma
MAX_PROGRAM_LEN = 64             # "gmp-ecm", "yafu", version strings
MAX_CHECKSUM_LEN = 128           # SHA-256 hex is 64; cap allows prefixes


class ParametersSchema(BaseModel):
    b1: int = Field(..., description="Stage 1 bound")
    b2: Optional[int] = Field(default=None, description="Stage 2 bound (optional)")
    curves: Optional[int] = Field(default=None, description="Number of curves requested")
    parametrization: Optional[int] = Field(default=None, ge=0, le=3, description="ECM parametrization type (0, 1, 2, or 3)")
    sigma: Optional[str] = Field(default=None, max_length=MAX_ID_LEN, description="ECM curve parameter (can include parametrization like '3:12345' or just '12345')")
    a: Optional[int] = Field(default=None, description="PP1 base parameter")

class FactorWithSigma(BaseModel):
    factor: str = Field(..., max_length=MAX_NUMBER_LEN, description="The factor value")
    sigma: Optional[str] = Field(default=None, max_length=MAX_ID_LEN, description="Sigma value that found this factor (ECM only)")

class ResultsSchema(BaseModel):
    factor_found: Optional[str] = Field(default=None, max_length=MAX_NUMBER_LEN, description="Factor found (if any) - DEPRECATED: use factors_found for multiple factors")
    factors_found: Optional[List[FactorWithSigma]] = Field(default=None, description="List of factors found with their sigmas (preferred over factor_found)")
    curves_completed: int = Field(..., description="Actual curves completed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")

class SubmitResultRequest(BaseModel):
    composite: str = Field(..., max_length=MAX_NUMBER_LEN, description="The number being factored")
    project: Optional[str] = Field(default=None, max_length=MAX_ID_LEN, description="Project name (optional)")
    client_id: str = Field(..., max_length=MAX_ID_LEN, description="Client identifier")
    method: Literal["ecm", "pm1", "pp1", "qs", "nfs"] = Field(..., description="Factorization method")
    program: str = Field(..., max_length=MAX_PROGRAM_LEN, description="Program used (e.g., 'gmp-ecm', 'yafu')")
    program_version: Optional[str] = Field(default=None, max_length=MAX_PROGRAM_LEN, description="Program version")
    parameters: ParametersSchema
    results: ResultsSchema
    raw_output: Optional[str] = Field(default=None, description="Full program output (truncated server-side at MAX_RAW_OUTPUT_LEN)")
    residue_checksum: Optional[str] = Field(default=None, max_length=MAX_CHECKSUM_LEN, description="SHA-256 checksum of residue file (for stage 2 work from residue pool)")

    @field_validator("raw_output")
    @classmethod
    def truncate_raw_output(cls, v: Optional[str]) -> Optional[str]:
        # Truncate rather than reject — losing log tail is fine, losing the
        # factorization result because logs are big is not.
        if v is not None and len(v) > MAX_RAW_OUTPUT_LEN:
            return v[:MAX_RAW_OUTPUT_LEN] + "\n...[truncated by server]"
        return v

class ErrorDetail(BaseModel):
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(default=None, description="Field that caused error (if applicable)")

class SubmitResultResponse(BaseModel):
    status: Literal["success", "error"] = Field(..., description="Request status")
    attempt_id: Optional[int] = Field(default=None, description="Created attempt ID")
    composite_id: Optional[int] = Field(default=None, description="Composite ID")
    message: str = Field(..., description="Status message")
    factor_status: Optional[Literal["new_factor", "known_factor", "no_factor", "duplicate"]] = Field(
        default=None, description="Factor discovery status"
    )
    residue_completed: bool = Field(
        default=False,
        description="True if the linked residue was completed server-side in this call "
                    "(client may skip the separate /residues/{id}/complete call)"
    )
    new_t_level: Optional[float] = Field(
        default=None,
        description="Composite's current t-level after this submission (set when a residue was completed)"
    )
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed error information")