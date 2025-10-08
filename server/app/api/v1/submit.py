from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Tuple
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...database import get_db
from ...schemas.submit import SubmitResultRequest, SubmitResultResponse
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...services.factors import FactorService
from ...utils.number_utils import is_trivial_factor, verify_factor_divides

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize rate limiter - 10 submissions per minute per IP
limiter = Limiter(key_func=get_remote_address)

@router.post("/submit_result", response_model=SubmitResultResponse)
@limiter.limit("10/minute")
async def submit_result(
    result_request: SubmitResultRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Submit factorization attempt result.

    This endpoint accepts results from distributed clients running ECM, P±1, and other
    factorization methods. It handles:
    - Creating/updating composite records
    - Recording the factorization attempt
    - Adding newly discovered factors
    - Idempotency for duplicate submissions
    - Factor validation
    """
    try:
        # Get client IP address for logging
        client_ip = request.client.host if request.client else "unknown"

        # Get or create composite
        composite, composite_created = CompositeService.get_or_create_composite(
            db, result_request.composite
        )

        # Get parametrization from explicit parameter or parse from sigma string
        parametrization = result_request.parameters.parametrization
        sigma = None

        # Parse sigma string (format: "3:123456" or just "123456")
        if result_request.parameters.sigma:
            sigma_str = str(result_request.parameters.sigma)
            if ':' in sigma_str:
                parts = sigma_str.split(':', 1)
                # If parametrization not explicitly provided, extract from sigma
                if parametrization is None:
                    parametrization = int(parts[0])
                    # Validate parametrization from sigma string
                    if parametrization not in [0, 1, 2, 3]:
                        raise ValueError(f"Invalid parametrization {parametrization} in sigma string. Must be 0, 1, 2, or 3.")
                sigma = int(parts[1])
            else:
                # Plain sigma value
                sigma = int(sigma_str)
                # Default to param 3 if not explicitly provided
                if parametrization is None:
                    parametrization = 3

        # Final validation of parametrization
        if parametrization is not None and parametrization not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid parametrization {parametrization}. Must be 0, 1, 2, or 3.")

        # Generate work hash for duplicate detection
        work_hash = ECMAttempt.generate_work_hash(
            result_request.composite,
            result_request.method,
            result_request.parameters.b1,
            result_request.parameters.b2,
            parametrization,
            sigma,
            result_request.parameters.curves
        )
        
        # Check for existing work
        existing_attempt = db.query(ECMAttempt).filter(ECMAttempt.work_hash == work_hash).first()
        if existing_attempt:
            return SubmitResultResponse(
                status="success",
                attempt_id=existing_attempt.id,
                composite_id=composite.id,
                message="Duplicate work detected - using existing attempt",
                factor_status="duplicate"
            )
        
        # Create attempt record with IP logging
        attempt = ECMAttempt(
            composite_id=composite.id,
            client_id=result_request.client_id,
            method=result_request.method,
            b1=result_request.parameters.b1,
            b2=result_request.parameters.b2,
            parametrization=parametrization,
            curves_requested=result_request.parameters.curves or 0,
            curves_completed=result_request.results.curves_completed,
            factor_found=result_request.results.factor_found,
            execution_time_seconds=result_request.results.execution_time,
            program=result_request.program,
            program_version=result_request.program_version,
            raw_output=result_request.raw_output,
            work_hash=work_hash,
            client_ip=client_ip
        )
        
        db.add(attempt)
        db.flush()  # Get ID without committing transaction
        db.refresh(attempt)
        
        # Handle factor discovery
        factor_status = "no_factor"
        if result_request.results.factor_found:
            factor_str = result_request.results.factor_found

            # Check if it's a trivial factor
            if is_trivial_factor(factor_str, result_request.composite):
                factor_status = "no_factor"  # Trivial factors don't count
            else:
                # SECURITY: Verify the factor actually divides the composite
                if not verify_factor_divides(factor_str, result_request.composite):
                    logger.warning(
                        f"Invalid factor submitted by client {result_request.client_id} "
                        f"from IP {client_ip}: factor {factor_str[:20]}... does not divide "
                        f"composite {result_request.composite[:20]}..."
                    )
                    db.rollback()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid factor: {factor_str} does not divide the composite"
                    )

                # Add factor to database
                factor, factor_created = FactorService.add_factor(
                    db, composite.id, factor_str, attempt.id, sigma
                )
                factor_status = "new_factor" if factor_created else "known_factor"

                # Check if we now have complete factorization
                if FactorService.verify_factorization(db, composite.id):
                    CompositeService.mark_fully_factored(db, composite.id)
        
        # Update t-level if this was an ECM attempt
        if result_request.method == 'ecm':
            try:
                CompositeService.update_t_level(db, composite.id)
            except Exception as e:
                # Log the error but don't fail the whole submission
                # The ECM result is still valid even if t-level update fails
                logger.warning(f"Failed to update t-level for composite {composite.id}: {str(e)}")

        # Commit the entire transaction at the end
        db.commit()

        return SubmitResultResponse(
            status="success",
            attempt_id=attempt.id,
            composite_id=composite.id,
            message="Result logged successfully",
            factor_status=factor_status
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions (like validation errors) without modification
        db.rollback()
        raise
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")