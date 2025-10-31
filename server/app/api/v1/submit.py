from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Tuple
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...database import get_db
from ...dependencies import get_composite_service
from ...schemas.submit import SubmitResultRequest, SubmitResultResponse
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...services.factors import FactorService
from ...utils.number_utils import is_trivial_factor, verify_factor_divides
from ...utils.transactions import transaction_scope

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize rate limiter - 10 submissions per minute per IP
limiter = Limiter(key_func=get_remote_address)

@router.post("/submit_result", response_model=SubmitResultResponse)
@limiter.limit("10/minute")
async def submit_result(
    result_request: SubmitResultRequest,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
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
    with transaction_scope(db, "submit_result"):
        try:
            # Get client IP address for logging
            client_ip = request.client.host if request.client else "unknown"

            # Get or create composite
            composite, composite_created, _ = composite_service.get_or_create_composite(
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

            # Handle factor discovery - support both single and multiple factors
            factor_status = "no_factor"
            factors_to_process = []

            # Debug: Log what we received
            logger.info(f"DEBUG: result_request.results.factors_found = {result_request.results.factors_found}")
            logger.info(f"DEBUG: result_request.results.factor_found = {result_request.results.factor_found}")

            # Collect factors from new or legacy format
            if result_request.results.factors_found:
                # New format: multiple factors with sigmas
                logger.info(f"Received {len(result_request.results.factors_found)} factors in batch submission")
                for factor_with_sigma in result_request.results.factors_found:
                    factors_to_process.append((factor_with_sigma.factor, factor_with_sigma.sigma))
                logger.info(f"Processing {len(factors_to_process)} factors: {[f[:20] + '...' for f, _ in factors_to_process]}")
            elif result_request.results.factor_found:
                # Legacy format: single factor with sigma from parameters
                logger.info(f"Received single factor (legacy format): {result_request.results.factor_found[:20]}...")
                factors_to_process.append((result_request.results.factor_found, sigma))

            # Process all factors in batch
            if factors_to_process:
                new_factors_count = 0
                known_factors_count = 0
                all_factors_valid = True

                # Validate and add all factors BEFORE updating composite
                for factor_str, factor_sigma in factors_to_process:
                    # Check if it's a trivial factor
                    if is_trivial_factor(factor_str, result_request.composite):
                        continue  # Skip trivial factors

                    # SECURITY: Verify the factor actually divides the composite
                    if not verify_factor_divides(factor_str, result_request.composite):
                        logger.warning(
                            f"Invalid factor submitted by client {result_request.client_id} "
                            f"from IP {client_ip}: factor {factor_str[:20]}... does not divide "
                            f"composite {result_request.composite[:20]}..."
                        )
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid factor: {factor_str} does not divide the composite"
                        )

                    # Parse sigma if it's a string (format: "3:12345")
                    parsed_sigma = None
                    if factor_sigma:
                        sigma_str = str(factor_sigma)
                        if ':' in sigma_str:
                            parsed_sigma = int(sigma_str.split(':', 1)[1])
                        else:
                            parsed_sigma = int(sigma_str)

                    # Add factor to database (with parametrization for group order calculation)
                    factor, factor_created = FactorService.add_factor(
                        db, composite.id, factor_str, attempt.id, parsed_sigma, parametrization
                    )

                    if factor_created:
                        new_factors_count += 1
                        logger.info(f"  ✓ Added new factor {new_factors_count}: {factor_str[:20]}...")
                    else:
                        known_factors_count += 1
                        logger.info(f"  ○ Factor already known: {factor_str[:20]}...")

                # Set factor status based on what was found
                if new_factors_count > 0:
                    factor_status = "new_factor"
                elif known_factors_count > 0:
                    factor_status = "known_factor"

                # Now update composite by dividing out all factors from the ORIGINAL composite
                if new_factors_count > 0 or known_factors_count > 0:
                    try:
                        # Calculate the product of all factors to divide out
                        from ...utils.number_utils import divide_factor

                        current_cofactor = result_request.composite
                        for factor_str, _ in factors_to_process:
                            if not is_trivial_factor(factor_str, result_request.composite):
                                # Check if this factor divides the current cofactor
                                if verify_factor_divides(factor_str, current_cofactor):
                                    # Check if dividing would result in 1 (final prime)
                                    new_cofactor = divide_factor(current_cofactor, factor_str)
                                    if new_cofactor == "1":
                                        # This is the final prime - don't divide it out
                                        logger.info(
                                            f"Rejecting final prime factor {factor_str[:20]}{'...' if len(factor_str) > 20 else ''} "
                                            f"- leaving as composite and marking as prime"
                                        )
                                        # We'll mark as prime after the loop
                                        continue

                                    # Divide from the running cofactor (starts as original composite)
                                    current_cofactor = new_cofactor
                                    logger.info(
                                        f"Divided out factor {factor_str[:20]}{'...' if len(factor_str) > 20 else ''}, "
                                        f"cofactor now has {len(current_cofactor)} digits"
                                    )
                                else:
                                    # Factor doesn't divide current cofactor (likely composite or duplicate)
                                    logger.warning(
                                        f"Skipping factor {factor_str[:20]}... - doesn't divide current cofactor "
                                        f"(likely composite factor or already divided out)"
                                    )

                        # Now update the composite with the final cofactor
                        from ...utils.number_utils import is_probably_prime, calculate_digit_length
                        from ...utils.calculations import ECMCalculations

                        composite.current_composite = current_cofactor
                        composite.digit_length = calculate_digit_length(current_cofactor)

                        # Test if cofactor is prime
                        if is_probably_prime(current_cofactor):
                            logger.info(f"Cofactor is prime - marking composite {composite.id} as fully factored")
                            composite.is_prime = True
                            composite.is_fully_factored = True
                        else:
                            # Cofactor is still composite - recalculate target t-level for new size
                            new_target_t_level = ECMCalculations.recommend_target_t_level(composite.digit_length)
                            logger.info(
                                f"Cofactor is composite ({composite.digit_length} digits) - "
                                f"updating target t-level to {new_target_t_level}"
                            )
                            composite.target_t_level = new_target_t_level

                            # Also update current t-level based on existing work
                            composite_service.update_t_level(db, composite.id)

                        db.flush()  # Make changes visible

                    except ValueError as e:
                        # Log but don't fail - the factors were still recorded
                        logger.warning(
                            f"Failed to update composite {composite.id} after factor division: {e}"
                        )

                    # Check if we now have complete factorization
                    if FactorService.verify_factorization(db, composite.id):
                        composite_service.mark_fully_factored(db, composite.id)

            # Update t-level if this was an ECM attempt
            if result_request.method == 'ecm':
                try:
                    composite_service.update_t_level(db, composite.id)
                except Exception as e:
                    # Log the error but don't fail the whole submission
                    # The ECM result is still valid even if t-level update fails
                    logger.warning(f"Failed to update t-level for composite {composite.id}: {str(e)}")

            return SubmitResultResponse(
                status="success",
                attempt_id=attempt.id,
                composite_id=composite.id,
                message="Result logged successfully",
                factor_status=factor_status
            )

        except HTTPException:
            # Re-raise HTTPExceptions (like validation errors) without modification
            raise
        except ValueError as e:
            # Client-side validation errors (safe to expose)
            logger.warning(f"Validation error from {client_ip if 'client_ip' in locals() else 'unknown'}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except (TypeError, AttributeError) as e:
            # Data structure errors (potentially from malformed requests)
            logger.error(f"Data structure error: {e}")
            raise HTTPException(status_code=400, detail="Invalid request format")
        except Exception as e:
            # Unexpected errors - log with full details but return generic message to client
            logger.exception(f"Unexpected error in submit_result: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error occurred while processing submission"
            )