from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Literal
import logging
from slowapi import Limiter

from ...rate_limit import get_real_client_ip
from ...database import get_db
from ...dependencies import get_composite_service, get_residue_manager
from ...schemas.submit import SubmitResultRequest, SubmitResultResponse
from ...models import Composite, ECMAttempt
from ...models.residues import ECMResidue
from ...services.composites import CompositeService
from ...services.factors import FactorService
from ...services.residue_manager import ResidueManager
from ...utils.number_utils import is_trivial_factor, verify_factor_divides, parse_sigma_with_parametrization
from ...utils.transactions import transaction_scope

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize rate limiter - 30 submissions per minute per IP
# Stage 2 consumers can submit rapidly when processing multiple residues
limiter = Limiter(key_func=get_real_client_ip)

@router.post("/submit_result", response_model=SubmitResultResponse)
@limiter.limit("30/minute")
def submit_result(
    result_request: SubmitResultRequest,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Submit factorization attempt result.

    This endpoint accepts results from distributed clients running ECM, P±1, and other
    factorization methods. It handles:
    - Validating composite exists in database (rejects unknown composites)
    - Recording the factorization attempt
    - Adding newly discovered factors
    - Idempotency for duplicate submissions
    - Factor validation

    Note: Only composites already registered in the database can receive submissions.
    Use --no-submit flag for local testing.
    """
    # Resolve real client IP (proxy-aware) up front so except blocks can log it
    client_ip = get_real_client_ip(request)

    with transaction_scope(db, "submit_result"):
        try:
            # SECURITY: Only accept submissions for composites already in the database
            # This prevents accidental pollution from local testing when users forget --no-submit
            composite = db.query(Composite).filter(
                Composite.current_composite == result_request.composite
            ).first()

            if not composite:
                logger.warning(
                    f"Submission rejected from {client_ip}: composite not in database "
                    f"({result_request.composite[:20]}...)"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Composite not found in database. Only registered composites can receive submissions. Use --no-submit for local testing."
                )

            # Get parametrization from explicit parameter or parse from sigma string
            explicit_parametrization = result_request.parameters.parametrization
            sigma = None
            parametrization = explicit_parametrization

            # Parse sigma string (format: "3:123456" or just "123456")
            if result_request.parameters.sigma:
                sigma, parsed_param = parse_sigma_with_parametrization(
                    result_request.parameters.sigma,
                    default_parametrization=3
                )
                # Use explicit parametrization if provided, otherwise use parsed value
                if explicit_parametrization is None:
                    parametrization = parsed_param

            # Final validation of parametrization
            if parametrization is not None and parametrization not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid parametrization {parametrization}. Must be 0, 1, 2, or 3.")

            # Generate work hash for duplicate detection
            # Convert sigma to int for hash generation (it can be a large number string)
            sigma_int = int(sigma) if sigma is not None else None
            work_hash = ECMAttempt.generate_work_hash(
                result_request.composite,
                result_request.method,
                result_request.parameters.b1,
                result_request.parameters.b2,
                parametrization,
                sigma_int,
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

            # Validate residue_checksum if provided (for stage 2 work from residue pool)
            # This verifies the client actually had the residue file
            residue_checksum = result_request.residue_checksum
            linked_residue = None
            if residue_checksum:
                linked_residue = db.query(ECMResidue).filter(
                    ECMResidue.checksum == residue_checksum,
                    ECMResidue.composite_id == composite.id
                ).first()
                if not linked_residue:
                    logger.warning(
                        f"Invalid residue_checksum {residue_checksum[:16]}... from {client_ip} "
                        f"for composite {composite.id} - not linking to residue"
                    )
                    # Don't fail submission - just don't link it (could be manual work)
                    residue_checksum = None

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
                client_ip=client_ip,
                residue_checksum=residue_checksum
            )

            db.add(attempt)
            db.flush()  # Get ID without committing transaction
            db.refresh(attempt)

            # Handle factor discovery - support both single and multiple factors
            factor_status: Literal["new_factor", "known_factor", "no_factor", "duplicate"] = "no_factor"
            factors_to_process = []

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
                # sigma is already a string (or None), no need to convert
                factors_to_process.append((result_request.results.factor_found, sigma))

            # Process all factors in batch
            if factors_to_process:
                new_factors_count = 0
                known_factors_count = 0

                # Validate and add all factors BEFORE updating composite
                # First pass: calculate running cofactor to identify final prime
                running_cofactor = result_request.composite
                factors_to_add = []  # Only factors that aren't the final prime

                from ...utils.number_utils import divide_factor

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

                    # Check if this factor divides the running cofactor
                    if verify_factor_divides(factor_str, running_cofactor):
                        # Calculate what the cofactor would be after dividing
                        new_cofactor = divide_factor(running_cofactor, factor_str)

                        # If dividing would result in 1, this is the final prime - don't add it
                        if new_cofactor == "1":
                            logger.info(
                                f"Skipping final prime factor {factor_str[:20]}{'...' if len(factor_str) > 20 else ''} "
                                f"- not adding to factors table"
                            )
                            # Mark that we found the final prime (will set is_complete=True later)
                            running_cofactor = factor_str  # The "cofactor" is now just this prime
                            continue

                        # Valid non-final factor - add to list and update running cofactor
                        factors_to_add.append((factor_str, factor_sigma))
                        running_cofactor = new_cofactor
                    else:
                        # Factor doesn't divide running cofactor (composite factor or already divided)
                        logger.warning(
                            f"Factor {factor_str[:20]}... doesn't divide running cofactor - skipping"
                        )

                # Second pass: add only the validated factors (excluding final prime)
                for factor_str, factor_sigma in factors_to_add:
                    # Parse sigma if it's a string (format: "3:12345")
                    parsed_sigma = None
                    if factor_sigma:
                        sigma_str = str(factor_sigma)
                        if ':' in sigma_str:
                            parsed_sigma = int(sigma_str.split(':', 1)[1])
                        else:
                            parsed_sigma = int(sigma_str)

                    # Add factor to database (with parametrization for group order calculation)
                    # Convert sigma to string for storage (supports large param 0 values)
                    sigma_for_storage = str(parsed_sigma) if parsed_sigma is not None else None
                    _, factor_created = FactorService.add_factor(
                        db, composite.id, factor_str, attempt.id, sigma_for_storage, parametrization,
                        method=result_request.method
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

                # Now update composite with the cofactor we calculated in the first pass
                if new_factors_count > 0 or known_factors_count > 0:
                    try:
                        composite_service.update_composite_to_cofactor(
                            db, composite.id, running_cofactor
                        )
                    except ValueError as e:
                        # Log but don't fail - the factors were still recorded
                        logger.warning(
                            f"Failed to update composite {composite.id} after factor division: {e}"
                        )

                    # Check if we now have complete factorization
                    if FactorService.verify_factorization(db, composite.id):
                        composite_service.mark_fully_factored(db, composite.id)

            # If this submission consumed a claimed residue, complete the residue
            # in the same transaction (supersedes stage 1 + any orphaned duplicate
            # attempts). This closes the window where the result was accepted but
            # the separate completion call never arrived. Old clients still call
            # /residues/{id}/complete afterwards; that's now an idempotent retry.
            residue_completed = False
            if linked_residue is not None:
                try:
                    # 'available' is completable too: a lapsed claim (released
                    # by expiry cleanup) shouldn't force the work to be redone,
                    # and the checksum match proves this client had the file.
                    # A residue claimed by a DIFFERENT client is left alone.
                    claim_ok = (
                        linked_residue.status == 'available'
                        or (linked_residue.status == 'claimed'
                            and linked_residue.claimed_by == result_request.client_id)
                    )
                    if (claim_ok
                            and residue_manager.completion_rejection_reason(linked_residue, attempt) is None):
                        residue_manager.complete_residue(
                            db, linked_residue.id, attempt.id, recalculate_t_level=False
                        )
                        residue_completed = True
                        logger.info(
                            f"Auto-completed residue {linked_residue.id} with attempt "
                            f"{attempt.id} for client {result_request.client_id}"
                        )
                    elif linked_residue.status == 'completed':
                        # Resubmission after a lost response: supersedes this
                        # duplicate attempt so the curves aren't counted twice
                        residue_manager.complete_residue(
                            db, linked_residue.id, attempt.id, recalculate_t_level=False
                        )
                        residue_completed = True
                except Exception as e:
                    # The submission is still valid even if completion fails;
                    # the client's separate completion call remains the fallback
                    logger.warning(f"Failed to auto-complete residue {linked_residue.id}: {e}")

            # Update t-level if this was an ECM attempt. Runs after residue
            # completion so the single recalculation excludes superseded attempts.
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
                factor_status=factor_status,
                residue_completed=residue_completed,
                new_t_level=composite.current_t_level if residue_completed else None
            )

        except HTTPException:
            # Re-raise HTTPExceptions (like validation errors) without modification
            raise
        except ValueError as e:
            # Client-side validation errors (safe to expose)
            logger.warning(f"Validation error from {client_ip}: {e}")
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