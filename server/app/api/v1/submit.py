from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Tuple

from ...database import get_db
from ...schemas.submit import SubmitResultRequest, SubmitResultResponse
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...services.factors import FactorService
from ...utils.number_utils import is_trivial_factor

router = APIRouter()

@router.post("/submit_result", response_model=SubmitResultResponse)
async def submit_result(
    request: SubmitResultRequest,
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
    """
    try:
        # Get or create composite
        composite, composite_created = CompositeService.get_or_create_composite(
            db, request.composite
        )
        
        # Generate work hash for duplicate detection
        work_hash = ECMAttempt.generate_work_hash(
            request.composite,
            request.method,
            request.parameters.b1,
            request.parameters.b2,
            request.parameters.sigma,
            request.parameters.curves
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
        
        # Create attempt record
        attempt = ECMAttempt(
            composite_id=composite.id,
            client_id=request.client_id,
            method=request.method,
            b1=request.parameters.b1,
            b2=request.parameters.b2,
            sigma=request.parameters.sigma,
            curves_requested=request.parameters.curves or 0,
            curves_completed=request.results.curves_completed,
            factor_found=request.results.factor_found,
            execution_time_seconds=request.results.execution_time,
            program=request.program,
            program_version=request.program_version,
            raw_output=request.raw_output,
            work_hash=work_hash
        )
        
        db.add(attempt)
        db.flush()  # Get ID without committing transaction
        db.refresh(attempt)
        
        # Handle factor discovery
        factor_status = "no_factor"
        if request.results.factor_found:
            factor_str = request.results.factor_found
            
            # Check if it's a trivial factor
            if is_trivial_factor(factor_str, request.composite):
                factor_status = "no_factor"  # Trivial factors don't count
            else:
                # Add factor to database
                factor, factor_created = FactorService.add_factor(
                    db, composite.id, factor_str, attempt.id
                )
                factor_status = "new_factor" if factor_created else "known_factor"
                
                # Check if we now have complete factorization
                if FactorService.verify_factorization(db, composite.id):
                    CompositeService.mark_fully_factored(db, composite.id)
        
        # Update t-level if this was an ECM attempt
        if request.method == 'ecm':
            try:
                CompositeService.update_t_level(db, composite.id)
            except Exception as e:
                # Log the error but don't fail the whole submission
                # The ECM result is still valid even if t-level update fails
                import logging
                logger = logging.getLogger(__name__)
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
        
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")