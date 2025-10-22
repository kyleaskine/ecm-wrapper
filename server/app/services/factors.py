from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from ..models.factors import Factor
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..utils.number_utils import validate_integer, verify_complete_factorization
from .group_order import GroupOrderCalculator
import logging

logger = logging.getLogger(__name__)

class FactorService:
    @staticmethod
    def add_factor(db: Session, composite_id: int, factor: str, attempt_id: Optional[int] = None,
                   sigma: Optional[int] = None, parametrization: Optional[int] = None) -> Tuple[Factor, bool]:
        """
        Add a factor to a composite.
        Returns (factor, created) where created is True if new factor was added.

        Args:
            sigma: The sigma value that found this factor (ECM only)
            parametrization: ECM parametrization type (0-3) for group order calculation
        """
        # Validate factor
        if not validate_integer(factor):
            raise ValueError(f"Invalid factor format: {factor}")

        # Check if factor already exists for this composite
        existing = db.query(Factor).filter(
            Factor.composite_id == composite_id,
            Factor.factor == factor
        ).first()

        if existing:
            return existing, False

        # Calculate group order if we have sigma (ECM factor)
        group_order = None
        group_order_factorization = None

        if sigma is not None:
            # Get parametrization from attempt if not provided
            if parametrization is None and attempt_id is not None:
                attempt = db.query(ECMAttempt).filter(ECMAttempt.id == attempt_id).first()
                if attempt and attempt.parametrization is not None:
                    parametrization = attempt.parametrization

            # Default to parametrization 3 if still not set
            if parametrization is None:
                parametrization = 3

            try:
                calculator = GroupOrderCalculator()
                result = calculator.calculate_group_order(factor, str(sigma), parametrization)
                if result:
                    group_order, group_order_factorization = result
                    logger.info(f"Calculated group order for factor {factor[:20]}...: {group_order}")
            except Exception as e:
                logger.warning(f"Failed to calculate group order for factor {factor[:20]}...: {e}")

        # Add new factor
        new_factor = Factor(
            composite_id=composite_id,
            factor=factor,
            found_by_attempt_id=attempt_id,
            sigma=sigma,
            group_order=group_order,
            group_order_factorization=group_order_factorization
        )

        db.add(new_factor)
        db.flush()  # Make visible within transaction
        db.refresh(new_factor)

        return new_factor, True

    @staticmethod
    def get_factors_for_composite(db: Session, composite_id: int) -> List[Factor]:
        """Get all factors for a composite."""
        return db.query(Factor).filter(Factor.composite_id == composite_id).all()

    @staticmethod
    def verify_factorization(db: Session, composite_id: int) -> bool:
        """
        Check if we have a complete factorization of the composite.
        Returns True if the product of all factors equals the original composite.
        """
        try:
            composite = db.query(Composite).filter(Composite.id == composite_id).first()
            if not composite:
                return False

            factors = FactorService.get_factors_for_composite(db, composite_id)
            if not factors:
                return False

            factor_strings = [factor.factor for factor in factors]
            return verify_complete_factorization(composite.number, factor_strings)

        except Exception as e:
            # Log error but don't raise - verification failure should be handled gracefully
            return False