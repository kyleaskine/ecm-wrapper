from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from ..models.factors import Factor
from ..models.composites import Composite
from ..utils.number_utils import validate_integer, verify_complete_factorization

class FactorService:
    @staticmethod
    def add_factor(db: Session, composite_id: int, factor: str, attempt_id: Optional[int] = None) -> Tuple[Factor, bool]:
        """
        Add a factor to a composite.
        Returns (factor, created) where created is True if new factor was added.
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
        
        # Add new factor
        new_factor = Factor(
            composite_id=composite_id,
            factor=factor,
            found_by_attempt_id=attempt_id
        )
        
        db.add(new_factor)
        db.commit()
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