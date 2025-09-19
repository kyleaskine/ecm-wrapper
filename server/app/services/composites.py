from sqlalchemy.orm import Session
from typing import Optional, Tuple
from ..models.composites import Composite
from ..utils.number_utils import calculate_digit_length, validate_integer

class CompositeService:
    @staticmethod
    def get_or_create_composite(db: Session, number: str) -> Tuple[Composite, bool]:
        """
        Get existing composite or create new one.
        Returns (composite, created) where created is True if new composite was created.
        """
        # Validate the number string
        if not validate_integer(number):
            raise ValueError(f"Invalid number format: {number}")
        
        # Check if composite already exists
        existing = db.query(Composite).filter(Composite.number == number).first()
        if existing:
            return existing, False
        
        # Create new composite
        digit_length = calculate_digit_length(number)

        composite = Composite(
            number=number,
            digit_length=digit_length
        )
        
        db.add(composite)
        db.commit()
        db.refresh(composite)
        
        return composite, True
    
    @staticmethod
    def get_composite_by_number(db: Session, number: str) -> Optional[Composite]:
        """Get composite by number string."""
        return db.query(Composite).filter(Composite.number == number).first()
    
    @staticmethod
    def mark_fully_factored(db: Session, composite_id: int) -> bool:
        """
        Mark composite as fully factored.
        Returns True if composite was found and updated, False otherwise.
        """
        try:
            rows_updated = db.query(Composite)\
                .filter(Composite.id == composite_id)\
                .update({"is_fully_factored": True})
            db.commit()
            return rows_updated > 0
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to mark composite {composite_id} as fully factored: {str(e)}")
    
    @staticmethod
    def mark_prime(db: Session, composite_id: int) -> bool:
        """
        Mark composite as prime.
        Returns True if composite was found and updated, False otherwise.
        """
        try:
            rows_updated = db.query(Composite)\
                .filter(Composite.id == composite_id)\
                .update({"is_prime": True})
            db.commit()
            return rows_updated > 0
        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to mark composite {composite_id} as prime: {str(e)}")