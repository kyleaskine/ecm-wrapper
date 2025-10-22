from sqlalchemy.orm import Session
from typing import Optional, Tuple
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..utils.number_utils import calculate_digit_length, validate_integer
from .t_level_calculator import TLevelCalculator

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

        # Check if composite already exists - match by either number or current_composite
        # This handles both:
        # 1. number = "2^1223-1", current_composite = "123456..." (uploaded via admin)
        # 2. number = "123456...", current_composite = "123456..." (submitted by client)
        existing = db.query(Composite).filter(
            (Composite.number == number) | (Composite.current_composite == number)
        ).first()
        if existing:
            return existing, False

        # Create new composite
        digit_length = calculate_digit_length(number)

        composite = Composite(
            number=number,
            current_composite=number,  # Initially same as number until factors are found
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
    def find_composite_by_identifier(db: Session, identifier: str) -> Optional[Composite]:
        """
        Find composite by ID, number (formula), or current_composite.

        Args:
            db: Database session
            identifier: Can be:
                - Numeric ID (e.g., "123")
                - Formula/number (e.g., "2^1223-1")
                - Current composite value (e.g., "179769313...")

        Returns:
            Composite if found, None otherwise
        """
        # Try to parse as ID first
        try:
            composite_id = int(identifier)
            composite = db.query(Composite).filter(Composite.id == composite_id).first()
            if composite:
                return composite
        except ValueError:
            # Not a valid integer, continue to string search
            pass

        # Search by number (formula) or current_composite
        composite = db.query(Composite).filter(
            (Composite.number == identifier) | (Composite.current_composite == identifier)
        ).first()

        return composite

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

    @staticmethod
    def update_t_level(db: Session, composite_id: int) -> bool:
        """
        Recalculate and update the current t-level and target t-level for a composite.

        This recalculates both:
        - Current t-level: Based on all ECM attempts completed
        - Target t-level: Based on current composite length and SNFS difficulty

        Returns True if composite was found and updated, False otherwise.
        """
        try:
            # Get the composite
            composite = db.query(Composite).filter(Composite.id == composite_id).first()
            if not composite:
                return False

            # Get all ECM attempts for this composite
            ecm_attempts = db.query(ECMAttempt).filter(
                ECMAttempt.composite_id == composite_id,
                ECMAttempt.method == 'ecm'
            ).all()

            # Calculate current t-level using the t-level calculator
            calculator = TLevelCalculator()
            current_t_level = calculator.get_current_t_level_from_attempts(ecm_attempts)

            # Recalculate target t-level based on current composite length and SNFS data
            # This is important when factors are found and current_composite is updated
            target_t_level = calculator.calculate_target_t_level(
                digit_length=composite.digit_length,
                snfs_difficulty=composite.snfs_difficulty
            )

            # Update the composite
            composite.current_t_level = current_t_level
            composite.target_t_level = target_t_level
            db.commit()

            return True

        except Exception as e:
            db.rollback()
            raise ValueError(f"Failed to update t-level for composite {composite_id}: {str(e)}")