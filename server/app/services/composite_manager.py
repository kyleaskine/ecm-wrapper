from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from typing import List, Dict, Any, Optional, Tuple, Union
import csv
import io
import re
import logging
from pathlib import Path

from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.work_assignments import WorkAssignment
from ..utils.number_utils import calculate_digit_length, validate_integer

logger = logging.getLogger(__name__)

class CompositeLoader:
    """Utility for loading composites from various sources."""

    @staticmethod
    def from_text_file(file_path: Union[str, Path]) -> List[str]:
        """
        Load composites from a text file.

        Args:
            file_path: Path to text file with one number per line

        Returns:
            List of valid composite numbers
        """
        numbers = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                # Extract number from line (handle various formats)
                number = CompositeLoader._extract_number(line)
                if number and validate_integer(number):
                    numbers.append(number)
                else:
                    logger.warning(f"Invalid number at line {line_num}: {line}")

        return numbers

    @staticmethod
    def from_csv_content(csv_content: str, number_column: str = 'number',
                        priority_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load composites from CSV content.

        Args:
            csv_content: CSV data as string
            number_column: Name of column containing numbers
            priority_column: Optional priority column name

        Returns:
            List of dictionaries with number and optional metadata
        """
        numbers = []
        reader = csv.DictReader(io.StringIO(csv_content))

        for row_num, row in enumerate(reader, 1):
            if number_column not in row:
                logger.error(f"Column '{number_column}' not found in CSV")
                break

            number = CompositeLoader._extract_number(row[number_column])
            if not number or not validate_integer(number):
                logger.warning(f"Invalid number at row {row_num}: {row[number_column]}")
                continue

            composite_data = {'number': number}

            # Add priority if specified
            if priority_column and priority_column in row:
                try:
                    composite_data['priority'] = int(row[priority_column])
                except ValueError:
                    composite_data['priority'] = 0

            # Add any other metadata
            for key, value in row.items():
                if key not in [number_column, priority_column]:
                    composite_data[key] = value

            numbers.append(composite_data)

        return numbers

    @staticmethod
    def from_number_list(numbers: List[str]) -> List[str]:
        """
        Validate a list of number strings.

        Args:
            numbers: List of number strings

        Returns:
            List of valid numbers
        """
        valid_numbers = []
        for i, number in enumerate(numbers):
            cleaned = CompositeLoader._extract_number(number)
            if cleaned and validate_integer(cleaned):
                valid_numbers.append(cleaned)
            else:
                logger.warning(f"Invalid number at index {i}: {number}")

        return valid_numbers

    @staticmethod
    def _extract_number(text: str) -> Optional[str]:
        """Extract a number from text, handling various formats."""
        if not text:
            return None

        # Remove whitespace
        text = text.strip()

        # Handle common prefixes/suffixes
        text = re.sub(r'^(composite|number|n)[:=\s]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(digits?|bits?)$', '', text, flags=re.IGNORECASE)

        # Extract just the digits
        match = re.search(r'\d+', text)
        if match:
            return match.group()

        return None


class CompositeManager:
    """Service for managing composites and work generation."""

    def __init__(self):
        self.loader = CompositeLoader()

    def bulk_load_composites(self, db: Session, data_source: Union[str, List[str], List[Dict[str, Any]]],
                           source_type: str = 'auto', default_priority: int = 0) -> Dict[str, Any]:
        """
        Bulk load composites from various sources.

        Args:
            db: Database session
            data_source: Data to load (file path, list of numbers, or list of dicts)
            source_type: 'file', 'csv', 'list', or 'auto'
            default_priority: Default priority for new composites

        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'total_processed': 0,
            'new_composites': 0,
            'existing_composites': 0,
            'invalid_numbers': 0,
            'errors': []
        }

        try:
            # Determine data type and load numbers
            if source_type == 'auto':
                source_type = self._detect_source_type(data_source)

            if source_type == 'file':
                numbers_data = [{'number': n} for n in self.loader.from_text_file(data_source)]
            elif source_type == 'csv':
                numbers_data = self.loader.from_csv_content(data_source)
            elif source_type == 'list':
                if isinstance(data_source[0], dict):
                    numbers_data = data_source
                else:
                    numbers_data = [{'number': n} for n in self.loader.from_number_list(data_source)]
            else:
                raise ValueError(f"Unknown source type: {source_type}")

            stats['total_processed'] = len(numbers_data)

            # Process each number
            for item in numbers_data:
                try:
                    number = item['number']
                    priority = item.get('priority', default_priority)

                    composite, created = self._get_or_create_composite(db, number)

                    if created:
                        stats['new_composites'] += 1
                        logger.info(f"Added new composite: {number} ({composite.digit_length} digits)")
                    else:
                        stats['existing_composites'] += 1

                except Exception as e:
                    stats['invalid_numbers'] += 1
                    stats['errors'].append(f"Error processing {item.get('number', 'unknown')}: {str(e)}")

        except Exception as e:
            stats['errors'].append(f"Fatal error during bulk load: {str(e)}")
            logger.error(f"Bulk load failed: {str(e)}")

        return stats

    def get_work_queue_status(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive status of the work queue."""

        # Composite statistics
        total_composites = db.query(Composite).count()
        unfactored_composites = db.query(Composite).filter(
            Composite.is_fully_factored == False
        ).count()

        # Work assignment statistics
        active_work = db.query(WorkAssignment).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).count()

        completed_work = db.query(WorkAssignment).filter(
            WorkAssignment.status == 'completed'
        ).count()

        # Composite size distribution
        size_distribution = db.query(
            func.floor(Composite.digit_length / 10) * 10,
            func.count()
        ).filter(
            Composite.is_fully_factored == False
        ).group_by(
            func.floor(Composite.digit_length / 10)
        ).all()

        size_dist_dict = {f"{int(size)}-{int(size)+9} digits": count
                         for size, count in size_distribution}

        return {
            'composites': {
                'total': total_composites,
                'unfactored': unfactored_composites,
                'factored': total_composites - unfactored_composites,
                'size_distribution': size_dist_dict
            },
            'work_assignments': {
                'active': active_work,
                'completed': completed_work,
            },
            'queue_health': {
                'available_work': max(0, unfactored_composites - active_work),
                'utilization': round((active_work / max(unfactored_composites, 1)) * 100, 1)
            }
        }

    def get_composite_details(self, db: Session, composite_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific composite."""
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return None

        # Get attempts
        attempts = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite_id
        ).order_by(ECMAttempt.created_at.desc()).all()

        # Get active work assignments
        active_work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.composite_id == composite_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
            )
        ).all()

        # Calculate progress statistics
        total_ecm_curves = sum(a.curves_completed for a in attempts if a.method == 'ecm')
        pm1_attempts = len([a for a in attempts if a.method == 'pm1'])

        factors_found = [a.factor_found for a in attempts if a.factor_found]

        return {
            'composite': {
                'id': composite.id,
                'number': composite.number,
                'digit_length': composite.digit_length,
                'target_t_level': composite.target_t_level,
                'current_t_level': composite.current_t_level,
                'priority': composite.priority,
                'is_prime': composite.is_prime,
                'is_fully_factored': composite.is_fully_factored,
                'created_at': composite.created_at,
                'updated_at': composite.updated_at
            },
            'progress': {
                'total_attempts': len(attempts),
                'total_ecm_curves': total_ecm_curves,
                'pm1_attempts': pm1_attempts,
                'factors_found': factors_found
            },
            'active_work': [
                {
                    'id': work.id,
                    'client_id': work.client_id,
                    'method': work.method,
                    'b1': work.b1,
                    'b2': work.b2,
                    'curves_requested': work.curves_requested,
                    'curves_completed': work.curves_completed,
                    'status': work.status,
                    'expires_at': work.expires_at
                }
                for work in active_work
            ],
            'recent_attempts': [
                {
                    'id': attempt.id,
                    'method': attempt.method,
                    'b1': attempt.b1,
                    'b2': attempt.b2,
                    'curves_completed': attempt.curves_completed,
                    'factor_found': attempt.factor_found,
                    'created_at': attempt.created_at,
                    'client_id': attempt.client_id
                }
                for attempt in attempts[:10]  # Last 10 attempts
            ]
        }

    def set_composite_priority(self, db: Session, composite_id: int, priority: int) -> bool:
        """Set priority for a composite (for future use in work assignment)."""
        # For now, we'll store this in a simple way
        # In a full implementation, you might add a priority column to Composite model
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return False

        # You could add a priority field to the Composite model or store in metadata
        logger.info(f"Priority {priority} set for composite {composite_id}")
        return True

    def mark_composite_complete(self, db: Session, composite_id: int,
                               reason: str = "manual") -> bool:
        """Mark a composite as fully factored."""
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return False

        composite.is_fully_factored = True

        # Cancel any active work assignments
        active_work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.composite_id == composite_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
            )
        ).all()

        for work in active_work:
            work.status = 'completed'  # Mark as completed rather than failed

        db.commit()

        logger.info(f"Marked composite {composite_id} as fully factored ({reason})")
        return True

    def _detect_source_type(self, data_source) -> str:
        """Detect the type of data source."""
        if isinstance(data_source, str):
            if '\n' in data_source or ',' in data_source:
                return 'csv'
            else:
                return 'file'
        elif isinstance(data_source, list):
            return 'list'
        else:
            raise ValueError("Unknown data source type")

    def _get_or_create_composite(self, db: Session, number: str) -> Tuple[Composite, bool]:
        """Get existing composite or create new one."""
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