import csv
import io
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.work_assignments import WorkAssignment
from ..models.projects import Project, ProjectComposite
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
        with open(file_path, 'r', encoding='utf-8') as f:
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

            # Handle specific SNFS and composite fields with proper type conversion
            if 'current_composite' in row and row['current_composite']:
                composite_data['current_composite'] = (
                    CompositeLoader._extract_number(row['current_composite'])
                )

            if 'has_snfs_form' in row and row['has_snfs_form']:
                value = (
                    row['has_snfs_form'].lower()
                    if isinstance(row['has_snfs_form'], str)
                    else str(row['has_snfs_form'])
                )
                composite_data['has_snfs_form'] = value in ('true', '1', 'yes', 't', 'y')

            if 'snfs_difficulty' in row and row['snfs_difficulty']:
                try:
                    composite_data['snfs_difficulty'] = int(row['snfs_difficulty'])
                except ValueError:
                    logger.warning(
                        "Invalid snfs_difficulty at row %s: %s",
                        row_num, row['snfs_difficulty']
                    )

            # Add any other metadata
            excluded_keys = [
                number_column, priority_column, 'current_composite',
                'has_snfs_form', 'snfs_difficulty'
            ]
            for key, value in row.items():
                if key not in excluded_keys:
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

    def bulk_load_composites(
        self,
        db: Session,
        data_source: Union[str, List[str], List[Dict[str, Any]]],
        source_type: str = 'auto',
        default_priority: int = 0,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Bulk load composites from various sources.

        Args:
            db: Database session
            data_source: Data to load (file path, list of numbers, or list of dicts)
            source_type: 'file', 'csv', 'list', or 'auto'
            default_priority: Default priority for new composites
            project_name: Optional project name to associate composites with

        Returns:
            Dictionary with loading statistics
        """
        logger.info(
            "bulk_load_composites called: source_type=%s, project_name=%s",
            source_type, project_name
        )

        stats = {
            'total_processed': 0,
            'new_composites': 0,
            'existing_composites': 0,
            'updated_composites': 0,
            'invalid_numbers': 0,
            'errors': [],
            'project_created': False,
            'composites_added_to_project': 0,
            'composites_already_in_project': 0
        }

        # Get or create project if specified
        project = None
        if project_name:
            project, created = self.get_or_create_project(db, project_name)
            stats['project_created'] = created
            stats['project_name'] = project_name
            logger.info(
                "Project '%s' %s (ID: %s)",
                project_name,
                "created" if created else "already exists",
                project.id
            )

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
            logger.info("Processing %d numbers from source", len(numbers_data))

            # Process each number
            for item in numbers_data:
                try:
                    number = item['number']
                    current_composite = item.get('current_composite', None)
                    has_snfs_form = item.get('has_snfs_form', False)
                    snfs_difficulty = item.get('snfs_difficulty', None)
                    is_prime = item.get('is_prime', None)
                    is_fully_factored = item.get('is_fully_factored', None)
                    priority = item.get('priority', None)

                    composite, created, updated = self._get_or_create_composite(
                        db, number,
                        current_composite=current_composite,
                        has_snfs_form=has_snfs_form,
                        snfs_difficulty=snfs_difficulty,
                        is_prime=is_prime,
                        is_fully_factored=is_fully_factored,
                        priority=priority
                    )

                    if created:
                        stats['new_composites'] += 1
                        logger.info(
                            "Created new composite: %s (%s digits) - ID: %s",
                            number[:20] + "..." if len(number) > 20 else number,
                            composite.digit_length,
                            composite.id
                        )
                    else:
                        stats['existing_composites'] += 1
                        if updated:
                            stats['updated_composites'] += 1
                        logger.info(
                            "Found existing composite: %s - ID: %s%s",
                            number[:20] + "..." if len(number) > 20 else number,
                            composite.id,
                            " (updated)" if updated else ""
                        )

                    # Associate with project if specified
                    if project:
                        _, association_created = self.add_composite_to_project(
                            db, composite.id, project.id, default_priority
                        )
                        if association_created:
                            stats['composites_added_to_project'] += 1
                        else:
                            stats['composites_already_in_project'] += 1

                except Exception as e:
                    stats['invalid_numbers'] += 1
                    error_msg = f"Error processing {item.get('number', 'unknown')[:50]}...: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)

        except Exception as e:
            stats['errors'].append(f"Fatal error during bulk load: {str(e)}")
            logger.error(f"Bulk load failed: {str(e)}")

        return stats

    def get_work_queue_status(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive status of the work queue."""

        # Composite statistics
        total_composites = db.query(Composite).count()
        unfactored_composites = db.query(Composite).filter(
            not Composite.is_fully_factored
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
            not Composite.is_fully_factored
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

        # Deduplicate factors found (same factor may appear in multiple attempts)
        # Sort numerically for consistent display
        factors_found = sorted(set(a.factor_found for a in attempts if a.factor_found), key=lambda x: int(x))

        # Get factors with full details (including group order info)
        from ..models.factors import Factor
        factors_with_details = db.query(Factor).filter(
            Factor.composite_id == composite_id
        ).all()

        return {
            'composite': {
                'id': composite.id,
                'number': composite.number,
                'current_composite': composite.current_composite,
                'digit_length': composite.digit_length,
                'has_snfs_form': composite.has_snfs_form,
                'snfs_difficulty': composite.snfs_difficulty,
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
                    'parametrization': attempt.parametrization,
                    'curves_completed': attempt.curves_completed,
                    'factor_found': attempt.factor_found,
                    'created_at': attempt.created_at,
                    'client_id': attempt.client_id
                }
                for attempt in attempts[:10]  # Last 10 attempts
            ],
            'factors_with_group_orders': [
                {
                    'factor': f.factor,
                    'sigma': f.sigma,
                    'group_order': f.group_order,
                    'group_order_factorization': f.group_order_factorization,
                    'is_prime': f.is_prime
                }
                for f in factors_with_details if f.group_order is not None
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

    def get_or_create_project(
        self, db: Session, project_name: str, description: Optional[str] = None
    ) -> Tuple[Project, bool]:
        """
        Get existing project or create new one.

        Args:
            db: Database session
            project_name: Unique project name
            description: Optional project description

        Returns:
            Tuple of (Project, created_flag)
        """
        existing = db.query(Project).filter(Project.name == project_name).first()
        if existing:
            return existing, False

        project = Project(name=project_name, description=description)
        db.add(project)
        db.commit()
        db.refresh(project)

        logger.info("Created new project: %s", project_name)
        return project, True

    def add_composite_to_project(
        self,
        db: Session,
        composite_id: int,
        project_id: int,
        priority: int = 0
    ) -> Tuple[ProjectComposite, bool]:
        """
        Associate a composite with a project.

        Args:
            db: Database session
            composite_id: ID of the composite
            project_id: ID of the project
            priority: Priority within this project (0-10 scale)

        Returns:
            Tuple of (ProjectComposite, created_flag)
        """
        # Check if association already exists
        existing = db.query(ProjectComposite).filter(
            and_(
                ProjectComposite.composite_id == composite_id,
                ProjectComposite.project_id == project_id
            )
        ).first()

        if existing:
            return existing, False

        # Create new association
        project_composite = ProjectComposite(
            project_id=project_id,
            composite_id=composite_id,
            priority=priority
        )

        db.add(project_composite)
        db.commit()
        db.refresh(project_composite)

        return project_composite, True

    def _get_or_create_composite(self, db: Session, number: str,
                                  current_composite: Optional[str] = None,
                                  has_snfs_form: bool = False,
                                  snfs_difficulty: Optional[int] = None,
                                  is_prime: Optional[bool] = None,
                                  is_fully_factored: Optional[bool] = None,
                                  priority: Optional[int] = None) -> Tuple[Composite, bool, bool]:
        """
        Get existing composite or create new one. Updates fields if composite exists.

        Returns:
            Tuple of (Composite, created, updated)
        """
        # Check if composite already exists (by number, which can be a formula)
        existing = db.query(Composite).filter(Composite.number == number).first()
        if existing:
            # Update existing composite with new metadata
            updated = False

            if current_composite is not None and existing.current_composite != current_composite:
                # Validate current_composite is an integer
                if not validate_integer(current_composite):
                    raise ValueError(f"Invalid current_composite format: {current_composite}")
                existing.current_composite = current_composite
                existing.digit_length = calculate_digit_length(current_composite)
                updated = True

            if existing.has_snfs_form != has_snfs_form:
                existing.has_snfs_form = has_snfs_form
                updated = True

            if snfs_difficulty is not None and existing.snfs_difficulty != snfs_difficulty:
                existing.snfs_difficulty = snfs_difficulty
                updated = True

            if is_prime is not None and existing.is_prime != is_prime:
                existing.is_prime = is_prime
                updated = True

            if is_fully_factored is not None and existing.is_fully_factored != is_fully_factored:
                existing.is_fully_factored = is_fully_factored
                updated = True

            if priority is not None and existing.priority != priority:
                existing.priority = priority
                updated = True

            if updated:
                db.commit()
                db.refresh(existing)
                logger.info(
                    "Updated composite: %s - ID: %s",
                    number[:20] + "..." if len(number) > 20 else number,
                    existing.id
                )

            return existing, False, updated

        # Create new composite
        # If current_composite is not provided, default to the original number
        if current_composite is None:
            current_composite = number

        # Validate current_composite is an integer (not the number field!)
        if not validate_integer(current_composite):
            raise ValueError(f"Invalid current_composite format: {current_composite}")

        digit_length = calculate_digit_length(current_composite)

        composite = Composite(
            number=number,
            current_composite=current_composite,
            digit_length=digit_length,
            has_snfs_form=has_snfs_form,
            snfs_difficulty=snfs_difficulty,
            is_prime=is_prime if is_prime is not None else False,
            is_fully_factored=is_fully_factored if is_fully_factored is not None else False,
            priority=priority if priority is not None else 0
        )

        db.add(composite)
        db.commit()
        db.refresh(composite)

        return composite, True, False
