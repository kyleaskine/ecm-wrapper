"""
Unified composite service that consolidates CompositeService and CompositeManager.

This service provides comprehensive composite management including:
- CRUD operations for composites
- Bulk loading from various sources
- Project associations
- Work queue management
- T-level calculations
"""

import logging
from typing import Optional, Tuple, Dict, Any, List, Union, cast

from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_

from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..models.work_assignments import WorkAssignment
from ..models.factors import Factor
from ..models.projects import Project, ProjectComposite
from ..utils.number_utils import calculate_digit_length, validate_integer
from .t_level_calculator import TLevelCalculator
from .composite_loader import CompositeLoader

logger = logging.getLogger(__name__)


class CompositeService:
    """
    Unified service for managing all composite operations.

    Replaces both the old CompositeService (static methods) and CompositeManager.
    """

    def __init__(self):
        """Initialize the service with dependencies."""
        self.loader = CompositeLoader()
        self.t_level_calculator = TLevelCalculator()

    # ==================== CRUD Operations ====================

    def get_or_create_composite(
        self,
        db: Session,
        number: str,
        current_composite: Optional[str] = None,
        has_snfs_form: Optional[bool] = None,
        snfs_difficulty: Optional[int] = None,
        is_prime: Optional[bool] = None,
        is_fully_factored: Optional[bool] = None,
        priority: Optional[int] = None
    ) -> Tuple[Composite, bool, bool]:
        """
        Get existing composite or create new one with full metadata support.

        This is the enhanced version that supports updating existing composites
        with new metadata (from bulk uploads, etc.).

        Args:
            db: Database session
            number: The composite number (can be formula like "2^1223-1" or actual number)
            current_composite: Current composite value after factorization
            has_snfs_form: Whether number has SNFS form
            snfs_difficulty: SNFS difficulty rating
            is_prime: Whether number is prime
            is_fully_factored: Whether number is fully factored
            priority: Priority level

        Returns:
            Tuple of (Composite, created, updated)
            - created: True if new composite was created
            - updated: True if existing composite was updated

        Raises:
            ValueError: If number format is invalid
        """
        # Check if composite already exists FIRST
        # Match by number OR current_composite to handle both upload scenarios
        existing = db.query(Composite).filter(
            (Composite.number == number) | (Composite.current_composite == number)
        ).first()

        if existing:
            # Update existing composite with new metadata
            updated = False
            t_level_needs_update = False

            if current_composite is not None and existing.current_composite != current_composite:
                if not validate_integer(current_composite):
                    raise ValueError(f"Invalid current_composite format: {current_composite}")
                existing.current_composite = current_composite
                existing.digit_length = calculate_digit_length(current_composite)
                updated = True
                t_level_needs_update = True  # digit_length affects target t-level

            if has_snfs_form is not None and existing.has_snfs_form != has_snfs_form:
                existing.has_snfs_form = has_snfs_form
                updated = True

            if snfs_difficulty is not None and existing.snfs_difficulty != snfs_difficulty:
                existing.snfs_difficulty = snfs_difficulty
                updated = True
                t_level_needs_update = True  # snfs_difficulty affects target t-level

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
                db.flush()  # Make changes visible within transaction
                db.refresh(existing)
                logger.info(
                    "Updated composite: %s - ID: %s",
                    number[:20] + "..." if len(number) > 20 else number,
                    existing.id
                )

                # Recalculate t-level if metadata that affects it changed
                if t_level_needs_update:
                    try:
                        self.update_t_level(db, existing.id)
                        logger.info(
                            "Recalculated t-level for composite %s (target: %.1f)",
                            existing.id,
                            existing.target_t_level
                        )
                    except Exception as e:
                        # Log but don't fail the update
                        logger.warning(
                            "Failed to update t-level for composite %s: %s",
                            existing.id, str(e)
                        )

            return existing, False, updated

        # Create new composite
        # If no current_composite provided, use number field (for backwards compatibility)
        if current_composite is None:
            current_composite = number

        # Validate that current_composite is a valid integer
        if not validate_integer(current_composite):
            raise ValueError(f"Invalid current_composite format: {current_composite}")

        digit_length = calculate_digit_length(current_composite)

        composite = Composite(
            number=number,
            current_composite=current_composite,
            digit_length=digit_length,
            has_snfs_form=has_snfs_form if has_snfs_form is not None else False,
            snfs_difficulty=snfs_difficulty,
            is_prime=is_prime if is_prime is not None else False,
            is_fully_factored=is_fully_factored if is_fully_factored is not None else False,
            priority=priority if priority is not None else 0
        )

        db.add(composite)
        db.flush()  # Get ID and make visible within transaction
        db.refresh(composite)

        logger.info(
            "Created new composite: %s (%s digits) - ID: %s",
            number[:20] + "..." if len(number) > 20 else number,
            composite.digit_length,
            composite.id
        )

        return composite, True, False

    def get_composite_by_number(self, db: Session, number: str) -> Optional[Composite]:
        """
        Get composite by number string.

        Args:
            db: Database session
            number: Composite number to search for

        Returns:
            Composite if found, None otherwise
        """
        return db.query(Composite).filter(Composite.number == number).first()

    def get_composite_by_id(self, db: Session, composite_id: int) -> Optional[Composite]:
        """
        Get composite by ID.

        Args:
            db: Database session
            composite_id: Composite ID

        Returns:
            Composite if found, None otherwise
        """
        return db.query(Composite).filter(Composite.id == composite_id).first()

    def find_composite_by_identifier(
        self,
        db: Session,
        identifier: str
    ) -> Optional[Composite]:
        """
        Find composite by ID, number (formula), or current_composite.

        This is a flexible lookup method useful for search/find operations.

        Args:
            db: Database session
            identifier: Can be numeric ID, formula (e.g., "2^1223-1"),
                       or current composite value

        Returns:
            Composite if found, None otherwise
        """
        # Try to parse as ID first
        try:
            composite_id = int(identifier)
            composite = self.get_composite_by_id(db, composite_id)
            if composite:
                return composite
        except ValueError:
            # Not a valid integer, continue to string search
            pass

        # Search by number (formula) or current_composite
        return db.query(Composite).filter(
            (Composite.number == identifier) | (Composite.current_composite == identifier)
        ).first()

    def delete_composite(
        self,
        db: Session,
        composite_id: int,
        reason: str = "admin_removal"
    ) -> Optional[Dict]:
        """
        Delete composite and all related data.

        This performs a cascading delete of:
        - All ECM attempts
        - All factors
        - All work assignments (active ones are cancelled)
        - The composite itself

        Args:
            db: Database session
            composite_id: ID of composite to delete
            reason: Reason for deletion (for logging)

        Returns:
            Dictionary with deletion details, or None if composite not found
        """
        composite = self.get_composite_by_id(db, composite_id)
        if not composite:
            return None

        # Cancel active work assignments
        active_work = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.composite_id == composite_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
            )
        ).all()

        for work in active_work:
            work.status = 'cancelled'

        # Delete related records in correct order (respecting foreign key constraints)
        # Factors reference attempts via found_by_attempt_id, so delete factors first
        factors_deleted = db.query(Factor).filter(
            Factor.composite_id == composite_id
        ).delete()

        attempts_deleted = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite_id
        ).delete()

        work_deleted = db.query(WorkAssignment).filter(
            WorkAssignment.composite_id == composite_id
        ).delete()

        # Delete the composite
        db.delete(composite)
        db.flush()  # Make changes visible within transaction

        logger.info(
            "Deleted composite %s (reason: %s): %d attempts, %d factors, %d work assignments",
            composite_id, reason, attempts_deleted, factors_deleted, work_deleted
        )

        return {
            "composite_id": composite_id,
            "status": "removed",
            "reason": reason,
            "cancelled_work_assignments": len(active_work),
            "deleted_attempts": attempts_deleted,
            "deleted_factors": factors_deleted,
            "deleted_work_assignments": work_deleted
        }

    # ==================== Status Updates ====================

    def mark_fully_factored(self, db: Session, composite_id: int) -> bool:
        """
        Mark composite as fully factored.

        Args:
            db: Database session
            composite_id: ID of composite to update

        Returns:
            True if composite was found and updated, False otherwise

        Raises:
            ValueError: If update fails
        """
        rows_updated = db.query(Composite)\
            .filter(Composite.id == composite_id)\
            .update({"is_fully_factored": True})
        db.flush()  # Make changes visible within transaction
        return rows_updated > 0

    def mark_prime(self, db: Session, composite_id: int) -> bool:
        """
        Mark composite as prime.

        Args:
            db: Database session
            composite_id: ID of composite to update

        Returns:
            True if composite was found and updated, False otherwise

        Raises:
            ValueError: If update fails
        """
        rows_updated = db.query(Composite)\
            .filter(Composite.id == composite_id)\
            .update({"is_prime": True})
        db.flush()  # Make changes visible within transaction
        return rows_updated > 0

    def mark_composite_complete(
        self,
        db: Session,
        composite_id: int,
        reason: str = "manual"
    ) -> bool:
        """
        Mark a composite as fully factored and cancel active work.

        Args:
            db: Database session
            composite_id: ID of composite to mark complete
            reason: Reason for completion

        Returns:
            True if successful, False if composite not found
        """
        composite = self.get_composite_by_id(db, composite_id)
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
            work.status = 'completed'

        db.flush()  # Make changes visible within transaction

        logger.info(
            "Marked composite %s as fully factored (%s), cancelled %d work assignments",
            composite_id, reason, len(active_work)
        )
        return True

    def set_composite_priority(
        self,
        db: Session,
        composite_id: int,
        priority: int
    ) -> bool:
        """
        Set priority for a composite.

        Args:
            db: Database session
            composite_id: ID of composite
            priority: Priority value

        Returns:
            True if successful, False if composite not found
        """
        composite = self.get_composite_by_id(db, composite_id)
        if not composite:
            return False

        composite.priority = priority
        db.flush()  # Make changes visible within transaction

        logger.info("Set priority %d for composite %s", priority, composite_id)
        return True

    def update_t_level(self, db: Session, composite_id: int) -> bool:
        """
        Recalculate and update current and target t-levels for a composite.

        Args:
            db: Database session
            composite_id: ID of composite to update

        Returns:
            True if successful, False if composite not found

        Raises:
            ValueError: If update fails
        """
        try:
            composite = self.get_composite_by_id(db, composite_id)
            if not composite:
                return False

            # Get all ECM attempts for this composite
            ecm_attempts = db.query(ECMAttempt).filter(
                ECMAttempt.composite_id == composite_id,
                ECMAttempt.method == 'ecm'
            ).all()

            # Calculate current t-level
            current_t_level = self.t_level_calculator.get_current_t_level_from_attempts(ecm_attempts)

            # Recalculate target t-level based on current composite length and SNFS data
            target_t_level = self.t_level_calculator.calculate_target_t_level(
                digit_length=composite.digit_length,
                snfs_difficulty=composite.snfs_difficulty
            )

            # Update the composite
            composite.current_t_level = current_t_level
            composite.target_t_level = target_t_level
            db.flush()  # Make changes visible within transaction

            return True

        except Exception as e:
            raise ValueError(f"Failed to update t-level for composite {composite_id}: {str(e)}")

    def update_after_factor_division(self, db: Session, composite_id: int, factor: str) -> bool:
        """
        Update composite after dividing out a factor.

        This method:
        1. Divides the factor out to get the cofactor
        2. Updates current_composite and digit_length
        3. Tests if cofactor is prime
        4. If prime, marks as fully factored
        5. If composite, recalculates target t-level for new size

        Args:
            db: Database session
            composite_id: ID of composite to update
            factor: Factor to divide out (as string)

        Returns:
            True if successful, False if composite not found

        Raises:
            ValueError: If factor doesn't divide composite or update fails
        """
        from ..utils.number_utils import divide_factor, is_probably_prime, calculate_digit_length
        from ..utils.calculations import ECMCalculations

        try:
            composite = self.get_composite_by_id(db, composite_id)
            if not composite:
                return False

            # Get the current composite value (use current_composite if set, otherwise original number)
            current_value = composite.current_composite or composite.number

            # Divide out the factor
            cofactor = divide_factor(current_value, factor)

            # Update composite with cofactor
            composite.current_composite = cofactor
            composite.digit_length = calculate_digit_length(cofactor)

            logger.info(
                "Divided factor %s out of composite %d: %s -> %s (%d digits)",
                factor[:20] + "..." if len(factor) > 20 else factor,
                composite_id,
                current_value[:20] + "..." if len(current_value) > 20 else current_value,
                cofactor[:20] + "..." if len(cofactor) > 20 else cofactor,
                composite.digit_length
            )

            # Test if cofactor is prime
            if is_probably_prime(cofactor):
                logger.info("Cofactor %s is prime - marking composite %d as fully factored",
                           cofactor[:20] + "..." if len(cofactor) > 20 else cofactor, composite_id)
                composite.is_prime = True
                composite.is_fully_factored = True
            else:
                # Cofactor is still composite - recalculate target t-level for new size
                new_target_t_level = ECMCalculations.recommend_target_t_level(composite.digit_length)

                logger.info(
                    "Cofactor %s is composite (%d digits) - updating target t-level from %.1f to %.1f",
                    cofactor[:20] + "..." if len(cofactor) > 20 else cofactor,
                    composite.digit_length,
                    composite.target_t_level or 0.0,
                    new_target_t_level
                )

                composite.target_t_level = new_target_t_level

                # Also update current t-level based on existing work
                self.update_t_level(db, composite_id)

            db.flush()  # Make changes visible within transaction
            return True

        except Exception as e:
            logger.error("Failed to update composite %d after factor division: %s", composite_id, str(e))
            raise ValueError(f"Failed to update composite {composite_id} after factor division: {str(e)}")

    # ==================== Bulk Operations ====================

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
            data_source: Data to load (file path, CSV string, list of numbers, or list of dicts)
            source_type: 'file', 'csv', 'list', or 'auto'
            default_priority: Default priority for new composites
            project_name: Optional project name to associate composites with

        Returns:
            Dictionary with loading statistics including:
            - total_processed: Number of items processed
            - new_composites: Number of new composites created
            - existing_composites: Number of existing composites found
            - updated_composites: Number of composites updated with new data
            - invalid_numbers: Number of invalid entries
            - errors: List of error messages
        """
        logger.info(
            "bulk_load_composites: source_type=%s, project_name=%s",
            source_type, project_name
        )

        stats: Dict[str, Any] = {
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

            numbers_data: List[Dict[str, Any]]
            if source_type == 'file':
                numbers_data = [{'number': n} for n in self.loader.from_text_file(data_source)]
            elif source_type == 'csv':
                numbers_data = self.loader.from_csv_content(data_source)
            elif source_type == 'list':
                if isinstance(data_source[0], dict):
                    numbers_data = cast(List[Dict[str, Any]], data_source)
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
                    composite, created, updated = self.get_or_create_composite(
                        db, number,
                        current_composite=item.get('current_composite'),
                        has_snfs_form=item.get('has_snfs_form'),  # None if not provided
                        snfs_difficulty=item.get('snfs_difficulty'),
                        is_prime=item.get('is_prime'),
                        is_fully_factored=item.get('is_fully_factored'),
                        priority=item.get('priority', default_priority)
                    )

                    if created:
                        stats['new_composites'] += 1
                    else:
                        stats['existing_composites'] += 1
                        if updated:
                            stats['updated_composites'] += 1

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
            logger.error("Bulk load failed: %s", str(e))

        return stats

    # ==================== Details and Statistics ====================

    def get_composite_details(
        self,
        db: Session,
        composite_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive details about a specific composite.

        Returns detailed information including:
        - Composite metadata
        - Progress statistics
        - Active work assignments
        - Recent attempts
        - Factors with group order information

        Args:
            db: Database session
            composite_id: ID of composite

        Returns:
            Dictionary with composite details, or None if not found
        """
        composite = self.get_composite_by_id(db, composite_id)
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

        # Get factors with full details from Factor table (not from attempts)
        factors_with_details = db.query(Factor).filter(
            Factor.composite_id == composite_id
        ).all()

        # Get unique factors sorted numerically
        factors_found = sorted(
            set(f.factor for f in factors_with_details),
            key=lambda x: int(x)
        )

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

    def get_work_queue_status(self, db: Session) -> Dict[str, Any]:
        """
        Get comprehensive status of the work queue.

        Returns statistics about:
        - Composite counts (total, factored, unfactored)
        - Size distribution
        - Active/completed work
        - Queue utilization

        Args:
            db: Database session

        Returns:
            Dictionary with queue status information
        """
        # Composite statistics
        total_composites = db.query(Composite).count()
        unfactored_composites = db.query(Composite).filter(
            ~Composite.is_fully_factored
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
            ~Composite.is_fully_factored
        ).group_by(
            func.floor(Composite.digit_length / 10)
        ).all()

        size_dist_dict = {
            f"{int(size)}-{int(size)+9} digits": count
            for size, count in size_distribution
        }

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

    # ==================== Project Management ====================

    def get_or_create_project(
        self,
        db: Session,
        project_name: str,
        description: Optional[str] = None
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
        db.flush()  # Get ID and make visible within transaction
        db.refresh(project)

        logger.info("Created new project: %s", project_name)
        return project, True

    def get_method_breakdown(self, composite_id: int, db: Session) -> Dict[str, Any]:
        """
        Get statistics grouped by method for a composite.

        Args:
            composite_id: ID of the composite
            db: Database session

        Returns:
            Dictionary with method-specific statistics
        """
        attempts = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite_id
        ).all()

        breakdown = {}

        # Prefetch all factors for these attempts to avoid N+1 queries
        from ..models import Factor
        attempt_ids = [a.id for a in attempts]
        factors_by_attempt = {}
        if attempt_ids:
            factors_query = db.query(Factor).filter(
                Factor.found_by_attempt_id.in_(attempt_ids)
            ).all()
            factors_by_attempt = {f.found_by_attempt_id: f for f in factors_query}

        for method in ['ecm', 'pm1', 'pp1']:
            method_attempts = [a for a in attempts if a.method == method]

            # Get factors found by this method
            factors = []
            for attempt in method_attempts:
                if attempt.factor_found:
                    # Get sigma from prefetched Factor model
                    factor_record = factors_by_attempt.get(attempt.id)

                    factors.append({
                        'factor': attempt.factor_found,
                        'sigma': factor_record.sigma if factor_record else None,
                        'b1': attempt.b1,
                        'b2': attempt.b2
                    })

            breakdown[method] = {
                'total_attempts': len(method_attempts),
                'total_curves': sum(a.curves_completed for a in method_attempts if a.curves_completed),
                'factors_found': factors,
                'max_b1': max((a.b1 for a in method_attempts if a.b1), default=0),
                'min_b1': min((a.b1 for a in method_attempts if a.b1), default=0),
                'avg_b1': sum(a.b1 for a in method_attempts if a.b1) / len(method_attempts) if method_attempts else 0,
                'attempts': method_attempts
            }

        return breakdown

    def get_milestone_groups(self, db: Session) -> Dict[int, Dict[str, list]]:
        """
        Group composites by t-level milestones (t30, t35, t40, etc.).

        Args:
            db: Database session

        Returns:
            Dictionary mapping milestone to grouped composites
        """
        milestones = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        groups = {}

        # Get all composites with t-level data
        composites = db.query(Composite).filter(
            or_(Composite.is_prime.is_(None), Composite.is_prime == False),
            Composite.is_fully_factored == False
        ).all()

        for milestone in milestones:
            complete = []
            in_progress = []
            not_started = []

            for comp in composites:
                # Only consider if targeting this milestone or higher
                if comp.target_t_level is not None and comp.target_t_level >= milestone:
                    current_t = comp.current_t_level if comp.current_t_level is not None else 0
                    if current_t >= milestone:
                        complete.append(comp)
                    elif current_t > 0:
                        in_progress.append(comp)
                    else:
                        not_started.append(comp)

            groups[milestone] = {
                'complete': complete,
                'in_progress': in_progress,
                'not_started': not_started,
                'total': len(complete) + len(in_progress) + len(not_started)
            }

        return groups

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
            priority: Priority within this project

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
        db.flush()  # Make visible within transaction
        db.refresh(project_composite)

        return project_composite, True

    # ==================== Private Helper Methods ====================

    def _detect_source_type(self, data_source) -> str:
        """
        Detect the type of data source.

        Args:
            data_source: The data source to analyze

        Returns:
            Source type: 'csv', 'file', or 'list'

        Raises:
            ValueError: If data source type cannot be determined
        """
        if isinstance(data_source, str):
            if '\n' in data_source or ',' in data_source:
                return 'csv'
            else:
                return 'file'
        elif isinstance(data_source, list):
            return 'list'
        else:
            raise ValueError("Unknown data source type")
