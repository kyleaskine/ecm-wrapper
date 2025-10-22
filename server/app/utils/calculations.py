"""
Calculation utilities for composite progress tracking and ECM effort grouping.

Provides centralized business logic for completion percentage calculations
and effort grouping to ensure consistency across the API.
"""

from typing import List, Dict
from ..models.composites import Composite
from ..models.attempts import ECMAttempt


class CompositeCalculations:
    """Business logic for composite-related calculations."""

    @staticmethod
    def get_completion_percentage(composite: Composite) -> float:
        """
        Calculate completion percentage for a composite based on t-level progress.

        Args:
            composite: Composite instance with target_t_level and current_t_level

        Returns:
            Completion percentage (0-100). Returns 0 if no target is set.

        Example:
            >>> comp = Composite(target_t_level=50.0, current_t_level=25.0)
            >>> CompositeCalculations.get_completion_percentage(comp)
            50.0
        """
        if composite.target_t_level and composite.target_t_level > 0:
            current_t = composite.current_t_level or 0.0
            return (current_t / composite.target_t_level) * 100
        return 0.0

    @staticmethod
    def sort_composites_by_progress(
        composites: List[Composite],
        reverse: bool = True
    ) -> List[Composite]:
        """
        Sort composites by completion percentage.

        Args:
            composites: List of Composite instances
            reverse: If True, sort descending (highest completion first)

        Returns:
            Sorted list of composites

        Example:
            >>> composites = [comp1, comp2, comp3]
            >>> sorted_comps = CompositeCalculations.sort_composites_by_progress(composites)
        """
        return sorted(
            composites,
            key=CompositeCalculations.get_completion_percentage,
            reverse=reverse
        )


class ECMCalculations:
    """Business logic for ECM-related calculations."""

    @staticmethod
    def group_attempts_by_b1(attempts: List[ECMAttempt]) -> Dict[int, int]:
        """
        Group ECM attempts by B1 bound and sum curves completed.

        Args:
            attempts: List of ECMAttempt instances

        Returns:
            Dictionary mapping B1 bound to total curves completed at that bound

        Example:
            >>> attempts = [
            ...     ECMAttempt(b1=50000, curves_completed=100),
            ...     ECMAttempt(b1=50000, curves_completed=50),
            ...     ECMAttempt(b1=250000, curves_completed=200)
            ... ]
            >>> ECMCalculations.group_attempts_by_b1(attempts)
            {50000: 150, 250000: 200}
        """
        effort_groups: Dict[int, int] = {}
        for attempt in attempts:
            b1 = attempt.b1
            effort_groups[b1] = effort_groups.get(b1, 0) + attempt.curves_completed
        return effort_groups

    @staticmethod
    def group_attempts_by_b1_sorted(attempts: List[ECMAttempt]) -> List[Dict[str, int]]:
        """
        Group ECM attempts by B1 and return sorted list of dicts.

        Args:
            attempts: List of ECMAttempt instances

        Returns:
            List of dicts with 'b1' and 'curves' keys, sorted by B1

        Example:
            >>> attempts = [...]
            >>> ECMCalculations.group_attempts_by_b1_sorted(attempts)
            [{'b1': 50000, 'curves': 150}, {'b1': 250000, 'curves': 200}]
        """
        effort_groups = ECMCalculations.group_attempts_by_b1(attempts)
        return [
            {"b1": b1, "curves": curves}
            for b1, curves in sorted(effort_groups.items())
        ]
