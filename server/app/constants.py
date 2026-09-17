"""
Shared constants for the ECM coordination server.

This module centralizes configuration constants used across multiple modules
to avoid duplication and ensure consistency.
"""

from typing import List, Tuple

# Work assignment statuses that indicate active/in-progress work.
# Used across query helpers, services, and route handlers to filter
# for assignments that should block new work or count as "active".
ACTIVE_WORK_STATUSES = ['assigned', 'claimed', 'running']

# Residue statuses that mean stage 2 is still outstanding: the file is waiting
# for a consumer, or one is processing it. Blocks new work on the composite
# (/ecm-work, /p1-work) and marks the residue as still in play.
PENDING_RESIDUE_STATUSES = ['available', 'claimed']

# ECM parameter table based on Paul Zimmerman's GMP-ECM 7 recommendations
# Source: https://www.rieselprime.de/ziki/Elliptic_curve_method
# Format: (max_digits, b1, b2, typical_curves)
# - max_digits: Maximum factor size (in digits) this B1 level is suitable for
# - b1: Stage 1 bound (from Zimmerman table)
# - b2: Stage 2 bound (approximated as 100*B1, GMP-ECM calculates optimal internally)
# - typical_curves: Expected curves for GMP-ECM 7 default parameters
#
# Note: These values are currently used for internal bookkeeping only.
# The client calculates actual parameters via the t-level binary.
ECM_BOUNDS: List[Tuple[int, int, int, int]] = [
    (20, 11000, 1100000, 107),
    (25, 50000, 5000000, 261),
    (30, 250000, 25000000, 513),
    (35, 1000000, 100000000, 1071),
    (40, 3000000, 300000000, 2753),
    (45, 11000000, 1100000000, 5208),
    (50, 43000000, 4300000000, 8704),
    (55, 110000000, 11000000000, 20479),
    (60, 260000000, 26000000000, 47888),
    (65, 850000000, 85000000000, 78923),
    (70, 2900000000, 290000000000, 115153),
    (75, 7600000000, 760000000000, 211681),
    (80, 25000000000, 2500000000000, 296479),
]


# Optimal B1 values for each t-level (Zimmermann's table)
# Format: (t_level, optimal_b1)
# Used by /p1-work endpoint to calculate PM1/PP1 B1 values
OPTIMAL_B1_TABLE: List[Tuple[int, int]] = [
    (20, 11000),
    (25, 50000),
    (30, 250000),
    (35, 1000000),
    (40, 3000000),
    (45, 11000000),
    (50, 43000000),
    (55, 110000000),
    (60, 260000000),
    (65, 850000000),
    (70, 2900000000),
    (75, 7600000000),
    (80, 25000000000),
]


def get_b1_above_tlevel(target_t_level: float) -> int:
    """
    Get B1 one step above the target t-level for PM1/PP1 sweeps.

    Finds the smallest entry in OPTIMAL_B1_TABLE where t_level >= target_t_level,
    then returns the B1 of the next entry (one step above). This ensures PM1/PP1
    runs at a B1 slightly beyond what ECM has already covered.

    Args:
        target_t_level: The composite's target t-level

    Returns:
        B1 value one step above the target t-level

    Example:
        >>> get_b1_above_tlevel(48)  # Next >= 48 is t50 (B1=43M), one above = t55
        110000000
        >>> get_b1_above_tlevel(55)  # Next >= 55 is t55 (B1=110M), one above = t60
        260000000
    """
    for i, (t_level, b1) in enumerate(OPTIMAL_B1_TABLE):
        if target_t_level <= t_level:
            if i + 1 < len(OPTIMAL_B1_TABLE):
                return OPTIMAL_B1_TABLE[i + 1][1]
            return b1
    return OPTIMAL_B1_TABLE[-1][1]
