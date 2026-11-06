"""
Mathematical utilities for ECM wrapper.

This module provides functions for:
- Trial division factorization
- Primality testing (Miller-Rabin)
- T-level calculations
- ECM parameter optimization
"""

import logging
import random
import re
import sys
from typing import Callable, List, Tuple, Optional
from .subprocess_utils import execute_subprocess_simple

TLEVEL_BINARY_DEFAULT = 'bin/t-level.exe' if sys.platform == 'win32' else 'bin/t-level'


logger = logging.getLogger(__name__)


# Zimmermann's optimal B1 values and expected curves for GMP-ECM 7
# Source: https://members.loria.fr/PZimmermann/records/ecmnet.html
# Format: (digits/t-level, optimal_b1, expected_curves_ecm7)
OPTIMAL_B1_TABLE = [
    (20, 11000, 107),
    (25, 50000, 261),
    (30, 250000, 513),
    (35, 1000000, 1071),
    (40, 3000000, 2753),
    (45, 11000000, 5208),
    (50, 43000000, 8704),
    (55, 110000000, 20479),
    (60, 260000000, 47888),
    (65, 850000000, 78923),
    (70, 2900000000, 115153),
    (75, 7600000000, 211681),
    (80, 25000000000, 296479),
]


def trial_division(n: int, limit: int = 10**7) -> Tuple[List[int], int]:
    """
    Fast trial division to find small prime factors.

    Uses wheel factorization to skip multiples of 2, 3, 5.

    Args:
        n: Number to factor
        limit: Trial division limit (default: 10^7)

    Returns:
        Tuple of (factors_found, cofactor)

    Example:
        >>> factors, cofactor = trial_division(360)
        >>> factors
        [2, 2, 2, 3, 3, 5]
        >>> cofactor
        1
    """
    factors = []
    cofactor = n

    # Trial division by 2
    while cofactor % 2 == 0:
        factors.append(2)
        cofactor //= 2

    # Trial division by 3
    while cofactor % 3 == 0:
        factors.append(3)
        cofactor //= 3

    # Trial division by 5
    while cofactor % 5 == 0:
        factors.append(5)
        cofactor //= 5

    # Trial division by odd numbers (wheel factorization: skip multiples of 2,3,5)
    i = 7
    while i * i <= cofactor and i <= limit:
        while cofactor % i == 0:
            factors.append(i)
            cofactor //= i
        i += 2

    return factors, cofactor


def split_into_distinct_primes(
    factors: List[str], sigmas: List[Optional[str]],
    fully_factor: Callable[[str], List[str]],
) -> Tuple[List[str], List[Optional[str]]]:
    """
    Fully factor each found factor and keep each distinct prime once, with the
    sigma of the curve that found it.

    A run can report a composite factor (two curves of a batch finding
    different primes, or one curve finding both), and the same prime can turn
    up in several expansions: if one curve finds p and another p*q, p is kept
    once (multiplicities are handled by the caller dividing them out).
    """
    primes: List[str] = []
    prime_sigmas: List[Optional[str]] = []
    for factor, sigma in zip(factors, sigmas):
        for prime in fully_factor(factor):
            if prime not in primes:
                primes.append(prime)
                prime_sigmas.append(sigma)
    return primes, prime_sigmas


def is_probably_prime(n: int, trials: int = 10) -> bool:
    """
    Miller-Rabin primality test.

    Probabilistic primality test with configurable confidence level.
    With 10 trials, false positive rate is approximately 1 in 4^10.

    Args:
        n: Number to test
        trials: Number of trials (default: 10, higher = more confident)

    Returns:
        True if probably prime, False if definitely composite

    Example:
        >>> is_probably_prime(17)
        True
        >>> is_probably_prime(18)
        False
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(trials):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def calculate_tlevel(curve_history: List[str], tlevel_binary: str = TLEVEL_BINARY_DEFAULT,
                     base_tlevel: float = 0.0) -> float:
    """
    Call t-level binary to calculate current t-level.

    Args:
        curve_history: List of curve strings like "100@1000000,p=1"
        tlevel_binary: Path to t-level executable
        base_tlevel: Starting t-level to add to (uses -w flag)

    Returns:
        Current t-level as float (0.0 on error, or base_tlevel if no curves)

    Example:
        >>> history = ["100@50000,p=1", "200@250000,p=1"]
        >>> t = calculate_tlevel(history)
        >>> t > 0
        True
        >>> # With base t-level
        >>> t = calculate_tlevel(["100@50000,p=1"], base_tlevel=35.0)
        >>> t > 35.0
        True
    """
    if not curve_history:
        return base_tlevel

    # Join curve strings with semicolons
    curve_input = ";".join(curve_history)

    try:
        # Call t-level binary using unified subprocess utility
        # Use -w to specify starting work if base_tlevel > 0
        if base_tlevel > 0:
            cmd = [tlevel_binary, '-w', str(base_tlevel), '-q', curve_input]
        else:
            cmd = [tlevel_binary, '-q', curve_input]
        stdout, _ = execute_subprocess_simple(cmd, timeout=10)

        # Parse output: "t40.234"
        match = re.search(r't([\d.]+)', stdout)
        if match:
            return float(match.group(1))

        logger.warning("Failed to parse t-level from: %s", stdout)
        return 0.0

    except (FileNotFoundError, ValueError) as e:
        logger.error("Error calculating t-level: %s", e)
        return 0.0


# Cached table for standard t-level transitions (5-digit increments)
# Format: (from_t, to_t, param): (b1, curves_needed)
# Calculated using t-level binary: t-level -w <from> -t <to> -b <optimal_b1> -p <param>
# Parametrization matters: p=1 (CPU) vs p=3 (GPU) give different t-levels for same curves!
TLEVEL_TRANSITION_CACHE = {
    # Parametrization 1 (CPU - Montgomery curves)
    (0, 20, 1): (11000, 107),      # t0 → t20: 107 curves at B1=11000, p=1
    (20, 25, 1): (50000, 261),     # t20 → t25: 261 curves at B1=50000, p=1
    (25, 30, 1): (250000, 513),    # t25 → t30: 513 curves at B1=250000, p=1
    (30, 35, 1): (1000000, 1071),  # t30 → t35: 1071 curves at B1=1000000, p=1
    (35, 40, 1): (3000000, 2753),  # t35 → t40: 2753 curves at B1=3000000, p=1
    (40, 45, 1): (11000000, 5208), # t40 → t45: 5208 curves at B1=11000000, p=1
    (45, 50, 1): (43000000, 8704), # t45 → t50: 8704 curves at B1=43000000, p=1

    # Parametrization 3 (GPU - Twisted Edwards curves)
    (0, 20, 3): (11000, 109),      # t0 → t20: 109 curves at B1=11000, p=3
    (20, 25, 3): (50000, 266),     # t20 → t25: 266 curves at B1=50000, p=3
    (25, 30, 3): (250000, 523),    # t25 → t30: 523 curves at B1=250000, p=3
    (30, 35, 3): (1000000, 1092),  # t30 → t35: 1092 curves at B1=1000000, p=3
    (35, 40, 3): (3000000, 2807),  # t35 → t40: 2807 curves at B1=3000000, p=3
    (40, 45, 3): (11000000, 5311), # t40 → t45: 5311 curves at B1=11000000, p=3
    (45, 50, 3): (43000000, 8875), # t45 → t50: 8875 curves at B1=43000000, p=3
}


def _query_tlevel_for_curves(curves: int, b1: int, b2: Optional[int], parametrization: int,
                             base_tlevel: float, tlevel_binary: str) -> float:
    """
    Query what t-level N curves at B1,B2,param would achieve starting from base_tlevel.

    Args:
        curves: Number of curves
        b1: B1 parameter
        b2: B2 parameter (None = GMP-ECM default)
        parametrization: ECM parametrization (0-3)
        base_tlevel: Starting t-level
        tlevel_binary: Path to t-level executable

    Returns:
        Resulting t-level
    """
    # Build curve string with explicit B2 if provided
    if b2 is not None:
        curve_str = f"{curves}@{b1},{b2},p={parametrization}"
    else:
        curve_str = f"{curves}@{b1},p={parametrization}"

    try:
        # Query t-level with starting work
        stdout, _ = execute_subprocess_simple(
            [tlevel_binary, '-w', str(base_tlevel), '-q', curve_str],
            timeout=10
        )

        # Parse output: "t40.234"
        match = re.search(r't([\d.]+)', stdout)
        if match:
            return float(match.group(1))

        return base_tlevel  # No change on parse failure

    except (FileNotFoundError, ValueError):
        return base_tlevel


def calculate_curves_to_target_direct(current_tlevel: float, target_tlevel: float,
                                      b1: int, parametrization: int = 1,
                                      b2: Optional[int] = None,
                                      tlevel_binary: str = TLEVEL_BINARY_DEFAULT) -> Optional[int]:
    """
    Use t-level binary to calculate exact curves needed to reach target.

    When b2 is None, calls t-level with -w (current work), -t (target), -b (B1),
    and -p (param) to get the precise number of curves required (assumes default B2).

    When b2 is specified, uses binary search to find the curves needed for the
    actual B2 value being used (important for two-stage mode with B2 = B1 * 100).

    Args:
        current_tlevel: Current t-level (e.g., 19.94)
        target_tlevel: Target t-level (e.g., 20.0)
        b1: B1 parameter for curves
        parametrization: ECM parametrization (0-4, default 1)
        b2: B2 parameter (None = use default, otherwise use specified value)
        tlevel_binary: Path to t-level executable

    Returns:
        Number of curves needed, or None if target already reached

    Example:
        >>> curves = calculate_curves_to_target_direct(19.94, 20.0, 11000, 1)
        >>> curves is not None and curves > 0
        True
    """
    if current_tlevel >= target_tlevel:
        return None

    # If B2 is specified, use binary search to find correct curve count
    if b2 is not None:
        return _calculate_curves_with_b2(current_tlevel, target_tlevel, b1, b2,
                                         parametrization, tlevel_binary)

    # Default behavior: use t-level suggestion mode (assumes default B2)
    try:
        # Call t-level binary with suggestion options
        stdout, _ = execute_subprocess_simple(
            [tlevel_binary, '-w', str(current_tlevel), '-t', str(target_tlevel),
             '-b', str(b1), '-p', str(parametrization)],
            timeout=10
        )

        # Parse output: "Running the following will get you to tXX.XXX:\n250@50e3"
        # or just "250@50e3"
        match = re.search(r'(\d+)@', stdout)
        if match:
            return int(match.group(1))

        logger.warning("Failed to parse curve suggestion from t-level: %s", stdout)
        return None

    except (FileNotFoundError, ValueError) as e:
        logger.error("Error calculating curves with t-level binary: %s", e)
        return None


def _calculate_curves_with_b2(current_tlevel: float, target_tlevel: float,
                              b1: int, b2: int, parametrization: int,
                              tlevel_binary: str) -> Optional[int]:
    """
    Calculate curves needed using binary search with explicit B2.

    This is needed for two-stage mode where B2 = B1 * 100 (weaker than default).

    Args:
        current_tlevel: Starting t-level
        target_tlevel: Target t-level
        b1: B1 parameter
        b2: B2 parameter (explicit value)
        parametrization: ECM parametrization
        tlevel_binary: Path to t-level executable

    Returns:
        Number of curves needed
    """
    # Get initial estimate from default B2 calculation
    try:
        stdout, _ = execute_subprocess_simple(
            [tlevel_binary, '-w', str(current_tlevel), '-t', str(target_tlevel),
             '-b', str(b1), '-p', str(parametrization)],
            timeout=10
        )
        match = re.search(r'(\d+)@', stdout)
        initial_estimate = int(match.group(1)) if match else 100
    except (FileNotFoundError, ValueError):
        initial_estimate = 100

    # Binary search to find correct curve count for the actual B2
    # Start with a range around the initial estimate (weaker B2 needs more curves)
    low = initial_estimate
    high = initial_estimate * 3  # Weaker B2 might need up to 3x more curves

    # First, ensure high is actually high enough
    high_tlevel = _query_tlevel_for_curves(high, b1, b2, parametrization, current_tlevel, tlevel_binary)
    while high_tlevel < target_tlevel:
        high *= 2
        high_tlevel = _query_tlevel_for_curves(high, b1, b2, parametrization, current_tlevel, tlevel_binary)
        if high > initial_estimate * 10:  # Safety limit
            logger.warning("Could not find curve count to reach t%.2f with B2=%d", target_tlevel, b2)
            return high

    # Binary search
    while high - low > 1:
        mid = (low + high) // 2
        mid_tlevel = _query_tlevel_for_curves(mid, b1, b2, parametrization, current_tlevel, tlevel_binary)

        if mid_tlevel >= target_tlevel:
            high = mid
        else:
            low = mid

    # Return high (guaranteed to reach target)
    logger.debug("B2-aware calculation: %d curves at B1=%d, B2=%d to reach t%.2f",
                 high, b1, b2, target_tlevel)
    return high


def get_b1_for_digit_length(digits: int) -> int:
    """
    Get recommended B1 value based on number of digits.

    Based on standard ECM parameter recommendations from GMP-ECM.

    Args:
        digits: Number of digits in the composite

    Returns:
        Recommended B1 value

    Example:
        >>> get_b1_for_digit_length(40)
        50000
        >>> get_b1_for_digit_length(60)
        3000000
    """
    if digits <= 20:
        return 2000
    if digits <= 25:
        return 11000
    if digits <= 30:
        return 50000
    if digits <= 35:
        return 250000
    if digits <= 40:
        return 1000000
    if digits <= 45:
        return 3000000
    if digits <= 50:
        return 11000000
    if digits <= 55:
        return 43000000
    if digits <= 60:
        return 110000000
    if digits <= 65:
        return 260000000
    # For larger numbers
    return 850000000


def get_optimal_b1_for_tlevel(target_tlevel: float) -> Tuple[int, int]:
    """
    Get optimal B1 and expected curve count for target t-level.

    Based on Zimmermann's optimal B1 values and expected curves for GMP-ECM 7.
    Source: https://members.loria.fr/PZimmermann/records/ecmnet.html

    Args:
        target_tlevel: Target t-level to achieve

    Returns:
        Tuple of (optimal_b1, expected_curves)

    Example:
        >>> b1, curves = get_optimal_b1_for_tlevel(30.0)
        >>> b1 >= 250000
        True
    """
    # Find exact match or next higher level
    for digits, b1, curves in OPTIMAL_B1_TABLE:
        if target_tlevel <= digits:
            return b1, curves

    # For t-levels beyond table, return highest entry
    return OPTIMAL_B1_TABLE[-1][1], OPTIMAL_B1_TABLE[-1][2]


def get_b1_above_tlevel(target_t_level: float) -> int:
    """
    Get B1 one step above the target t-level for PM1/PP1 sweeps.

    Finds the smallest entry in OPTIMAL_B1_TABLE where digits >= target_t_level,
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
    for i, (digits, b1, _curves) in enumerate(OPTIMAL_B1_TABLE):
        if target_t_level <= digits:
            # Return next entry's B1 if available, else this one
            if i + 1 < len(OPTIMAL_B1_TABLE):
                return OPTIMAL_B1_TABLE[i + 1][1]
            return b1
    return OPTIMAL_B1_TABLE[-1][1]


def get_max_tlevel_for_workers(workers: int) -> float:
    """
    Get a reasonable max target t-level for adaptive CPU mode based on worker count.

    Higher t-level steps require exponentially more curves and time. This function
    caps the target t-level so CPU workers don't get assigned work that would take
    unreasonably long (e.g., weeks for a single step).

    Args:
        workers: Number of parallel CPU workers

    Returns:
        Maximum target t-level to request from server
    """
    if workers >= 16:
        return 55.0
    elif workers >= 5:
        return 50.0
    elif workers >= 3:
        return 45.0
    else:
        return 40.0


def calculate_target_tlevel(digit_length: int) -> float:
    """
    Calculate target t-level for a composite based on its digit length.

    Uses the 4/13 rule: target t-level is 4/13 of the composite's digit length.
    This provides a reasonable balance between ECM effort and probability of success.

    Args:
        digit_length: Number of digits in the composite

    Returns:
        Target t-level (float)

    Example:
        >>> calculate_target_tlevel(65)  # 65-digit composite
        20.0  # Target t20
        >>> calculate_target_tlevel(100)  # 100-digit composite
        30.769...  # Target ~t31
    """
    return (4.0 / 13.0) * digit_length
