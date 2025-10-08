"""
T-level calculation service for ECM factorization targets.

Integrates with existing t-level software and provides target calculations
based on composite size and special form detection.
"""
import math
import re
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import subprocess
import json

logger = logging.getLogger(__name__)

class TLevelCalculator:
    """Calculate target t-levels and ECM parameters for composites."""

    def __init__(self):
        # Get t-level binary path from config
        from ..config import get_settings
        settings = get_settings()
        self.T_LEVEL_BINARY = settings.t_level_binary_path
        self.t_level_available = self._check_t_level_software()

    def _check_t_level_software(self) -> bool:
        """Check if external t-level software is available."""
        try:
            t_level_path = Path(self.T_LEVEL_BINARY)
            return t_level_path.exists()
        except Exception:
            return False

    def calculate_target_t_level(self, digit_length: int,
                                special_form: Optional[str] = None,
                                snfs_difficulty: Optional[int] = None) -> float:
        """
        Calculate target t-level for a composite based on its size and form.

        Uses the standard formula: target_t = 4/13 * effective_digits
        Where effective_digits = min(digit_length, snfs_difficulty) if snfs_difficulty is set.

        Args:
            digit_length: Number of decimal digits in the current composite
            special_form: Optional special form type ('fermat', 'mersenne', etc.) - deprecated
            snfs_difficulty: GNFS-equivalent digit count for SNFS numbers

        Returns:
            Target t-level as float
        """
        # Use the easier of actual size or SNFS difficulty
        effective_digits = digit_length
        if snfs_difficulty is not None:
            effective_digits = min(digit_length, snfs_difficulty)
            logger.info(f"Using SNFS difficulty: min({digit_length}, {snfs_difficulty}) = {effective_digits} digits")

        # Base formula: 4/13 * effective_digits
        base_target = (4.0 / 13.0) * effective_digits

        # Apply SNFS discount for special forms (deprecated - prefer snfs_difficulty)
        if special_form and snfs_difficulty is None:
            discount = self._get_snfs_discount(special_form, digit_length)
            target_t = base_target * (1.0 - discount)
            logger.info(f"Applied SNFS discount of {discount*100:.1f}% for {special_form} form")
        else:
            target_t = base_target

        # Reasonable bounds
        target_t = max(10.0, min(target_t, 85.0))

        logger.info(f"Target t-level for {effective_digits}-digit number: t{target_t:.1f}")
        return target_t

    def _get_snfs_discount(self, special_form: str, digit_length: int) -> float:
        """
        Calculate SNFS discount factor for special number forms.

        Args:
            special_form: Type of special form
            digit_length: Size of the number

        Returns:
            Discount factor (0.0 to 1.0)
        """
        # Conservative discounts based on SNFS effectiveness
        discounts = {
            'fermat': 0.15,      # Numbers of form 2^n + 1
            'mersenne': 0.20,    # Numbers of form 2^n - 1
            'aurifeuillean': 0.15, # Lucas/Fibonacci numbers
            'cunningham': 0.10,   # Numbers of form a^n ± b^n
            'generalized_fermat': 0.12, # Numbers of form a^(2^n) + b^(2^n)
            'repunit': 0.18,     # Numbers of form (10^n - 1)/9
        }

        base_discount = discounts.get(special_form.lower(), 0.0)

        # Larger numbers benefit more from SNFS
        if digit_length > 100:
            size_multiplier = 1.3
        elif digit_length > 80:
            size_multiplier = 1.2
        elif digit_length > 60:
            size_multiplier = 1.1
        else:
            size_multiplier = 1.0

        return min(base_discount * size_multiplier, 0.25)  # Cap at 25% discount

    def _format_number_for_tlevel(self, number: float) -> str:
        """
        Format number for t-level executable input.

        The t-level executable accepts formats like:
        - 110000000 (plain integers)
        - 11e7 (integer scientific notation)

        But rejects:
        - 1.1e+08 (decimal scientific notation)
        """
        if number is None:
            return "0"

        num = int(number)

        # For small numbers, use plain format
        if num < 1000000:
            return str(num)

        # For larger numbers, find appropriate scientific notation
        # Convert to string and count zeros
        num_str = str(num)

        # Try to express as integer * 10^n (like 11e7 for 110000000)
        if num_str.endswith('000000'):  # At least 6 zeros
            # Find how many trailing zeros
            trailing_zeros = len(num_str) - len(num_str.rstrip('0'))
            if trailing_zeros >= 6:
                # Express as significand * 10^exponent
                significand = num // (10 ** trailing_zeros)
                return f"{significand}e{trailing_zeros}"

        # Fallback to plain number
        return str(num)

    def detect_special_form(self, number_str: str) -> Optional[str]:
        """
        Detect if a number has a special form suitable for SNFS.

        Args:
            number_str: String representation of the number

        Returns:
            Special form type if detected, None otherwise
        """
        try:
            # Convert to integer for analysis
            n = int(number_str)

            # Check for small cases first
            if n < 1000:
                return None

            # Check for Fermat numbers: 2^(2^k) + 1
            if self._is_fermat_form(n):
                return 'fermat'

            # Check for Mersenne-like: 2^k - 1 or 2^k + 1
            if self._is_mersenne_like(n):
                return 'mersenne'

            # Check for repunits: (10^k - 1)/9
            if self._is_repunit(n):
                return 'repunit'

            # Check for Cunningham form: a^n ± b^n
            cunningham_form = self._detect_cunningham_form(number_str)
            if cunningham_form:
                return 'cunningham'

            return None

        except (ValueError, OverflowError):
            # Number too large for direct analysis
            return self._detect_special_form_string(number_str)

    def _is_fermat_form(self, n: int) -> bool:
        """Check if number is of Fermat form 2^(2^k) + 1."""
        if n <= 3:
            return False

        # Check if n-1 is a power of 2
        m = n - 1
        if m & (m - 1) != 0:  # Not a power of 2
            return False

        # Check if the exponent is also a power of 2
        exp = m.bit_length() - 1
        return exp > 0 and (exp & (exp - 1)) == 0

    def _is_mersenne_like(self, n: int) -> bool:
        """Check if number is close to 2^k ± 1."""
        # Check 2^k - 1
        k = (n + 1).bit_length()
        if (1 << k) - 1 == n and k > 10:
            return True

        # Check 2^k + 1
        k = (n - 1).bit_length()
        if (1 << k) + 1 == n and k > 10:
            return True

        return False

    def _is_repunit(self, n: int) -> bool:
        """Check if number is a repunit (10^k - 1)/9."""
        # Repunits are numbers with all digits being 1
        str_n = str(n)
        return len(str_n) > 3 and all(d == '1' for d in str_n)

    def _detect_cunningham_form(self, number_str: str) -> bool:
        """
        Detect Cunningham form a^n ± b^n using heuristics.

        This is a simplified detection - full detection would require
        more sophisticated factorization attempts.
        """
        # Look for numbers that might be of form a^n ± 1 for small a
        try:
            n = int(number_str)
            digit_length = len(number_str)

            # Only check for reasonably sized numbers
            if digit_length < 20 or digit_length > 500:
                return False

            # Check for a^n + 1 and a^n - 1 for small bases a
            for a in range(2, 21):  # Check bases 2 through 20
                # Estimate what power would give this size
                estimated_power = digit_length / math.log10(a)

                # Check nearby integer powers
                for exp in range(max(2, int(estimated_power) - 2),
                               int(estimated_power) + 3):
                    try:
                        val_plus = a ** exp + 1
                        val_minus = a ** exp - 1

                        if val_plus == n or val_minus == n:
                            logger.info(f"Detected Cunningham form: {a}^{exp} ± 1")
                            return True

                    except OverflowError:
                        break

            return False

        except (ValueError, OverflowError):
            return False

    def _detect_special_form_string(self, number_str: str) -> Optional[str]:
        """
        Detect special forms using string patterns for very large numbers.
        """
        # Look for pattern clues in the string representation
        if len(set(number_str)) == 1 and number_str[0] == '1':
            return 'repunit'

        # Add more pattern-based detection as needed
        return None

    def get_current_t_level_from_attempts(self, attempts: list) -> float:
        """
        Calculate current t-level achieved from previous ECM attempts.

        Args:
            attempts: List of ECMAttempt objects with curves_completed, b1, b2, method

        Returns:
            Current t-level achieved
        """
        if not attempts:
            return 0.0

        # Filter for ECM attempts only
        ecm_attempts = [a for a in attempts if a.method == 'ecm']

        if not ecm_attempts:
            return 0.0

        # Use external t-level software if available
        if self.t_level_available:
            return self._calculate_t_level_external(ecm_attempts)
        else:
            return self._calculate_t_level_estimate(ecm_attempts)

    def _calculate_t_level_external(self, attempts: list) -> float:
        """Calculate t-level using external t-level software."""
        try:
            # Convert attempts to curve string format for t-level executable
            curve_strings = []
            for attempt in attempts:
                if attempt.curves_completed > 0:
                    # Format: curves@B1,B2,param
                    # Format numbers to avoid decimals in scientific notation (use 11e7 not 1.1e+08)
                    b1_str = self._format_number_for_tlevel(attempt.b1)
                    # Use actual b2 value (including 0), only default to b1*100 if b2 is None
                    b2_str = self._format_number_for_tlevel(attempt.b2) if attempt.b2 is not None else self._format_number_for_tlevel(attempt.b1 * 100)
                    # Use actual parametrization from attempt, default to 3 if not set
                    param = str(attempt.parametrization) if attempt.parametrization is not None else "3"

                    curve_str = f"{attempt.curves_completed}@{b1_str},{b2_str},{param}"
                    curve_strings.append(curve_str)

            if not curve_strings:
                return 0.0

            # Join with semicolons for multiple entries
            input_string = ";".join(curve_strings)

            # Call external t-level calculator
            result = subprocess.run(
                [self.T_LEVEL_BINARY, "-q", input_string],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            if result.returncode == 0:
                # Parse output: expected format is "t45.185"
                output = result.stdout.strip()
                if output.startswith('t'):
                    t_level = float(output[1:])  # Remove 't' prefix and convert to float
                    logger.info(f"External t-level calculation: {input_string} -> {output}")
                    return t_level
                else:
                    logger.warning(f"Unexpected t-level output format: {output}")
                    return 0.0
            else:
                logger.warning(f"T-level calculation failed: {result.stderr}")
                return 0.0

        except subprocess.TimeoutExpired:
            logger.warning("T-level calculation timed out")
            return 0.0
        except Exception as e:
            logger.warning(f"External t-level calculation failed: {e}")
            return 0.0

    def _calculate_t_level_estimate(self, attempts: list) -> float:
        """
        Fallback t-level estimation when external calculator unavailable.

        Returns 0.0 for now - proper calculation requires the external t-level binary.
        """
        # Without proper probability tables, any estimate would be misleading
        # Better to return 0.0 and rely on the external calculator
        logger.info("Using fallback t-level estimation (returns 0.0)")
        return 0.0

    def suggest_next_ecm_parameters(self, target_t_level: float,
                                  current_t_level: float,
                                  digit_length: int) -> Dict[str, Any]:
        """
        Suggest next ECM parameters to work toward target t-level.

        Args:
            target_t_level: Desired t-level to achieve
            current_t_level: Current t-level already achieved
            digit_length: Size of the composite in digits

        Returns:
            Dictionary with suggested B1, B2, and curves
        """
        if current_t_level >= target_t_level:
            return {
                'status': 'target_reached',
                'message': f'Target t-level {target_t_level:.1f} already achieved (current: {current_t_level:.1f})'
            }

        # Calculate remaining t-level needed
        remaining_t = target_t_level - current_t_level

        # Select B1 based on digit length and remaining work
        b1 = self._select_optimal_b1(digit_length, remaining_t)

        # Calculate B2 (typically B1 * 100 to B1 * 1000)
        b2 = int(b1 * 500)  # Middle ground

        # Estimate curves needed
        curves = self._estimate_curves_needed(b1, remaining_t)

        return {
            'status': 'suggestion',
            'b1': b1,
            'b2': b2,
            'curves': curves,
            'estimated_t_level_gain': remaining_t,
            'target_t_level': target_t_level,
            'current_t_level': current_t_level,
            'message': f'Work toward t{target_t_level:.1f} (currently t{current_t_level:.1f})'
        }

    def _select_optimal_b1(self, digit_length: int, remaining_t: float) -> int:
        """Select optimal B1 value based on number size and remaining t-level."""
        # Base B1 selection on digit length
        if digit_length <= 30:
            base_b1 = 11000
        elif digit_length <= 40:
            base_b1 = 50000
        elif digit_length <= 50:
            base_b1 = 250000
        elif digit_length <= 60:
            base_b1 = 1000000
        elif digit_length <= 70:
            base_b1 = 3000000
        elif digit_length <= 80:
            base_b1 = 11000000
        else:
            base_b1 = 43000000

        # Adjust based on remaining t-level work
        if remaining_t > 20:
            multiplier = 2.0
        elif remaining_t > 10:
            multiplier = 1.5
        elif remaining_t > 5:
            multiplier = 1.2
        else:
            multiplier = 1.0

        return int(base_b1 * multiplier)

    def _estimate_curves_needed(self, b1: int, remaining_t: float) -> int:
        """Estimate number of curves needed for remaining t-level."""
        # Very rough estimate: more t-level needs more curves
        # This is a placeholder - real calculation would use probability tables
        base_curves = max(10, int(remaining_t * 20))

        # Adjust for B1 level
        if b1 >= 10000000:
            curve_factor = 0.5  # Fewer curves needed for higher B1
        elif b1 >= 1000000:
            curve_factor = 0.7
        elif b1 >= 100000:
            curve_factor = 1.0
        else:
            curve_factor = 1.5

        curves = int(base_curves * curve_factor)
        return max(10, min(curves, 1000))  # Reasonable bounds