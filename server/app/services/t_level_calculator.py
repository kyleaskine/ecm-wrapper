"""
T-level calculation service for ECM factorization targets.

Integrates with existing t-level software and provides target calculations
based on composite size and special form detection.
"""
import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, NamedTuple, Tuple

from ..utils.process_executor import ExternalProgramExecutor
from . import t_level_library

logger = logging.getLogger(__name__)

# (b1, b2, param) -> total curves; see _aggregate_curve_classes
CurveTotals = Dict[Tuple[int, Optional[int], int], int]


class CurveClassRow(NamedTuple):
    """Summed curves of one (B1, B2, parametrization) class. It has the
    attributes get_current_t_level_from_attempts reads from an attempt."""
    curves_completed: int
    b1: int
    b2: Optional[int]
    parametrization: Optional[int]
    id: int  # first attempt in the class, for logging
    method: str = 'ecm'


class TLevelCalculator:
    """Calculate target t-levels and ECM parameters for composites."""

    def __init__(self) -> None:
        # Get t-level binary path from config
        from ..config import get_settings
        settings = get_settings()
        self.T_LEVEL_BINARY = settings.t_level_binary_path

        # Initialize executor for t-level binary
        self.executor = ExternalProgramExecutor(
            self.T_LEVEL_BINARY,
            binary_name="t-level"
        )

        # In-process library by default; the binary is the fallback, chosen by
        # setting or used when the package cannot be imported (the library
        # module logs that once). is_available() imports the package on first
        # use, so with T_LEVEL_BACKEND=binary it is never loaded.
        self.use_library = (settings.t_level_backend == "library"
                            and t_level_library.is_available())

        # Probe the binary only when it may be needed. check_binary_exists()
        # logs a warning when it is missing, and this class is instantiated per
        # request, so probing eagerly would warn on every request wherever the
        # binary is not deployed.
        self._binary_available: Optional[bool] = None
        self.t_level_available = self.use_library or self._binary_is_available()

    def _binary_is_available(self) -> bool:
        if self._binary_available is None:
            self._binary_available = self.executor.check_binary_exists()
        return self._binary_available

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

    def get_current_t_level_from_attempts(self, attempts: list,
                                          starting_t_level: float = 0.0) -> Optional[float]:
        """
        Calculate current t-level achieved from previous ECM attempts.

        Args:
            attempts: List of ECMAttempt objects with curves_completed, b1, b2, method
            starting_t_level: Base t-level to start from (e.g., prior work done before import)

        Returns:
            Current t-level achieved (includes starting_t_level), or None if it
            could not be calculated. On None, callers keep the composite's
            existing current_t_level. Writing starting_t_level instead would
            throw away the progress recorded so far.
        """
        if not attempts:
            return starting_t_level

        # Filter for ECM attempts only
        ecm_attempts = [a for a in attempts if a.method == 'ecm']

        if not ecm_attempts:
            return starting_t_level

        # Use external t-level software if available
        if self.t_level_available:
            return self._calculate_t_level_external(ecm_attempts, starting_t_level)
        else:
            return self._calculate_t_level_estimate(ecm_attempts, starting_t_level)

    def _calculate_t_level_external(self, attempts: list,
                                      starting_t_level: float = 0.0) -> Optional[float]:
        """Calculate t-level with the t-level software: in-process library or binary.

        Args:
            attempts: List of ECMAttempt objects
            starting_t_level: Base t-level to start from (the CLI's -w flag)

        Returns:
            Calculated t-level (includes starting_t_level), or None on failure.
        """
        try:
            logger.info(
                f"Processing {len(attempts)} attempts for t-level calculation "
                f"(starting_t_level={starting_t_level})"
            )
            aggregated_curves = self._aggregate_curve_classes(attempts)
            if not aggregated_curves:
                return starting_t_level

            if starting_t_level > 0 and not (
                t_level_library.MIN_STARTING_T_LEVEL
                <= starting_t_level
                <= t_level_library.MAX_STARTING_T_LEVEL
            ):
                # The CLI's -w only accepts 5-100. The server used to give up
                # on priors outside that range and keep just the prior, throwing
                # away every ECM attempt. Below 5, a prior is negligible work;
                # above 100, it cannot be expressed as curves. Either way the
                # combined t-level is at least the larger of the two parts, so
                # calculate the attempts alone and take the maximum.
                attempts_t_level = self._calculate_from_classes(aggregated_curves, 0.0)
                if attempts_t_level is None:
                    return None
                return max(attempts_t_level, starting_t_level)

            return self._calculate_from_classes(aggregated_curves, starting_t_level)
        except Exception as e:
            logger.warning(f"External t-level calculation failed: {e}")
            return None

    def _calculate_from_classes(self, aggregated_curves: CurveTotals,
                                starting_t_level: float) -> Optional[float]:
        if self.use_library:
            return self._calculate_t_level_library(aggregated_curves, starting_t_level)
        return self._calculate_t_level_binary(aggregated_curves, starting_t_level)

    def _aggregate_curve_classes(self, attempts: list) -> CurveTotals:
        """Total the curves of each (B1, B2, parametrization) class.

        The t-level of N batches at identical bounds equals one batch of the
        summed curves. ECM curves are independent trials with the same
        per-curve success probability, so P(no factor) is (1-p)^(total curves)
        however the curves are grouped. Collapsing keeps the calculation's input
        O(distinct bound-classes) instead of O(attempts); a composite hammered
        with hundreds of identical stage-1 batches would otherwise build a
        multi-KB curve string, all while the caller holds the composite row
        lock. B2 None (GMP-ECM default) and B2 0 (stage 1 only) are distinct
        work and never merge. Insertion order is the order the classes are
        combined in, as before.
        """
        aggregated_curves: CurveTotals = {}
        for attempt in attempts:
            if attempt.curves_completed <= 0:
                continue
            # Validate required fields
            if attempt.b1 is None or attempt.b1 <= 0:
                logger.error(
                    f"Skipping attempt with invalid B1: {attempt.b1} "
                    f"(attempt_id={attempt.id if hasattr(attempt, 'id') else 'unknown'})"
                )
                continue

            # Use actual parametrization from attempt, default to 3 if not set
            param = attempt.parametrization if attempt.parametrization is not None else 3
            key = (attempt.b1, attempt.b2, param)
            aggregated_curves[key] = aggregated_curves.get(key, 0) + attempt.curves_completed
        return aggregated_curves

    def _calculate_t_level_library(self, aggregated_curves: CurveTotals,
                                   starting_t_level: float) -> Optional[float]:
        """Calculate in-process with the t-level package (cached; see t_level_library)."""
        classes: List[t_level_library.CurveClass] = [
            (curves, b1, b2, param) for (b1, b2, param), curves in aggregated_curves.items()
        ]
        try:
            t_level = t_level_library.calculate_t_level(classes, starting_t_level)
        except sqlite3.Error:
            # The package's bundled probability database failed: an
            # infrastructure fault, not a property of this input. The binary
            # carries its own copy, so it can still answer.
            logger.exception(
                "t-level package database error; falling back to the t-level binary"
            )
            if self._binary_is_available():
                return self._calculate_t_level_binary(aggregated_curves, starting_t_level)
            return None
        except Exception:
            # Anything else depends on this input. The binary runs the same
            # calculation and would fail the same way, so retrying there would
            # only add a 200-1,400 ms process spawn to every recalculation.
            logger.exception(f"In-process t-level calculation failed for {classes}")
            return None

        if t_level is None:
            # Input the CLI would also have rejected (the library module logged why).
            return None

        work = f" (starting from t{starting_t_level})" if starting_t_level > 0 else ""
        logger.info(f"In-process t-level calculation{work}: {classes} -> t{t_level:.3f}")
        return t_level

    def _calculate_t_level_binary(self, aggregated_curves: CurveTotals,
                                  starting_t_level: float) -> Optional[float]:
        """Calculate by piping a curve string to the t-level binary (the fallback).

        Returns None on failure. Callers pass a non-empty aggregated_curves.
        """
        try:
            curve_strings = []
            for (b1, b2, param), curves in aggregated_curves.items():
                # Format: curves@B1[,B2],p=param
                # Format numbers to avoid decimals in scientific notation (use 11e7 not 1.1e+08)
                b1_str = self._format_number_for_tlevel(b1)

                # Omit B2 if None (let t-level use GMP-ECM default)
                if b2 is not None:
                    b2_str = self._format_number_for_tlevel(b2)
                    curve_strings.append(f"{curves}@{b1_str},{b2_str},p={param}")
                else:
                    # No B2 specified - let t-level binary use GMP-ECM defaults
                    curve_strings.append(f"{curves}@{b1_str},p={param}")

            # Join with semicolons for multiple entries
            input_string = ";".join(curve_strings)

            logger.info(
                f"Generated {len(curve_strings)} valid curve strings: {input_string}"
            )

            # Build command args - use -w flag if we have a starting t-level
            # Pass curve string via stdin (not -q flag) to avoid argument parsing issues
            if starting_t_level > 0:
                args = ["-w", str(starting_t_level)]
            else:
                args = []

            logger.info(
                f"Calling t-level binary with args: {args}, passing {len(input_string)} chars via stdin"
            )

            # Call external t-level calculator using executor
            # Curve string is passed via stdin (like: echo "curves" | t-level -w X)
            success, output = self.executor.execute_and_get_last_line(
                args=args,
                input_data=input_string,
                timeout=30
            )

            if not success or not output:
                return None

            # Parse output: expected format is "t45.185"
            if output.startswith('t'):
                t_level = float(output[1:])  # Remove 't' prefix and convert to float
                if starting_t_level > 0:
                    logger.info(f"External t-level calculation (starting from t{starting_t_level}): {input_string} -> {output}")
                else:
                    logger.info(f"External t-level calculation: {input_string} -> {output}")
                return t_level
            else:
                logger.warning(f"Unexpected t-level output format: {output}")
                return None

        except Exception as e:
            logger.warning(f"External t-level calculation failed: {e}")
            return None

    def _calculate_t_level_estimate(self, attempts: list,
                                     starting_t_level: float = 0.0) -> Optional[float]:
        """
        Fallback when neither the t-level package nor the binary is available.

        Returns None, meaning not calculated. Without the probability tables,
        any estimate would be misleading, and callers keep the existing value.
        """
        logger.warning("No t-level backend available (package or binary); t-level not calculated")
        return None

    def _curve_classes_for_composite(self, db, composite_id: int) -> List["CurveClassRow"]:
        """The composite's ECM work as one row per (B1, B2, parametrization) class.

        - Unsuperseded ECM attempts, summed per class.
        - Leftover stage-1 curves: a stage-1 attempt (B2 = 0) superseded by a
          stage 2 that completed fewer curves keeps the difference, credited at
          B1 only. Its parametrization is the stage 1's raw value (None becomes
          3 in _aggregate_curve_classes, as for real attempts; 0 is a real
          value). Superseded stage-2 attempts (orphans of a residue another
          client completed) earn no leftover: their curves are the same curves
          as the stage 1's.

        Rows come in order of each class's first attempt id. Attempts with no
        curves are left out here, as _aggregate_curve_classes would skip them;
        invalid B1s are kept so that it logs them.
        """
        from sqlalchemy import func
        from sqlalchemy.orm import aliased
        from ..models.attempts import ECMAttempt

        first_id = func.min(ECMAttempt.id)
        unsuperseded = db.query(
            first_id, ECMAttempt.b1, ECMAttempt.b2, ECMAttempt.parametrization,
            func.sum(ECMAttempt.curves_completed)
        ).filter(
            ECMAttempt.composite_id == composite_id,
            ECMAttempt.superseded_by.is_(None),
            ECMAttempt.method == 'ecm',
            ECMAttempt.curves_completed > 0,
        ).group_by(
            ECMAttempt.b1, ECMAttempt.b2, ECMAttempt.parametrization
        ).order_by(first_id).all()

        stage2 = aliased(ECMAttempt)
        leftover_curves = func.sum(ECMAttempt.curves_completed - stage2.curves_completed)
        leftovers = db.query(
            first_id, ECMAttempt.b1, ECMAttempt.parametrization, leftover_curves
        ).join(
            stage2, stage2.id == ECMAttempt.superseded_by
        ).filter(
            ECMAttempt.composite_id == composite_id,
            ECMAttempt.method == 'ecm',
            ECMAttempt.b2 == 0,
            stage2.curves_completed < ECMAttempt.curves_completed,
        ).group_by(
            ECMAttempt.b1, ECMAttempt.parametrization
        ).order_by(first_id).all()

        rows = [CurveClassRow(int(curves), b1, b2, param, attempt_id)
                for attempt_id, b1, b2, param, curves in unsuperseded]
        for attempt_id, b1, param, curves in leftovers:
            rows.append(CurveClassRow(int(curves), b1, 0, param, attempt_id))
            logger.info(
                f"Partial supersession on composite {composite_id}: {curves} leftover "
                f"stage-1 curves at B1={b1} (first stage 1 attempt {attempt_id}), "
                f"credited at B1-only"
            )
        return rows

    def calculate_composite_t_level(self, db, composite) -> Optional[float]:
        """The composite's t-level from its ECM work and prior_t_level, not stored.

        The one definition of a composite's t-level: unsuperseded attempts
        plus partial-supersession leftovers (_curve_classes_for_composite),
        starting from prior_t_level. Returns None if the calculation failed
        (already logged); callers then keep the stored value.
        """
        attempts = self._curve_classes_for_composite(db, composite.id)
        return self.get_current_t_level_from_attempts(attempts, composite.prior_t_level or 0.0)

    def recalculate_composite_t_level(self, db, composite) -> float:
        """
        Recalculate the t-level for a composite based on all its ECM attempts.

        Uses prior_t_level as the starting point (work done before import),
        then adds the t-level contribution from ECM attempts in this system.

        Excludes attempts that have been superseded (e.g., stage 1 attempts
        that were replaced by full stage 1+2 attempts).

        For partial supersession (stage 2 completed fewer curves than stage 1),
        credits leftover curves at B1-only.

        Two queries, whatever the number of attempts: the curve totals per
        (B1, B2, parametrization) class are summed in SQL.

        Args:
            db: Database session
            composite: Composite model instance

        Returns:
            Updated t-level value (includes prior_t_level)
        """
        starting_t_level = composite.prior_t_level or 0.0
        new_t_level = self.calculate_composite_t_level(db, composite)

        if new_t_level is None:
            # The calculation failed (already logged). Keep the last good value;
            # falling back to the prior would discard recorded progress and
            # make the composite look under-worked.
            logger.warning(
                f"T-level recalculation failed for composite {composite.id}; "
                f"keeping current t{composite.current_t_level}"
            )
            return composite.current_t_level

        # Update composite
        composite.current_t_level = new_t_level
        db.flush()  # Make changes visible within transaction

        if starting_t_level > 0:
            logger.info(f"Recalculated t-level for composite {composite.id}: t{new_t_level:.2f} (starting from prior t{starting_t_level:.2f})")
        else:
            logger.info(f"Recalculated t-level for composite {composite.id}: t{new_t_level:.2f}")
        return new_t_level

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
        # Defensive null checks
        if target_t_level is None or current_t_level is None:
            # Fallback to default parameters if t-levels are not set
            logger.warning(
                f"suggest_next_ecm_parameters called with None values: "
                f"target={target_t_level}, current={current_t_level}"
            )
            # Use default t-level 30 for the target if not set
            target_t_level = target_t_level or 30.0
            current_t_level = current_t_level or 0.0

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
