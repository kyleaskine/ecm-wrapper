"""
T-level calculation service for ECM factorization targets.

Integrates with existing t-level software and provides target calculations
based on composite size and special form detection.
"""
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.process_executor import ExternalProgramExecutor

logger = logging.getLogger(__name__)

class TLevelCalculator:
    """Calculate target t-levels and ECM parameters for composites."""

    def __init__(self):
        # Get t-level binary path from config
        from ..config import get_settings
        settings = get_settings()
        self.T_LEVEL_BINARY = settings.t_level_binary_path

        # Initialize executor for t-level binary
        self.executor = ExternalProgramExecutor(
            self.T_LEVEL_BINARY,
            binary_name="t-level"
        )
        self.t_level_available = self.executor.check_binary_exists()

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
                                          starting_t_level: float = 0.0) -> float:
        """
        Calculate current t-level achieved from previous ECM attempts.

        Args:
            attempts: List of ECMAttempt objects with curves_completed, b1, b2, method
            starting_t_level: Base t-level to start from (e.g., prior work done before import)

        Returns:
            Current t-level achieved (includes starting_t_level)
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
                                      starting_t_level: float = 0.0) -> float:
        """Calculate t-level using external t-level software.

        Args:
            attempts: List of ECMAttempt objects
            starting_t_level: Base t-level to start from (uses -w flag)

        Returns:
            Calculated t-level (includes starting_t_level)
        """
        try:
            # Convert attempts to curve string format for t-level executable
            logger.info(
                f"Processing {len(attempts)} attempts for t-level calculation "
                f"(starting_t_level={starting_t_level})"
            )

            # Aggregate curves by (B1, B2, parametrization) before formatting.
            # The t-level of N batches at identical bounds equals one batch of
            # the summed curves - ECM curves are independent trials with the
            # same per-curve success probability, so P(no factor) is
            # (1-p)^(total curves) regardless of how the curves are grouped.
            # Collapsing keeps the binary's input O(distinct bound-classes)
            # instead of O(attempts): a composite hammered with hundreds of
            # identical stage-1 batches otherwise builds a multi-KB curve
            # string that the binary must parse - all while the caller holds
            # the composite row lock. B2 None (GMP-ECM default) and B2 0
            # (stage 1 only) are distinct work and never merge.
            aggregated_curves: Dict[tuple, int] = {}
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

            if not curve_strings:
                return starting_t_level

            # Join with semicolons for multiple entries
            input_string = ";".join(curve_strings)

            # Defensive check: ensure input_string is not empty or just semicolons
            if not input_string or not input_string.strip() or input_string.strip(";").strip() == "":
                logger.error(
                    f"Generated empty curve string from {len(curve_strings)} curve_strings "
                    f"(curve_strings={curve_strings}), returning starting t-level {starting_t_level}"
                )
                return starting_t_level

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
                return starting_t_level

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
                return starting_t_level

        except Exception as e:
            logger.warning(f"External t-level calculation failed: {e}")
            return starting_t_level

    def _calculate_t_level_estimate(self, attempts: list,
                                     starting_t_level: float = 0.0) -> float:
        """
        Fallback t-level estimation when external calculator unavailable.

        Returns starting_t_level for now - proper calculation requires the external t-level binary.
        """
        # Without proper probability tables, any estimate would be misleading
        # Better to return starting_t_level and rely on the external calculator
        logger.info("Using fallback t-level estimation (returns starting_t_level)")
        return starting_t_level

    def recalculate_composite_t_level(self, db, composite) -> float:
        """
        Recalculate the t-level for a composite based on all its ECM attempts.

        Uses prior_t_level as the starting point (work done before import),
        then adds the t-level contribution from ECM attempts in this system.

        Excludes attempts that have been superseded (e.g., stage 1 attempts
        that were replaced by full stage 1+2 attempts).

        For partial supersession (stage 2 completed fewer curves than stage 1),
        credits leftover curves at B1-only.

        Args:
            db: Database session
            composite: Composite model instance

        Returns:
            Updated t-level value (includes prior_t_level)
        """
        from ..models.attempts import ECMAttempt
        from ..models.residues import ECMResidue

        from sqlalchemy.orm import defer

        # Get all ECM attempts for this composite, excluding superseded ones
        # (defer raw_output: it can be ~1MB per row and is never read here)
        attempts = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output)).filter(
            ECMAttempt.composite_id == composite.id,
            ECMAttempt.superseded_by.is_(None)  # Exclude superseded attempts
        ).all()

        # Create synthetic "attempts" for leftover curves (B1-only credit)
        class LeftoverCurves:
            """Synthetic attempt object for leftover curves from partial stage 2."""
            def __init__(self, curves, b1, parametrization, source_desc):
                self.curves_completed = curves
                self.b1 = b1
                self.b2 = 0  # B1-only (no stage 2)
                self.parametrization = parametrization
                self.method = 'ecm'
                self._source = source_desc  # For logging

        # Find partial supersession cases from ATTEMPTS directly
        # This handles cases where stage 2 completed fewer curves than stage 1,
        # regardless of whether residue records exist
        superseded_attempts = db.query(ECMAttempt).options(defer(ECMAttempt.raw_output)).filter(
            ECMAttempt.composite_id == composite.id,
            ECMAttempt.superseded_by.isnot(None),  # Has been superseded
            ECMAttempt.method == 'ecm'
        ).all()

        for stage1 in superseded_attempts:
            # Find the stage 2 attempt that superseded this one
            stage2 = db.query(ECMAttempt).filter(
                ECMAttempt.id == stage1.superseded_by
            ).first()

            if stage2 and stage2.curves_completed < stage1.curves_completed:
                # Partial completion - stage 2 did fewer curves than stage 1
                leftover = stage1.curves_completed - stage2.curves_completed
                leftover_attempt = LeftoverCurves(
                    leftover,
                    stage1.b1,
                    # Preserve the real parametrization: 0 is a valid value, so
                    # `or 3` collapsed genuine param-0 leftovers into 3 - costing
                    # them with the wrong probability model and (post-aggregation)
                    # merging them into the p=3 bucket. None normalizes to 3
                    # downstream, same as real attempts; the residue-leftover path
                    # below already passes the raw value.
                    stage1.parametrization,
                    f"attempt {stage1.id}"
                )
                attempts.append(leftover_attempt)
                logger.info(
                    f"Partial supersession: stage1 attempt {stage1.id} had {stage1.curves_completed} curves, "
                    f"stage2 attempt {stage2.id} completed {stage2.curves_completed}, "
                    f"adding {leftover} leftover at B1-only"
                )

        # Also check completed residues for additional context (may have different curve counts)
        completed_residues = db.query(ECMResidue).filter(
            ECMResidue.composite_id == composite.id,
            ECMResidue.status == 'completed',
            ECMResidue.stage1_attempt_id.isnot(None)
        ).all()

        # Track which stage1 attempts we've already handled via direct supersession check
        handled_stage1_ids = {a.id for a in superseded_attempts}

        for residue in completed_residues:
            # Skip if we already handled this stage1 attempt above
            if residue.stage1_attempt_id in handled_stage1_ids:
                continue

            # Find the stage 1 attempt
            stage1 = db.query(ECMAttempt).filter(
                ECMAttempt.id == residue.stage1_attempt_id
            ).first()

            if stage1 and stage1.superseded_by:
                # Find the stage 2 attempt that superseded it
                stage2 = db.query(ECMAttempt).filter(
                    ECMAttempt.id == stage1.superseded_by
                ).first()

                # Use residue.curve_count which may differ from stage1.curves_completed
                if stage2 and stage2.curves_completed < residue.curve_count:
                    # Partial completion - calculate leftover curves
                    leftover = residue.curve_count - stage2.curves_completed
                    leftover_attempt = LeftoverCurves(
                        leftover,
                        residue.b1,
                        residue.parametrization,
                        f"residue {residue.id}"
                    )
                    attempts.append(leftover_attempt)
                    logger.info(
                        f"Partial supersession: residue {residue.id} had {residue.curve_count} curves, "
                        f"stage2 completed {stage2.curves_completed}, adding {leftover} leftover at B1-only"
                    )

        # Use prior_t_level as starting point (work done before import)
        starting_t_level = composite.prior_t_level or 0.0

        # Calculate new t-level, starting from prior work
        new_t_level = self.get_current_t_level_from_attempts(attempts, starting_t_level)

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
