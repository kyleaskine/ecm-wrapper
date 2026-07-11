#!/usr/bin/env python3
"""
Aliquot Sequence Calculator using YAFU for factorization.

An aliquot sequence starting with n is defined as:
- a(0) = n
- a(k+1) = s(a(k)) where s(n) = σ(n) - n (sum of proper divisors)

The sequence terminates at 1, or may enter a cycle (sociable chain).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from lib.base_wrapper import BaseWrapper
from lib.ecm_math import trial_division, is_probably_prime, calculate_target_tlevel
from lib.ecm_config import TLevelConfig
from lib.ecm_executor import ECMWrapper
from lib.tracker_client import AliquotTrackerClient, TrackerSequence
from cado_wrapper import CADOWrapper
from yafu_wrapper import YAFUWrapper


class AliquotSequence:
    """Represents an aliquot sequence with tracking and cycle detection."""

    def __init__(self, start: int):
        self.start = start
        self.sequence: List[Optional[int]] = [start]
        self.factorizations: Dict[int, Dict[int, int]] = {}
        self.terminated = False
        self.cycle_start: Optional[int] = None
        self.cycle_length: Optional[int] = None

    def add_term(self, term: int, factorization: Dict[int, int]):
        """Add a term to the sequence with its factorization."""
        self.sequence.append(term)
        self.factorizations[term] = factorization

        # Check for cycles (excluding the first term)
        if term in self.sequence[:-1]:
            cycle_idx = self.sequence.index(term)
            self.cycle_start = cycle_idx
            self.cycle_length = len(self.sequence) - cycle_idx - 1
            self.terminated = True

    def check_termination(self) -> Tuple[bool, str]:
        """Check if sequence has terminated and return reason."""
        current = self.sequence[-1]

        if current == 1:
            return True, "terminated (reached 1)"

        if self.cycle_start is not None:
            cycle_terms = self.sequence[self.cycle_start:-1]
            return True, f"cyclic (period {self.cycle_length}): {' → '.join(map(str, cycle_terms))}"

        # Check if current term is prime (factorization has only one factor with exponent 1)
        if current in self.factorizations:
            factors = self.factorizations[current]
            if len(factors) == 1 and list(factors.values())[0] == 1:
                return True, f"terminated (prime: {current})"

        return False, ""


class AliquotWrapper(BaseWrapper):
    """Wrapper for computing aliquot sequences using CADO-NFS and ECM."""

    def __init__(self, config_path: str, factorizer: str = 'cado', hybrid_threshold: int = 100,
                 siqs_threshold: int = 100, ecm_program: str = 'gmp-ecm', threads: Optional[int] = None,
                 verbose: bool = False, use_two_stage: bool = False, max_batch_curves: Optional[int] = None,
                 progress_interval: int = 0, use_tracker: bool = False,
                 tracker_start: Optional[int] = None, small_method: str = 'siqs'):
        """Initialize aliquot wrapper with specified factorization engine.

        Args:
            config_path: Path to configuration file
            factorizer: Either 'cado' or 'hybrid' (default: 'cado')
            hybrid_threshold: Digit length below which ECM pretesting is skipped and the
                cofactor goes straight to the final method (default: 100)
            siqs_threshold: Digit length threshold selecting the final method: below it
                YAFU SIQS (or ECM to completion with small_method='ecm'), at/above it
                CADO-NFS (default: 100)
            ecm_program: ECM program to use: 'gmp-ecm' or 'yafu' (default: 'gmp-ecm')
            threads: Optional thread/worker count for parallel execution
            verbose: Enable verbose output from factorization programs
            use_two_stage: Use GPU two-stage mode for ECM (GPU stage 1 + CPU stage 2, default: False)
            max_batch_curves: Max curves per GPU batch in two-stage t-level mode (default: None = use config)
            progress_interval: Show progress every N completed curves (0 = disabled)
            use_tracker: Submit factors via the aliquot tracker (which forwards to
                FactorDB) and upload ECM t-level progress to the ECM server
            tracker_start: Start number of the sequence on the tracker (required
                when use_tracker is True)
            small_method: How to finish cofactors below siqs_threshold: 'siqs'
                (YAFU SIQS) or 'ecm' (GMP-ECM to completion, for platforms
                without YAFU such as macOS)
        """
        super().__init__(config_path)
        self.config_path = config_path  # Store for lazy initialization of sub-wrappers
        self.factorizer_name = factorizer
        self.hybrid_threshold = hybrid_threshold
        self.siqs_threshold = siqs_threshold
        self.small_method = small_method
        self.ecm_program = ecm_program
        self.threads = threads
        self.verbose = verbose
        self.use_two_stage = use_two_stage
        self.max_batch_curves = max_batch_curves
        self.progress_interval = progress_interval

        self.use_tracker = use_tracker
        self.tracker_start = tracker_start
        # ECM t-level upload rides tracker mode (kept as a separate switch so
        # the two concerns stay independently controllable)
        self.submit_ecm_results = use_tracker
        self.tracker: Optional[AliquotTrackerClient] = None
        if use_tracker:
            tracker_config = self.typed_config.aliquot_tracker
            if not tracker_config.url:
                raise ValueError(
                    "--tracker requires aliquot_tracker.url to be set in "
                    "client.yaml or client.local.yaml")
            self.tracker = AliquotTrackerClient(
                base_url=tracker_config.url,
                api_key=tracker_config.api_key,
                submitter=tracker_config.submitter or self.typed_config.client.username,
                timeout=self.typed_config.api.timeout,
                retry_attempts=self.typed_config.api.retry_attempts,
                logger=self.logger,
            )

        # Lazy initialization of factorizers (created on first access)
        self._cado = None
        self._ecm = None
        self._yafu = None

        # Set primary factorizer
        if factorizer == 'hybrid':
            self.factorizer = None  # Will be selected dynamically
        else:
            self.factorizer = self.cado

    @property
    def cado(self):
        """Lazy initialization of CADOWrapper."""
        if self._cado is None:
            self._cado = CADOWrapper(self.config_path)
        return self._cado

    @property
    def ecm(self):
        """Lazy initialization of ECMWrapper."""
        if self._ecm is None:
            self._ecm = ECMWrapper(self.config_path)
        return self._ecm

    @property
    def yafu(self):
        """Lazy initialization of YAFUWrapper."""
        if self._yafu is None:
            self._yafu = YAFUWrapper(self.config_path)
        return self._yafu

    def parse_factorization(self, factors_found: List[str]) -> Dict[int, int]:
        """
        Parse list of prime factors into a dictionary of {prime: exponent}.

        Args:
            factors_found: List of prime factors (may contain duplicates)

        Returns:
            Dictionary mapping prime to its exponent
        """
        if not factors_found:
            return {}

        # Count occurrences of each prime
        factor_counts = Counter(int(f) for f in factors_found)
        return dict(factor_counts)

    def calculate_divisor_sum(self, factorization: Dict[int, int]) -> int:
        """
        Calculate σ(n) - sum of all divisors including n.

        For n = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ:
        σ(n) = σ(p₁^a₁) × σ(p₂^a₂) × ... × σ(pₖ^aₖ)
        where σ(p^a) = (p^(a+1) - 1) / (p - 1)

        Args:
            factorization: Dictionary of {prime: exponent}

        Returns:
            Sum of all divisors
        """
        if not factorization:
            return 0

        sigma = 1
        for prime, exponent in factorization.items():
            # σ(p^a) = (p^(a+1) - 1) / (p - 1)
            sigma *= (prime**(exponent + 1) - 1) // (prime - 1)

        return sigma

    def calculate_next_term(self, n: int, factorization: Dict[int, int]) -> int:
        """
        Calculate next term in aliquot sequence: s(n) = σ(n) - n.

        Args:
            n: Current term
            factorization: Prime factorization of n

        Returns:
            Sum of proper divisors (next term in sequence)
        """
        sigma = self.calculate_divisor_sum(factorization)
        return sigma - n

    def factor_number(self, n: int) -> Tuple[bool, Dict[int, int], Dict]:
        """
        Factor a number completely using the hybrid factorization strategy.

        Strategy (always uses progressive approach, never jumps to SIQS/NFS):
        1. Trial division up to 10^7 (very fast, catches small factors)
        2. Progressive ECM in 3 phases (1/13, 2/13, 4/13 of digit length)
           - Each phase uses optimal B1 values from GMP-ECM plans
           - Stops early if fully factored or cofactor < hybrid_threshold
        3. CADO-NFS only if cofactor remains after ECM

        This ensures we ALWAYS attempt ECM before resorting to expensive
        SIQS or NFS methods.

        Args:
            n: Number to factor

        Returns:
            Tuple of (success, factorization_dict, raw_results)
        """
        digit_length = len(str(n))
        self.logger.info(f"Factoring {n} ({digit_length} digits)...")

        # Always use hybrid strategy (trial division + progressive ECM + CADO if needed)
        return self._factor_hybrid(n, digit_length)

    def _build_tlevel_config(self, composite: int, target_t_level: float,
                             start_t_level: float) -> TLevelConfig:
        """TLevelConfig for a progressive GMP-ECM run, honoring the wrapper's
        threading/GPU/batching settings and tracker-mode submission."""
        # Determine max_batch_curves: CLI arg > config value > None
        max_batch = self.max_batch_curves
        if max_batch is None:
            max_batch = self.typed_config.programs.gmp_ecm.max_batch

        return TLevelConfig(
            composite=str(composite),
            target_t_level=target_t_level,
            start_t_level=start_t_level,  # Continue from achieved t-level
            threads=self.threads if self.threads else 1,
            verbose=self.verbose,
            use_two_stage=self.use_two_stage,  # GPU two-stage mode if enabled
            max_batch_curves=max_batch,  # Enable batching for pipelined GPU/CPU execution
            progress_interval=self.progress_interval,
            project='Aliquot' if self.submit_ecm_results else None,
            # Tracker mode uploads t-level progress to the ECM server
            # (composites are registered there as aliquot:{start}:i{index});
            # otherwise aliquot handles its own submissions
            no_submit=not self.submit_ecm_results,
        )

    def _ecm_factor_completely(self, n: int, start_t_level: float,
                               already_found: List[str]) -> List[str]:
        """Factor a composite cofactor to completion with GMP-ECM alone.

        Used below siqs_threshold with small_method='ecm' (platforms without
        YAFU, e.g. macOS). The smallest prime factor of a d-digit composite
        has at most ceil(d/2) digits, so each pass targets a t-level just past
        that, escalating by 5 whenever a pass comes up empty - the miss
        probability shrinks geometrically, so this terminates.

        Args:
            n: Composite to factor (must not be prime)
            start_t_level: T-level already achieved on this cofactor
            already_found: Factors of the enclosing term found so far (for
                tracker mid-run sync)

        Returns:
            All prime factors of n (with multiplicity). Raises
            KeyboardInterrupt if ECM was interrupted.
        """
        current = n
        current_t = start_t_level
        factors: List[str] = []

        while current > 1:
            if is_probably_prime(current):
                factors.append(str(current))
                break

            digits = len(str(current))
            min_target = (digits + 1) // 2 + 2
            target = float(max(min_target, int(current_t) + 5))
            self.logger.info(f"ECM to completion: running to t{target:.1f} on C{digits}")

            if self.use_tracker:
                self._sync_tracker_factors(already_found + factors)

            result = self.ecm.run_tlevel_v2(
                self._build_tlevel_config(current, target, current_t))

            if result.curve_summary:
                result.print_curve_summary(show_parametrization=self.verbose)

            current_t = result.t_level_achieved

            if result.interrupted:
                self.logger.info("ECM was interrupted, stopping aliquot factorization")
                raise KeyboardInterrupt("ECM interrupted")

            for factor in result.factors:
                while current % int(factor) == 0:
                    current //= int(factor)
                    factors.append(factor)

            if result.factors and current > 1:
                self.logger.info(f"ECM to completion: cofactor C{len(str(current))} "
                                 f"remains (continuing from t{current_t:.2f})")

        return factors

    def _factor_hybrid(self, n: int, digit_length: int) -> Tuple[bool, Dict[int, int], Dict]:
        """
        Hybrid factorization: Trial division + ECM + SIQS/CADO-NFS.

        Strategy:
        1. Trial division up to 10^7 (very fast, catches small factors)
        2. ECM pretesting for cofactors >= hybrid_threshold digits (configurable
           via ecm_program):
           - GMP-ECM: Progressive approach with t-level targeting 4/13 of digit length
           - YAFU: Intelligent pretesting with -pretest flag
        3. Final factorization based on cofactor size (configurable via siqs_threshold):
           - Cofactors < siqs_threshold digits: YAFU SIQS, or GMP-ECM to
             completion with small_method='ecm' (default: 100)
           - CADO-NFS for larger cofactors >= siqs_threshold digits

        This approach optimizes factorization by using the best tool for each size range.

        Args:
            n: Number to factor
            digit_length: Number of digits in n (unused, kept for compatibility)

        Returns:
            Tuple of (success, factorization_dict, raw_results)
        """
        all_factors = []
        current_composite = n

        # Step 0: Trial division with small primes (very fast)
        self.logger.info(f"Running trial division up to 10^7...")
        trial_factors, current_composite = trial_division(current_composite)
        if trial_factors:
            self.logger.info(f"Trial division found {len(trial_factors)} small factor(s)")
            all_factors.extend([str(f) for f in trial_factors])

        if current_composite == 1:
            self.logger.info("Fully factored by trial division")
            factorization = self.parse_factorization(all_factors)
            return True, factorization, {'success': True, 'method': 'trial_division'}

        cofactor_digits = len(str(current_composite))
        self.logger.info(f"Cofactor after trial division: {current_composite} ({cofactor_digits} digits)")

        # Check if cofactor is prime before attempting ECM
        if is_probably_prime(current_composite):
            self.logger.info(f"Cofactor C{cofactor_digits} is prime, factorization complete")
            all_factors.append(str(current_composite))
            factorization = self.parse_factorization(all_factors)
            return True, factorization, {'success': True, 'method': 'trial_division+primality_test'}

        # Step 1: Progressive ECM (ALWAYS attempt ECM, regardless of size)
        cofactor_digits = len(str(current_composite))

        # Initialize ecm_factors and ecm_results for both branches
        ecm_factors: List[str] = []
        ecm_results: Dict[str, Any] = {}
        # Achieved t-level; carries into ECM-to-completion for small cofactors
        current_t_level = 0.0

        if self.ecm_program == 'yafu':
            # Use YAFU for ECM with intelligent pretesting
            self.logger.info(f"Running YAFU ECM pretest on C{cofactor_digits}")

            ecm_results = self.yafu.run_yafu_ecm(
                composite=str(current_composite),
                b1=None,  # Use -pretest (no explicit B1)
                method='ecm',
                verbose=self.verbose
            )

            # Collect YAFU ECM factors
            ecm_factors = ecm_results.get('factors_found', [])
            if ecm_factors:
                self.logger.info(f"YAFU ECM found {len(ecm_factors)} factor(s)")
                all_factors.extend(ecm_factors)

                # Calculate cofactor by dividing out found factors
                for factor_str in ecm_factors:
                    current_composite //= int(factor_str)

                if current_composite > 1:
                    self.logger.info(f"Cofactor after YAFU ECM: C{len(str(current_composite))}")
                else:
                    current_composite = 1
        else:
            # Use GMP-ECM's progressive approach with t-level (v2 API)
            # Keep running ECM until target is reached or cofactor is small enough for SIQS/CADO
            # Track achieved t-level to carry over when factor found (work done applies to cofactor too)
            while current_composite > 1:
                cofactor_digits = len(str(current_composite))

                # Check if cofactor is prime
                if is_probably_prime(current_composite):
                    self.logger.info(f"Cofactor C{cofactor_digits} is prime")
                    all_factors.append(str(current_composite))
                    current_composite = 1
                    break

                # Below hybrid_threshold, skip ECM pretesting - the final
                # method (SIQS/CADO/ECM-to-completion) takes it from here
                if cofactor_digits < self.hybrid_threshold:
                    self.logger.info(f"Cofactor C{cofactor_digits} below hybrid threshold, exiting ECM pretest")
                    break

                target_t_level = calculate_target_tlevel(cofactor_digits)

                # Skip if we've already reached the target for this size
                if current_t_level >= target_t_level:
                    self.logger.info(f"Already at t{current_t_level:.2f} >= target t{target_t_level:.1f}, moving to SIQS/CADO")
                    break

                self.logger.info(f"Running progressive GMP-ECM to t{target_t_level:.1f} on C{cofactor_digits}")

                # In tracker mode, push factors found so far (trial division,
                # earlier ECM passes) to the tracker BEFORE running ECM. The
                # tracker forwards them to FactorDB and re-registers the ECM
                # server's composite with the reduced cofactor, so the t-level
                # submissions below match a registered composite.
                if self.use_tracker:
                    self._sync_tracker_factors(all_factors)

                config = self._build_tlevel_config(current_composite, target_t_level, current_t_level)
                ecm_result = self.ecm.run_tlevel_v2(config)

                # Print curve summary for this ECM run
                if ecm_result.curve_summary:
                    ecm_result.print_curve_summary(show_parametrization=self.verbose)

                # Update achieved t-level (carries over to cofactor if factor found)
                current_t_level = ecm_result.t_level_achieved

                # Check if ECM was interrupted - propagate to caller
                if ecm_result.interrupted:
                    self.logger.info("ECM was interrupted, stopping aliquot factorization")
                    raise KeyboardInterrupt("ECM interrupted")

                # Collect ECM factors (all are guaranteed to be prime)
                ecm_factors = ecm_result.factors
                if ecm_factors:
                    self.logger.info(f"Progressive GMP-ECM found {len(ecm_factors)} prime factor(s)")

                    # Divide out found factors, recording one entry per
                    # division so p^k contributes k entries and the final
                    # product-vs-n verification holds
                    for factor in ecm_factors:
                        while current_composite % int(factor) == 0:
                            current_composite //= int(factor)
                            all_factors.append(factor)

                    if current_composite > 1:
                        self.logger.info(f"Cofactor after GMP-ECM: C{len(str(current_composite))} (continuing from t{current_t_level:.2f})")
                    else:
                        # Fully factored
                        self.logger.info(f"Fully factored by GMP-ECM")
                else:
                    # No factors found after reaching target t-level, exit ECM loop
                    self.logger.info(f"Reached t{target_t_level:.1f} with no factor, moving to SIQS/CADO")
                    break

        # Check if fully factored
        if current_composite == 1:
            self.logger.info(f"Fully factored by {'YAFU' if self.ecm_program == 'yafu' else 'GMP'} ECM")
            factorization = self.parse_factorization(all_factors)
            # Create dummy dict for backward compatibility (not used by caller)
            ecm_results_compat = {'factors_found': ecm_factors, 'success': True}
            return True, factorization, ecm_results_compat

        # Check if cofactor is prime before continuing
        cofactor_digits = len(str(current_composite))
        if is_probably_prime(current_composite):
            self.logger.info(f"Cofactor C{cofactor_digits} is prime, factorization complete")
            all_factors.append(str(current_composite))
            factorization = self.parse_factorization(all_factors)
            return True, factorization, ecm_results

        # Choose the final method by cofactor size: below siqs_threshold use
        # YAFU SIQS (or ECM to completion with --small-method ecm), otherwise
        # CADO-NFS
        if cofactor_digits < self.siqs_threshold and self.small_method == 'ecm':
            # Finish with GMP-ECM alone (no YAFU on this platform)
            self.logger.info(f"Cofactor is {cofactor_digits} digits (composite), running ECM to completion")
            completion_factors = self._ecm_factor_completely(
                current_composite, current_t_level, all_factors)
            all_factors.extend(completion_factors)
            final_results = {'success': True, 'method': 'ecm_completion'}
        elif cofactor_digits < self.siqs_threshold:
            # Use YAFU SIQS for smaller cofactors
            self.logger.info(f"Cofactor is {cofactor_digits} digits (composite), using YAFU SIQS")
            siqs_results = self.yafu.run_yafu_auto(
                composite=str(current_composite),
                method='siqs',
                verbose=self.verbose
            )

            # Check if SIQS succeeded
            if not siqs_results.get('success', False):
                self.logger.error(f"YAFU SIQS failed to factor C{cofactor_digits}")
                return False, {}, siqs_results

            # Parse the raw output to detect both primes and composites
            # YAFU SIQS can return composite factors that need further factorization
            from lib.parsing_utils import parse_yafu_output_with_composites
            raw_output = siqs_results.get('raw_output', '')
            parsed_factors = parse_yafu_output_with_composites(raw_output)

            prime_factors = parsed_factors.get('primes', [])
            composite_factors = parsed_factors.get('composites', [])

            if not prime_factors and not composite_factors:
                self.logger.error("YAFU SIQS succeeded but found no factors")
                return False, {}, siqs_results

            # Add prime factors
            if prime_factors:
                self.logger.info(f"YAFU SIQS found {len(prime_factors)} prime factor(s)")
                all_factors.extend(prime_factors)

            # Recursively factor any composite results
            if composite_factors:
                self.logger.warning(f"YAFU SIQS returned {len(composite_factors)} composite factor(s) - factoring recursively")
                for composite_factor in composite_factors:
                    self.logger.info(f"Recursively factoring C{len(composite_factor)} = {composite_factor}")
                    success, sub_factorization, _ = self.factor_number(int(composite_factor))

                    if not success:
                        self.logger.error(f"Failed to recursively factor C{len(composite_factor)}")
                        return False, {}, siqs_results

                    # Add all prime factors from the sub-factorization
                    for prime, exponent in sub_factorization.items():
                        for _ in range(exponent):
                            all_factors.append(str(prime))

            final_results = siqs_results
        else:
            # Use CADO-NFS for larger cofactors
            self.logger.info(f"Cofactor is {cofactor_digits} digits (composite), using CADO-NFS")
            cado_results = self.cado.run_cado_nfs(composite=str(current_composite), threads=self.threads, verbose=self.verbose)

            # Check if CADO succeeded
            if not cado_results.get('success', False):
                self.logger.error(f"CADO-NFS failed to factor C{cofactor_digits}")
                return False, {}, cado_results

            cado_factors = cado_results.get('factors_found', [])
            if cado_factors:
                all_factors.extend(cado_factors)
            else:
                self.logger.error("CADO-NFS succeeded but found no factors")
                return False, {}, cado_results

            final_results = cado_results

        factorization = self.parse_factorization(all_factors)
        self.logger.info(f"Final factorization: {self.format_factorization(factorization)}")

        # Verify factorization
        product = self._factorization_product(factorization)
        if product != n:
            self.logger.error(f"Factorization verification failed: {product} != {n}")
            return False, {}, final_results

        return True, factorization, final_results

    def format_factorization(self, factorization: Dict[int, int]) -> str:
        """Format factorization as string like '2^3 × 3 × 23'."""
        parts = []
        for prime in sorted(factorization.keys()):
            exp = factorization[prime]
            if exp == 1:
                parts.append(str(prime))
            else:
                parts.append(f"{prime}^{exp}")
        return " × ".join(parts)

    def compute_sequence(self, start: int, max_iterations: int = 100,
                        submit: bool = False) -> AliquotSequence:
        """
        Compute aliquot sequence starting from given number.

        Args:
            start: Starting number
            max_iterations: Maximum number of iterations
            submit: Whether to submit factorizations (via the tracker when
                --tracker is set, otherwise directly to FactorDB)

        Returns:
            AliquotSequence object with full sequence data
        """
        seq = AliquotSequence(start)

        # Factor the starting number
        success, factorization, _ = self.factor_number(start)
        if not success:
            self.logger.error("Failed to factor starting number")
            return seq

        seq.factorizations[start] = factorization

        # Compute sequence
        current = start
        for iteration in range(max_iterations):
            # Calculate next term
            next_term = self.calculate_next_term(current, seq.factorizations[current])

            self.logger.info(f"Iteration {iteration + 1}: {current} → {next_term}")
            print(f"\nStep {iteration + 1}:")
            print(f"  Current: {current}")
            print(f"  Factorization: {self.format_factorization(seq.factorizations[current])}")
            print(f"  σ({current}) = {self.calculate_divisor_sum(seq.factorizations[current])}")
            print(f"  Next term: {next_term}")

            # Check for termination before factoring next term
            if next_term == 0:
                self.logger.info("Sequence terminated (reached 0 - perfect number)")
                seq.terminated = True
                break

            if next_term == 1:
                seq.add_term(next_term, {1: 1})
                self.logger.info("Sequence terminated (reached 1)")
                seq.terminated = True
                break

            # Factor next term
            success, factorization, results = self.factor_number(next_term)
            if not success:
                self.logger.error(f"Failed to factor {next_term}, stopping sequence")
                seq.add_term(next_term, {})
                break

            # Submit factorization if requested
            if submit and factorization:
                self.submit_factors(next_term, factorization)

            # Add to sequence
            seq.add_term(next_term, factorization)

            # Check for cycles or other termination
            terminated, reason = seq.check_termination()
            if terminated:
                self.logger.info(f"Sequence {reason}")
                print(f"\n  Status: {reason}")
                break

            current = next_term
        else:
            self.logger.warning(f"Reached maximum iterations ({max_iterations})")
            print(f"\nReached maximum iterations ({max_iterations})")

        return seq

    def fetch_factordb_last_term(self, start: int) -> Optional[Tuple[int, int, str, Dict[int, int]]]:
        """
        Fetch the last known term from FactorDB for an aliquot sequence.

        Args:
            start: Starting number of the aliquot sequence

        Returns:
            Tuple of (iteration, composite, status, known_factors) or None if fetch failed
            - iteration: The sequence iteration number
            - composite: The full number at this iteration
            - status: FactorDB status ("C", "CF", "FF", "P", etc.)
            - known_factors: Dict of {prime: exponent} for known prime factors
        """
        import requests
        import re

        try:
            url = f"https://factordb.com/sequences.php?se=1&aq={start}&action=last&fr=0&to=100"
            self.logger.info(f"Fetching last known term from FactorDB for sequence {start}...")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            html = response.text

            # Parse iteration number: <td bgcolor="#DDDDDD">2157</td>
            iteration_match = re.search(r'<td bgcolor="#DDDDDD">(\d+)</td>', html)
            if not iteration_match:
                self.logger.warning("Could not find iteration number in FactorDB response")
                return None

            iteration = int(iteration_match.group(1))

            # Parse the number ID to fetch full composite
            id_match = re.search(r'id=(\d+).*?<font color="#\w+">(.*?)</font>', html)
            if not id_match:
                self.logger.warning("Could not find composite ID in FactorDB response")
                return None

            composite_id = id_match.group(1)

            # Fetch full number from FactorDB API
            api_url = f"https://factordb.com/api?id={composite_id}"
            self.logger.info(f"Fetching full number from FactorDB API (ID: {composite_id})...")

            # Use cookie for authenticated requests (may help with rate limiting)
            cookies = self._factordb_cookies()
            api_response = requests.get(api_url, cookies=cookies, timeout=30)
            api_response.raise_for_status()

            api_result = api_response.json()

            # Extract composite number and factor info from API response
            # API returns: {"id": "...", "status": "C"/"CF"/"FF", "factors": [[prime, exp], ...]}
            if 'factors' in api_result and api_result['factors']:
                status = api_result.get('status', 'C')

                # Reconstruct number and collect known factors
                composite = 1
                known_factors: Dict[int, int] = {}

                for factor_pair in api_result['factors']:
                    factor = int(factor_pair[0])
                    exponent = int(factor_pair[1])
                    composite *= factor ** exponent

                    # Check if this factor is a proven prime (use Miller-Rabin)
                    # For "CF" status, one factor will be composite (the cofactor)
                    if is_probably_prime(factor):
                        known_factors[factor] = exponent

                composite_str = str(composite)

                # Log what we found
                if status == "FF":
                    self.logger.info(f"FactorDB: Iteration {iteration} is FULLY FACTORED ({len(known_factors)} prime factors)")
                elif status == "CF" and known_factors:
                    cofactor = composite
                    for p, e in known_factors.items():
                        cofactor //= p ** e
                    self.logger.info(f"FactorDB: Iteration {iteration} has {len(known_factors)} known prime factors, "
                                   f"{len(str(cofactor))}-digit cofactor remains")
                else:
                    self.logger.info(f"FactorDB: Iteration {iteration} with {len(composite_str)}-digit composite (status={status})")

                return (iteration, composite, status, known_factors)
            else:
                self.logger.warning(f"Could not extract factors from FactorDB API (response: {api_result})")
                return None

        except requests.RequestException as e:
            self.logger.error(f"FactorDB fetch failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing FactorDB response: {e}")
            return None

    @staticmethod
    def _factorization_product(factorization: Dict[int, int]) -> int:
        """Reconstruct n from a {prime: exponent} factorization."""
        product = 1
        for prime, exp in factorization.items():
            product *= prime ** exp
        return product

    def _tracker_current_composite(self, state: 'TrackerSequence') -> Optional[int]:
        """The tracker's current composite as a positive int, or None when the
        payload is missing/malformed (never let a bad tracker response abort a
        multi-hour run - the caller falls back to direct FactorDB)."""
        if not state.current_composite:
            return None
        try:
            current = int(state.current_composite)
        except ValueError:
            self.logger.warning(
                f"Tracker returned a non-numeric composite for sequence "
                f"{self.tracker_start}: {state.current_composite[:40]!r}")
            return None
        if current <= 1:
            self.logger.warning(
                f"Tracker returned an invalid composite ({current}) for "
                f"sequence {self.tracker_start}")
            return None
        return current

    def submit_factors(self, n: int, factorization: Dict[int, int]) -> bool:
        """
        Submit a term's factorization to the configured destination.

        Tracker mode: submit via the aliquot tracker (which forwards to
        FactorDB and auto-advances the sequence), falling back to direct
        FactorDB submission if the tracker can't take the work. Otherwise:
        direct FactorDB submission.
        """
        if self.use_tracker:
            if self.submit_via_tracker(n, factorization):
                return True
            self.logger.warning("Tracker submission incomplete - falling back to direct FactorDB")
            print("  Tracker submission failed - falling back to direct FactorDB")
        return self.submit_to_factordb(n, factorization)

    def submit_via_tracker(self, n: int, factorization: Dict[int, int]) -> bool:
        """
        Submit the factorization of term n via the aliquot tracker.

        The tracker only accepts factors of ITS current composite, so factors
        are submitted one at a time, re-reading the tracker's state from each
        response (the current composite shrinks as FactorDB divides factors
        out). The final prime cofactor is never submitted - FactorDB derives
        it, flips the term to fully-factored, and the tracker auto-advances.

        Returns:
            True if the tracker handled the submission (advanced, or nothing
            new to submit); False if the caller should fall back to direct
            FactorDB submission.
        """
        if self.tracker is None or self.tracker_start is None:
            return False

        # Verify factorization before submitting anywhere
        product = self._factorization_product(factorization)
        if product != n:
            self.logger.error(f"Factor verification failed: {product} != {n}")
            return False

        state = self.tracker.get_sequence(self.tracker_start)
        if state is None:
            return False
        current = self._tracker_current_composite(state)
        if state.status != 'active' or current is None:
            self.logger.info(
                f"Tracker sequence {self.tracker_start} has no active composite "
                f"(status={state.status}) - nothing to submit via tracker")
            return False
        if n % current != 0:
            self.logger.warning(
                f"Tracker is on a different term (its C{len(str(current))} "
                f"at index {state.current_index} does not divide our term)")
            return False

        primes = [p for p, e in sorted(factorization.items()) for _ in range(e)]
        submitted, final_state = self._tracker_submit_new_factors(primes, state)
        if submitted is None:
            return False

        if final_state is not None and final_state.current_index > state.current_index:
            print(f"  Tracker: submitted {submitted} factor(s), sequence advanced "
                  f"to index {final_state.current_index}")
        else:
            print(f"  Tracker: submitted {submitted} factor(s)")
        return True

    def _tracker_submit_new_factors(
            self, primes: List[int],
            state: Optional['TrackerSequence'] = None
    ) -> Tuple[Optional[int], Optional['TrackerSequence']]:
        """
        Submit whichever of `primes` divide the tracker's current composite.

        One factor per request, smallest first, refreshing state between
        submissions. A prime equal to the current composite is never sent
        (the tracker rejects factor == composite; FactorDB resolves the final
        cofactor itself).

        Returns:
            (submitted_count, final_state) on success - success includes
            "nothing left to submit". (None, last_state) when the tracker
            was unreachable or rejected a factor.
        """
        assert self.tracker is not None and self.tracker_start is not None
        if state is None:
            state = self.tracker.get_sequence(self.tracker_start)
            if state is None:
                return None, None

        remaining = sorted(primes)
        submitted = 0
        while True:
            if state.status != 'active':
                return submitted, state
            current = self._tracker_current_composite(state)
            if current is None:
                return submitted, state
            candidate = next(
                (p for p in remaining if p != current and current % p == 0), None)
            if candidate is None:
                return submitted, state

            result = self.tracker.submit_factor(state.id, str(candidate))
            if not result.accepted:
                if result.permanent:
                    # The tracker validated the factor and said no (or our
                    # credentials were rejected) - retrying won't change it
                    self.logger.error(
                        f"Tracker rejected factor {candidate}: {result.error}")
                else:
                    self.logger.warning(
                        f"Tracker unavailable while submitting factor "
                        f"{candidate}: {result.error}")
                return None, state
            submitted += 1
            remaining.remove(candidate)
            self.logger.info(
                f"Tracker accepted factor {candidate} for sequence {self.tracker_start}")

            if result.auto_advanced:
                self.logger.info(
                    f"Tracker auto-advanced sequence {self.tracker_start}")
                return submitted, result.sequence

            if result.sequence is not None:
                state = result.sequence
            else:
                # Degraded response (tracker's FactorDB refresh failed after
                # accepting the factor) - re-fetch to see the updated state
                state = self.tracker.get_sequence(self.tracker_start)
                if state is None:
                    return None, None

    def _sync_tracker_factors(self, factors: List[str]) -> None:
        """Best-effort mid-run push of already-found factors to the tracker.

        Keeps FactorDB and the ECM server's composite registration in step
        with our working cofactor while long ECM runs are still in progress.
        Failures are ignored - the term-end submit_factors() is the safety net.
        """
        if self.tracker is None or self.tracker_start is None or not factors:
            return
        try:
            self._tracker_submit_new_factors([int(f) for f in factors])
        except Exception as e:
            # Never let a progress push kill a multi-hour factorization
            self.logger.warning(f"Tracker mid-run factor sync failed: {e}")

    def _factordb_cookies(self) -> Dict[str, str]:
        """FactorDB auth cookie from config (factordb.cookie), if set."""
        cookie = self.typed_config.factordb.cookie
        return {"fdbuser": cookie} if cookie else {}

    def submit_to_factordb(self, n: int, factorization: Dict[int, int]) -> bool:
        """
        Submit factorization to FactorDB using the reportfactor.php API.

        Only submits NEW factors that FactorDB doesn't already have.
        Skips the final cofactor (FactorDB will calculate it automatically).

        Args:
            n: Number that was factored
            factorization: Prime factorization as {prime: exponent}

        Returns:
            True if submission succeeded
        """
        import requests

        # Reconstruct number from factors to verify
        product = self._factorization_product(factorization)
        if product != n:
            self.logger.error(f"Factor verification failed: {product} != {n}")
            return False

        try:
            # Use cookie for authenticated requests
            cookies = self._factordb_cookies()

            # Step 1: Query FactorDB to see what factors they already have
            query_url = f"https://factordb.com/api?query={n}"
            query_response = requests.get(query_url, cookies=cookies, timeout=30)
            query_response.raise_for_status()
            fdb_data = query_response.json()

            # Parse existing factors from FactorDB
            # Response format: {"id": "...", "status": "C"/"CF"/"FF", "factors": [["prime", exp], ...]}
            existing_factors: Dict[int, int] = {}
            if 'factors' in fdb_data and fdb_data['factors']:
                for factor_pair in fdb_data['factors']:
                    prime = int(factor_pair[0])
                    exp = int(factor_pair[1])
                    existing_factors[prime] = existing_factors.get(prime, 0) + exp

            # Step 2: Determine NEW factors to submit (exclude largest prime - the final cofactor)
            sorted_primes = sorted(factorization.keys())
            largest_prime = sorted_primes[-1] if sorted_primes else None

            new_factors_to_submit = {}
            for prime, exp in factorization.items():
                # Skip the largest prime (final cofactor - FactorDB will calculate it)
                if prime == largest_prime:
                    continue

                # Only submit if FactorDB doesn't have this factor yet
                existing_exp = existing_factors.get(prime, 0)
                if existing_exp < exp:
                    # Submit the missing occurrences
                    new_factors_to_submit[prime] = exp - existing_exp

            if not new_factors_to_submit:
                self.logger.info(f"FactorDB: Already has all factors for {n} ({len(str(n))} digits)")
                print(f"  FactorDB: Already has all factors")
                print(f"  View at: https://factordb.com/index.php?query={n}")
                return True

            # Step 3: Submit only the NEW factors
            self.logger.info(f"FactorDB: Submitting {sum(new_factors_to_submit.values())} new factor(s) for {n} ({len(str(n))} digits)")
            submission_url = "https://factordb.com/reportfactor.php"
            success_count = 0

            failed_factors = []
            for prime, exp in sorted(new_factors_to_submit.items()):
                # Submit each occurrence of this prime factor
                for occurrence in range(exp):
                    submitted = False
                    last_error = None

                    # Retry up to 3 times with exponential backoff
                    for attempt in range(3):
                        try:
                            form_data = {
                                "number": str(n),
                                "factor": str(prime)
                            }

                            response = requests.post(
                                submission_url,
                                data=form_data,
                                cookies=cookies,
                                timeout=30
                            )
                            response.raise_for_status()
                            success_count += 1
                            submitted = True
                            if attempt > 0:
                                self.logger.info(f"FactorDB: Submitted factor {prime} for {n} (succeeded on retry {attempt+1})")
                            else:
                                self.logger.debug(f"FactorDB: Submitted factor {prime} for {n}")
                            break  # Success, exit retry loop
                        except requests.RequestException as factor_err:
                            last_error = factor_err
                            if attempt < 2:  # Don't sleep after last attempt
                                import time
                                wait_time = 2 ** attempt  # 1s, 2s exponential backoff
                                self.logger.warning(f"FactorDB: Retry {attempt+1}/3 failed for factor {prime} (occurrence {occurrence+1}/{exp}): {factor_err}. Retrying in {wait_time}s...")
                                time.sleep(wait_time)

                    if not submitted:
                        failed_factors.append((prime, str(last_error)))
                        self.logger.error(f"FactorDB: Failed to submit factor {prime} after 3 attempts (occurrence {occurrence+1}/{exp}): {last_error}")

            if failed_factors:
                self.logger.warning(f"FactorDB: Partial submission - {success_count} succeeded, {len(failed_factors)} failed for {n}")
                print(f"  FactorDB: WARNING - {success_count} factor(s) submitted, {len(failed_factors)} failed")
                print(f"  View at: https://factordb.com/index.php?query={n}")
                return False

            self.logger.info(f"FactorDB: Successfully submitted {success_count} factor(s) for {n} - https://factordb.com/index.php?query={n}")
            print(f"  FactorDB: Submitted {success_count} NEW factor(s)")
            print(f"  View at: https://factordb.com/index.php?query={n}")

            return True

        except requests.RequestException as e:
            self.logger.error(f"FactorDB submission failed for {n}: {e}")
            print(f"  Error: Failed to submit to FactorDB - {e}")
            return False
        except Exception as e:
            self.logger.error(f"FactorDB submission unexpected error for {n}: {e}")
            print(f"  Error: Unexpected error submitting to FactorDB - {e}")
            return False

    def save_sequence(self, seq: AliquotSequence, output_file: Optional[Path] = None) -> Path:
        """
        Save sequence data to JSON file in data/aliquot_sequences/.

        Args:
            seq: AliquotSequence to save
            output_file: Optional output path

        Returns:
            Path where sequence was saved
        """
        if output_file is None:
            output_dir = Path("data/aliquot_sequences")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"aliquot_{seq.start}_{timestamp}.json"
        else:
            # Ensure it's in data/ directory
            if not str(output_file).startswith('data/'):
                output_file = Path("data/aliquot_sequences") / output_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'start': seq.start,
            'sequence': seq.sequence,
            'length': len(seq.sequence),
            'factorizations': {
                str(n): {str(p): e for p, e in factors.items()}
                for n, factors in seq.factorizations.items()
            },
            'terminated': seq.terminated,
            'cycle_start': seq.cycle_start,
            'cycle_length': seq.cycle_length,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Sequence saved to {output_file}")
        return output_file

    def cleanup_temp_files(self):
        """Clean up temporary files created by CADO-NFS."""
        import glob

        # CADO-NFS working directory files (if run from client/)
        cado_temp_patterns = [
            'cado-nfs.*',
            '*.poly',
            '*.roots*'
        ]

        cleaned_files = []
        for pattern in cado_temp_patterns:
            for filepath in glob.glob(pattern):
                try:
                    Path(filepath).unlink()
                    cleaned_files.append(filepath)
                except Exception as e:
                    self.logger.debug(f"Could not remove {filepath}: {e}")

        if cleaned_files:
            self.logger.info(f"Cleaned up {len(cleaned_files)} temporary file(s): {', '.join(cleaned_files)}")

        return cleaned_files

    def print_summary(self, seq: AliquotSequence):
        """Print summary of the sequence."""
        print("\n" + "="*80)
        print("ALIQUOT SEQUENCE SUMMARY")
        print("="*80)
        print(f"Starting number: {seq.start}")
        print(f"Sequence length: {len(seq.sequence)}")
        print(f"Sequence: {' → '.join(map(str, seq.sequence[:10]))}")
        if len(seq.sequence) > 10:
            print(f"          ... ({len(seq.sequence) - 10} more terms)")

        terminated, reason = seq.check_termination()
        if terminated:
            print(f"Status: {reason.capitalize()}")
        else:
            print(f"Status: Open (not yet terminated)")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate aliquot sequences using CADO-NFS and ECM for factorization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate aliquot sequence starting from 276
  python3 aliquot_wrapper.py --start 276

  # Use pure CADO-NFS for all factorizations
  python3 aliquot_wrapper.py --start 276 --factorizer cado

  # Use hybrid mode (ECM + CADO-NFS) for large numbers
  python3 aliquot_wrapper.py --start 276 --factorizer hybrid

  # Use YAFU for ECM pretesting (faster than GMP-ECM)
  python3 aliquot_wrapper.py --start 276 --ecm-program yafu

  # Use YAFU SIQS for numbers < 100 digits (default)
  python3 aliquot_wrapper.py --start 276 --siqs-threshold 100

  # No YAFU (e.g. macOS): finish small cofactors with GMP-ECM instead of SIQS
  python3 aliquot_wrapper.py --start 276 --small-method ecm

  # Calculate with more iterations
  python3 aliquot_wrapper.py --start 1248 --max-iterations 50

  # Submit results to FactorDB
  python3 aliquot_wrapper.py --start 138 --factordb

  # Submit results via the aliquot tracker (forwards to FactorDB, advances the
  # tracked sequence, and uploads ECM t-level progress to the ECM server)
  python3 aliquot_wrapper.py --start 138 --tracker --resume-factordb

  # Quiet mode (no factor spam)
  python3 aliquot_wrapper.py --start 276 --quiet-factors

  # Resume from FactorDB (fetches last known term automatically)
  python3 aliquot_wrapper.py --start 276 --resume-factordb --quiet-factors

  # Manual resume from specific iteration
  python3 aliquot_wrapper.py --start 276 --resume-iteration 2157 --resume-composite 175258998...

  # Use 8 threads/workers for parallel execution
  python3 aliquot_wrapper.py --start 276 --workers 8 --quiet-factors

  # Use GPU two-stage mode for faster ECM (requires GMP-ECM with GPU support)
  python3 aliquot_wrapper.py --start 276 --two-stage --workers 8 --quiet-factors
  python3 aliquot_wrapper.py --start 276 --gpu --workers 8  # alias for --two-stage

  # Verbose mode (show detailed output from ECM and CADO-NFS)
  python3 aliquot_wrapper.py --start 276 -v --workers 8

  # Show progress every 100 curves
  python3 aliquot_wrapper.py --start 276 --two-stage --workers 8 --progress-interval 100

Common test sequences:
  276 → 396 → 696 → 1104 → 1872 → 3770 → ... (terminates at 1)
  220 → 284 → 220 (amicable pair, cycle of length 2)
  138 → long open sequence
        """
    )

    parser.add_argument('--start', type=int, required=True,
                       help='Starting number for the aliquot sequence')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Maximum number of iterations (default: 100)')
    parser.add_argument('--config', type=str, default='client.yaml',
                       help='Configuration file path (default: client.yaml)')
    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument('--factordb', action='store_true',
                       help='Submit factorizations directly to FactorDB')
    submit_group.add_argument('--tracker', action='store_true',
                       help='Submit factorizations via the aliquot tracker (forwards to '
                            'FactorDB, auto-advances the sequence) and upload ECM t-level '
                            'progress to the ECM server. Requires aliquot_tracker.url in config. '
                            'Falls back to direct FactorDB submission if the tracker is unavailable.')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for sequence data')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save sequence to file')
    parser.add_argument('--quiet-factors', action='store_true',
                       help='Disable factor logging to factors_found.txt (reduces spam for aliquot sequences)')
    parser.add_argument('--factorizer', type=str, choices=['cado', 'hybrid'], default='hybrid',
                       help='Factorization strategy: cado (pure CADO-NFS) or hybrid (default: hybrid - uses ECM+CADO for large numbers)')
    parser.add_argument('--hybrid-threshold', type=int, default=100,
                       help='Skip ECM pretesting for cofactors below this many digits - they go '
                            'straight to the final method (default: 100)')
    parser.add_argument('--siqs-threshold', type=int, default=100,
                       help='Final-method selector: cofactors below this many digits use YAFU SIQS '
                            '(or ECM with --small-method ecm), larger ones use CADO-NFS (default: 100)')
    parser.add_argument('--small-method', type=str, choices=['siqs', 'ecm'], default='siqs',
                       help='How to finish cofactors below --siqs-threshold: siqs (YAFU) or ecm '
                            '(GMP-ECM to completion; use on platforms without YAFU, e.g. macOS) '
                            '(default: siqs)')
    parser.add_argument('--ecm-program', type=str, choices=['gmp-ecm', 'yafu'], default='gmp-ecm',
                       help='ECM program: gmp-ecm (progressive t-level) or yafu (intelligent pretest) (default: gmp-ecm)')
    parser.add_argument('--resume-factordb', action='store_true',
                       help='Resume from last known term in FactorDB')
    parser.add_argument('--resume-iteration', type=int,
                       help='Resume from specific iteration with composite given via --resume-composite')
    parser.add_argument('--resume-composite', type=str,
                       help='Composite number to resume from (use with --resume-iteration)')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (ECM: stage2 threads or multiprocess workers, YAFU/CADO: threads)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Use GPU two-stage mode for ECM (GPU stage 1 + CPU stage 2, requires GMP-ECM with GPU support)')
    parser.add_argument('--gpu', action='store_true',
                       help='Alias for --two-stage (use GPU acceleration for ECM)')
    parser.add_argument('--max-batch', type=int,
                       help='Max curves per GPU batch in two-stage t-level mode (enables chunking for earlier factor discovery)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output from factorization programs (ECM, CADO-NFS)')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Show progress updates every N completed curves (0 = disabled)')

    args = parser.parse_args()

    # Normalize GPU flag to two-stage
    if args.gpu:
        args.two_stage = True

    # Initialize wrapper with selected factorizer
    try:
        wrapper = AliquotWrapper(args.config, factorizer=args.factorizer, hybrid_threshold=args.hybrid_threshold,
                                siqs_threshold=args.siqs_threshold, ecm_program=args.ecm_program,
                                threads=args.workers, verbose=args.verbose, use_two_stage=args.two_stage,
                                max_batch_curves=args.max_batch, progress_interval=args.progress_interval,
                                use_tracker=args.tracker, tracker_start=args.start,
                                small_method=args.small_method)
    except ValueError as config_error:
        print(f"Error: {config_error}")
        sys.exit(1)

    # Any submission destination enabled?
    submit_enabled = args.factordb or args.tracker

    # In tracker mode, show the tracker's view of the sequence up front so a
    # misconfigured URL or untracked sequence is visible before hours of work
    if args.tracker and wrapper.tracker is not None:
        tracker_state = wrapper.tracker.get_sequence(args.start)
        if tracker_state is None:
            print(f"Warning: tracker unreachable or sequence {args.start} not tracked.")
            print(f"         Factor submissions will fall back to direct FactorDB.")
        else:
            print(f"Tracker: sequence {args.start} at index {tracker_state.current_index} "
                  f"(status={tracker_state.status}, "
                  f"C{len(tracker_state.current_composite) if tracker_state.current_composite else '?'})")

    # Override factor logging config if requested
    if args.quiet_factors:
        wrapper.typed_config.logging.log_factors_found = False
        wrapper.cado.typed_config.logging.log_factors_found = False
        wrapper.ecm.typed_config.logging.log_factors_found = False
        wrapper.yafu.typed_config.logging.log_factors_found = False

    print(f"\nComputing aliquot sequence starting from {args.start}")
    print("="*80)

    # Handle resume options
    resume_iteration = None
    resume_composite = None

    # Track known factors from FactorDB (used to skip redundant factorization)
    fdb_known_factors: Dict[int, int] = {}
    fdb_status: Optional[str] = None

    if args.resume_factordb:
        # Warn if --resume-iteration is also specified (it will be ignored)
        if args.resume_iteration is not None:
            print(f"Warning: --resume-iteration is ignored with --resume-factordb (fetches latest term)")
            print(f"         To resume from a specific iteration, use: --resume-iteration N --resume-composite X")
            print()

        # Fetch last known term from FactorDB (now includes known factors)
        result = wrapper.fetch_factordb_last_term(args.start)
        if result:
            resume_iteration, resume_composite, fdb_status, fdb_known_factors = result
            print(f"Resuming from FactorDB: iteration {resume_iteration}, {len(str(resume_composite))}-digit number (status={fdb_status})")
            if fdb_known_factors:
                print(f"  FactorDB already knows {len(fdb_known_factors)} prime factor(s)")
                for p, e in sorted(fdb_known_factors.items()):
                    if e > 1:
                        print(f"    {p}^{e}")
                    else:
                        print(f"    {p}")
        else:
            print("Failed to fetch from FactorDB. Exiting.")
            print("To start from scratch, run without --resume-factordb flag.")
            wrapper.cleanup_temp_files()
            sys.exit(1)
    elif args.resume_iteration is not None and args.resume_composite:
        # Manual resume
        resume_iteration = args.resume_iteration
        resume_composite = int(args.resume_composite)
        print(f"Resuming from manual input: iteration {resume_iteration}, {len(str(resume_composite))}-digit composite")

    # Initialize sequence appropriately
    interrupted = False
    seq = None

    try:
        if resume_iteration is not None and resume_composite is not None:
            # Create sequence starting at resume point
            seq = AliquotSequence(args.start)
            # Mark iterations up to resume point as already done
            for i in range(resume_iteration):
                seq.sequence.append(None)  # Placeholder for unknown intermediates
            seq.sequence.append(resume_composite)

            # Factor the resume composite (using known factors from FactorDB if available)
            if fdb_status == "FF" and fdb_known_factors:
                # FactorDB says it's fully factored - use their factors directly
                print(f"  Using FactorDB's complete factorization (no local factoring needed)")
                factorization = fdb_known_factors.copy()
                success = True
            elif fdb_status == "CF" and fdb_known_factors:
                # FactorDB has partial factorization - only factor the cofactor
                cofactor = resume_composite
                for p, e in fdb_known_factors.items():
                    cofactor //= p ** e

                if is_probably_prime(cofactor):
                    # Cofactor is prime - we're done!
                    print(f"  FactorDB's cofactor C{len(str(cofactor))} is prime - no factoring needed")
                    factorization = fdb_known_factors.copy()
                    factorization[cofactor] = 1
                    success = True
                else:
                    # Factor only the cofactor
                    print(f"  Factoring only the {len(str(cofactor))}-digit cofactor (FactorDB has {len(fdb_known_factors)} known factors)")
                    success, cofactor_factorization, _ = wrapper.factor_number(cofactor)
                    if success:
                        # Combine known factors with cofactor factorization
                        factorization = fdb_known_factors.copy()
                        for p, e in cofactor_factorization.items():
                            if p in factorization:
                                factorization[p] += e
                            else:
                                factorization[p] = e
                    else:
                        factorization = {}
            else:
                # No useful info from FactorDB - factor from scratch
                success, factorization, _ = wrapper.factor_number(resume_composite)

            if success:
                seq.factorizations[resume_composite] = factorization

                # Submit the resume composite factorization first
                # This allows FactorDB to calculate the next term and maintain sequence linkage
                if submit_enabled and factorization:
                    wrapper.submit_factors(resume_composite, factorization)

                # Continue from this point
                current = resume_composite
                for iteration in range(args.max_iterations):
                    next_term = wrapper.calculate_next_term(current, seq.factorizations[current])

                    wrapper.logger.info(f"Iteration {resume_iteration + iteration + 1}: {current} → {next_term}")
                    print(f"\nStep {resume_iteration + iteration + 1}:")
                    print(f"  Current: {current}")
                    print(f"  Factorization: {wrapper.format_factorization(seq.factorizations[current])}")
                    print(f"  σ({current}) = {wrapper.calculate_divisor_sum(seq.factorizations[current])}")
                    print(f"  Next term: {next_term}")

                    if next_term == 0 or next_term == 1:
                        seq.add_term(next_term, {1: 1} if next_term == 1 else {})
                        seq.terminated = True
                        break

                    # Factor next term
                    success, factorization, results = wrapper.factor_number(next_term)
                    if not success:
                        wrapper.logger.error(f"Failed to factor {next_term}, stopping")
                        seq.add_term(next_term, {})
                        break

                    # Submit factorization
                    if submit_enabled and factorization:
                        wrapper.submit_factors(next_term, factorization)

                    seq.add_term(next_term, factorization)

                    terminated, reason = seq.check_termination()
                    if terminated:
                        wrapper.logger.info(f"Sequence {reason}")
                        print(f"\n  Status: {reason}")
                        break

                    current = next_term
            else:
                print("Failed to factor resume composite")
                wrapper.cleanup_temp_files()
                sys.exit(1)
        else:
            # Normal computation from start
            seq = wrapper.compute_sequence(
                start=args.start,
                max_iterations=args.max_iterations,
                submit=submit_enabled
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        wrapper.logger.info("Aliquot sequence computation interrupted by user")
        interrupted = True
        # Create minimal sequence if none exists
        if seq is None:
            seq = AliquotSequence(args.start)

    # Print summary (even if interrupted)
    if seq is not None:
        wrapper.print_summary(seq)

        # Save sequence unless disabled
        if not args.no_save:
            output_path = Path(args.output) if args.output else None
            saved_path = wrapper.save_sequence(seq, output_path)
            print(f"\nSequence saved to: {saved_path}")

    # Clean up temporary files created by YAFU/CADO
    wrapper.cleanup_temp_files()

    # Exit with appropriate code
    if interrupted:
        print("\nExiting due to interrupt.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    sys.exit(0 if (seq and seq.terminated) else 1)


if __name__ == '__main__':
    main()
