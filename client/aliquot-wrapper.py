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
from typing import Dict, List, Optional, Tuple
from collections import Counter

from base_wrapper import BaseWrapper
import importlib.util

# Import cado-wrapper.py
spec = importlib.util.spec_from_file_location("cado_wrapper", "cado-wrapper.py")
if spec is None or spec.loader is None:
    raise ImportError("Failed to load cado-wrapper.py")
cado_module = importlib.util.module_from_spec(spec)
sys.modules["cado_wrapper"] = cado_module
spec.loader.exec_module(cado_module)
CADOWrapper = cado_module.CADOWrapper

# Import ecm-wrapper.py
spec = importlib.util.spec_from_file_location("ecm_wrapper", "ecm-wrapper.py")
if spec is None or spec.loader is None:
    raise ImportError("Failed to load ecm-wrapper.py")
ecm_module = importlib.util.module_from_spec(spec)
sys.modules["ecm_wrapper"] = ecm_module
spec.loader.exec_module(ecm_module)
ECMWrapper = ecm_module.ECMWrapper


class AliquotSequence:
    """Represents an aliquot sequence with tracking and cycle detection."""

    def __init__(self, start: int):
        self.start = start
        self.sequence = [start]
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

    def __init__(self, config_path: str, factorizer: str = 'cado', hybrid_threshold: int = 100, threads: Optional[int] = None, verbose: bool = False):
        """Initialize aliquot wrapper with specified factorization engine.

        Args:
            config_path: Path to configuration file
            factorizer: Either 'cado' or 'hybrid' (default: 'cado')
            hybrid_threshold: Digit length threshold for switching to ECM+CADO (default: 100)
            threads: Optional thread/worker count for parallel execution
            verbose: Enable verbose output from factorization programs
        """
        super().__init__(config_path)
        self.factorizer_name = factorizer
        self.hybrid_threshold = hybrid_threshold
        self.threads = threads
        self.verbose = verbose

        # Initialize factorizers
        self.cado = CADOWrapper(config_path)
        self.ecm = ECMWrapper(config_path)

        # Set primary factorizer
        if factorizer == 'hybrid':
            self.factorizer = None  # Will be selected dynamically
        else:
            self.factorizer = self.cado

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

    def _trial_division(self, n: int, limit: int = 10**7) -> Tuple[List[int], int]:
        """
        Fast trial division to find small prime factors.

        Args:
            n: Number to factor
            limit: Trial division limit (default: 10^7)

        Returns:
            Tuple of (factors_found, cofactor)
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
        # This is faster than checking every number
        i = 7
        while i * i <= cofactor and i <= limit:
            while cofactor % i == 0:
                factors.append(i)
                cofactor //= i
            i += 2
            # Skip multiples of 3 and 5
            if i % 3 == 0 or i % 5 == 0:
                continue

        return factors, cofactor

    def _factor_hybrid(self, n: int, digit_length: int) -> Tuple[bool, Dict[int, int], Dict]:
        """
        Hybrid factorization: Trial division + Progressive ECM + CADO-NFS.

        Strategy:
        1. Trial division up to 10^7 (very fast, catches small factors)
        2. Progressive ECM using run_ecm_with_tlevel:
           - Target: 4/13 * digit_length (e.g., t30 for 98-digit number)
           - Starts at t20, increments by 5 digits: t20 → t25 → t30 → t35 ...
           - Uses Zimmermann's optimal B1 and curve counts at each level
           - Automatically handles factors and cofactor reduction
        3. CADO-NFS only for remaining cofactor after ECM completes

        This progressive approach finds cheap factors first before investing
        compute in higher B1 values, optimizing for aliquot factorization.

        Args:
            n: Number to factor
            digit_length: Number of digits in n

        Returns:
            Tuple of (success, factorization_dict, raw_results)
        """
        all_factors = []
        current_composite = n

        # Step 0: Trial division with small primes (very fast)
        self.logger.info(f"Running trial division up to 10^7...")
        trial_factors, current_composite = self._trial_division(current_composite)
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
        if self.ecm._is_probably_prime(current_composite):
            self.logger.info(f"Cofactor C{cofactor_digits} is prime, factorization complete")
            all_factors.append(str(current_composite))
            factorization = self.parse_factorization(all_factors)
            return True, factorization, {'success': True, 'method': 'trial_division+primality_test'}

        # Step 1: Progressive ECM (ALWAYS attempt ECM, regardless of size)
        # Use ECM's run_ecm_with_tlevel which handles progressive approach automatically
        cofactor_digits = len(str(current_composite))
        target_t_level = (4.0 / 13.0) * cofactor_digits  # Target: 4/13 of digit length

        self.logger.info(f"Running progressive ECM to t{target_t_level:.1f} on C{cofactor_digits}")

        ecm_results = self.ecm.run_ecm_with_tlevel(
            composite=str(current_composite),
            target_tlevel=target_t_level,
            workers=self.threads if self.threads else 1,
            verbose=self.verbose
        )

        # Collect ECM factors (all are guaranteed to be prime)
        ecm_factors = ecm_results.get('factors_found', [])
        if ecm_factors:
            self.logger.info(f"Progressive ECM found {len(ecm_factors)} prime factor(s)")
            all_factors.extend(ecm_factors)

            # Get final cofactor from ECM results
            final_cofactor = ecm_results.get('final_cofactor')
            if final_cofactor:
                current_composite = int(final_cofactor)
                self.logger.info(f"Cofactor after ECM: C{len(str(current_composite))}")
            else:
                # Fully factored
                current_composite = 1

        # Check if fully factored
        if current_composite == 1:
            self.logger.info("Fully factored by progressive ECM")
            factorization = self.parse_factorization(all_factors)
            return True, factorization, ecm_results

        # Check if cofactor is prime before using CADO-NFS
        cofactor_digits = len(str(current_composite))
        if self.ecm._is_probably_prime(current_composite):
            self.logger.info(f"Cofactor C{cofactor_digits} is prime, factorization complete")
            all_factors.append(str(current_composite))
            factorization = self.parse_factorization(all_factors)
            return True, factorization, ecm_results

        # Use CADO-NFS for remaining cofactor
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

        factorization = self.parse_factorization(all_factors)
        self.logger.info(f"Final factorization: {self.format_factorization(factorization)}")

        # Verify factorization
        product = 1
        for factor_str in all_factors:
            product *= int(factor_str)

        if product != n:
            self.logger.error(f"Factorization verification failed: {product} != {n}")
            return False, {}, ecm_results

        return True, factorization, ecm_results

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
                        submit_to_factordb: bool = False) -> AliquotSequence:
        """
        Compute aliquot sequence starting from given number.

        Args:
            start: Starting number
            max_iterations: Maximum number of iterations
            submit_to_factordb: Whether to submit to FactorDB

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

            # Submit to FactorDB if requested
            if submit_to_factordb and factorization:
                self.submit_to_factordb(next_term, factorization)

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

    def fetch_factordb_last_term(self, start: int) -> Optional[Tuple[int, int]]:
        """
        Fetch the last known term from FactorDB for an aliquot sequence.

        Args:
            start: Starting number of the aliquot sequence

        Returns:
            Tuple of (iteration, composite) or None if fetch failed
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
            cookies = {"fdbuser": "49842c2d25d13890591f62931240e7ba"}
            api_response = requests.get(api_url, cookies=cookies, timeout=30)
            api_response.raise_for_status()

            api_result = api_response.json()

            # Extract composite number from API response
            # API returns: {"id": "...", "status": "C"/"CF", "factors": [[prime, exp], ...]}
            # Reconstruct the number from factors
            if 'factors' in api_result and api_result['factors']:
                # Reconstruct number: multiply all prime^exponent
                composite = 1
                for factor_pair in api_result['factors']:
                    prime = int(factor_pair[0])
                    exponent = int(factor_pair[1])
                    composite *= prime ** exponent

                composite_str = str(composite)
                self.logger.info(f"FactorDB: Found iteration {iteration} with {len(composite_str)}-digit composite")
                return (iteration, composite)
            else:
                self.logger.warning(f"Could not extract factors from FactorDB API (response: {api_result})")
                return None

        except requests.RequestException as e:
            self.logger.error(f"FactorDB fetch failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing FactorDB response: {e}")
            return None

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
        product = 1
        for prime, exp in factorization.items():
            product *= prime ** exp

        if product != n:
            self.logger.error(f"Factor verification failed: {product} != {n}")
            return False

        try:
            # Use cookie for authenticated requests
            cookies = {"fdbuser": "49842c2d25d13890591f62931240e7ba"}

            # Step 1: Query FactorDB to see what factors they already have
            query_url = f"https://factordb.com/api?query={n}"
            query_response = requests.get(query_url, cookies=cookies, timeout=30)
            query_response.raise_for_status()
            fdb_data = query_response.json()

            # Parse existing factors from FactorDB
            # Response format: {"id": "...", "status": "C"/"CF"/"FF", "factors": [["prime", exp], ...]}
            existing_factors = {}
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
  python3 aliquot-wrapper.py --start 276

  # Use pure CADO-NFS for all factorizations
  python3 aliquot-wrapper.py --start 276 --factorizer cado

  # Use hybrid mode (ECM + CADO-NFS) for large numbers
  python3 aliquot-wrapper.py --start 276 --factorizer hybrid

  # Calculate with more iterations
  python3 aliquot-wrapper.py --start 1248 --max-iterations 50

  # Submit results to FactorDB
  python3 aliquot-wrapper.py --start 138 --factordb

  # Quiet mode (no factor spam)
  python3 aliquot-wrapper.py --start 276 --quiet-factors

  # Resume from FactorDB (fetches last known term automatically)
  python3 aliquot-wrapper.py --start 276 --resume-factordb --quiet-factors

  # Manual resume from specific iteration
  python3 aliquot-wrapper.py --start 276 --resume-iteration 2157 --resume-composite 175258998...

  # Use 8 threads/workers for parallel execution
  python3 aliquot-wrapper.py --start 276 --threads 8 --quiet-factors

  # Verbose mode (show detailed output from ECM and CADO-NFS)
  python3 aliquot-wrapper.py --start 276 -v --threads 8

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
    parser.add_argument('--factordb', action='store_true',
                       help='Submit factorizations to FactorDB')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for sequence data')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save sequence to file')
    parser.add_argument('--quiet-factors', action='store_true',
                       help='Disable factor logging to factors_found.txt (reduces spam for aliquot sequences)')
    parser.add_argument('--factorizer', type=str, choices=['cado', 'hybrid'], default='hybrid',
                       help='Factorization strategy: cado (pure CADO-NFS) or hybrid (default: hybrid - uses ECM+CADO for large numbers)')
    parser.add_argument('--hybrid-threshold', type=int, default=100,
                       help='Digit length threshold for hybrid ECM+CADO strategy (default: 100)')
    parser.add_argument('--resume-factordb', action='store_true',
                       help='Resume from last known term in FactorDB')
    parser.add_argument('--resume-iteration', type=int,
                       help='Resume from specific iteration with composite given via --resume-composite')
    parser.add_argument('--resume-composite', type=str,
                       help='Composite number to resume from (use with --resume-iteration)')
    parser.add_argument('--threads', type=int,
                       help='Number of threads/workers for parallel execution (ECM: multiprocess workers, YAFU/CADO: threads)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output from factorization programs (ECM, CADO-NFS)')

    args = parser.parse_args()

    # Initialize wrapper with selected factorizer
    wrapper = AliquotWrapper(args.config, factorizer=args.factorizer, hybrid_threshold=args.hybrid_threshold, threads=args.threads, verbose=args.verbose)

    # Override factor logging config if requested
    if args.quiet_factors:
        wrapper.config['logging']['log_factors_found'] = False
        wrapper.cado.config['logging']['log_factors_found'] = False
        wrapper.ecm.config['logging']['log_factors_found'] = False

    print(f"\nComputing aliquot sequence starting from {args.start}")
    print("="*80)

    # Handle resume options
    resume_iteration = None
    resume_composite = None

    if args.resume_factordb:
        # Fetch last known term from FactorDB
        result = wrapper.fetch_factordb_last_term(args.start)
        if result:
            resume_iteration, resume_composite = result
            print(f"Resuming from FactorDB: iteration {resume_iteration}, {len(str(resume_composite))}-digit composite")
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
    if resume_iteration is not None and resume_composite is not None:
        # Create sequence starting at resume point
        seq = AliquotSequence(args.start)
        # Mark iterations up to resume point as already done
        for i in range(resume_iteration):
            seq.sequence.append(None)  # Placeholder for unknown intermediates
        seq.sequence.append(resume_composite)

        # Factor the resume composite
        success, factorization, _ = wrapper.factor_number(resume_composite)
        if success:
            seq.factorizations[resume_composite] = factorization

            # Submit the resume composite factorization to FactorDB first
            # This allows FactorDB to calculate the next term and maintain sequence linkage
            if args.factordb and factorization:
                wrapper.submit_to_factordb(resume_composite, factorization)

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

                # Submit to FactorDB
                if args.factordb and factorization:
                    wrapper.submit_to_factordb(next_term, factorization)

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
            submit_to_factordb=args.factordb
        )

    # Print summary
    wrapper.print_summary(seq)

    # Save sequence unless disabled
    if not args.no_save:
        output_path = Path(args.output) if args.output else None
        saved_path = wrapper.save_sequence(seq, output_path)
        print(f"\nSequence saved to: {saved_path}")

    # Clean up temporary files created by YAFU/CADO
    wrapper.cleanup_temp_files()

    # Exit with success if sequence completed
    sys.exit(0 if seq.terminated else 1)


if __name__ == '__main__':
    main()
