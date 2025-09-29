#!/usr/bin/env python3
"""
ECM Performance Benchmark Suite

Compares performance of:
1. Raw GMP-ECM binary execution
2. ECM-wrapper.py (original wrapper)
3. ECM-wrapper-optimized.py (optimized wrapper)

Generates test composite numbers and measures execution time, factor discovery, and overhead.
"""
import subprocess
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import yaml

class CompositeGenerator:
    """Generate test composite numbers using OpenSSL for cryptographically secure randomness."""

    @staticmethod
    def generate_prime_product(bit_size: int) -> str:
        """Generate a composite number as a product of two primes using OpenSSL."""
        # Generate two primes of approximately half the target bit size
        prime_bits = bit_size // 2

        try:
            # Generate first prime
            result1 = subprocess.run([
                'openssl', 'prime', '-generate', '-bits', str(prime_bits), '-hex'
            ], capture_output=True, text=True, check=True)
            prime1_hex = result1.stdout.strip()
            prime1 = int(prime1_hex, 16)

            # Generate second prime
            result2 = subprocess.run([
                'openssl', 'prime', '-generate', '-bits', str(prime_bits), '-hex'
            ], capture_output=True, text=True, check=True)
            prime2_hex = result2.stdout.strip()
            prime2 = int(prime2_hex, 16)

            # Return product as string
            composite = prime1 * prime2
            return str(composite)

        except subprocess.CalledProcessError as e:
            print(f"Error generating composite with OpenSSL: {e}")
            # Fallback to simple method
            return CompositeGenerator._fallback_composite(bit_size)

    @staticmethod
    def _fallback_composite(bit_size: int) -> str:
        """Fallback method using Python's random if OpenSSL fails."""
        import random

        # Generate a number in the target bit range
        min_val = 2**(bit_size-1)
        max_val = 2**bit_size - 1

        # Generate odd composite (avoid even numbers)
        composite = random.randrange(min_val | 1, max_val, 2)

        # Ensure it's composite by making it divisible by a small prime if it's prime
        if CompositeGenerator._is_prime_simple(composite):
            composite += random.choice([3, 7, 11, 13])

        return str(composite)

    @staticmethod
    def _is_prime_simple(n: int) -> bool:
        """Simple primality test for small numbers."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def get_test_numbers() -> List[Tuple[str, str, int]]:
        """Generate test composite numbers of increasing difficulty."""
        test_cases = [
            ("Small", "Quick test (30-40 digits)", 128),    # ~40 digits
            ("Medium", "Moderate difficulty (50-60 digits)", 200),  # ~60 digits
            ("Large", "Challenging (70-80 digits)", 256),   # ~80 digits
        ]

        results = []
        for name, description, bits in test_cases:
            composite = CompositeGenerator.generate_prime_product(bits)
            results.append((name, description, len(composite), composite))

        return results

class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, method: str, composite: str):
        self.method = method
        self.composite = composite
        self.execution_time: Optional[float] = None
        self.factor_found: Optional[str] = None
        self.curves_completed: int = 0
        self.success: bool = False
        self.error: Optional[str] = None
        self.raw_output: str = ""

    def __repr__(self):
        return f"BenchmarkResult({self.method}, success={self.success}, time={self.execution_time:.2f}s)"

class ECMBenchmarker:
    """Main benchmarking class."""

    def __init__(self, config_path: str = "client.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.ecm_path = self.config['programs']['gmp_ecm']['path']

        # Benchmark parameters
        self.b1 = 50000  # Small B1 for quick tests
        self.b2 = self.config['programs']['gmp_ecm']['default_b2']  # Default B2 from config
        self.curves = 10  # Limited curves for speed
        self.timeout = 60  # 60 second timeout per test
        self.verbose = False  # Verbose output flag

        print(f"ECM Benchmarker initialized")
        print(f"GMP-ECM path: {self.ecm_path}")
        print(f"Test parameters: B1={self.b1}, B2={self.b2}, curves={self.curves}, timeout={self.timeout}s")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def benchmark_raw_ecm_cpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark raw GMP-ECM binary execution (CPU, Stage 1 only)."""
        result = BenchmarkResult("Raw ECM (CPU, Stage 1)", composite)

        # Build command: ecm -c curves b1 0 (B2=0 means Stage 1 only)
        cmd = [self.ecm_path, '-c', str(self.curves), str(self.b1), '0']

        try:
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send composite number and get output
            stdout, _ = process.communicate(input=composite, timeout=self.timeout)

            result.execution_time = time.time() - start_time
            result.raw_output = stdout
            result.success = True

            # Parse for factors
            factor = self._parse_ecm_output(stdout)
            if factor:
                result.factor_found = factor

            # Count completed curves
            result.curves_completed = stdout.count("Step 1 took")

        except subprocess.TimeoutExpired:
            process.kill()
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_raw_ecm_cpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark raw GMP-ECM binary execution (CPU, Stage 1+2)."""
        result = BenchmarkResult("Raw ECM (CPU, Stage 1+2)", composite)

        # Build command: ecm -c curves b1 b2 (full ECM with both stages)
        cmd = [self.ecm_path, '-c', str(self.curves), str(self.b1), str(self.b2)]

        try:
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send composite number and get output
            stdout, _ = process.communicate(input=composite, timeout=self.timeout)

            result.execution_time = time.time() - start_time
            result.raw_output = stdout
            result.success = True

            # Parse for factors
            factor = self._parse_ecm_output(stdout)
            if factor:
                result.factor_found = factor

            # Count completed curves
            result.curves_completed = stdout.count("Step 1 took")

        except subprocess.TimeoutExpired:
            process.kill()
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_raw_ecm_gpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark raw GMP-ECM binary execution (GPU, Stage 1 only)."""
        result = BenchmarkResult("Raw ECM (GPU, Stage 1)", composite)

        # Build command: ecm -gpu -c curves b1 0 (B2=0 means Stage 1 only)
        cmd = [self.ecm_path, '-gpu', '-c', str(self.curves), str(self.b1), '0']

        try:
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send composite number and get output
            stdout, _ = process.communicate(input=composite, timeout=self.timeout)

            result.execution_time = time.time() - start_time
            result.raw_output = stdout
            result.success = True

            # Parse for factors
            factor = self._parse_ecm_output(stdout)
            if factor:
                result.factor_found = factor

            # Count completed curves
            result.curves_completed = stdout.count("Step 1 took")

        except subprocess.TimeoutExpired:
            process.kill()
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_raw_ecm_gpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark raw GMP-ECM binary execution (GPU, Stage 1+2)."""
        result = BenchmarkResult("Raw ECM (GPU, Stage 1+2)", composite)

        # Build command: ecm -gpu -c curves b1 b2 (full ECM with both stages)
        cmd = [self.ecm_path, '-gpu', '-c', str(self.curves), str(self.b1), str(self.b2)]

        try:
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send composite number and get output
            stdout, _ = process.communicate(input=composite, timeout=self.timeout)

            result.execution_time = time.time() - start_time
            result.raw_output = stdout
            result.success = True

            # Parse for factors
            factor = self._parse_ecm_output(stdout)
            if factor:
                result.factor_found = factor

            # Count completed curves
            result.curves_completed = stdout.count("Step 1 took")

        except subprocess.TimeoutExpired:
            process.kill()
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_cpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark ECM-wrapper.py (CPU, Stage 1 only)."""
        result = BenchmarkResult("ECM-wrapper (CPU, Stage 1)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', '0',  # Stage 1 only
            '--curves', str(self.curves),
            '--no-gpu',  # Force CPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_cpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark original ECM-wrapper.py (CPU, Stage 1+2)."""
        result = BenchmarkResult("ECM-wrapper (CPU, Stage 1+2)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', str(self.b2),  # Use custom B2 for Stage 1+2
            '--curves', str(self.curves),
            '--no-gpu',  # Force CPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_gpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark original ECM-wrapper.py (GPU, Stage 1+2)."""
        result = BenchmarkResult("ECM-wrapper (GPU, Stage 1+2)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', str(self.b2),  # Use custom B2 for Stage 1+2
            '--curves', str(self.curves),
            '--gpu',  # Force GPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_optimized_cpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark optimized ECM-wrapper-optimized.py (CPU, Stage 1+2)."""
        result = BenchmarkResult("ECM-wrapper-optimized (CPU, Stage 1+2)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper-optimized.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', str(self.b2),  # Use custom B2 for Stage 1+2
            '--curves', str(self.curves),
            '--no-gpu',  # Force CPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_optimized_gpu_stage12(self, composite: str) -> BenchmarkResult:
        """Benchmark optimized ECM-wrapper-optimized.py (GPU, Stage 1+2)."""
        result = BenchmarkResult("ECM-wrapper-optimized (GPU, Stage 1+2)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper-optimized.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', str(self.b2),  # Use custom B2 for Stage 1+2
            '--curves', str(self.curves),
            '--gpu',  # Force GPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_optimized_cpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark optimized ECM-wrapper-optimized.py (CPU, Stage 1 only)."""
        result = BenchmarkResult("ECM-wrapper-optimized (CPU, Stage 1)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper-optimized.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', '0',  # Stage 1 only
            '--curves', str(self.curves),
            '--no-gpu',  # Force CPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_optimized_gpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark optimized ECM-wrapper-optimized.py (GPU, Stage 1 only)."""
        result = BenchmarkResult("ECM-wrapper-optimized (GPU, Stage 1)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper-optimized.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', '0',  # Stage 1 only
            '--curves', str(self.curves),
            '--gpu',  # Force GPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def benchmark_ecm_wrapper_gpu_stage1(self, composite: str) -> BenchmarkResult:
        """Benchmark ECM-wrapper.py (GPU, Stage 1 only)."""
        result = BenchmarkResult("ECM-wrapper (GPU, Stage 1)", composite)

        cmd = [
            sys.executable, 'ecm-wrapper.py',
            '--composite', composite,
            '--b1', str(self.b1),
            '--b2', '0',  # Stage 1 only
            '--curves', str(self.curves),
            '--gpu',  # Force GPU mode
            '--no-submit',  # Don't submit to API during benchmarking
            '--config', self.config_path
        ]

        try:
            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent
            )

            result.execution_time = time.time() - start_time
            result.raw_output = process.stdout + process.stderr
            result.success = (process.returncode == 0)

            if not result.success:
                result.error = f"Exit code {process.returncode}"

            # Parse wrapper output for results
            self._parse_wrapper_output(result)

        except subprocess.TimeoutExpired:
            result.execution_time = self.timeout
            result.error = "Timeout"
            result.success = False
        except Exception as e:
            result.error = str(e)
            result.success = False
            if result.execution_time is None:
                result.execution_time = 0

        return result

    def _parse_ecm_output(self, output: str) -> Optional[str]:
        """Parse raw ECM output for factors."""
        import re

        # Look for factor patterns in ECM output
        patterns = [
            r'Factor found in step \d+: (\d+)',
            r'Using B1=\d+, B2=\d+, polynomial .+, sigma=.+\n(\d+)',
            r'^(\d+)$'  # Simple number on its own line
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            if matches:
                return matches[0]

        return None

    def _parse_wrapper_output(self, result: BenchmarkResult) -> None:
        """Parse wrapper output for factors and completion info."""
        output = result.raw_output

        # Look for factor announcements in wrapper logs - multiple patterns
        import re

        # Pattern 1: 🎉 FACTOR FOUND: number
        match = re.search(r'🎉 FACTOR FOUND: (\d+)', output)
        if match:
            result.factor_found = match.group(1)

        # Pattern 2: Factor found: number
        elif "Factor found:" in output:
            match = re.search(r'Factor found: (\d+)', output)
            if match:
                result.factor_found = match.group(1)

        # Pattern 3: Look in ECM output for factor patterns
        elif "********** Factor found" in output:
            match = re.search(r'\*+ Factor found .+: (\d+)', output)
            if match:
                result.factor_found = match.group(1)

        # Count curves from wrapper output
        curve_count = output.count("curves completed") + output.count("Step 1 took")
        if curve_count > 0:
            result.curves_completed = min(curve_count, self.curves)
        else:
            result.curves_completed = self.curves if result.success else 0

    def run_benchmark_suite(self, test_numbers: List[Tuple[str, str, int, str]]) -> Dict[str, List[BenchmarkResult]]:
        """Run complete benchmark suite on all test numbers."""
        results = {}

        for name, description, digit_count, composite in test_numbers:
            print(f"\n{'='*60}")
            print(f"Benchmarking {name} ({digit_count} digits)")
            print(f"Description: {description}")
            print(f"Composite: {composite[:20]}...{composite[-20:]}")
            print(f"{'='*60}")

            test_results = []

            # Test each method - organized by Stage 1 only vs Stage 1+2 for fair comparison
            methods = [
                # Stage 1 only tests (fair comparison - same computational work)
                ("Raw ECM (CPU, Stage 1)", self.benchmark_raw_ecm_cpu_stage1),
                ("Raw ECM (GPU, Stage 1)", self.benchmark_raw_ecm_gpu_stage1),
                ("ECM-wrapper (CPU, Stage 1)", self.benchmark_ecm_wrapper_cpu_stage1),
                ("ECM-wrapper (GPU, Stage 1)", self.benchmark_ecm_wrapper_gpu_stage1),
                ("ECM-wrapper-optimized (CPU, Stage 1)", self.benchmark_ecm_wrapper_optimized_cpu_stage1),
                ("ECM-wrapper-optimized (GPU, Stage 1)", self.benchmark_ecm_wrapper_optimized_gpu_stage1),

                # Stage 1+2 tests (more computational work, higher factor finding probability)
                ("Raw ECM (CPU, Stage 1+2)", self.benchmark_raw_ecm_cpu_stage12),
                ("Raw ECM (GPU, Stage 1+2)", self.benchmark_raw_ecm_gpu_stage12),
                ("ECM-wrapper (CPU, Stage 1+2)", self.benchmark_ecm_wrapper_cpu_stage12),
                ("ECM-wrapper (GPU, Stage 1+2)", self.benchmark_ecm_wrapper_gpu_stage12),
                ("ECM-wrapper-optimized (CPU, Stage 1+2)", self.benchmark_ecm_wrapper_optimized_cpu_stage12),
                ("ECM-wrapper-optimized (GPU, Stage 1+2)", self.benchmark_ecm_wrapper_optimized_gpu_stage12)
            ]

            for method_name, benchmark_func in methods:
                print(f"\nTesting {method_name}...")
                result = benchmark_func(composite)
                test_results.append(result)

                if result.success:
                    status = "✓ SUCCESS"
                    if result.factor_found:
                        status += f" (factor: {result.factor_found})"
                    status += f" - {result.execution_time:.2f}s"
                else:
                    status = f"✗ FAILED - {result.error}"

                print(f"  {status}")

                # Show verbose output if requested
                if self.verbose and result.raw_output:
                    print(f"  --- Verbose Output for {method_name} ---")
                    print(result.raw_output)
                    print(f"  --- End Output ---")

            results[name] = test_results

        return results

    def print_summary_report(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print formatted summary of benchmark results."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY REPORT")
        print(f"{'='*80}")

        # Summary table
        print(f"\n{'Method':<25} {'Success Rate':<12} {'Avg Time':<12} {'Factors Found':<15}")
        print(f"{'-'*70}")

        method_stats = {}
        for test_name, results in all_results.items():
            for result in results:
                method = result.method
                if method not in method_stats:
                    method_stats[method] = {'times': [], 'successes': 0, 'factors': 0, 'total': 0}

                method_stats[method]['total'] += 1
                if result.success:
                    method_stats[method]['successes'] += 1
                    method_stats[method]['times'].append(result.execution_time)
                if result.factor_found:
                    method_stats[method]['factors'] += 1

        for method, stats in method_stats.items():
            success_rate = f"{stats['successes']}/{stats['total']}"
            avg_time = f"{sum(stats['times'])/len(stats['times']):.2f}s" if stats['times'] else "N/A"
            factors_found = f"{stats['factors']}/{stats['total']}"

            print(f"{method:<25} {success_rate:<12} {avg_time:<12} {factors_found:<15}")

        # Detailed results by test size
        print(f"\n{'='*80}")
        print("DETAILED RESULTS BY TEST SIZE")
        print(f"{'='*80}")

        for test_name, results in all_results.items():
            print(f"\n{test_name} Test Results:")
            print(f"{'Method':<25} {'Time (s)':<10} {'Status':<15} {'Factor Found':<20}")
            print(f"{'-'*75}")

            for result in results:
                time_str = f"{result.execution_time:.2f}" if result.execution_time else "N/A"
                status = "Success" if result.success else f"Failed ({result.error})"
                factor_str = result.factor_found[:18] + "..." if result.factor_found and len(result.factor_found) > 20 else (result.factor_found or "None")

                print(f"{result.method:<25} {time_str:<10} {status:<15} {factor_str:<20}")

        # Performance comparison
        print(f"\n{'='*80}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*80}")

        if 'Raw ECM' in method_stats and method_stats['Raw ECM']['times']:
            raw_avg = sum(method_stats['Raw ECM']['times']) / len(method_stats['Raw ECM']['times'])

            print(f"Overhead Analysis (vs Raw ECM baseline):")
            for method, stats in method_stats.items():
                if method != 'Raw ECM' and stats['times']:
                    method_avg = sum(stats['times']) / len(stats['times'])
                    overhead = ((method_avg - raw_avg) / raw_avg) * 100
                    print(f"  {method}: +{overhead:.1f}% overhead ({method_avg:.2f}s vs {raw_avg:.2f}s)")

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark ECM wrapper performance")
    parser.add_argument('--config', default='client.yaml', help='Configuration file path')
    parser.add_argument('--b1', type=int, default=50000, help='B1 parameter for tests')
    parser.add_argument('--b2', type=int, help='B2 parameter for Stage 1+2 tests (default: use config file)')
    parser.add_argument('--curves', type=int, default=10, help='Number of curves per test')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout per test in seconds')
    parser.add_argument('--quick', action='store_true', help='Run only small test for quick validation')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output from each test')

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = ECMBenchmarker(args.config)
    benchmarker.b1 = args.b1
    benchmarker.b2 = args.b2 if args.b2 is not None else benchmarker.config['programs']['gmp_ecm']['default_b2']
    benchmarker.curves = args.curves
    benchmarker.timeout = args.timeout
    benchmarker.verbose = args.verbose

    # Generate test numbers
    print("Generating test composite numbers...")
    test_numbers = CompositeGenerator.get_test_numbers()

    if args.quick:
        test_numbers = test_numbers[:1]  # Only small test
        print("Running in quick mode (small test only)")

    # Display test numbers
    print(f"\nGenerated {len(test_numbers)} test numbers:")
    for name, desc, digits, composite in test_numbers:
        print(f"  {name}: {digits} digits - {composite[:15]}...{composite[-15:]}")

    # Run benchmarks
    results = benchmarker.run_benchmark_suite(test_numbers)

    # Print summary
    benchmarker.print_summary_report(results)

if __name__ == '__main__':
    main()