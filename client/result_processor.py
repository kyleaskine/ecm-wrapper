#!/usr/bin/env python3
"""
High-performance consolidated result processor for factorization programs.
Eliminates code duplication and optimizes result handling across all wrappers.
"""
import time
import json
import hashlib
import datetime
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from execution_engine import ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class FactorResult:
    """Standardized factor result."""
    factor: str
    sigma: Optional[str] = None
    method: str = "ecm"
    curves_used: int = 0
    discovery_time: float = 0

@dataclass
class ProcessingResult:
    """Consolidated result from all processing."""
    composite: str
    method: str
    program: str
    success: bool
    execution_time: float

    # Factor information
    factors_found: List[FactorResult] = field(default_factory=list)
    primary_factor: Optional[str] = None

    # Execution metrics
    curves_requested: int = 0
    curves_completed: int = 0
    b1: Optional[int] = None
    b2: Optional[int] = None

    # Raw data
    raw_output: str = ""
    error: Optional[str] = None

    # API submission
    submitted: bool = False
    submission_attempts: int = 0

class OptimizedParser:
    """High-performance parser with pre-compiled patterns and streaming support."""

    def __init__(self):
        # Import and cache all parsing patterns at initialization
        from parsing_utils import ECMPatterns, YAFUPatterns
        self.ecm_patterns = ECMPatterns()
        self.yafu_patterns = YAFUPatterns()

    def parse_ecm_output(self, output: str, method: str = "ecm") -> List[FactorResult]:
        """Optimized ECM output parsing with minimal overhead."""
        factors = []

        # Check for prime factors first (most reliable)
        prime_matches = self.ecm_patterns.PRIME_FACTOR.findall(output)
        if prime_matches:
            for factor in prime_matches:
                # Extract sigma for this specific factor
                sigma = self._extract_sigma_for_factor(output, factor)
                factors.append(FactorResult(
                    factor=factor,
                    sigma=sigma,
                    method=method
                ))

        # Also check for standard "Factor found in step" format
        standard_matches = self.ecm_patterns.STANDARD_FACTOR.findall(output)
        for factor in standard_matches:
            # Avoid duplicates
            if not any(f.factor == factor for f in factors):
                factors.append(FactorResult(
                    factor=factor,
                    method=method
                ))

        # Fallback to GPU format
        if not factors:
            gpu_matches = self.ecm_patterns.GPU_FACTOR.findall(output)
            for match in gpu_matches:
                factor = match[0]
                sigma = f"3:{match[1]}" if match[1] else None
                factors.append(FactorResult(
                    factor=factor,
                    sigma=sigma,
                    method=method
                ))

        return factors

    def parse_yafu_output(self, output: str, method: str = "ecm") -> List[FactorResult]:
        """Optimized YAFU output parsing."""
        factors = []

        # Direct factor announcements
        factor_matches = self.yafu_patterns.FACTOR_FOUND.findall(output)
        for factor in factor_matches:
            factors.append(FactorResult(factor=factor, method=method))

        # P/Q notation factors
        pq_matches = self.yafu_patterns.PQ_NOTATION.findall(output)
        for factor in pq_matches:
            if len(factor) > 1:  # Skip trivial factors
                factors.append(FactorResult(factor=factor, method=method))

        return factors

    def parse_curves_completed(self, output: str, program: str) -> int:
        """Extract curves completed from output."""
        if program.startswith('gmp-ecm'):
            return output.count("Step 1 took")
        elif program.startswith('yafu'):
            curves_match = self.yafu_patterns.CURVES_COMPLETED.search(output)
            if curves_match:
                return int(curves_match.group(1))
        return 0

    def _extract_sigma_for_factor(self, output: str, factor: str) -> Optional[str]:
        """Extract sigma value for a specific factor."""
        import re
        # Look for GPU line mentioning this factor
        pattern = rf'GPU: factor {factor} found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?'
        match = re.search(pattern, output)
        return f"3:{match.group(1)}" if match and match.group(1) else None

class ConsolidatedResultProcessor:
    """
    High-performance unified result processor.
    Eliminates all duplicate result handling across wrappers.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client_id = config['client']['id']
        self.api_endpoint = config['api']['endpoint']
        self.parser = OptimizedParser()
        self.logger = logger

    def process_execution_result(self, exec_result: ExecutionResult,
                               composite: str, method: str, program: str,
                               **params) -> ProcessingResult:
        """
        Process execution result into standardized format.

        Args:
            exec_result: Result from execution engine
            composite: Number being factored
            method: Factorization method
            program: Program used
            **params: Additional parameters (b1, b2, curves, etc.)
        """
        # Create base result
        result = ProcessingResult(
            composite=composite,
            method=method,
            program=program,
            success=exec_result.success,
            execution_time=exec_result.execution_time,
            raw_output=exec_result.stdout,
            error=exec_result.error,
            curves_requested=params.get('curves', 0),
            b1=params.get('b1'),
            b2=params.get('b2')
        )

        # Parse factors even if return code indicates "failure" (ECM exits non-zero when factor found)
        factors = []
        if program.startswith('gmp-ecm'):
            factors = self.parser.parse_ecm_output(exec_result.stdout, method)
        elif program.startswith('yafu'):
            factors = self.parser.parse_yafu_output(exec_result.stdout, method)

        result.factors_found = factors
        result.primary_factor = factors[0].factor if factors else None

        # If factors were found, consider it a success regardless of return code
        if factors:
            result.success = True

        # Parse curves completed
        result.curves_completed = self.parser.parse_curves_completed(
            exec_result.stdout, program
        )

        return result

    def log_factors(self, result: ProcessingResult):
        """Optimized factor logging with minimal I/O."""
        if not result.factors_found:
            return

        factors_file = Path("data/factors_found.txt")
        factors_file.parent.mkdir(parents=True, exist_ok=True)

        # Build log entry efficiently
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entries = []

        for factor_result in result.factors_found:
            log_entries.append(
                f"\n{'='*80}\n"
                f"FACTOR FOUND: {timestamp}\n"
                f"{'='*80}\n"
                f"Composite ({len(result.composite)} digits): {result.composite}\n"
                f"Factor: {factor_result.factor}\n"
                f"Parameters: B1={result.b1}, B2={result.b2}, Curves={result.curves_completed}\n"
                f"Program: {result.program} ({result.method.upper()} mode)\n"
                f"{'='*80}\n"
            )

        # Single file write for all factors
        with open(factors_file, 'a') as f:
            f.writelines(log_entries)

        # Console output
        for factor_result in result.factors_found:
            print(f"\n🎉 FACTOR FOUND: {factor_result.factor}")
        print(f"📋 Logged to: {factors_file}")

    def submit_results(self, result: ProcessingResult,
                      project: Optional[str] = None) -> bool:
        """
        Optimized API submission with retry logic and batch support.
        Consolidates all submission logic from base wrapper.
        """
        if not result.factors_found and not self._should_submit_no_factor():
            self.logger.debug("Skipping submission - no factors found and no-factor submission disabled")
            return True

        # Build primary payload efficiently
        payload = self._build_api_payload(result, project)

        # Submit primary result
        success = self._submit_single_payload(payload, result)

        if success:
            result.submitted = True

            # Submit additional factors if present
            if len(result.factors_found) > 1:
                self._submit_additional_factors(result, project)

        return success

    def _build_api_payload(self, result: ProcessingResult,
                          project: Optional[str]) -> Dict[str, Any]:
        """Build API payload efficiently."""
        factor_found = result.primary_factor if result.factors_found else None

        return {
            'composite': result.composite,
            'project': project,
            'client_id': self.client_id,
            'method': result.method,
            'program': result.program,
            'program_version': self._get_program_version(result.program),
            'parameters': {
                'b1': result.b1,
                'b2': result.b2,
                'curves': result.curves_requested,
                'sigma': result.factors_found[0].sigma if result.factors_found else None
            },
            'results': {
                'factor_found': factor_found,
                'curves_completed': result.curves_completed,
                'execution_time': result.execution_time
            },
            'raw_output': result.raw_output
        }

    def _submit_single_payload(self, payload: Dict[str, Any],
                              result: ProcessingResult) -> bool:
        """Submit single payload with optimized retry logic."""
        url = f"{self.api_endpoint}/submit_result"
        retry_count = self.config['api']['retry_attempts']

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.config['api']['timeout']
                )

                if response.status_code == 200:
                    self.logger.info(f"Successfully submitted results: {response.json()}")
                    return True
                else:
                    self.logger.warning(f"API submission failed (HTTP {response.status_code}): {response.text}")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"API submission failed (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # Save failed submission
        self._save_failed_submission(result, payload)
        return False

    def _submit_additional_factors(self, result: ProcessingResult,
                                  project: Optional[str]):
        """Submit additional factors found in same run."""
        for factor_result in result.factors_found[1:]:
            additional_payload = self._build_api_payload(result, project)
            additional_payload['results']['factor_found'] = factor_result.factor
            additional_payload['results']['curves_completed'] = 0
            additional_payload['results']['execution_time'] = 0
            additional_payload['raw_output'] = f"Additional factor from same run: {factor_result.factor}"

            self._submit_single_payload(additional_payload, result)

    def _save_failed_submission(self, result: ProcessingResult,
                               payload: Dict[str, Any]):
        """Save failed submission for later retry."""
        try:
            data_dir = Path("data/results")
            data_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            composite_hash = hashlib.md5(result.composite.encode()).hexdigest()[:8]
            filename = f"failed_submission_{timestamp}_{composite_hash}.json"

            save_data = {
                'composite': result.composite,
                'method': result.method,
                'program': result.program,
                'factors_found': [f.factor for f in result.factors_found],
                'execution_time': result.execution_time,
                'api_payload': payload,
                'submitted': False,
                'failed_at': datetime.datetime.now().isoformat(),
                'retry_count': self.config['api']['retry_attempts']
            }

            filepath = data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)

            self.logger.info(f"Saved failed submission to: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save submission data: {e}")

    def _should_submit_no_factor(self) -> bool:
        """Check if no-factor results should be submitted."""
        return self.config.get('api', {}).get('submit_no_factor', True)

    def _get_program_version(self, program: str) -> str:
        """Get program version (cached for performance)."""
        # This could be enhanced with caching
        return "unknown"

    def save_raw_output(self, result: ProcessingResult):
        """Save raw output if configured."""
        if not self.config.get('execution', {}).get('save_raw_output', False):
            return

        output_dir = Path(self.config['execution']['output_dir'])
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"{result.program}_{result.method}_{timestamp}_{result.curves_completed}curves.txt"

        with open(filename, 'w') as f:
            f.write(f"Composite: {result.composite}\n")
            f.write(f"B1: {result.b1}, B2: {result.b2}\n")
            f.write(f"Method: {result.method}\n")
            f.write(f"Program: {result.program}\n")
            f.write(f"Factors found: {[f.factor for f in result.factors_found]}\n")
            f.write(f"Curves completed: {result.curves_completed}\n")
            f.write(f"Execution time: {result.execution_time:.2f}s\n\n")
            f.write(result.raw_output)