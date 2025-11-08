#!/usr/bin/env python3
"""
ECMWorkerProcess - Encapsulates single-process ECM execution with factor detection.

This module eliminates ~110 lines of duplicated subprocess logic in the global
_run_worker_ecm_process() function by providing a reusable, testable class.

Design note: multiprocessing requires pickleable functions, so we provide both:
- A class-based implementation (testable, maintainable)
- A thin global function wrapper (for multiprocessing compatibility)
"""
from typing import Optional, Dict, Any
from .parsing_utils import parse_ecm_output, ECMPatterns
from .subprocess_utils import execute_subprocess


class ECMWorkerProcess:
    """Encapsulates single-process ECM execution for multiprocessing pools."""

    def __init__(self, worker_id: int, composite: str, b1: int, b2: Optional[int],
                 curves: int, verbose: bool, method: str, ecm_path: str,
                 progress_interval: int = 0, progress_queue=None):
        """
        Initialize ECM worker process.

        Args:
            worker_id: Worker identifier for logging
            composite: Composite number to factor
            b1: B1 parameter
            b2: B2 parameter (None to use GMP-ECM default)
            curves: Number of curves to run
            verbose: Enable verbose output
            method: Method name (ecm, pm1, pp1)
            ecm_path: Path to GMP-ECM binary
            progress_interval: Report progress every N curves (0 to disable)
            progress_queue: Queue for sending progress updates to parent
        """
        self.worker_id = worker_id
        self.composite = composite
        self.b1 = b1
        self.b2 = b2
        self.curves = curves
        self.verbose = verbose
        self.method = method
        self.ecm_path = ecm_path
        self.progress_interval = progress_interval
        self.progress_queue = progress_queue

    def execute(self, stop_event=None) -> Dict[str, Any]:
        """
        Execute ECM and return results.

        Args:
            stop_event: Optional multiprocessing.Event for early termination

        Returns:
            Dict with keys:
            - worker_id: Worker identifier
            - factor_found: Factor string or None
            - sigma_found: Sigma value that found the factor
            - sigma_values: List of all sigma values used
            - curves_completed: Number of curves completed
            - raw_output: Full program output
        """
        # Build command for this worker
        cmd = [self.ecm_path]

        # Add method-specific parameters
        if self.method == "pm1":
            cmd.append('-pm1')
        elif self.method == "pp1":
            cmd.append('-pp1')

        if self.verbose:
            cmd.append('-v')

        # Run specified number of curves
        cmd.extend(['-c', str(self.curves), str(self.b1)])
        if self.b2 is not None:
            cmd.append(str(self.b2))

        try:
            print(f"Worker {self.worker_id} starting {self.curves} curves")

            # State for line-by-line processing
            curves_completed = 0
            factor_found = None
            sigma_found = None
            sigma_values = []
            current_curve_sigma = None  # Track sigma for the curve currently running
            last_progress_report = 0

            def process_line(line: str, output_lines: list) -> None:
                """Process each output line for curve tracking and factor detection."""
                nonlocal curves_completed, factor_found, sigma_found, sigma_values, current_curve_sigma, last_progress_report

                # Track progress
                if "Step 1 took" in line:
                    curves_completed += 1

                    # Send progress updates if enabled
                    if self.progress_interval > 0 and self.progress_queue is not None:
                        if curves_completed - last_progress_report >= self.progress_interval:
                            self.progress_queue.put({
                                'worker_id': self.worker_id,
                                'curves_completed': curves_completed
                            })
                            last_progress_report = curves_completed

                # Collect sigma values and track current curve's sigma
                sigma_match = ECMPatterns.SIGMA_COLON_FORMAT.search(line) or \
                             ECMPatterns.SIGMA_DASH_FORMAT.search(line)
                if sigma_match:
                    sigma_val = sigma_match.group(1)
                    current_curve_sigma = sigma_val  # This is the sigma for the current curve
                    if sigma_val not in sigma_values:
                        sigma_values.append(sigma_val)

                # Check for factor (just pattern match the line, don't use parse_ecm_output on single line)
                if not factor_found:
                    # Check for standard factor pattern in this line
                    factor_match = ECMPatterns.STANDARD_FACTOR.search(line) or \
                                  ECMPatterns.PRIME_FACTOR.search(line)
                    if factor_match:
                        factor_found = factor_match.group(1)
                        # Use the sigma from the current curve (captured earlier)
                        sigma_found = current_curve_sigma

            # Execute subprocess using unified utility
            result = execute_subprocess(
                cmd=cmd,
                composite=self.composite,
                verbose=self.verbose,
                line_callback=process_line,
                log_prefix=f"Worker {self.worker_id}",
                stop_event=stop_event
            )

            # If no factor found during streaming, check full output
            if not factor_found and not result['terminated_early']:
                factor_found, sigma_found = parse_ecm_output(result['stdout'])
                curves_completed = self.curves  # All curves completed
            elif factor_found and not sigma_found:
                # Factor was found during streaming but sigma was None (found on first curve)
                # Re-parse the full output to get the sigma
                _, sigma_found = parse_ecm_output(result['stdout'])

            # Output completion status
            if result['terminated_early']:
                print(f"Worker {self.worker_id} terminated early after {curves_completed}/{self.curves} curves")
            elif factor_found:
                factor_display = factor_found[:20] + "..." if len(factor_found) > 20 else factor_found
                print(f"Worker {self.worker_id} completed {curves_completed} curves - FACTOR FOUND: {factor_display}")
            else:
                print(f"Worker {self.worker_id} completed {curves_completed} curves - no factor")

            return {
                'worker_id': self.worker_id,
                'factor_found': factor_found,
                'sigma_found': sigma_found,
                'sigma_values': sigma_values,
                'curves_completed': curves_completed if not result['terminated_early'] else curves_completed,
                'raw_output': result['stdout']
            }

        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
            return {
                'worker_id': self.worker_id,
                'factor_found': None,
                'sigma_found': None,
                'sigma_values': [],
                'curves_completed': 0,
                'raw_output': f"Worker failed: {e}"
            }


def run_worker_ecm_process(worker_id: int, composite: str, b1: int, b2: Optional[int],
                           curves: int, verbose: bool, method: str, ecm_path: str,
                           result_queue, stop_event, progress_interval: int = 0,
                           progress_queue=None) -> None:
    """
    Global wrapper function for multiprocessing compatibility.

    This thin wrapper allows ECMWorkerProcess to be used with multiprocessing
    by providing a pickleable top-level function.
    """
    worker = ECMWorkerProcess(worker_id, composite, b1, b2, curves, verbose, method, ecm_path,
                            progress_interval, progress_queue)
    result = worker.execute(stop_event)
    result_queue.put(result)
