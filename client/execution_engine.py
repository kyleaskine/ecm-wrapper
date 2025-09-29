#!/usr/bin/env python3
"""
High-performance unified subprocess execution engine for factorization programs.
Optimized for minimal overhead and maximum throughput.
"""
import subprocess
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionParams:
    """Unified parameters for subprocess execution."""
    cmd: List[str]
    input_data: Optional[str] = None
    timeout: int = 3600
    capture_output: bool = True
    stream_output: bool = False
    working_dir: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None

@dataclass
class ExecutionResult:
    """Unified result from subprocess execution."""
    returncode: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool
    timeout_occurred: bool = False
    error: Optional[str] = None

class StreamingOutputHandler:
    """High-performance streaming output handler with parsing callbacks."""

    def __init__(self, parse_callback: Optional[Callable[[str], Any]] = None):
        self.parse_callback = parse_callback
        self.output_buffer = []
        self.parsed_results = []

    def handle_line(self, line: str) -> bool:
        """
        Handle a single output line.
        Returns True if execution should continue, False to terminate early.
        """
        self.output_buffer.append(line)

        if self.parse_callback:
            result = self.parse_callback(line)
            if result:
                self.parsed_results.append(result)
                # Early termination on factor found (configurable)
                return False
        return True

    def get_output(self) -> str:
        """Get complete output."""
        return '\n'.join(self.output_buffer)

class ProcessPool:
    """Optimized process pool for batch operations."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._pool = None

    def __enter__(self):
        self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            self._pool.shutdown(wait=True)

class UnifiedExecutionEngine:
    """
    High-performance unified execution engine for factorization programs.
    Consolidates all subprocess patterns with minimal overhead.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

    def execute_single(self, params: ExecutionParams,
                      stream_handler: Optional[StreamingOutputHandler] = None) -> ExecutionResult:
        """
        Execute single subprocess with optimal performance.

        Args:
            params: Execution parameters
            stream_handler: Optional streaming output handler for real-time parsing

        Returns:
            ExecutionResult with timing and output data
        """
        start_time = time.time()

        try:
            process = subprocess.Popen(
                params.cmd,
                stdin=subprocess.PIPE if params.input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=params.working_dir,
                env=params.env_vars
            )

            if params.stream_output and stream_handler:
                stdout, stderr = self._stream_with_handler(process, params, stream_handler)
            else:
                stdout, stderr = self._communicate_with_timeout(process, params)

            execution_time = time.time() - start_time

            return ExecutionResult(
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr or "",
                execution_time=execution_time,
                success=process.returncode == 0,
                timeout_occurred=False
            )

        except subprocess.TimeoutExpired:
            process.kill()
            execution_time = time.time() - start_time

            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr="Process timed out",
                execution_time=execution_time,
                success=False,
                timeout_occurred=True
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                success=False,
                error=str(e)
            )

    def _communicate_with_timeout(self, process: subprocess.Popen,
                                 params: ExecutionParams) -> tuple[str, str]:
        """Optimized communicate with timeout handling."""
        try:
            if params.input_data:
                stdout, stderr = process.communicate(
                    input=params.input_data,
                    timeout=params.timeout
                )
            else:
                stdout, stderr = process.communicate(timeout=params.timeout)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            process.kill()
            # Try to get partial output
            try:
                stdout, stderr = process.communicate(timeout=1)
                return stdout or "", stderr or ""
            except:
                return "", "Timeout expired"

    def _stream_with_handler(self, process: subprocess.Popen,
                           params: ExecutionParams,
                           handler: StreamingOutputHandler) -> tuple[str, str]:
        """Stream output with real-time parsing."""
        if params.input_data:
            process.stdin.write(params.input_data)
            process.stdin.close()

        stderr_lines = []

        # Read stdout line by line
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break

                line = line.rstrip()
                if line:
                    # Let handler process the line
                    continue_execution = handler.handle_line(line)
                    if not continue_execution:
                        # Early termination requested
                        process.terminate()
                        break
        except Exception as e:
            self.logger.warning(f"Error during streaming: {e}")

        # Wait for process completion
        process.wait()

        # Get any remaining stderr
        if process.stderr:
            stderr_output = process.stderr.read()
            if stderr_output:
                stderr_lines.append(stderr_output)

        return handler.get_output(), '\n'.join(stderr_lines)

    def execute_batch(self, batch_params: List[ExecutionParams],
                     max_workers: int = 4) -> List[ExecutionResult]:
        """
        Execute multiple subprocesses in parallel with optimal resource usage.

        Args:
            batch_params: List of execution parameters
            max_workers: Maximum concurrent processes

        Returns:
            List of ExecutionResults in same order as input
        """
        if not batch_params:
            return []

        results = [None] * len(batch_params)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.execute_single, params): i
                for i, params in enumerate(batch_params)
            }

            # Collect results maintaining order
            for future in future_to_index:
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    # Create error result
                    results[index] = ExecutionResult(
                        returncode=-1,
                        stdout="",
                        stderr=str(e),
                        execution_time=0,
                        success=False,
                        error=str(e)
                    )

        return results

    def execute_multiprocess_ecm(self, composite: str, total_curves: int,
                               workers: int, b1: int, b2: Optional[int],
                               ecm_path: str, method: str = "ecm") -> Dict[str, Any]:
        """
        Optimized multiprocess ECM execution with early termination.
        Replaces the complex multiprocess logic in ECM wrapper.
        """
        # Distribute curves across workers
        curves_per_worker = total_curves // workers
        remaining = total_curves % workers

        worker_params = []
        for worker_id in range(workers):
            curves = curves_per_worker + (1 if worker_id < remaining else 0)
            if curves > 0:
                cmd = [ecm_path]
                if method == "pm1":
                    cmd.append('-pm1')
                elif method == "pp1":
                    cmd.append('-pp1')
                cmd.extend(['-c', str(curves), str(b1)])
                if b2:
                    cmd.append(str(b2))

                worker_params.append(ExecutionParams(
                    cmd=cmd,
                    input_data=composite,
                    timeout=3600
                ))

        # Execute all workers in parallel
        results = self.execute_batch(worker_params, max_workers=workers)

        # Aggregate results
        total_curves_completed = 0
        factor_found = None
        all_outputs = []

        for result in results:
            if result.success:
                all_outputs.append(result.stdout)
                # Parse for factors (using existing parsing logic)
                # This will be enhanced when we integrate with the result processor

        return {
            'curves_completed': total_curves_completed,
            'factor_found': factor_found,
            'raw_outputs': all_outputs,
            'execution_time': max(r.execution_time for r in results),
            'success': any(r.success for r in results)
        }