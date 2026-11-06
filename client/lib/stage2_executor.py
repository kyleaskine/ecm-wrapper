#!/usr/bin/env python3
"""
Stage2Executor - Unified Stage 2 execution with worker pool management.

This class eliminates ~270 lines of duplicated Stage 2 logic across:
- execution_engine.py (two-stage and pipelined modes)
- scripts/run_batch_pipeline.py cpu_worker() function

Handles:
- Residue file splitting
- Multi-threaded worker pool execution
- Early termination when factor found
- Progress tracking and reporting
- Process cleanup
"""
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Set, Tuple, List, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ecm_command import build_ecm_command, gpu_stage2_supported
from .parsing_utils import (
    ECMPatterns, count_completed_curves, parse_b2_used, parse_ecm_output,
    parse_ecm_output_multiple,
)

if TYPE_CHECKING:
    from ecm_wrapper import ECMWrapper


def create_stage2_executor(
    wrapper: 'ECMWrapper', residue_file: Path, b1: int, b2: Optional[int],
    k: Optional[int], workers: int, verbose: bool = False,
    pin_threads: bool = False, *, use_gpu: bool = False,
    gpu_device: Optional[int] = None, gpu_curves: Optional[int] = None,
    cpu_b2: Optional[int] = None,
) -> 'Stage2Executor | GPUStage2Executor':
    """
    Pick the stage 2 executor: one GPU process, or a pool of CPU workers.

    use_gpu falls back to CPU workers (with a warning) when the configured
    GMP-ECM binary predates GPU stage 2 - such builds reject -gpu -resume.
    A GPU executor also falls back to these CPU workers if its run fails.

    cpu_b2, when given, replaces b2 for any run on CPU workers (b2 may be -1,
    GMP-ECM's default, which is cheap on the GPU but far larger on the CPU).
    """
    if use_gpu:
        ecm_path = wrapper.typed_config.programs.gmp_ecm.path
        if gpu_stage2_supported(ecm_path):
            return GPUStage2Executor(wrapper, residue_file, b1, b2, verbose,
                                     gpu_device=gpu_device, gpu_curves=gpu_curves,
                                     fallback_k=k, fallback_workers=workers,
                                     fallback_b2=cpu_b2, pin_threads=pin_threads)
        wrapper.logger.warning(
            f"GMP-ECM at {ecm_path} has no GPU stage 2 (no -cpustage2 in its "
            f"usage text); running stage 2 on {workers} CPU workers instead")
    return Stage2Executor(wrapper, residue_file, b1,
                          cpu_b2 if cpu_b2 is not None else b2, k, workers, verbose,
                          pin_threads=pin_threads)


Stage2Outcome = Tuple[Optional[str], List[str], int, float, Optional[str]]


def residue_params(residue_file: Path) -> Set[int]:
    """The -param values (PARAM=) of the curves in a residue file."""
    try:
        with open(residue_file, 'r', encoding='utf-8') as f:
            return {int(p) for p in re.findall(r'\bPARAM=(\d+)', f.read())}
    except OSError:
        return set()


def save_stage2_log(name: str, output: str, logger) -> Optional[Path]:
    """Save stage 2 ECM output to <tmp>/ecm_stage2_logs/<name>_<time>.log."""
    try:
        import tempfile
        output_dir = Path(tempfile.gettempdir()) / "ecm_stage2_logs"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{name}_{int(time.time())}.log"
        with open(output_file, 'w') as f:
            f.write(output)
        return output_file
    except Exception as e:
        logger.warning(f"Failed to save {name} stage 2 output: {e}")
        return None


class GPUStage2Executor:
    """
    Stage 2 for a whole residue file in a single GPU process.

    Runs `ecm -gpu -resume FILE B1-B1 B2`: GMP-ECM gathers consecutive
    residues (same N, same B1, -param 3) into GPU batches itself, so the file
    is never split. Same execute() contract and raw_output attribute as
    Stage2Executor, so callers can use either.

    Differences from the CPU executor:
    - No -k: k is tuned for CPU memory; the GPU picks its own stage 2 shape.
    - The B2 GMP-ECM actually ran is parsed into b2_used: an explicit B2 is
      rounded up (e.g. 1.5e9 -> 2.14e9), and where the polynomial GPU stage 2
      cannot run the default B2 is 100*B1. Callers should report b2_used.
    - Residues GMP-ECM cannot batch (other -param, other N/B1) run one at a
      time on the CPU inside the same process; they still count.
    - Ctrl+C level 2 ("stop after current curve") stops after the current
      GPU batch: a batch is all-or-nothing, so killing it mid-way discards
      every curve in it. With no --gpu-curves cap, a file is one batch.
    - If the GPU run fails (exit 1: out of GPU memory, device error, ...) the
      residue file is rerun on fallback_workers CPU workers, when given;
      otherwise `failed` is set and the caller must not report success.
    """

    def __init__(self, wrapper: 'ECMWrapper', residue_file: Path, b1: int,
                 b2: Optional[int], verbose: bool = False, *,
                 gpu_device: Optional[int] = None,
                 gpu_curves: Optional[int] = None,
                 fallback_k: Optional[int] = None,
                 fallback_workers: int = 0,
                 fallback_b2: Optional[int] = None,
                 pin_threads: bool = False) -> None:
        self.wrapper = wrapper
        self.residue_file = residue_file
        self.b1 = b1
        self.b2 = b2
        self.verbose = verbose
        self.gpu_device = gpu_device
        self.gpu_curves = gpu_curves
        self.fallback_k = fallback_k
        self.fallback_workers = fallback_workers
        self.fallback_b2 = fallback_b2
        self.pin_threads = pin_threads
        self.logger = wrapper.logger
        self.ecm_path = wrapper.typed_config.programs.gmp_ecm.path
        self.stop_event = wrapper.stop_event
        self.raw_output: str = ""
        self.b2_used: Optional[int] = None
        self.failed = False
        self.error_message: Optional[str] = None
        self.factor_sigmas: List[Tuple[str, Optional[str]]] = []

    def build_command(self, b1: int, early_termination: bool) -> List[str]:
        return build_ecm_command(
            self.ecm_path, b1, b2=self.b2,
            use_gpu=True, gpu_device=self.gpu_device, gpu_curves=self.gpu_curves,
            residue_load=self.residue_file,
            verbose=self.verbose, one=early_termination,
            b1done=b1,
        )

    def execute(self, early_termination: bool = True,
                progress_interval: int = 0) -> Stage2Outcome:
        """
        Run GPU stage 2 over the residue file.

        progress_interval only applies to a CPU fallback run; GPU progress is
        reported per batch (every completed batch is logged).

        Returns:
            (factor, all_factors, curves_completed, execution_time, sigma),
            as Stage2Executor.execute(). Every factor is in all_factors; the
            sigma of each is in self.factor_sigmas.
        """
        start_time = time.time()
        self.raw_output = ""
        self.b2_used = None
        self.failed = False
        self.error_message = None
        self.factor_sigmas = []

        # Ctrl+C level 2 before stage 2 started (e.g. during the residue
        # download): honour it rather than starting a long GPU run.
        if self.wrapper.graceful_shutdown_requested is True:
            self.logger.info("GPU stage 2: stop requested before start, not running")
            return (None, [], 0, 0.0, None)
        # A previous executor may have left stop_event set (factor found)
        self.stop_event.clear()

        params = residue_params(self.residue_file)
        if params != {3}:
            # GPU stage 2 only batches -param 3 residues (what GPU stage 1
            # makes); others would run one at a time on a single core.
            param_list = ", ".join(str(p) for p in sorted(params)) or "unknown"
            return self._run_on_cpu(
                f"Residues are -param {param_list}, GPU stage 2 needs -param 3",
                early_termination, progress_interval, start_time)

        residue_info = self.wrapper._parse_residue_file(self.residue_file)
        actual_b1 = residue_info['b1']
        if actual_b1 > 0 and actual_b1 != self.b1:
            self.logger.info(f"Using B1={actual_b1} from residue file (overriding parameter B1={self.b1})")
        b1_to_use = actual_b1 if actual_b1 > 0 else self.b1
        total = residue_info.get('curve_count', 0)

        cmd = self.build_command(b1_to_use, early_termination)
        self.logger.info(f"GPU stage 2: {total or '?'} residues, B1={b1_to_use}, "
                         f"B2={'default' if self.b2 in (None, -1) else self.b2}")
        self.logger.debug(f"GPU stage 2 command: {' '.join(cmd)}")

        lines: List[str] = []
        done = 0
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,  # Ctrl+C is handled by the wrapper, not ECM
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                if (m := ECMPatterns.GPU_STAGE2_BATCH.match(line)):
                    done += int(m.group(1))  # progress only; final count below
                    self.logger.info(f"GPU stage 2: {done}/{total or '?'} curves done "
                                     f"({line.strip()})")
                elif "GPU: Error" in line or line.startswith("GPU: stage 2 on the CPU") \
                        or line.startswith("GPU: baby-step giant-step"):
                    self.logger.warning(f"GPU stage 2: {line.strip()}")
                if self.stop_event.is_set() and (
                        ECMPatterns.GPU_STAGE2_BATCH.match(line)
                        or ECMPatterns.GPU_BATCH_CPU_STAGE2_CURVE.match(line)
                        or ECMPatterns.CPU_STAGE2_CURVE.match(line)):
                    self.logger.info("GPU stage 2: stopping after the completed curves")
                    process.terminate()
                    break
            process.wait()
        finally:
            # Level-3 Ctrl+C raises KeyboardInterrupt out of the read loop:
            # the ECM process is in its own session, so it must be killed here.
            if process.poll() is None:
                process.kill()
                process.wait()
            self.raw_output = "".join(lines)
            log_file = save_stage2_log("gpu", self.raw_output, self.logger)
            if log_file:
                self.logger.info(f"GPU stage 2 output saved to: {log_file}")

        exit_code = process.returncode
        stopped = self.stop_event.is_set()
        if (exit_code == 1 or exit_code < 0) and not stopped:
            # GMP-ECM: 1 = error; 0 = no factor; even codes = factor found
            # (bit flags for prime factor / prime cofactor / input found).
            # Negative = killed by a signal we did not send.
            self.logger.error(f"GPU stage 2 exited with code {exit_code}; "
                              f"last output:\n{''.join(lines[-10:])}")
            return self._run_on_cpu(f"GPU stage 2 failed (exit code {exit_code})",
                                    early_termination, progress_interval, start_time)

        curves = count_completed_curves(self.raw_output)
        self.b2_used = parse_b2_used(self.raw_output)
        found = parse_ecm_output_multiple(self.raw_output)
        self.factor_sigmas = list(found)
        execution_time = time.time() - start_time
        self.logger.info(f"GPU stage 2 finished: {curves} curves in {execution_time:.1f}s"
                         + (f", B2={self.b2_used}" if self.b2_used else "")
                         + (f", factor {found[0][0]}" if found else ""))
        if found:
            factor, sigma = found[0]
            return (factor, [f for f, _ in found], curves, execution_time, sigma)
        return (None, [], curves, execution_time, None)

    def _run_on_cpu(self, reason: str, early_termination: bool,
                    progress_interval: int, start_time: float) -> Stage2Outcome:
        """Run the whole residue file on CPU workers, or flag the failure."""
        gpu_output = self.raw_output
        if self.fallback_workers <= 0:
            self.logger.error(f"{reason}; no CPU workers to run stage 2 on")
            self.failed = True
            self.error_message = reason
            return (None, [], 0, time.time() - start_time, None)

        # The GPU's partial work (if any) is discarded: the CPU rerun covers
        # every residue, so its curve count is the whole truth.
        self.logger.warning(f"{reason}; running the residue file on "
                            f"{self.fallback_workers} CPU workers")
        cpu_b2 = self.fallback_b2 if self.fallback_b2 is not None else self.b2
        cpu = Stage2Executor(self.wrapper, self.residue_file, self.b1, cpu_b2,
                             self.fallback_k, self.fallback_workers, self.verbose,
                             pin_threads=self.pin_threads)
        factor, all_factors, curves, _, sigma = cpu.execute(
            early_termination=early_termination, progress_interval=progress_interval)
        self.raw_output = (gpu_output + "\n" + cpu.raw_output) if gpu_output else cpu.raw_output
        self.b2_used = cpu.b2_used
        self.factor_sigmas = list(cpu.factor_sigmas)
        return (factor, all_factors, curves, time.time() - start_time, sigma)


class Stage2Executor:
    """Manages Stage 2 execution with configurable worker pools."""

    def __init__(self, wrapper: 'ECMWrapper', residue_file: Path, b1: int, b2: Optional[int],
                 k: Optional[int], workers: int, verbose: bool = False,
                 pin_threads: bool = False) -> None:
        """
        Initialize Stage 2 executor.

        Args:
            wrapper: ECMWrapper instance (for config and residue manager)
            residue_file: Path to residue file from Stage 1
            b1: B1 parameter (will be extracted from residue file if available)
            b2: B2 parameter for Stage 2
            workers: Number of parallel workers
            verbose: Enable verbose output
            pin_threads: If True, pin each worker's ECM subprocess to its own CPU core
        """
        self.wrapper = wrapper
        self.residue_file = residue_file
        self.b1 = b1
        self.b2 = b2
        self.k = k
        self.workers = workers
        self.verbose = verbose
        self.pin_threads = pin_threads
        self.pin_cpus: Optional[List[int]] = None  # Set in execute() when pin_threads is True
        self.logger = wrapper.logger
        self.ecm_path = wrapper.typed_config.programs.gmp_ecm.path

        # Shared state for worker coordination
        self.factor_found: Optional[Tuple[str, str]] = None
        self.factor_lock = threading.Lock()
        # Use wrapper's stop_event for multi-level shutdown support
        # (Ctrl+C level 2 sets this to stop after current curve)
        self.stop_event = wrapper.stop_event
        self.running_processes: List[subprocess.Popen] = []
        self.process_lock = threading.Lock()
        self.curves_completed_total = 0
        self.curves_lock = threading.Lock()
        # Aggregated raw output from all workers
        self.raw_output: str = ""
        self.output_lock = threading.Lock()
        # Same result fields as GPUStage2Executor. The CPU runs exactly the
        # B2 it is given (b2_used None only for GMP-ECM's default) and does
        # not fail as a whole (failed False).
        self.b2_used: Optional[int] = b2 if b2 not in (None, -1) else None
        self.failed = False
        self.error_message: Optional[str] = None
        self.factor_sigmas: List[Tuple[str, Optional[str]]] = []

    def execute(self, early_termination: bool = True,
                progress_interval: int = 0) -> Tuple[Optional[str], List[str], int, float, Optional[str]]:
        """
        Execute Stage 2 with worker pool.

        Args:
            early_termination: If True, stop all workers when first factor found
            progress_interval: Report progress every N curves (0 = no progress reporting)

        Returns:
            Tuple of (factor, all_factors, curves_completed, execution_time, sigma)
            - factor: First factor found (or None)
            - all_factors: List of all factors found (empty list if none)
            - curves_completed: Total curves processed
            - execution_time: Total execution time in seconds
            - sigma: Sigma value that found the factor (or None)
        """
        start_time = time.time()

        # CRITICAL: Clear stop_event at start of each execution
        # This prevents a factor found in a previous run from stopping this run
        self.stop_event.clear()
        self.factor_found = None  # Also reset factor state
        self.raw_output = ""  # Reset aggregated output
        self.factor_sigmas = []

        # Extract B1 from residue file to ensure consistency
        residue_info = self.wrapper._parse_residue_file(self.residue_file)
        actual_b1 = residue_info['b1']
        if actual_b1 > 0 and actual_b1 != self.b1:
            self.logger.info(f"Using B1={actual_b1} from residue file (overriding parameter B1={self.b1})")
        b1_to_use = actual_b1 if actual_b1 > 0 else self.b1

        # Split residue file into chunks for workers
        residue_chunks = self._split_residue_file()

        if not residue_chunks:
            self.logger.error("Failed to split residue file")
            execution_time = time.time() - start_time
            return (None, [], 0, execution_time, None)

        # Resolve CPU pin assignments if requested (one per chunk/worker actually used)
        if self.pin_threads:
            from .thread_pinning import resolve_pin_assignments
            self.pin_cpus = resolve_pin_assignments(len(residue_chunks))
            self.logger.info(f"Pinning stage 2 workers to CPUs: {self.pin_cpus}")

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i, chunk_file in enumerate(residue_chunks):
                pin_cpu = self.pin_cpus[i] if self.pin_cpus else None
                future = executor.submit(
                    self._worker_stage2, chunk_file, i + 1, b1_to_use,
                    early_termination, progress_interval, pin_cpu
                )
                futures.append(future)

            # Wait for completion or first factor
            graceful_shutdown = False
            for future in as_completed(futures):
                # Check for graceful shutdown request from main thread
                if hasattr(self.wrapper, 'graceful_shutdown_requested') and self.wrapper.graceful_shutdown_requested:
                    self.logger.info("Stage 2: Graceful shutdown requested - letting workers finish current chunks")
                    graceful_shutdown = True
                    # Don't set stop_event here - let workers finish their current chunks naturally
                    # Just stop waiting for remaining futures
                    break

                try:
                    result = future.result()
                    if result:
                        self.factor_found = result  # This is (factor, sigma) tuple
                        self.stop_event.set()  # Ensure all workers are signaled to stop
                        break
                except Exception as e:
                    self.logger.error(f"Worker thread error: {e}")

            # Ensure all remaining processes are terminated (but not during graceful shutdown)
            if not graceful_shutdown:
                with self.process_lock:
                    for process in self.running_processes:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
            else:
                # During graceful shutdown, wait for workers to complete naturally
                self.logger.info("Stage 2: Waiting for workers to complete their current chunks...")
                # The ThreadPoolExecutor context manager will wait for all workers to finish

        # Cleanup temporary chunk files and directory
        self._cleanup_chunks(residue_chunks)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Return factor info along with curves completed
        if self.factor_found:
            # Extract factor and sigma from tuple (factor, sigma)
            factor = self.factor_found[0]
            sigma = self.factor_found[1]
            all_factors = [factor] if factor else []
            self.factor_sigmas = [(f, sigma) for f in all_factors]
            return (factor, all_factors, self.curves_completed_total, execution_time, sigma)

        return (None, [], self.curves_completed_total, execution_time, None)

    def _split_residue_file(self) -> List[Path]:
        """Split residue file into chunks for parallel processing."""
        import tempfile
        chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
        self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

        # Use ResidueFileManager to split the file
        chunk_paths = self.wrapper.residue_manager.split_into_chunks(
            str(self.residue_file), self.workers, chunk_dir
        )

        # Convert string paths to Path objects
        return [Path(p) for p in chunk_paths]

    def _worker_stage2(self, chunk_file: Path, worker_id: int, b1: int,
                       early_termination: bool, progress_interval: int,
                       pin_cpu: Optional[int] = None) -> Optional[Tuple[str, str]]:
        """Worker function for Stage 2 processing."""
        cmd = build_ecm_command(
            self.ecm_path, b1, b2=self.b2, k=self.k,
            residue_load=chunk_file,
            verbose=self.verbose, one=True,
            b1done=b1,
        )

        # Count total lines in this worker's chunk for progress reporting and diagnostics
        total_lines = 0
        try:
            with open(chunk_file, 'r') as f:
                total_lines = sum(1 for _ in f)
        except:
            total_lines = 0

        # Initialize process to None in case exception occurs before Popen
        process = None

        # Build preexec_fn to pin the ECM subprocess to its CPU before exec.
        # Using preexec_fn (not setting affinity post-fork in Python) ensures the
        # affinity is set before the ECM binary starts running — no migration window.
        preexec_fn = None
        if pin_cpu is not None:
            import os
            cpu = pin_cpu  # bind for closure
            def _pin() -> None:
                os.sched_setaffinity(0, {cpu})
            preexec_fn = _pin

        try:
            self.logger.info(f"Worker {worker_id} starting Stage 2" +
                           (f" ({total_lines} curves)" if self.verbose and total_lines > 0 else "") +
                           (f" pinned to CPU {pin_cpu}" if pin_cpu is not None else ""))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,  # Isolate from terminal SIGINT so Ctrl+C doesn't kill workers
                preexec_fn=preexec_fn,
            )

            # Register process for potential termination
            with self.process_lock:
                self.running_processes.append(process)

            # Stream output line by line to enable curve-level shutdown
            # (stop_event is checked after each curve completes)
            full_output = self._stream_worker_output(
                process, worker_id, total_lines, progress_interval, early_termination
            )

            # Check if we were stopped due to shutdown request
            if early_termination and self.stop_event.is_set():
                self.logger.info(f"Worker {worker_id} stopped due to shutdown request")
                # Don't return None - we want to count the curves we did complete

            # Count curve completions from output
            curves_completed = full_output.count("Step 2 took")

            # Add to total curves completed (thread-safe)
            with self.curves_lock:
                self.curves_completed_total += curves_completed

            # Aggregate raw output (thread-safe)
            with self.output_lock:
                if self.raw_output:
                    self.raw_output += f"\n\n=== Worker {worker_id} ===\n"
                else:
                    self.raw_output = f"=== Worker {worker_id} ===\n"
                self.raw_output += full_output

            # Progress reporting in verbose mode
            if self.verbose and total_lines > 0:
                percentage = (curves_completed / total_lines) * 100
                self.logger.info(f"Worker {worker_id} progress: {curves_completed}/{total_lines} curves - {percentage:.1f}% complete")

            # Save raw output to file for debugging
            self._save_worker_output(worker_id, full_output)

            # Check for factor - enable debug mode if worker stopped early
            enable_debug = self._should_enable_debug(curves_completed, total_lines)
            factor, sigma_from_output = parse_ecm_output(full_output, debug=enable_debug)

            if factor:
                # Ensure sigma is not None for type safety
                sigma_value = sigma_from_output if sigma_from_output is not None else ""
                with self.factor_lock:
                    if not self.factor_found:  # First factor wins
                        self.factor_found = (factor, sigma_value)
                        if early_termination:
                            self.stop_event.set()  # Signal other workers to stop
                        self.logger.info(f"Worker {worker_id} found factor: {factor} (sigma: {sigma_value})")
                        # Kill other processes if early termination enabled
                        if early_termination:
                            self._terminate_other_processes(process)
                return (factor, sigma_value)

            # If no factor found, report completion with diagnostic
            output_size = len(full_output)
            self.logger.info(f"Worker {worker_id} completed (no factor) - {curves_completed} curves, {output_size} bytes output")

            # Show diagnostic if worker stopped early
            if self._should_enable_debug(curves_completed, total_lines):
                if total_lines > 0:
                    self.logger.warning(f"Worker {worker_id} stopped early at {curves_completed}/{total_lines} curves")
                else:
                    self.logger.warning(f"Worker {worker_id} stopped early at {curves_completed} curves (expected count unknown)")
                self.logger.debug(f"Worker {worker_id} output preview (first 500 chars):\n{full_output[:500]}")
                self.logger.debug(f"Worker {worker_id} output preview (last 500 chars):\n{full_output[-500:]}")

            return None

        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
            return None
        finally:
            # Remove process from tracking (only if it was created)
            if process is not None:
                with self.process_lock:
                    if process in self.running_processes:
                        self.running_processes.remove(process)

    def _stream_worker_output(self, process: subprocess.Popen, worker_id: int,
                              total_lines: int, progress_interval: int,
                              early_termination: bool) -> str:
        """Stream output from worker with progress tracking."""
        full_output = ""
        last_progress_report = 0
        curves_completed = 0

        if not process.stdout:
            return full_output

        while True:
            line = process.stdout.readline()
            if not line:
                break

            full_output += line

            # Check if we should terminate early
            if early_termination and self.stop_event.is_set():
                process.terminate()
                self.logger.info(f"Worker {worker_id} terminating due to factor found elsewhere")
                break

            # Check for curve completion and progress reporting
            if "Step 2 took" in line:
                # Incremented, not recounted: full_output.count(...) rescanned
                # the whole accumulated transcript once per curve, which is
                # quadratic in a worker's output and burns the same cores
                # stage 2 is using.
                curves_completed += 1

                # Report progress at intervals. The > 0 test matters: with
                # progress_interval == 0 the difference test alone is always
                # true, which logged a line for every single curve instead of
                # disabling reporting as documented above.
                if (progress_interval > 0
                        and curves_completed - last_progress_report >= progress_interval):
                    if total_lines > 0:
                        percentage = (curves_completed / total_lines) * 100
                        self.logger.info(f"Worker {worker_id}: {curves_completed}/{total_lines} curves ({percentage:.1f}%)")
                    else:
                        self.logger.info(f"Worker {worker_id}: {curves_completed} curves completed")
                    last_progress_report = curves_completed

        process.wait()

        # CRITICAL: Drain any remaining buffered output after process exits
        if process.stdout:
            remaining = process.stdout.read()
            if remaining:
                full_output += remaining
                self.logger.debug(f"Worker {worker_id}: Drained {len(remaining)} chars from buffer after process exit")

        return full_output

    def _terminate_other_processes(self, current_process: subprocess.Popen):
        """Terminate all processes except the current one."""
        with self.process_lock:
            processes_to_terminate = [p for p in self.running_processes
                                     if p != current_process and p.poll() is None]
            for p in processes_to_terminate:
                p.terminate()

    def _should_enable_debug(self, curves_completed: int, total_lines: int) -> bool:
        """Determine if debug output should be enabled based on completion rate."""
        if total_lines > 0:
            return curves_completed < total_lines * 0.9
        else:
            # If we don't know total_lines, assume each chunk should do ~300+ curves
            return curves_completed < 300

    def _save_worker_output(self, worker_id: int, output: str):
        """Save worker output to file for debugging."""
        output_file = save_stage2_log(f"worker_{worker_id}", output, self.logger)
        if output_file:
            self.logger.debug(f"Worker {worker_id} output saved to: {output_file}")

    def _cleanup_chunks(self, residue_chunks: List[Path]):
        """Cleanup temporary chunk files and directories."""
        chunk_dirs_to_cleanup = set()
        for chunk_file in residue_chunks:
            try:
                chunk_dirs_to_cleanup.add(chunk_file.parent)
                chunk_file.unlink()
            except:
                pass

        # Clean up temporary chunk directories
        for chunk_dir in chunk_dirs_to_cleanup:
            try:
                import shutil
                shutil.rmtree(chunk_dir)
                self.logger.debug(f"Cleaned up chunk directory: {chunk_dir}")
            except:
                pass
