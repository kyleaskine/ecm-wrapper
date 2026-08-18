#!/usr/bin/env python3
"""
CompositeExecutionEngine - Unified execution orchestration for ECM.

Owns the GPU-producer + CPU-consumer orchestration pattern. All execution
methods in ECMWrapper become thin facades over this engine.

Methods:
  run_cpu_workers() - Multiprocess CPU execution
  run_two_stage()   - GPU stage 1 + CPU stage 2 pipeline
  run_pipelined()   - GPU producer + CPU consumer with batch producer
"""
import multiprocessing as mp
import multiprocessing.managers
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple, TYPE_CHECKING

from .ecm_config import ExecutionBatch, BatchResult, FactorResult, ECMPrimitiveResult
from .ecm_worker_process import run_worker_ecm_process

if TYPE_CHECKING:
    from .ecm_executor import ECMWrapper
    from .stage2_executor import Stage2Executor


class CompositeExecutionEngine:
    """
    Unified execution engine for ECM factorization.

    Encapsulates the orchestration patterns shared across execution modes:
    - Multiprocess worker spawning and result collection
    - GPU stage 1 + CPU stage 2 pipeline
    - Pipelined GPU producer + CPU consumer with backpressure

    The engine takes an ECMWrapper reference for access to primitives
    (_execute_ecm_primitive, _run_stage1_primitive, etc.) and shared state
    (stop_event, interrupted, etc.).
    """

    def __init__(self, wrapper: 'ECMWrapper'):
        self.wrapper = wrapper
        self.logger = wrapper.logger
        self.ecm_path = wrapper.typed_config.programs.gmp_ecm.path

    # ==================== Phase 1: Multiprocess CPU Workers ====================

    def run_cpu_workers(self, batch: ExecutionBatch, num_workers: int,
                        pin_threads: bool = False) -> BatchResult:
        """
        Execute ECM across multiple CPU workers using multiprocessing.

        Distributes curves evenly across workers, collects results, handles
        early termination on factor found, and performs process cleanup with
        SIGTERM -> SIGKILL escalation.

        Args:
            batch: Execution parameters (composite, b1, b2, curves, etc.)
            num_workers: Number of parallel worker processes
            pin_threads: If True, pin each worker to a distinct CPU core (Linux only).

        Returns:
            BatchResult with discovered factors (recursively factored to primes)
        """
        self.logger.info(f"Running multi-process ECM: {num_workers} workers, {batch.curves} total curves")
        start_time = time.time()

        # Distribute curves across workers
        curves_per_worker = batch.curves // num_workers
        remaining_curves = batch.curves % num_workers
        worker_assignments: List[Tuple[int, int]] = []

        for worker_id in range(num_workers):
            worker_curves = curves_per_worker + (1 if worker_id < remaining_curves else 0)
            if worker_curves > 0:
                worker_assignments.append((worker_id + 1, worker_curves))

        # Resolve CPU pin assignments if requested (one CPU per active worker)
        pin_cpus: Optional[List[int]] = None
        if pin_threads:
            from .thread_pinning import resolve_pin_assignments
            pin_cpus = resolve_pin_assignments(len(worker_assignments))
            self.logger.info(f"Pinning workers to CPUs: {pin_cpus}")

        # Create shared variables for worker coordination
        manager = mp.Manager()
        result_queue = manager.Queue()
        progress_queue = manager.Queue()
        stop_event = manager.Event()

        # Start worker processes
        processes: List[mp.Process] = []
        for idx, (worker_id, worker_curves) in enumerate(worker_assignments):
            pin_cpu = pin_cpus[idx] if pin_cpus else None
            p = mp.Process(
                target=run_worker_ecm_process,
                args=(worker_id, batch.composite, batch.b1, batch.b2, worker_curves,
                      batch.verbose, batch.method, self.ecm_path,
                      result_queue, stop_event, batch.progress_interval, progress_queue,
                      pin_cpu)
            )
            p.start()
            processes.append(p)

        # Collect results from workers
        all_factors: List[str] = []
        all_sigmas: List[Optional[str]] = []
        all_raw_outputs: List[str] = []
        total_curves_completed = 0
        completed_workers = 0
        interrupted = False

        def process_result(result: dict) -> None:
            nonlocal total_curves_completed
            if result is None:
                return
            total_curves_completed += result.get('curves_completed', 0)

            if result.get('factor_found'):
                factor = result['factor_found']
                # Deduplicate: multiple workers can find the same factor
                if factor not in all_factors:
                    all_factors.append(factor)
                    all_sigmas.append(result.get('sigma_found'))
                    self.logger.info(f"Worker {result['worker_id']} found factor: {factor}")
                else:
                    self.logger.info(f"Worker {result['worker_id']} found same factor: {factor} (already recorded)")
                stop_event.set()  # Signal workers to stop on factor found

            # Collect raw output from each worker
            if 'raw_output' in result:
                all_raw_outputs.append(f"=== Worker {result['worker_id']} ===\n{result['raw_output']}")

        try:
            while completed_workers < len(processes):
                try:
                    result = result_queue.get(timeout=0.5)
                    process_result(result)
                    completed_workers += 1
                except queue.Empty:
                    # Check if stop was requested externally (e.g. work mode signal handler)
                    if self.wrapper.stop_event.is_set() or self.wrapper.interrupted:
                        self.logger.info("Multiprocess ECM stop detected via signal handler")
                        interrupted = True
                        self.wrapper.interrupted = True
                        try:
                            stop_event.set()
                        except Exception:
                            pass
                        break
                    # Timeout waiting for result - check if processes are still alive
                    if not any(p.is_alive() for p in processes):
                        # Drain any remaining results from the queue before breaking
                        # (processes may have exited after putting results in queue)
                        while True:
                            try:
                                result = result_queue.get_nowait()
                                process_result(result)
                                completed_workers += 1
                            except queue.Empty:
                                break
                        break
                    continue
        except KeyboardInterrupt:
            self.logger.info("Multiprocess ECM interrupted by user")
            interrupted = True
            self.wrapper.interrupted = True
            try:
                stop_event.set()  # Signal workers to stop
            except Exception:
                pass  # Manager connection may be broken

        # If interrupted, drain remaining worker results before cleanup
        if interrupted:
            try:
                deadline = time.time() + 15  # Give workers up to 15s to wrap up
                while completed_workers < len(processes) and time.time() < deadline:
                    try:
                        result = result_queue.get(timeout=1.0)
                        process_result(result)
                        completed_workers += 1
                    except queue.Empty:
                        if not any(p.is_alive() for p in processes):
                            # All dead, drain anything left
                            while True:
                                try:
                                    result = result_queue.get_nowait()
                                    process_result(result)
                                    completed_workers += 1
                                except queue.Empty:
                                    break
                            break
                    except KeyboardInterrupt:
                        # Another Ctrl+C during drain — give up waiting
                        self.logger.info("Interrupt during drain, skipping result collection")
                        break
            except (BrokenPipeError, EOFError, ConnectionError, OSError):
                # Manager connection broken — can't drain results
                self.logger.info("Manager connection lost, skipping result collection")

        # Clean up processes and manager
        self._cleanup_processes(processes, manager)

        if interrupted:
            print(f"\nMultiprocess ECM stopped. Completed {total_curves_completed} curves before interrupt.")

        # Build BatchResult with recursive factoring of any composite factors
        # Deduplicate primes after full factoring: if Worker A finds prime p and
        # Worker B finds composite p*q, the prime p appears in both expansions.
        # Only count each distinct prime once (multiplicities are handled by the
        # caller via division, not by counting occurrences in this list).
        batch_result = BatchResult()
        seen_primes: set = set()
        for factor, sigma in zip(all_factors, all_sigmas):
            prime_factors = self.wrapper._fully_factor_composite(factor)
            for prime in prime_factors:
                if prime not in seen_primes:
                    seen_primes.add(prime)
                    batch_result.factors.append(prime)
                    batch_result.sigmas.append(sigma)

        batch_result.curves_run = total_curves_completed
        batch_result.execution_time = time.time() - start_time
        batch_result.success = len(batch_result.factors) > 0
        batch_result.raw_output = '\n\n'.join(all_raw_outputs) if all_raw_outputs else None
        batch_result.interrupted = interrupted

        return batch_result

    def _cleanup_processes(self, processes: List[mp.Process], manager: multiprocessing.managers.SyncManager) -> None:
        """
        Clean up worker processes with SIGTERM -> SIGKILL escalation.

        Suppresses SIGINT during cleanup to prevent additional Ctrl+C from
        breaking out of the kill loop and leaving orphaned child processes.
        """
        prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # Wait for all processes to finish.
            # Use os.killpg() to kill the entire process group (worker + ECM binary)
            # since each worker calls os.setpgrp() and ECM inherits that group.
            for p in processes:
                try:
                    p.join(timeout=10)
                except Exception:
                    pass
                if p.is_alive():
                    # SIGTERM the entire process group (worker + its ECM child)
                    try:
                        if p.pid is not None:
                            if sys.platform != 'win32':
                                os.killpg(p.pid, signal.SIGTERM)
                            else:
                                p.terminate()
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        p.join(timeout=5)
                    except Exception:
                        pass
                if p.is_alive():
                    # Escalate to SIGKILL if SIGTERM didn't work
                    try:
                        if p.pid is not None:
                            if sys.platform != 'win32':
                                os.killpg(p.pid, signal.SIGKILL)
                            else:
                                p.kill()
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        p.join(timeout=2)
                    except Exception:
                        pass

            # Shutdown the manager in a daemon thread to avoid hanging
            def _shutdown_manager() -> None:
                try:
                    manager.shutdown()
                except Exception:
                    pass

            shutdown_thread = threading.Thread(target=_shutdown_manager, daemon=True)
            shutdown_thread.start()
            shutdown_thread.join(timeout=5)
        finally:
            signal.signal(signal.SIGINT, prev_handler)

    # ==================== Phase 2: Two-Stage GPU+CPU Pipeline ====================

    def run_two_stage(
        self,
        composite: str,
        b1: int,
        b2: Optional[int],
        stage1_curves: int,
        stage2_workers: int,
        *,
        stage1_parametrization: int = 3,
        verbose: bool = False,
        progress_interval: int = 0,
        use_gpu: bool = True,
        gpu_device: Optional[int] = None,
        gpu_curves: Optional[int] = None,
        resume_file: Optional[Path] = None,
        save_residues: Optional[str] = None,
    ) -> BatchResult:
        """
        Execute two-stage ECM pipeline: GPU stage 1 + CPU stage 2.

        If resume_file is provided, skips stage 1 and processes existing residue.
        Installs 3-level Ctrl+C handler for graceful shutdown.

        Args:
            composite: Number to factor
            b1: B1 bound
            b2: B2 bound (None = B1 * 100)
            stage1_curves: Number of curves for stage 1
            stage2_workers: Number of CPU threads for stage 2
            stage1_parametrization: Parametrization for stage 1 (default 3 = GPU)
            verbose: Enable verbose output
            progress_interval: Progress reporting interval
            use_gpu: Use GPU for stage 1
            gpu_device: GPU device number
            gpu_curves: Curves per GPU batch
            resume_file: Path to existing residue file (skip stage 1)
            save_residues: Path to save residue file

        Returns:
            BatchResult with discovered factors
        """
        start_time = time.time()

        # Install 3-level Ctrl+C handler
        self._install_graceful_handler()

        try:
            if resume_file:
                residue_file, actual_b1, stage1_result = self._handle_resume(
                    resume_file, b1
                )
                if residue_file is None:
                    # Resume file not found
                    return BatchResult(execution_time=time.time() - start_time)
            else:
                # Normal flow: create residue path and run stage 1
                residue_file = self._create_residue_path_for_two_stage(
                    composite, save_residues
                )
                actual_b1 = b1

                stage1_result = self.wrapper._run_stage1_primitive(
                    composite=composite,
                    b1=b1,
                    curves=stage1_curves,
                    residue_file=residue_file,
                    use_gpu=use_gpu,
                    param=stage1_parametrization,
                    verbose=verbose,
                    gpu_device=gpu_device,
                    gpu_curves=gpu_curves,
                )

                # Early return if factor found in stage 1
                if stage1_result['factors']:
                    return self._stage1_result_to_batch_result(
                        stage1_result, start_time
                    )

                # Skip stage 2 if b2=0 (stage 1 only mode)
                if b2 == 0:
                    self.logger.info("B2=0: Skipping stage 2 (stage 1 only mode)")
                    return self._stage1_result_to_batch_result(
                        stage1_result, start_time
                    )

            # Stage 2: Multi-threaded CPU processing
            actual_b2 = b2 or (actual_b1 * 100)
            self.logger.info(
                f"Starting stage 2: B1={actual_b1:,}, B2={actual_b2:,}, workers={stage2_workers}"
            )

            from .stage2_executor import Stage2Executor

            executor = Stage2Executor(
                self.wrapper, residue_file, actual_b1, actual_b2, None,
                stage2_workers, verbose
            )
            self.wrapper._active_stage2_executor = executor
            try:
                _factor, all_factors, curves_completed, _stage2_time, sigma = (
                    executor.execute(
                        early_termination=True,
                        progress_interval=progress_interval,
                    )
                )
            finally:
                self.wrapper._active_stage2_executor = None

            # Build result
            result = BatchResult()
            if all_factors:
                for f in all_factors:
                    result.factors.append(f)
                    result.sigmas.append(sigma)
            result.curves_run = curves_completed
            result.execution_time = time.time() - start_time
            result.success = len(all_factors) > 0

            # Check if execution was gracefully shutdown
            if self.wrapper.shutdown_level >= 1:
                result.interrupted = True
                self.logger.info(
                    f"Graceful shutdown completed - processed {curves_completed} curves"
                )

            return result
        finally:
            self._restore_graceful_handler()

    def _handle_resume(
        self, resume_file: Path, config_b1: int
    ) -> Tuple[Optional[Path], int, Optional[ECMPrimitiveResult]]:
        """
        Handle resume from existing residue file.

        Returns:
            (residue_file, actual_b1, stage1_result) - residue_file is None if not found
        """
        if not resume_file.exists():
            self.logger.error(f"Resume file not found: {resume_file}")
            return (None, config_b1, None)

        residue_info = self.wrapper._parse_residue_file(resume_file)
        stage1_b1 = residue_info['b1']
        stage1_curves = residue_info['curve_count']

        if stage1_b1 > 0:
            if stage1_b1 != config_b1:
                self.logger.info(
                    f"Using B1={stage1_b1} from residue file (overriding config B1={config_b1})"
                )
            actual_b1 = stage1_b1
        else:
            actual_b1 = config_b1

        self.logger.info(f"Resuming from residue file: {resume_file}")
        self.logger.info(f"Stage 1 completed {stage1_curves} curves at B1={actual_b1}")

        return (resume_file, actual_b1, None)

    def _create_residue_path_for_two_stage(
        self, composite: str, save_residues: Optional[str]
    ) -> Path:
        """Create residue file path for two-stage mode."""
        if save_residues:
            return Path(save_residues)

        import hashlib

        residue_dir = Path(self.wrapper.typed_config.execution.residue_dir)
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        composite_hash = hashlib.md5(composite.encode()).hexdigest()[:8]
        return residue_dir / f"stage1_{composite_hash}_{timestamp}.txt"

    def _stage1_result_to_batch_result(
        self, prim_result: ECMPrimitiveResult, start_time: float
    ) -> BatchResult:
        """Convert a stage 1 primitive result to BatchResult, with recursive factoring."""
        result = BatchResult()
        original_factors = list(
            zip(prim_result.get('factors', []), prim_result.get('sigmas', []))
        )
        for factor, sigma in original_factors:
            prime_factors = self.wrapper._fully_factor_found_result(factor, quiet=False)
            for prime in prime_factors:
                result.factors.append(prime)
                result.sigmas.append(sigma)

        result.curves_run = prim_result.get('curves_completed', 0)
        result.execution_time = time.time() - start_time
        result.success = len(result.factors) > 0
        result.raw_output = prim_result.get('raw_output')
        result.interrupted = prim_result.get('interrupted', False)
        return result

    def _install_graceful_handler(self) -> None:
        """Install 3-level Ctrl+C handler for graceful shutdown."""

        def handler(signum: int, frame: Any) -> None:
            self.wrapper.shutdown_level += 1
            if self.wrapper.shutdown_level == 1:
                print("\n[Ctrl+C] Will complete after this assignment finishes...")
                print("[Ctrl+C] Press again to stop after current curve, twice more to abort")
                self.logger.info("Shutdown level 1: will complete current assignment")
            elif self.wrapper.shutdown_level == 2:
                self.wrapper.graceful_shutdown_requested = True
                self.wrapper.stop_event.set()
                self.wrapper._signal_subprocesses_interrupt()
                print("\n[Ctrl+C] Stopping after current curve...")
                print("[Ctrl+C] Press again to abort immediately")
                self.logger.info("Shutdown level 2: stopping after current curve")
            else:
                print("\n[Ctrl+C] Immediate abort requested")
                self.logger.info("Shutdown level 3: immediate abort")
                if self.wrapper._original_sigint_handler:
                    signal.signal(signal.SIGINT, self.wrapper._original_sigint_handler)
                raise KeyboardInterrupt()

        self.wrapper._original_sigint_handler = signal.signal(signal.SIGINT, handler)

    def _restore_graceful_handler(self) -> None:
        """Restore original signal handler and reset shutdown state."""
        if self.wrapper._original_sigint_handler:
            signal.signal(signal.SIGINT, self.wrapper._original_sigint_handler)
        self.wrapper.graceful_shutdown_requested = False
        self.wrapper.shutdown_level = 0
        self.wrapper.stop_event.clear()

    # ==================== Phase 3: Pipelined GPU+CPU Execution ====================

    def run_pipelined(
        self,
        batch_producer: 'TLevelBatchProducer',
        composite: str,
        stage2_workers: int,
        *,
        verbose: bool = False,
        progress_interval: int = 0,
        gpu_device: Optional[int] = None,
        gpu_curves: Optional[int] = None,
        no_submit: bool = False,
        project: Optional[str] = None,
        work_id: Optional[str] = None,
        start_t_level: float = 0.0,
    ) -> BatchResult:
        """
        Execute pipelined GPU producer + CPU consumer.

        GPU thread produces residues by calling batch_producer.next_batch().
        CPU thread consumes residues from queue and runs stage 2.
        T-level planning stays in the batch producer.

        Args:
            batch_producer: Produces ExecutionBatch objects for stage 1
            composite: Composite being factored
            stage2_workers: Number of CPU threads for stage 2
            verbose: Enable verbose output
            progress_interval: Progress reporting interval
            gpu_device: GPU device number
            gpu_curves: Curves per GPU batch
            no_submit: Skip API submission
            project: Project name for submission
            work_id: Work ID for submission
            start_t_level: Starting t-level for calculations

        Returns:
            BatchResult with discovered factors and t-level info
        """
        from .ecm_math import calculate_tlevel

        self.logger.info(
            f"Starting PIPELINED progressive ECM (GPU+CPU concurrent)"
        )
        start_time = time.time()

        # Shared state between threads
        residue_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        shutdown_event = threading.Event()
        self.wrapper.stop_event.clear()
        self.wrapper.interrupted = False

        # Results accumulation (protected by lock)
        result_lock = threading.Lock()
        all_factors: List[str] = []
        all_sigmas: List[Optional[str]] = []
        total_curves = 0
        curve_history: List[str] = []

        # Current t-level tracking
        current_t_level = start_t_level

        # Set by the CPU thread when a per-batch submission fails and is queued.
        # The caller needs it to avoid completing the assignment with no progress
        # recorded (see FactorResult.submission_failed).
        submit_failed = threading.Event()

        def gpu_producer() -> None:
            """GPU thread: Run stage 1 for each batch, put residues in queue."""
            self.logger.info("[GPU Thread] Starting production")

            while True:
                # Check for shutdown
                if shutdown_event.is_set() or self.wrapper.interrupted:
                    self.logger.info("[GPU Thread] Stopping (shutdown or interrupt)")
                    break

                # Get next batch from producer
                next_batch = batch_producer.next_batch()
                if next_batch is None:
                    self.logger.info("[GPU Thread] No more batches, done")
                    break

                batch, step_target, b2_for_step = next_batch

                self.logger.info(
                    f"[GPU Thread] Starting stage 1: {batch.curves} curves "
                    f"at B1={batch.b1} (target t{step_target:.1f})"
                )

                # Create residue file
                residue_file = Path(
                    f"/tmp/tlevel_residue_{int(time.time() * 1000)}.txt"
                )

                # Run stage 1
                stage1_start = time.time()
                try:
                    stage1_result = self.wrapper._run_stage1_primitive(
                        composite=composite,
                        b1=batch.b1,
                        curves=batch.curves,
                        residue_file=residue_file,
                        use_gpu=True,
                        verbose=verbose,
                        gpu_device=gpu_device,
                        gpu_curves=gpu_curves,
                    )
                    stage1_time = time.time() - stage1_start

                    # Get actual curve count from residue file
                    if residue_file.exists():
                        residue_info = self.wrapper._parse_residue_file(residue_file)
                        actual_curves = residue_info.get('curve_count', batch.curves)
                        if actual_curves != batch.curves:
                            self.logger.debug(
                                f"[GPU Thread] Residue file has {actual_curves} curves "
                                f"(requested {batch.curves})"
                            )
                    else:
                        actual_curves = batch.curves

                    # Check for shutdown before queuing
                    if shutdown_event.is_set():
                        self.logger.info(
                            "[GPU Thread] Shutdown detected after stage 1, discarding batch"
                        )
                        if residue_file.exists():
                            residue_file.unlink()
                        break

                    # Extract factor info from stage1_result
                    stage1_factors = stage1_result.get('factors', [])
                    stage1_sigmas = stage1_result.get('sigmas', [])
                    stage1_factor = stage1_factors[-1] if stage1_factors else None
                    all_stage1_factors = list(zip(stage1_factors, stage1_sigmas))

                    # Put work item in queue
                    residue_queue.put({
                        'residue_file': residue_file,
                        'b1': batch.b1,
                        'b2': b2_for_step,
                        'curves': actual_curves,
                        'stage1_factor': stage1_factor,
                        'all_factors': all_stage1_factors,
                        'stage1_time': stage1_time,
                        'stage1_output': stage1_result.get('raw_output', ''),
                        'step_target': step_target,
                        'stage1_success': stage1_result.get('success', False),
                    })

                    # Update projected t-level
                    batch_producer.update_projected(
                        actual_curves, batch.b1, b2_for_step
                    )
                    self.logger.info(
                        f"[GPU Thread] Stage 1 batch complete in {stage1_time:.1f}s, "
                        f"projected t-level now {batch_producer.projected_t_level:.2f}"
                    )

                    # If factor found, stop producing
                    if stage1_factor:
                        self.logger.info(
                            "[GPU Thread] Factor found in stage 1, stopping production"
                        )
                        break

                except Exception as e:
                    self.logger.error(f"[GPU Thread] Error in stage 1: {e}")
                    break

            # Send sentinel
            if not shutdown_event.is_set():
                self.logger.info("[GPU Thread] All batches complete, signaling CPU thread")
                residue_queue.put(None)
            else:
                self.logger.info(
                    "[GPU Thread] All batches complete, CPU already stopped"
                )

        def cpu_consumer() -> None:
            """CPU thread: Process stage 2 from residue queue."""
            nonlocal current_t_level, total_curves

            self.logger.info(f"[CPU Thread] Starting with {stage2_workers} workers")

            while True:
                if self.wrapper.interrupted:
                    self.logger.info("[CPU Thread] User interrupt, draining queue")
                    shutdown_event.set()
                    break

                # Get next work item (poll with timeout for shutdown check)
                self.logger.info("[CPU Thread] Waiting for next work item from queue...")
                work_item = None
                got_item = False
                while not got_item:
                    if self.wrapper.interrupted or shutdown_event.is_set():
                        self.logger.info(
                            "[CPU Thread] Shutdown detected while waiting for work"
                        )
                        break
                    try:
                        work_item = residue_queue.get(timeout=1.0)
                        got_item = True
                    except queue.Empty:
                        continue

                if not got_item:
                    break

                # Check for sentinel
                if work_item is None:
                    self.logger.info("[CPU Thread] Received stop signal")
                    residue_queue.task_done()
                    break

                try:
                    residue_file = work_item['residue_file']
                    b1 = work_item['b1']
                    b2 = work_item['b2']
                    curves = work_item['curves']
                    stage1_factor = work_item['stage1_factor']
                    all_stage1_factors = work_item['all_factors']
                    stage1_time = work_item['stage1_time']
                    step_target = work_item['step_target']
                    stage1_success = work_item['stage1_success']

                    # Handle stage 1 factor
                    if stage1_factor:
                        self.logger.info(
                            f"[CPU Thread] Factor found in stage 1: {stage1_factor}"
                        )
                        with result_lock:
                            for f, s in all_stage1_factors:
                                all_factors.append(f)
                                all_sigmas.append(s)
                            total_curves += curves
                            curve_history.append(f"{curves}@{b1},0,p=3")
                            current_t_level = calculate_tlevel(
                                curve_history, base_tlevel=start_t_level
                            )

                        # Submit stage 1 factor results
                        if not no_submit and curves > 0:
                            s1_result = FactorResult()
                            for f, s in all_stage1_factors:
                                s1_result.add_factor(f, s)
                            s1_result.curves_run = curves
                            s1_result.execution_time = stage1_time

                            step_results = s1_result.to_dict(composite, 'ecm')
                            step_results['b1'] = b1
                            step_results['b2'] = 0
                            step_results['parametrization'] = 3
                            if work_id:
                                step_results['work_id'] = work_id
                            if not self.wrapper.submit_result(
                                step_results, project, 'gmp-ecm-ecm'
                            ):
                                submit_failed.set()
                                self.logger.warning(
                                    f"[CPU Thread] Failed to submit stage-1 factor "
                                    f"results for B1={b1}"
                                )

                        shutdown_event.set()
                        residue_queue.task_done()
                        if residue_file.exists():
                            residue_file.unlink()
                        break

                    # Run stage 2
                    if stage1_success and residue_file.exists():
                        self.logger.info(
                            f"[CPU Thread] Starting stage 2: {curves} curves "
                            f"at B1={b1}, B2={b2}"
                        )

                        from .stage2_executor import Stage2Executor

                        executor = Stage2Executor(
                            self.wrapper, residue_file, b1, b2, None,
                            stage2_workers, verbose
                        )
                        self.wrapper._active_stage2_executor = executor
                        try:
                            _, all_stage2_factors, curves_completed, stage2_time, sigma = (
                                executor.execute(
                                    early_termination=True,
                                    progress_interval=progress_interval,
                                )
                            )
                        finally:
                            self.wrapper._active_stage2_executor = None

                        self.logger.info(
                            f"[CPU Thread] Stage 2 complete in {stage2_time:.1f}s"
                        )

                        with result_lock:
                            if all_stage2_factors:
                                all_factors.extend(all_stage2_factors)
                                all_sigmas.extend(
                                    [sigma] * len(all_stage2_factors)
                                )
                                self.logger.info(
                                    f"[CPU Thread] Factor found in stage 2: "
                                    f"{all_stage2_factors}"
                                )
                                shutdown_event.set()
                                self.wrapper.stop_event.set()
                                self.wrapper._signal_subprocesses_interrupt()

                            total_curves += curves_completed
                            curve_history.append(
                                f"{curves_completed}@{b1},{int(b2)},p=3"
                            )
                            current_t_level = calculate_tlevel(
                                curve_history, base_tlevel=start_t_level
                            )
                            self.logger.info(
                                f"[CPU Thread] Current t-level: {current_t_level:.2f}"
                            )

                            if all_stage2_factors:
                                # Submit factor results
                                if not no_submit and curves_completed > 0:
                                    s2_result = FactorResult()
                                    for f in all_stage2_factors:
                                        s2_result.add_factor(f, sigma)
                                    s2_result.curves_run = curves_completed
                                    s2_result.execution_time = (
                                        stage1_time + stage2_time
                                    )
                                    step_results = s2_result.to_dict(
                                        composite, 'ecm'
                                    )
                                    step_results['b1'] = b1
                                    step_results['b2'] = b2
                                    step_results['parametrization'] = 3
                                    if work_id:
                                        step_results['work_id'] = work_id
                                    if not self.wrapper.submit_result(
                                        step_results, project, 'gmp-ecm-ecm'
                                    ):
                                        submit_failed.set()
                                        self.logger.warning(
                                            "[CPU Thread] Failed to submit stage-2 "
                                            f"factor results for B1={b1}"
                                        )
                                residue_queue.task_done()
                                if residue_file.exists():
                                    residue_file.unlink()
                                break

                        # Submit no-factor results
                        if not no_submit and curves_completed > 0:
                            step_results = {
                                'success': True,
                                'factors_found': [],
                                'curves_completed': curves_completed,
                                'execution_time': stage1_time + stage2_time,
                                'composite': composite,
                                'method': 'ecm',
                                'b1': b1,
                                'b2': b2,
                                'parametrization': 3,
                            }
                            if work_id:
                                step_results['work_id'] = work_id
                            submit_response = self.wrapper.submit_result(
                                step_results, project, 'gmp-ecm-ecm'
                            )
                            if not submit_response:
                                submit_failed.set()
                                self.logger.warning(
                                    f"[CPU Thread] Failed to submit results for B1={b1}"
                                )
                    else:
                        self.logger.warning(
                            "[CPU Thread] Stage 1 failed or no residue file"
                        )

                    # Clean up residue file
                    if residue_file.exists():
                        residue_file.unlink()

                    residue_queue.task_done()
                    self.logger.info(
                        "[CPU Thread] Batch complete, looping for next work item"
                    )

                except Exception as e:
                    self.logger.error(f"[CPU Thread] Error processing batch: {e}")
                    residue_queue.task_done()
                    continue

            # Drain queue to prevent GPU thread from blocking on put()
            drained = 0
            while True:
                try:
                    item = residue_queue.get_nowait()
                    if item is not None:
                        rf = item.get('residue_file')
                        if rf and Path(rf).exists():
                            Path(rf).unlink()
                        drained += 1
                    residue_queue.task_done()
                except queue.Empty:
                    break
            if drained > 0:
                self.logger.info(
                    f"[CPU Thread] Drained {drained} unprocessed items from queue"
                )

        # Start both threads
        gpu_thread = threading.Thread(target=gpu_producer, name="GPU-Stage1")
        cpu_thread = threading.Thread(target=cpu_consumer, name="CPU-Stage2")

        gpu_thread.start()
        cpu_thread.start()

        # Wait for completion
        try:
            gpu_thread.join()
            cpu_thread.join()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info(
                "Pipelined execution interrupted - terminating subprocesses..."
            )
            self.wrapper.interrupted = True
            self.wrapper.stop_event.set()
            shutdown_event.set()
            self.wrapper._terminate_all_subprocesses()
            s2_exec = getattr(self.wrapper, '_active_stage2_executor', None)
            if s2_exec is not None:
                with s2_exec.process_lock:
                    for p in s2_exec.running_processes:
                        if p.poll() is None:
                            try:
                                p.terminate()
                            except OSError:
                                pass
            gpu_thread.join(timeout=5)
            cpu_thread.join(timeout=5)

        # Build result
        result = BatchResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.factors.append(factor)
            result.sigmas.append(sigma)
        result.curves_run = total_curves
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0
        result.interrupted = self.wrapper.interrupted
        result.submission_failed = submit_failed.is_set()

        # Attach t-level info via extra attributes
        result.t_level_achieved = current_t_level  # type: ignore[attr-defined]
        result.curve_history = curve_history  # type: ignore[attr-defined]

        self.logger.info(
            f"Pipelined t-level execution complete: {total_curves} curves, "
            f"t{current_t_level:.2f} achieved"
        )

        return result


class TLevelBatchProducer:
    """
    Produces ExecutionBatch objects for pipelined t-level execution.

    Owns the t-level planning logic: step targets, projected vs actual t-level
    tracking, and curve calculation. Cleanly separates "what to run next"
    from "how to run GPU+CPU concurrently" (engine).
    """

    def __init__(
        self,
        start_t_level: float,
        target_t_level: float,
        composite: str,
        b2_multiplier: float = 500.0,
        b2_dictionary: Optional[Dict[int, int]] = None,
        max_batch_curves: Optional[int] = None,
        logger: Optional[Any] = None,
    ):
        from .ecm_math import get_optimal_b1_for_tlevel, calculate_curves_to_target_direct

        self._get_optimal_b1 = get_optimal_b1_for_tlevel
        self._calculate_curves = calculate_curves_to_target_direct
        self.composite = composite
        self.b2_multiplier = b2_multiplier
        self.b2_dictionary = b2_dictionary
        self.max_batch_curves = max_batch_curves
        self.logger = logger
        self.start_t_level = start_t_level
        self.target_t_level = target_t_level

        # Projected t-level tracking (includes queued but not completed work)
        self._projected_lock = threading.Lock()
        self._projected_curve_history: List[str] = []
        self.projected_t_level = start_t_level

        # Build step targets
        self._step_targets: List[float] = []
        t = 20.0
        while t < target_t_level:
            if t > start_t_level:
                self._step_targets.append(t)
            t += 5.0
        if target_t_level not in self._step_targets and target_t_level > start_t_level:
            self._step_targets.append(target_t_level)

        # Iterator state for batch production
        self._step_index = 0
        self._remaining_curves_in_step = 0
        self._current_b1 = 0
        self._current_b2 = 0
        self._current_step_target = 0.0

    def next_batch(self) -> Optional[Tuple[ExecutionBatch, float, int]]:
        """
        Get the next batch to execute.

        Returns:
            Tuple of (ExecutionBatch, step_target, b2) or None when done.
        """
        # If we have remaining curves from a chunked step, produce next chunk
        if self._remaining_curves_in_step > 0:
            batch_curves = min(
                self._remaining_curves_in_step,
                self.max_batch_curves or self._remaining_curves_in_step,
            )
            self._remaining_curves_in_step -= batch_curves

            batch = ExecutionBatch(
                composite=self.composite,
                b1=self._current_b1,
                curves=batch_curves,
            )
            return (batch, self._current_step_target, self._current_b2)

        # Move to next step target
        while self._step_index < len(self._step_targets):
            step_target = self._step_targets[self._step_index]
            self._step_index += 1

            # Use projected t-level for lookahead
            with self._projected_lock:
                current_projected = self.projected_t_level

            if current_projected >= step_target:
                if self.logger:
                    self.logger.info(
                        f"[GPU Thread] Skipping t{step_target:.1f} "
                        f"(projected t{current_projected:.2f} already exceeds)"
                    )
                continue

            # Get optimal B1 and calculate curves
            b1, _ = self._get_optimal_b1(step_target)
            if self.b2_dictionary and b1 in self.b2_dictionary:
                b2 = self.b2_dictionary[b1]
            else:
                b2 = int(b1 * self.b2_multiplier)

            curves = self._calculate_curves(
                current_projected, step_target, b1, 3, b2=b2
            )
            if self.logger:
                self.logger.info(
                    f"[GPU Thread] B2-aware: t{current_projected:.2f} -> "
                    f"t{step_target:.1f} = {curves} curves at B1={b1}, "
                    f"B2={b2}, p=3"
                )

            if curves is None or curves <= 0:
                # For sub-t20 targets, just run the full t0→t20 entry
                if step_target <= 20 and current_projected < 20:
                    from .ecm_math import TLEVEL_TRANSITION_CACHE
                    full_key = (0, 20, 3)  # GPU parametrization
                    if full_key in TLEVEL_TRANSITION_CACHE:
                        b1, curves = TLEVEL_TRANSITION_CACHE[full_key]
                        b2 = int(b1 * self.b2_multiplier)
                        self._current_b1 = b1
                        self._current_b2 = b2
                        if self.logger:
                            self.logger.info(f"[GPU Thread] Using t0→t20 curves for sub-t20 target: {curves} curves at B1={b1}")
                # Tiny gap: target B1 too large for the gap, binary says 0 curves.
                # Fall back to 1 curve at the current bracket's B1.
                if curves is None or curves <= 0:
                    current_b1, _ = self._get_optimal_b1(current_projected)
                    if current_b1 < b1:
                        if self.logger:
                            self.logger.info(
                                f"[GPU Thread] Tiny t-level gap "
                                f"(t{current_projected:.3f} -> t{step_target:.1f}), "
                                f"running 1 curve at B1={current_b1}"
                            )
                        b1 = current_b1
                        b2 = int(b1 * self.b2_multiplier)
                        curves = 1
                if curves is None or curves <= 0:
                    if self.logger:
                        self.logger.warning(
                            f"[GPU Thread] Could not calculate curves for "
                            f"t{current_projected:.3f} -> t{step_target:.1f}, skipping"
                        )
                    continue

            self._current_b1 = b1
            self._current_b2 = b2
            self._current_step_target = step_target

            # Chunk if max_batch_curves is set
            if self.max_batch_curves and curves > self.max_batch_curves:
                batch_curves = self.max_batch_curves
                self._remaining_curves_in_step = curves - batch_curves
            else:
                batch_curves = curves
                self._remaining_curves_in_step = 0

            batch = ExecutionBatch(
                composite=self.composite,
                b1=b1,
                curves=batch_curves,
            )
            return (batch, step_target, b2)

        # No more steps
        return None

    def update_projected(self, actual_curves: int, b1: int, b2: int) -> None:
        """Update projected t-level after stage 1 completes."""
        from .ecm_math import calculate_tlevel

        with self._projected_lock:
            self._projected_curve_history.append(
                f"{actual_curves}@{b1},{int(b2)},p=3"
            )
            self.projected_t_level = calculate_tlevel(
                self._projected_curve_history,
                base_tlevel=self.start_t_level,
            )
