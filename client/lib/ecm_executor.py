#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from lib.base_wrapper import BaseWrapper
from lib.parsing_utils import ECMPatterns
from lib.residue_manager import ResidueFileManager
from lib.result_processor import ResultProcessor
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult, ExecutionBatch, ECMPrimitiveResult
from lib.execution_engine import CompositeExecutionEngine, TLevelBatchProducer
from lib.ecm_command import build_ecm_command
from lib.ecm_math import (
    trial_division, is_probably_prime, OPTIMAL_B1_TABLE
)
from lib.api_client import PermanentUploadError

class ECMWrapper(BaseWrapper):
    """
    Wrapper for GMP-ECM factorization with multiple execution modes.

    Supports:
    - Standard ECM execution (run_ecm_v2)
    - Two-stage GPU/CPU pipeline (run_two_stage_v2)
    - Multiprocess parallelization (run_multiprocess_v2)
    - Progressive t-level targeting (run_tlevel_v2)
    - Server-coordinated auto-work mode

    All methods use type-safe configuration objects (ECMConfig, TwoStageConfig, etc.)
    and return FactorResult objects with discovered factors and metadata.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.residue_manager = ResidueFileManager()
        # Graceful shutdown support (3 levels). stop_event is inherited from
        # BaseWrapper and shared with run_subprocess_with_parsing / Stage2Executor.
        self.interrupted = False
        self.shutdown_level = 0  # 0=none, 1=complete batch, 2=complete curve, 3=abort
        self.graceful_shutdown_requested = False  # First Ctrl+C: finish current work
        self._original_sigint_handler: Optional[Any] = None  # Store original handler for restoration
        self._active_stage2_executor: Optional[Any] = None  # Set by execution_engine.py for interrupt handling
        # Unified execution engine
        self._engine = CompositeExecutionEngine(self)

    # ==================== NEW CONFIG-BASED METHODS ====================
    # These methods use configuration objects for cleaner interfaces

    def run_ecm_v2(self, config: ECMConfig) -> FactorResult:
        """
        Execute ECM with configuration object (simplified interface).

        This is the new, recommended method that uses ECMConfig dataclass.
        It has better type safety, validation, and testability.

        For GPU mode or when saving residues, uses _execute_ecm_primitive directly.
        For CPU mode without residues, delegates to run_multiprocess_v2 with 1 worker.

        Args:
            config: ECM configuration object

        Returns:
            FactorResult with discovered factors and metadata

        Example:
            >>> config = ECMConfig(composite="123456789", b1=50000, curves=100)
            >>> result = wrapper.run_ecm_v2(config)
            >>> if result.success:
            ...     print(f"Found factors: {result.factors}")
        """
        # Use primitive directly for GPU mode or when saving residues
        # (multiprocess doesn't support these features)
        if config.use_gpu or config.save_residues:
            start_time = time.time()
            residue_path = Path(config.save_residues) if config.save_residues else None

            prim_result = self._execute_ecm_primitive(
                composite=config.composite,
                b1=config.b1,
                b2=config.b2,
                curves=config.curves,
                residue_save=residue_path,
                sigma=config.sigma,
                param=config.parametrization,
                method=config.method,
                use_gpu=config.use_gpu,
                gpu_device=config.gpu_device,
                gpu_curves=config.gpu_curves,
                verbose=config.verbose,
                maxmem=config.maxmem,
            )

            return FactorResult.from_primitive_result(prim_result, time.time() - start_time)

        # For CPU mode without residues, delegate to multiprocess for proper handling
        mp_config = MultiprocessConfig(
            composite=config.composite,
            b1=config.b1,
            b2=config.b2,
            total_curves=config.curves,
            curves_per_process=config.curves,
            num_processes=1,
            parametrization=config.parametrization,
            method=config.method,
            verbose=config.verbose,
            progress_interval=config.progress_interval
        )
        return self.run_multiprocess_v2(mp_config)

    def run_ecm_from_dict(self, params: Dict[str, Any]) -> FactorResult:
        """
        Execute ECM from dictionary parameters.

        Convenient wrapper for config-based execution when you have
        a dictionary of parameters (e.g., from JSON config file).

        Args:
            params: Dictionary with ECMConfig fields

        Returns:
            FactorResult object

        Example:
            >>> params = {"composite": "12345", "b1": 50000, "curves": 10}
            >>> result = wrapper.run_ecm_from_dict(params)
        """
        config = ECMConfig(**params)
        return self.run_ecm_v2(config)

    def run_two_stage_v2(self, config: TwoStageConfig) -> FactorResult:
        """
        Execute two-stage ECM pipeline with configuration object.

        Stage 1: GPU-accelerated residue generation
        Stage 2: CPU processing of residues

        Delegates to CompositeExecutionEngine.run_two_stage().

        Args:
            config: Two-stage configuration object
                   - If config.resume_file is provided, skip stage 1 and process existing residue

        Returns:
            FactorResult with discovered factors

        Example:
            >>> config = TwoStageConfig(
            ...     composite="12345...",
            ...     b1=50000,
            ...     stage1_curves=100,
            ...     stage2_curves_per_residue=1000
            ... )
            >>> result = wrapper.run_two_stage_v2(config)
        """
        batch_result = self._engine.run_two_stage(
            composite=config.composite,
            b1=config.b1,
            b2=config.b2,
            stage1_curves=config.stage1_curves,
            stage2_workers=config.threads,
            stage1_parametrization=config.stage1_parametrization,
            verbose=config.verbose,
            progress_interval=config.progress_interval,
            use_gpu=(config.stage1_device == "GPU"),
            gpu_device=config.gpu_device,
            gpu_curves=config.gpu_curves,
            resume_file=Path(config.resume_file) if config.resume_file else None,
            save_residues=config.save_residues,
        )
        return batch_result.to_factor_result()

    def run_multiprocess_v2(self, config: MultiprocessConfig) -> FactorResult:
        """
        Execute multiprocess ECM with configuration object.

        Distributes curves across multiple CPU cores for parallel execution.
        Delegates to CompositeExecutionEngine.run_cpu_workers().

        Args:
            config: Multiprocess configuration object

        Returns:
            FactorResult with discovered factors

        Example:
            >>> config = MultiprocessConfig(
            ...     composite="12345...",
            ...     b1=50000,
            ...     total_curves=1000,
            ...     curves_per_process=100
            ... )
            >>> result = wrapper.run_multiprocess_v2(config)
        """
        import multiprocessing as mp

        num_processes = config.num_processes or mp.cpu_count()

        batch = ExecutionBatch(
            composite=config.composite,
            b1=config.b1,
            b2=config.b2,
            curves=config.total_curves,
            method=config.method,
            verbose=config.verbose,
            progress_interval=config.progress_interval,
        )

        batch_result = self._engine.run_cpu_workers(
            batch, num_processes, pin_threads=config.pin_threads
        )
        return batch_result.to_factor_result()

    def _run_tlevel_pipelined(self, config: TLevelConfig) -> FactorResult:
        """
        Execute T-level targeting in pipelined mode (GPU + CPU running concurrently).

        GPU thread produces residues, CPU thread consumes and processes them.
        Delegates to CompositeExecutionEngine.run_pipelined() with a TLevelBatchProducer.

        Args:
            config: T-level configuration object with use_two_stage=True

        Returns:
            FactorResult with discovered factors
        """
        self.logger.info(f"Starting PIPELINED progressive ECM with target t{config.target_t_level:.1f} (GPU+CPU concurrent)")

        producer = TLevelBatchProducer(
            start_t_level=config.start_t_level,
            target_t_level=config.target_t_level,
            composite=config.composite,
            b2_multiplier=config.b2_multiplier,
            b2_dictionary=config.b2_dictionary,
            max_batch_curves=config.max_batch_curves,
            logger=self.logger,
        )

        batch_result = self._engine.run_pipelined(
            batch_producer=producer,
            composite=config.composite,
            stage2_workers=config.threads,
            verbose=config.verbose,
            progress_interval=config.progress_interval,
            gpu_device=config.gpu_device,
            gpu_curves=config.gpu_curves,
            no_submit=config.no_submit,
            project=config.project,
            work_id=config.work_id,
            start_t_level=config.start_t_level,
        )

        # Convert BatchResult to FactorResult with t-level info
        result = batch_result.to_factor_result()
        result.t_level_achieved = getattr(batch_result, 't_level_achieved', 0.0)
        curve_history = getattr(batch_result, 'curve_history', [])
        result.curve_summary = self._parse_curve_history(curve_history)

        return result

    def run_tlevel_v2(self, config: TLevelConfig) -> FactorResult:
        """
        Execute T-level targeting with configuration object.

        Progressively runs ECM with optimized B1 values to reach target t-level.

        Args:
            config: T-level configuration object

        Returns:
            FactorResult with discovered factors

        Example:
            >>> config = TLevelConfig(
            ...     composite="12345...",
            ...     target_t_level=30.0,
            ...     b1_strategy='optimal'
            ... )
            >>> result = wrapper.run_tlevel_v2(config)
        """
        # If starting t-level already meets target, nothing to do
        if config.start_t_level >= config.target_t_level:
            self.logger.info(f"Target already met: t{config.start_t_level:.2f} >= t{config.target_t_level:.1f}")
            return FactorResult(success=True, curves_run=0)

        # Use pipelined mode for two-stage (GPU+CPU concurrent)
        if config.use_two_stage:
            return self._run_tlevel_pipelined(config)

        from lib.ecm_math import (get_optimal_b1_for_tlevel, calculate_tlevel,
                                  calculate_curves_to_target_direct, TLEVEL_TRANSITION_CACHE)

        if config.start_t_level > 0:
            self.logger.info(f"Starting progressive ECM with target t{config.target_t_level:.1f} (starting from t{config.start_t_level:.2f})")
        else:
            self.logger.info(f"Starting progressive ECM with target t{config.target_t_level:.1f}")
        start_time = time.time()

        # Accumulated results
        all_factors = []
        all_sigmas = []
        total_curves = 0
        curve_history = []  # Track B1 values and curves for t-level calculation
        # Per-batch submissions happen inside this loop, so a failure here is the
        # only signal the caller gets that the work was done but not reported.
        submission_failed = False

        # Progressive loop: start from start_t_level (default 0) and work up to target
        current_t_level = config.start_t_level
        step_targets = []

        # Build step targets (every 5 t-levels from 20 to target, plus final target)
        # Skip steps that are below our starting t-level
        t = 20.0
        while t < config.target_t_level:
            if t > current_t_level:  # Only add steps above our starting point
                step_targets.append(t)
            t += 5.0
        # Always add the final target (might not be a multiple of 5)
        if config.target_t_level not in step_targets and config.target_t_level > current_t_level:
            step_targets.append(config.target_t_level)

        # Run ECM at each step
        try:
            for step_target in step_targets:
                # Check for interruption at start of each step
                if self.interrupted or self.shutdown_level >= 1:
                    self.logger.info("T-level execution interrupted, returning partial results")
                    break

                # Skip this step if we've already surpassed it
                if current_t_level >= step_target:
                    self.logger.info(f"Skipping t{step_target:.1f} (already at t{current_t_level:.2f})")
                    continue

                # Get optimal B1 for this t-level
                b1, _ = get_optimal_b1_for_tlevel(step_target)

                # Calculate curves needed to reach this step from current position
                # First try the cached table for standard 5-digit increments
                current_rounded = round(current_t_level)
                target_rounded = round(step_target)
                cache_key = (current_rounded, target_rounded, config.parametrization)

                curves: Optional[int] = None  # Will be set below
                if cache_key in TLEVEL_TRANSITION_CACHE:
                    # Use cached value for standard transition
                    cached_b1, cached_curves = TLEVEL_TRANSITION_CACHE[cache_key]
                    if cached_b1 == b1:  # Only use cache if B1 matches
                        curves = cached_curves
                        self.logger.info(f"Using cached transition: t{current_rounded} → t{target_rounded} = {curves} curves at B1={b1}, p={config.parametrization}")
                    else:
                        # B1 doesn't match, call binary
                        curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)
                else:
                    # Non-standard transition, call t-level binary directly
                    curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)

                if curves is None or curves <= 0:
                    # For sub-t20 targets, just run the full t0→t20 entry
                    if step_target <= 20 and current_t_level < 20:
                        full_key = (0, 20, config.parametrization)
                        if full_key in TLEVEL_TRANSITION_CACHE:
                            b1, curves = TLEVEL_TRANSITION_CACHE[full_key]
                            self.logger.info(f"Using t0→t20 curves for sub-t20 target: {curves} curves at B1={b1}")
                    # Tiny gap: the target B1 is too large for the gap so the binary
                    # suggests 0 curves. Fall back to 1 curve at the current bracket's
                    # B1, which is the minimum work that will close the gap.
                    if curves is None or curves <= 0:
                        current_b1, _ = get_optimal_b1_for_tlevel(current_t_level)
                        if current_b1 < b1:
                            self.logger.info(f"Tiny t-level gap (t{current_t_level:.3f} → t{step_target:.1f}), running 1 curve at B1={current_b1}")
                            b1 = current_b1
                            curves = 1
                    if curves is None or curves <= 0:
                        self.logger.warning(f"Could not calculate curves for t{current_t_level:.3f} → t{step_target:.1f}, skipping to next level")
                        # Skip this target and continue to next one
                        continue

                self.logger.info(f"Running {curves} curves at B1={b1} (targeting t{step_target:.1f}, currently at t{current_t_level:.2f})")

                # Run this batch based on execution mode
                if config.use_two_stage:
                    # Two-stage mode: GPU stage 1 + CPU stage 2
                    self.logger.info(f"Using two-stage mode (GPU stage 1 + CPU stage 2) with {config.threads} workers")
                    step_result = self.run_two_stage_v2(TwoStageConfig(
                        composite=config.composite,
                        b1=b1,
                        b2=None,  # Use default B2 (B1 * 100)
                        stage1_curves=curves,
                        stage1_parametrization=3,  # GPU uses twisted Edwards
                        stage2_parametrization=1,  # CPU uses Montgomery
                        threads=config.threads,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval,
                        project=config.project,
                        no_submit=True  # T-level mode handles its own submissions
                    ))
                elif config.threads > 1:
                    # Multiprocess mode (CPU only)
                    self.logger.info(f"Using multiprocess mode with {config.threads} workers")
                    step_result = self.run_multiprocess_v2(MultiprocessConfig(
                        composite=config.composite,
                        b1=b1,
                        total_curves=curves,
                        num_processes=config.threads,
                        parametrization=config.parametrization,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval
                    ))
                else:
                    # Single process mode
                    step_result = self.run_ecm_v2(ECMConfig(
                        composite=config.composite,
                        b1=b1,
                        curves=curves,
                        parametrization=config.parametrization,
                        threads=1,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval
                    ))

                # Check for interruption after execution
                if self.interrupted or self.shutdown_level >= 1:
                    self.logger.info("T-level execution interrupted after ECM run, returning partial results")
                    # Still accumulate results from this batch
                    all_factors.extend(step_result.factors)
                    all_sigmas.extend(step_result.sigmas)
                    total_curves += step_result.curves_run

                    # Track partial curves in history for accurate t-level
                    if step_result.curves_run > 0:
                        effective_param = 3 if config.use_two_stage else config.parametrization
                        if config.use_two_stage:
                            actual_b2 = config.get_b2_for_b1(b1)
                            curve_history.append(f"{step_result.curves_run}@{b1},{actual_b2},p={effective_param}")
                        else:
                            curve_history.append(f"{step_result.curves_run}@{b1},p={effective_param}")
                        try:
                            current_t_level = calculate_tlevel(curve_history, base_tlevel=config.start_t_level)
                            self.logger.info(f"T-level after partial run: {current_t_level:.2f}")
                        except (subprocess.CalledProcessError, ValueError, OSError):
                            pass
                    break

                # Accumulate results
                all_factors.extend(step_result.factors)
                all_sigmas.extend(step_result.sigmas)
                total_curves += step_result.curves_run

                # Track curve history for t-level calculation (format: "curves@b1,b2,p=parametrization")
                # Two-stage mode uses GPU (p=3) for stage 1, but the full curve credit is for the combined run
                # For t-level calculation purposes, two-stage effectively uses p=3 parametrization
                effective_param = 3 if config.use_two_stage else config.parametrization

                # CRITICAL: Calculate actual B2 used for accurate t-level tracking
                # Two-stage mode uses B1 * multiplier, which gives less t-level per curve
                # than GMP's default B2. Must include B2 in curve history to prevent overestimating progress.
                # Example: 100@11e6,p=3 → t33.9 (wrong!), 100@11e6,11e8,p=3 → t32.2 (correct)
                if config.use_two_stage:
                    actual_b2 = config.get_b2_for_b1(b1)
                    # Format B2 as integer (no decimals) for t-level binary compatibility
                    curve_history.append(f"{step_result.curves_run}@{b1},{actual_b2},p={effective_param}")
                else:
                    # Standard mode: omit B2 to let t-level binary use GMP default
                    curve_history.append(f"{step_result.curves_run}@{b1},p={effective_param}")

                # Submit this batch's results if enabled
                if not config.no_submit and step_result.curves_run > 0:
                    # Calculate B2 for submission (two-stage uses B1*multiplier, else use GMP default)
                    submission_b2 = config.get_b2_for_b1(b1) if config.use_two_stage else None

                    step_results = step_result.to_dict(config.composite, 'ecm')
                    step_results['b1'] = b1
                    step_results['b2'] = submission_b2
                    step_results['parametrization'] = effective_param
                    if config.work_id:
                        step_results['work_id'] = config.work_id

                    program_name = 'gmp-ecm-ecm'
                    submit_response = self.submit_result(step_results, config.project, program_name)
                    if not submit_response:
                        submission_failed = True
                        self.logger.warning(f"Failed to submit results for B1={b1}")

                # Break if factor found
                if step_result.factors:
                    self.logger.info(f"Factor found after {total_curves} curves")
                    # Build and return result immediately
                    result = FactorResult()
                    for factor, sigma in zip(all_factors, all_sigmas):
                        result.add_factor(factor, sigma)
                    result.curves_run = total_curves
                    result.execution_time = time.time() - start_time
                    result.success = True
                    result.t_level_achieved = current_t_level  # Track achieved t-level
                    result.curve_summary = self._parse_curve_history(curve_history)
                    result.submission_failed = submission_failed
                    return result

                # Update current t-level after this batch
                try:
                    current_t_level = calculate_tlevel(curve_history, base_tlevel=config.start_t_level)
                    self.logger.info(f"Current t-level: {current_t_level:.2f}")
                except (subprocess.CalledProcessError, ValueError, OSError) as e:
                    # If t-level calculation fails (binary error, parse error), approximate it
                    self.logger.warning(f"T-level calculation failed ({type(e).__name__}), using step target: {step_target:.2f}")
                    current_t_level = step_target

                # Check if we've reached overall target
                if current_t_level >= config.target_t_level:
                    self.logger.info(f"Reached target t-level: {current_t_level:.2f} >= {config.target_t_level:.1f}")
                    break

        except KeyboardInterrupt:
            self.logger.info("T-level execution interrupted by Ctrl+C")
            self.interrupted = True
            # Fall through to build partial result

        # Build final FactorResult
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = total_curves
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0
        result.t_level_achieved = current_t_level  # Track achieved t-level
        result.interrupted = self.interrupted  # Signal that execution was interrupted
        result.curve_summary = self._parse_curve_history(curve_history)
        result.submission_failed = submission_failed

        return result

    def _parse_curve_history(self, curve_history: List[str]) -> List[Dict[str, Any]]:
        """
        Parse curve history strings into structured summary data.

        Args:
            curve_history: List of strings in format "{curves}@{b1},{b2},p={param}" or "{curves}@{b1},p={param}"

        Returns:
            List of dictionaries with B1, B2, curves, and parametrization
        """
        import re
        summary = []

        for entry in curve_history:
            # Pattern: "{curves}@{b1},{b2},p={param}" or "{curves}@{b1},p={param}"
            # Also handle: "{curves}@{b1},0,p={param}" for stage1-only
            match = re.match(r'(\d+)@(\d+),?(\d+)?,p=(\d+)', entry)
            if match:
                curves = int(match.group(1))
                b1 = int(match.group(2))
                b2_str = match.group(3)
                param = int(match.group(4))

                # Parse B2 (None = default, 0 = stage1 only, number = explicit B2)
                if b2_str is None:
                    b2 = None  # GMP-ECM default
                else:
                    b2 = int(b2_str)

                # Determine mode
                if b2 == 0:
                    mode = "Stage1 Only"
                elif b2 is None:
                    mode = "ECM (default B2)"
                else:
                    mode = "Two-Stage"

                summary.append({
                    'b1': b1,
                    'b2': b2,
                    'curves': curves,
                    'parametrization': param,
                    'mode': mode
                })

        return summary

    # ==================== LEGACY METHODS (BACKWARD COMPATIBLE) ====================

    def _log_and_store_factors(self, all_factors: List[Tuple[str, Optional[str]]],
                               results: Dict[str, Any], composite: str, b1: int,
                               b2: Optional[int], curves: int, method: str,
                               program: str) -> Optional[str]:
        """
        Deduplicate factors, log them, and store in results dictionary.

        This is now a thin wrapper around ResultProcessor for backward compatibility.

        Args:
            all_factors: List of (factor, sigma) tuples
            results: Results dictionary to update
            composite: Composite number being factored
            b1, b2, curves: ECM parameters
            method: Method name (ecm, pm1, pp1)
            program: Program name for logging

        Returns:
            First factor (for compatibility)
        """
        processor = ResultProcessor(self, composite, method, b1, b2, curves, program)
        return processor.log_and_store_factors(all_factors, results, quiet=False)

    # ==================== PRIMITIVE ECM EXECUTION ====================
    # Core primitive that all execution methods build upon

    def _execute_ecm_primitive(
        self,
        composite: str,
        b1: int,
        b2: Optional[int] = None,
        curves: int = 1,
        # Stage separation
        residue_save: Optional[Path] = None,
        residue_load: Optional[Path] = None,
        # ECM parameters
        sigma: Optional[Union[str, int]] = None,
        param: Optional[int] = None,
        method: str = "ecm",
        # GPU support
        use_gpu: bool = False,
        gpu_device: Optional[int] = None,
        gpu_curves: Optional[int] = None,
        # Progress and control
        progress_callback: Optional[Callable[[str, List[str]], None]] = None,
        stop_on_factor: bool = True,
        # Execution
        verbose: bool = False,
        maxmem: Optional[int] = None,
    ) -> ECMPrimitiveResult:
        """
        Execute GMP-ECM primitive operation.

        Single source of truth for all ECM execution. Handles 3 modes:
        1. Full ECM (B1 + B2) - default
        2. Stage 1 only - when residue_save is set
        3. Stage 2 only - when residue_load is set

        Note: No timeout is enforced - ECM factorization can run for extended periods.

        Args:
            composite: Number to factor
            b1: B1 bound
            b2: B2 bound (None = GMP-ECM default, 0 = stage1 only)
            curves: Number of curves to run
            residue_save: Path to save residue file (stage 1 only)
            residue_load: Path to load residue from (stage 2 only)
            sigma: Sigma value
            param: Parametrization (0-3)
            method: Method ("ecm", "pm1", "pp1")
            use_gpu: Use GPU acceleration
            gpu_device: GPU device number
            gpu_curves: Curves per GPU batch
            progress_callback: Called for each output line with (line, all_lines)
            stop_on_factor: Stop when first factor found
            verbose: Verbose output

        Returns:
            {
                'success': bool,
                'factors': List[str],
                'sigmas': List[Optional[str]],
                'curves_completed': int,
                'raw_output': str,
                'parametrization': Optional[int],
                'exit_code': int,
                'interrupted': bool
            }
        """
        ecm_path = self.typed_config.programs.gmp_ecm.path

        # Build command using shared builder
        cmd = build_ecm_command(
            ecm_path, b1,
            b2=b2, curves=curves, method=method,
            use_gpu=use_gpu, gpu_device=gpu_device, gpu_curves=gpu_curves,
            residue_save=residue_save, residue_load=residue_load,
            verbose=verbose, param=param, sigma=sigma,
            maxmem=maxmem,
        )

        # Execute subprocess with streaming
        try:
            process, output_lines = self._stream_subprocess_output(
                cmd=cmd,
                composite=composite if not residue_load else None,
                log_prefix=method.upper(),
                line_callback=progress_callback
            )

            exit_code = process.returncode
            raw_output = '\n'.join(output_lines)

            # Check for interruption
            interrupted = self.interrupted or exit_code == -15  # SIGTERM

            # Parse factors
            from lib.parsing_utils import parse_ecm_output_multiple
            factors_with_sigmas = parse_ecm_output_multiple(raw_output)

            # Separate factors and sigmas
            factors = [f[0] for f in factors_with_sigmas]
            sigmas = [f[1] for f in factors_with_sigmas]

            # Extract parametrization from output
            parametrization_extracted = None
            if sigmas and sigmas[0]:
                first_sigma = sigmas[0]
                if ':' in first_sigma:
                    parametrization_extracted = int(first_sigma.split(':')[0])
            if parametrization_extracted is None:
                parametrization_extracted = param if param is not None else 3

            # Determine success
            success = exit_code in [0, 8, 14] and not interrupted

            # Count curves completed (extract from GPU output, fallback to requested)
            curves_completed = curves  # Default fallback
            if use_gpu:
                # GPU mode: extract actual curve count from output
                # Use findall to capture ALL batches (GPU may run multiple batches)
                # Each batch prints "(N curves)" - sum them all for total
                curve_matches = ECMPatterns.CURVE_COUNT.findall(raw_output)
                if curve_matches:
                    curves_completed = sum(int(m) for m in curve_matches)
                    self.logger.debug(f"GPU completed {curves_completed} curves across {len(curve_matches)} batch(es) (requested {curves})")

            return {
                'success': success,
                'factors': factors,
                'sigmas': sigmas,
                'curves_completed': curves_completed,
                'raw_output': raw_output,
                'parametrization': parametrization_extracted,
                'exit_code': exit_code,
                'interrupted': interrupted
            }

        except KeyboardInterrupt:
            # KeyboardInterrupt doesn't inherit from Exception in Python 3
            # Handle it explicitly so we can return a proper result
            self.logger.info(f"{method.upper()} execution interrupted by user")
            self.interrupted = True
            return {
                'success': False,
                'factors': [],
                'sigmas': [],
                'curves_completed': 0,
                'raw_output': 'Interrupted by user',
                'parametrization': param if param is not None else 3,
                'exit_code': -2,
                'interrupted': True
            }

        except Exception as e:
            self.logger.error(f"{method.upper()} execution failed: {e}")
            return {
                'success': False,
                'factors': [],
                'sigmas': [],
                'curves_completed': 0,
                'raw_output': str(e),
                'parametrization': param if param is not None else 3,
                'exit_code': -1,
                'interrupted': False
            }

    def _run_stage1_primitive(
        self,
        composite: str,
        b1: int,
        curves: int,
        residue_file: Path,
        **kwargs
    ) -> ECMPrimitiveResult:
        """
        Run stage 1 only, save residue to file.

        Helper wrapper around _execute_ecm_primitive() for stage 1 execution.

        Args:
            composite: Number to factor
            b1: B1 bound
            curves: Number of curves
            residue_file: Path to save residue file
            **kwargs: Pass through sigma, param, use_gpu, verbose, etc.

        Returns:
            Result dict from primitive
        """
        return self._execute_ecm_primitive(
            composite=composite,
            b1=b1,
            b2=0,  # Stage 1 only
            curves=curves,
            residue_save=residue_file,
            **kwargs
        )

    def _run_stage2_primitive(
        self,
        residue_file: Path,
        b1: int,
        b2: Optional[int],
        **kwargs
    ) -> ECMPrimitiveResult:
        """
        Run stage 2 from residue file.

        Helper wrapper around _execute_ecm_primitive() for stage 2 execution.
        Automatically parses residue metadata to extract composite and curve count.

        Args:
            residue_file: Path to residue file
            b1: B1 bound (must match stage1)
            b2: B2 bound for stage 2
            **kwargs: Pass through verbose, etc.

        Returns:
            Result dict from primitive
        """
        # Parse residue metadata to get composite and curve count
        metadata = self.residue_manager.parse_metadata(str(residue_file))
        if not metadata:
            raise ValueError(f"Could not parse residue file: {residue_file}")

        composite, _, curve_count = metadata

        return self._execute_ecm_primitive(
            composite=composite,
            b1=b1,
            b2=b2,
            curves=curve_count,
            residue_load=residue_file,
            **kwargs
        )

    def _preserve_failed_upload(self, residue_file: Path) -> None:
        """
        Preserve a copy of a residue file that failed to upload.

        Saves the file to failed_uploads_dir with timestamp for manual retry later.

        Args:
            residue_file: Path to the residue file to preserve
        """
        import shutil

        try:
            failed_dir = Path(self.typed_config.execution.failed_uploads_dir)
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            preserved_name = f"{residue_file.stem}_failed_{timestamp}{residue_file.suffix}"
            preserved_path = failed_dir / preserved_name

            # Copy the file
            shutil.copy2(residue_file, preserved_path)

            self.logger.info(f"Preserved failed upload: {preserved_path}")
            print(f"Residue file preserved for manual retry: {preserved_path}")

        except Exception as e:
            self.logger.error(f"Failed to preserve residue file: {e}")

    def _upload_residue_if_needed(
        self,
        residue_file: Path,
        stage1_attempt_id: Optional[int],
        factor_found: Optional[str],
        client_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Upload residue file to server only if no factor was found.

        When a factor is found in stage 1, there's no point in running stage 2,
        so we skip the residue upload.

        Args:
            residue_file: Path to the residue file
            stage1_attempt_id: Attempt ID (int) from stage1 submission (None if submission failed)
            factor_found: Factor found in stage 1 (None if no factor)
            client_id: Client identifier for upload

        Returns:
            Upload result dict if uploaded, None if skipped or failed
        """
        if not factor_found and stage1_attempt_id:
            # No factor found and we have attempt ID - upload residue for potential stage 2
            print(f"Uploading residue file ({residue_file.stat().st_size} bytes)...")
            api_client = self._get_api_client()
            try:
                upload_result = api_client.upload_residue(
                    client_id=client_id,
                    residue_file_path=str(residue_file),
                    stage1_attempt_id=stage1_attempt_id,
                    expiry_days=7
                )
            except PermanentUploadError as e:
                # Server permanently rejected the upload (composite factored,
                # stage-1 attempt gone, etc.). Don't queue it - it can't succeed.
                self.logger.error(f"Residue upload permanently rejected, not queuing: {e}")
                print("Residue upload rejected by server - skipping (will not retry)")
                return None

            if upload_result:
                print(f"Residue uploaded: ID {upload_result['residue_id']}, "
                      f"{upload_result['curve_count']} curves")
                return upload_result
            else:
                self.logger.error("Failed to upload residue file - queuing for retry")

                # Preserve residue file in submission queue for automatic retry
                self.submission_queue.enqueue_residue_upload(
                    residue_file=residue_file,
                    client_id=client_id,
                    stage1_attempt_id=stage1_attempt_id,
                    expiry_days=7
                )

                return None
        elif factor_found:
            # Factor found - no need for stage 2
            print("Factor found - skipping residue upload")
            return None
        else:
            # No stage1_attempt_id - can't upload
            return None

    def _split_residue_file(self, residue_file: Path, num_chunks: int) -> List[Path]:
        """Split residue file into chunks for parallel processing"""
        # Create unique temporary directory for chunk files to avoid conflicts between concurrent jobs
        import tempfile
        chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
        self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

        # Use ResidueFileManager to split the file
        chunk_paths = self.residue_manager.split_into_chunks(
            str(residue_file), num_chunks, chunk_dir
        )

        # Convert string paths to Path objects
        return [Path(p) for p in chunk_paths]



    def _parse_residue_file(self, residue_path: Path) -> Dict[str, Any]:
        """
        Parse ECM residue file and extract all metadata in a single pass.

        Returns:
            Dict with keys: composite, b1, curve_count
        """
        metadata = self.residue_manager.parse_metadata(str(residue_path))

        if metadata:
            composite, b1, curve_count = metadata
            return {
                'composite': composite,
                'b1': b1,
                'curve_count': curve_count
            }
        else:
            # Return default values on parse failure
            return {
                'composite': 'unknown',
                'b1': 0,
                'curve_count': 0
            }

    def _correlate_factor_to_sigma(self, factor: str, residue_path: Path) -> Optional[str]:
        """
        Try to determine which sigma value found the factor by testing each residue.
        This is a fallback when ECM output doesn't contain sigma information.
        """
        sigma = self.residue_manager.correlate_factor_to_sigma(factor, str(residue_path))

        if sigma:
            # Format as "3:sigma" (ECM format with parametrization 3)
            # If sigma already contains parametrization prefix, use as-is
            if ':' not in sigma:
                return f"3:{sigma}"
            return sigma

        return None




    def get_program_version(self, program: str) -> str:
        """Override base class method to get GMP-ECM version."""
        return self.get_ecm_version()

    def get_ecm_version(self) -> str:
        """Get GMP-ECM version."""
        from lib.parsing_utils import get_binary_version
        return get_binary_version(
            self.typed_config.programs.gmp_ecm.path,
            'ecm'
        )

    def _fully_factor_with_runner(
        self,
        factor: str,
        runner: Callable[[str, int, int], List[str]],
        max_ecm_attempts: int = 5,
        quiet: bool = False,
    ) -> List[str]:
        """
        Recursively factor a value via trial division + ECM with increasing B1.

        Shared body for `_fully_factor_found_result` and `_fully_factor_composite`,
        which differ only in the ECM runner they use (high-level `run_ecm_v2`
        vs primitive `_execute_ecm_primitive`).

        Args:
            factor: Factor to fully factor (may be composite)
            runner: Callable(composite, b1, curves) -> list of factor strings.
                Recursion reuses the same runner, which is what keeps the
                primitive-only path free of run_ecm_v2 -> run_multiprocess_v2
                -> _fully_factor_composite recursion.
            max_ecm_attempts: Maximum ECM attempts with increasing B1
            quiet: Reserved for log suppression; currently passed through
                recursion but not honored (preserved for API compat).

        Returns:
            List of prime factors (as strings)
        """
        factor_int = int(factor)

        # Trial division catches small factors quickly (2, 3, 5, 7, ... up to 10^7)
        small_primes, cofactor = trial_division(factor_int, limit=10**7)
        all_primes = [str(p) for p in small_primes]

        if cofactor == 1:
            return all_primes

        if is_probably_prime(cofactor):
            self.logger.info(f"Cofactor {cofactor} is prime")
            all_primes.append(str(cofactor))
            return all_primes

        digit_length = len(str(cofactor))
        self.logger.info(f"Cofactor remaining: C{digit_length}, using ECM to complete factorization")

        current_cofactor = cofactor
        for attempt in range(max_ecm_attempts):
            if current_cofactor == 1:
                break

            cofactor_digits = len(str(current_cofactor))
            # Target factors up to half the digit length (smallest factor can't be larger)
            target_digits = (cofactor_digits + 1) // 2
            b1, expected_curves = OPTIMAL_B1_TABLE[0][1], OPTIMAL_B1_TABLE[0][2]
            for digits, table_b1, table_curves in OPTIMAL_B1_TABLE:
                if target_digits <= digits:
                    b1, expected_curves = table_b1, table_curves
                    break
            # Spread expected curves across attempts, minimum 50 per attempt
            curves = max(50, expected_curves // max_ecm_attempts)

            self.logger.info(f"ECM attempt {attempt+1}/{max_ecm_attempts} on C{cofactor_digits} "
                             f"with B1={b1}, {curves} curves (target P{target_digits})")

            try:
                found_factors = runner(str(current_cofactor), b1, curves)

                if found_factors:
                    self.logger.info(f"ECM found {len(found_factors)} factor(s): {found_factors}")

                    for found_factor in found_factors:
                        sub_primes = self._fully_factor_with_runner(
                            found_factor, runner, max_ecm_attempts, quiet
                        )
                        all_primes.extend(sub_primes)

                        for prime in sub_primes:
                            while current_cofactor % int(prime) == 0:
                                current_cofactor //= int(prime)

                    if current_cofactor == 1:
                        break

                    if is_probably_prime(current_cofactor):
                        self.logger.info(f"Remaining cofactor {current_cofactor} is prime")
                        all_primes.append(str(current_cofactor))
                        current_cofactor = 1
                        break
                else:
                    self.logger.info(f"No factor found in attempt {attempt+1}")

            except Exception as e:
                self.logger.error(f"ECM factorization error: {e}")
                break

        if current_cofactor > 1:
            self.logger.warning(f"Could not fully factor C{len(str(current_cofactor))}: {current_cofactor}")
            all_primes.append(str(current_cofactor))

        return all_primes

    def _fully_factor_found_result(self, factor: str, max_ecm_attempts: int = 5, quiet: bool = False) -> List[str]:
        """
        Recursively factor a result from ECM until all prime factors found.

        Uses the high-level `run_ecm_v2` API (which routes through multiprocess
        when appropriate). Use this from contexts that are not already inside a
        multiprocess worker — otherwise see `_fully_factor_composite`.
        """
        def runner(composite: str, b1: int, curves: int) -> List[str]:
            ecm_result = self.run_ecm_v2(ECMConfig(
                composite=composite, b1=b1, curves=curves, verbose=False,
            ))
            return ecm_result.factors if ecm_result.factors else []

        return self._fully_factor_with_runner(factor, runner, max_ecm_attempts, quiet)

    def _fully_factor_composite(self, factor: str, max_ecm_attempts: int = 5) -> List[str]:
        """
        Recursively factor a composite using `_execute_ecm_primitive` directly.

        Called from inside `run_multiprocess_v2` to factor any composite factors
        found. Bypasses `run_ecm_v2` to avoid the
        run_ecm_v2 -> run_multiprocess_v2 -> _fully_factor_composite recursion.
        """
        def runner(composite: str, b1: int, curves: int) -> List[str]:
            prim_result = self._execute_ecm_primitive(
                composite=composite, b1=b1, curves=curves, verbose=False,
            )
            return prim_result.get('factors', [])

        return self._fully_factor_with_runner(factor, runner, max_ecm_attempts)
