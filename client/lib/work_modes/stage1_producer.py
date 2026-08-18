#!/usr/bin/env python3
"""
Stage 1 Producer mode: GPU stage 1 execution, upload residues to server.
"""

from pathlib import Path
from typing import Any, Optional, Dict
import signal
import time

from ..ecm_config import FactorResult
from ..ecm_math import calculate_tlevel, get_optimal_b1_for_tlevel
from ..work_helpers import print_work_header, request_ecm_work
from ..stage1_helpers import submit_stage1_complete_workflow
from ..results_builder import results_for_stage1
from ..arg_parser import resolve_gpu_settings
from ..ecm_arg_helpers import parse_sigma_arg, resolve_param
from .base import WorkMode, WorkLoopContext, SubmissionFailedError


class Stage1ProducerMode(WorkMode):
    """
    Stage 1 Producer mode: GPU execution, upload residues to server.

    This mode:
    1. Requests regular ECM work from server
    2. Runs stage 1 only (B2=0) to generate residues
    3. Submits stage 1 results
    4. Uploads residue file for stage 2 consumers

    GPU-specific Ctrl+C handling (2 levels):
    - 1st Ctrl+C: Print message, GPU keeps running. Submit + upload when done, then exit.
    - 2nd Ctrl+C: Immediate abort.
    """

    mode_name = "Stage 1 Producer (GPU)"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self.residue_file: Optional[Path] = None

    def _setup_signal_handler(self) -> None:
        """
        GPU-specific signal handler with 2 levels.

        GPU can't be interrupted mid-batch, so level 1 just flags for exit after
        GPU finishes. Level 2 aborts immediately.
        """
        def handler(signum, frame):
            if not self._first_interrupt_received:
                # First interrupt: GPU keeps running, submit when done, then exit
                self._first_interrupt_received = True
                self.ctx.finish_after_current = True
                print("\n")
                print("=" * 60)
                print("GPU batch will complete. Results will be submitted, then exit.")
                print("Press Ctrl+C again to abort immediately.")
                print("=" * 60)
            else:
                # Second interrupt: immediate abort
                raise KeyboardInterrupt()

        self._original_sigint_handler = signal.signal(signal.SIGINT, handler)

    def _print_startup_banner(self) -> None:
        """Print GPU mode startup banner."""
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
        print("Ctrl+C once: finish GPU batch, submit results, then exit")
        print("Ctrl+C twice: abort immediately")
        print("=" * 60)
        print()

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_ecm_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    # Minimum B1 for stage1-only mode to prevent overloading server with too-fast submissions
    # Low B1 values (e.g., 11000 for t20) complete in seconds on GPU, causing submission spam
    MIN_STAGE1_B1 = 250000  # ~t30 level

    def _calculate_stage1_params(self, work: Dict[str, Any]) -> tuple:
        """
        Calculate optimal B1/curves for stage 1 based on t-level info.

        If --b1 and --curves are specified, uses those.
        Otherwise picks appropriate B1 for current t-level and runs one GPU batch.

        For stage1-only mode, the goal is simple:
        - Pick the right B1 for where we are
        - Run one batch (GPU batch size)
        - Let server track actual t-level achieved

        Returns:
            Tuple of (b1, curves)
        """
        # If B1 is explicitly specified, use it (but still enforce minimum)
        if self.args.b1 is not None:
            b1 = self.args.b1
            if b1 < self.MIN_STAGE1_B1:
                self.logger.warning(f"B1={b1} below minimum {self.MIN_STAGE1_B1} for stage1-only, using minimum")
                b1 = self.MIN_STAGE1_B1
            curves = self.args.curves if self.args.curves is not None else \
                     self.wrapper.typed_config.programs.gmp_ecm.default_curves
            return b1, curves

        # Get current t-level to determine appropriate B1
        current_t = work.get('current_t_level', 0.0) or 0.0

        # Get optimal B1 for the current t-level (rounds up to next standard level)
        # e.g., t54.7 -> use B1 for t55
        target_for_b1 = max(20, int(current_t) + 1)  # At least t20
        b1, _ = get_optimal_b1_for_tlevel(target_for_b1)

        # Enforce minimum B1 to prevent submission spam from too-fast GPU runs
        if b1 < self.MIN_STAGE1_B1:
            self.logger.info(f"B1={b1} (t{target_for_b1}) below minimum, using B1={self.MIN_STAGE1_B1}")
            b1 = self.MIN_STAGE1_B1

        # Use GPU batch size for curves (one batch per work unit)
        # Check args.curves first, then config, then default
        # NOTE: GMP-ECM GPU rounds up to its natural batch size (e.g., 2304, 3072).
        # If we request MORE than the batch size, it runs multiple full batches.
        # Default to 1000 which is always <= any GPU batch size, ensuring exactly one batch.
        if self.args.curves is not None:
            curves = self.args.curves
        else:
            curves = self.wrapper.typed_config.programs.gmp_ecm.gpu.curves_per_batch

        self.logger.info(f"Stage 1: t{current_t:.1f} using B1={b1}, curves={curves} (one batch)")
        return b1, curves

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        # Store work for execute_work to use
        self._current_work = work

        # Calculate B1/curves (may use t-level info)
        b1, curves = self._calculate_stage1_params(work)
        self._stage1_b1 = b1
        self._stage1_curves = curves

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={'B1': b1, 'curves': curves,
                    'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"}
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']

        # Use pre-calculated B1/curves from on_work_started
        b1 = self._stage1_b1
        curves = self._stage1_curves

        # Resolve GPU settings
        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.typed_config)

        # Generate residue file path
        residue_dir = Path(self.wrapper.typed_config.execution.residue_dir)
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.residue_file = residue_dir / f"stage1_{timestamp}_{composite[:20]}.txt"

        # Resolve parameters
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        print(f"Running ECM stage 1 (B1={b1}, curves={curves})...")
        print(f"Saving residues to: {self.residue_file}")

        # Run stage 1
        stage1_result = self.wrapper._run_stage1_primitive(
            composite=composite,
            b1=b1,
            curves=curves,
            residue_file=self.residue_file,
            sigma=sigma,
            param=param,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=self.args.verbose
        )

        # Extract fields from primitive result dict
        success = stage1_result['success']
        factor = stage1_result['factors'][-1] if stage1_result['factors'] else None
        actual_curves = stage1_result['curves_completed']
        raw_output = stage1_result['raw_output']
        all_factors = list(zip(stage1_result['factors'], stage1_result['sigmas']))

        # Build FactorResult
        result = FactorResult()
        result.success = success
        result.curves_run = actual_curves
        result.raw_output = raw_output

        for f, s in all_factors:
            result.add_factor(f, s)

        # Store factor for submit_results
        self._last_factor = factor
        self._last_param = param if param is not None else 3
        self._last_curves = actual_curves
        self._last_output = raw_output
        self._last_all_factors = all_factors

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        if not result.success:
            self.logger.error("Stage 1 execution failed")
            return False

        composite = work['composite']

        # Build results using ResultsBuilder (use pre-calculated B1)
        builder = (results_for_stage1(composite, self._stage1_b1, self._last_curves, self._last_param)
            .with_curves(self._last_curves, self._last_curves)
            .with_factors(self._last_all_factors)
            .add_raw_output(self._last_output)
            .with_execution_time(result.execution_time))

        if self.current_work_id:
            builder.with_work_id(self.current_work_id)

        results = builder.build()

        # Predict whether stage 1 alone already meets the composite's target
        # t-level. If so, the server would never hand the residue out to a
        # stage 2 consumer, so there's no point uploading it.
        upload_residue = True
        target_t_level = work.get('target_t_level')
        if target_t_level is not None and self._last_factor is None:
            current_t_level = work.get('current_t_level') or 0.0
            try:
                # Include B2=0 so the t-level binary credits stage 1 only.
                # Without it, the binary uses default B2 and predicts a full
                # stage 1+2 t-level, which inflates the prediction and causes
                # residues to be skipped when they're actually still needed.
                curve_str = f"{self._last_curves}@{self._stage1_b1},0,p={self._last_param}"
                predicted_t = calculate_tlevel([curve_str], base_tlevel=current_t_level)
                if predicted_t >= target_t_level:
                    self.logger.info(
                        f"Stage 1 reached target t{target_t_level:.1f} "
                        f"(predicted t{predicted_t:.2f}), skipping residue upload"
                    )
                    upload_residue = False
            except Exception as e:
                # Fall back to uploading: better an unused residue than a dropped one
                self.logger.warning(f"T-level prediction failed, uploading residue anyway: {e}")

        # Submit stage 1 results and handle workflow
        assert self.residue_file is not None  # Set in execute_work
        stage1_attempt_id = submit_stage1_complete_workflow(
            wrapper=self.wrapper,
            results=results,
            residue_file=self.residue_file,
            work_id=self.current_work_id,
            project=self.args.project,
            client_id=self.ctx.client_id,
            factor_found=self._last_factor,
            cleanup_residue=True,
            upload_residue=upload_residue
        )

        return stage1_attempt_id is not None

    # complete_work() inherited from WorkMode base class

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        if self.current_work_id:
            queue = self.wrapper.submission_queue
            if isinstance(error, SubmissionFailedError) and \
                    queue.has_pending_result_for_work(self.current_work_id):
                # A multi-hour GPU batch ran; only reporting it failed, and the
                # queue now holds the result plus a preserved copy of the
                # residue. Hold the assignment so another GPU doesn't redo the
                # batch - the queued result's residue_upload chain uploads the
                # residue and completes the assignment when it drains, and the
                # server's 1-day expiry is the backstop.
                #
                # This is why the hold can't go through attach_work_completion:
                # the result already carries the stage-1 chain.
                self.logger.info(
                    f"Holding work {self.current_work_id} assignment - "
                    "completed stage-1 result is queued for retry"
                )
            elif not self.wrapper.abandon_work(self.current_work_id, reason="stage1_failed"):
                # Network likely down - queue abandonment so the assignment gets
                # released on reconnect.
                queue.enqueue_work_abandonment(
                    self.current_work_id, self.ctx.client_id
                )
            self.current_work_id = None

        # Clean up residue file
        if self.residue_file and self.residue_file.exists():
            self.residue_file.unlink()
            self.residue_file = None
