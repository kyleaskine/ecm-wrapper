#!/usr/bin/env python3
"""
Adaptive CPU mode: prioritise stage 2 residue work, fall back to ECM.
"""

from pathlib import Path
from typing import Any, Optional, Dict
import hashlib
import time

from ..ecm_config import MultiprocessConfig, FactorResult
from ..ecm_math import get_optimal_b1_for_tlevel, get_max_tlevel_for_workers
from ..work_helpers import print_work_header
from ..arg_parser import resolve_worker_count, resolve_pin_threads
from ..api_client import ResourceNotFoundError
from ..cleanup_helpers import handle_shutdown
from .base import WorkMode, WorkLoopContext


class AdaptiveCPUMode(WorkMode):
    """
    Adaptive CPU mode: prioritizes stage 2 residue work, falls back to ECM.

    Each loop iteration:
    1. Check for stage 2 residue work (highest priority - completes GPU work)
    2. If residues available: download and process stage 2
    3. If no residues: request ECM work (progressive ordering, t-level capped
       by worker count) and run multiprocess ECM
    4. Loop back to step 1

    This ensures CPU clients are never idle when useful work exists, and they
    prioritize finishing stage 2 work (the bottleneck when GPUs produce residues
    faster than CPUs can consume them).
    """

    mode_name = "Adaptive CPU"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)

        # Resolve worker count once
        self._workers = resolve_worker_count(self.args, self.wrapper.typed_config)

        # Resolve pin-threads once (validates platform support up-front)
        self._pin_threads = resolve_pin_threads(self.args)

        # Calculate t-level cap based on worker count (only applies to ECM fallback)
        # User can override with --max-target-tlevel
        user_max = self.args.max_target_tlevel
        self._max_tlevel = user_max if user_max is not None else get_max_tlevel_for_workers(self._workers)

        # Default progress_interval to 100 to avoid spamming console with thousands of lines
        self._progress_interval = self.args.progress_interval if self.args.progress_interval > 0 else 100

        # Current work type tracking
        self._current_mode: Optional[str] = None  # 'stage2' or 'ecm'

        # Stage 2 state (reused across iterations)
        self._s2_b2: Optional[int] = None
        self._s2_k: Optional[int] = None
        self._s2_local_residue_file: Optional[Path] = None
        self._s2_residue_checksum: Optional[str] = None
        self._s2_expected_curves: int = 0
        self._s2_curves_completed: int = 0
        self._s2_found_factor: bool = False
        self._s2_raw_output: str = ""
        self._s2_factor: Optional[str] = None
        self._s2_sigma: Optional[str] = None
        self._s2_stage2_attempt_id: Optional[int] = None
        self._s2_primary_submission_failed: bool = False

        # ECM state
        self._ecm_results_dict: Optional[Dict[str, Any]] = None

        # Import stage2 executor
        from ..stage2_executor import Stage2Executor
        self.Stage2Executor = Stage2Executor

    def _print_startup_banner(self) -> None:
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
        print(f"Workers: {self._workers}")
        print(f"Priority: stage 2 residues > ECM (progressive, max t{self._max_tlevel:.0f})")
        print("Ctrl+C once: finish current assignment, then exit")
        print("Ctrl+C twice: stop after current curve")
        print("Ctrl+C three times: abort immediately")
        print("=" * 60)
        print()

    def _cleanup_s2_residue(self) -> None:
        """Clean up local stage 2 residue file if it exists."""
        if self._s2_local_residue_file and self._s2_local_residue_file.exists():
            self._s2_local_residue_file.unlink()
            self.logger.info(f"Deleted local residue file: {self._s2_local_residue_file}")
        self._s2_local_residue_file = None

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    # --- Work request: try stage 2 first, then ECM ---

    def request_work(self) -> Optional[Dict[str, Any]]:
        # CLI --max-b1 takes priority, then config stage2_max_b1
        max_b1 = self.args.max_b1
        if max_b1 is None:
            max_b1 = self.wrapper.typed_config.programs.gmp_ecm.stage2_max_b1

        # Try stage 2 residues first (no wait on failure - fall through to ECM)
        residue_work = self.api_client.get_residue_work(
            client_id=self.ctx.client_id,
            min_b1=self.args.min_b1,
            max_b1=max_b1,
            claim_timeout_hours=24,
            project=self.args.project
        )

        if residue_work:
            self._current_mode = 'stage2'
            self.logger.info("Stage 2 residue work available, prioritizing")
            return residue_work

        # No residues - try ECM work with progressive ordering
        work = self.api_client.get_ecm_work(
            client_id=self.ctx.client_id,
            min_target_tlevel=self.args.min_target_tlevel,
            max_target_tlevel=self._max_tlevel,
            priority=self.args.priority,
            min_digits=self.args.min_digits,
            max_digits=self.args.max_digits,
            work_type='progressive',
            project=self.args.project
        )

        if work:
            self._current_mode = 'ecm'
            return work

        # Nothing available at all
        if self.args.exit_on_no_work:
            self.logger.info("No work available. Exiting (--exit-on-no-work).")
            import sys; sys.exit(0)
        self.logger.info("No work available (stage 2 or ECM), waiting 30 seconds...")
        time.sleep(30)
        return None

    # --- Work started ---

    def on_work_started(self, work: Dict[str, Any]) -> None:
        if self._current_mode == 'stage2':
            self._on_stage2_started(work)
        else:
            self._on_ecm_started(work)

    def _on_stage2_started(self, work: Dict[str, Any]) -> None:
        self.current_residue_id = work['residue_id']
        self.current_work_id = None

        self._s2_expected_curves = work['curve_count']
        self._s2_curves_completed = 0
        self._s2_found_factor = False
        self._s2_raw_output = ""
        self._s2_primary_submission_failed = False

        b1 = work['b1']

        # Determine B2
        if self.args.b2 is not None:
            b2 = self.args.b2
        elif self.args.b2_multiplier is not None:
            b2 = int(b1 * self.args.b2_multiplier)
            print(f"Using dynamic B2 = B1 * {self.args.b2_multiplier} = {b2}")
        else:
            b2 = work.get('suggested_b2', b1 * 500)

        self._s2_b2 = b2
        self._s2_k = None
        b2_display = "GMP-ECM default" if b2 == -1 else str(b2)

        print_work_header(
            work_id=str(self.current_residue_id),
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'Mode': 'Stage 2 (adaptive)',
                'B1': b1,
                'B2': b2_display,
                'curves': work['curve_count'],
                'Stage 1 attempt ID': work.get('stage1_attempt_id')
            }
        )

    def _on_ecm_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'Mode': 'ECM multiprocess (adaptive)',
                'Workers': self._workers,
                'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"
            }
        )

    # --- Execution ---

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        if self._current_mode == 'stage2':
            return self._execute_stage2(work)
        else:
            return self._execute_ecm(work)

    def _execute_stage2(self, work: Dict[str, Any]) -> FactorResult:
        # Download residue file
        residue_dir = Path(self.wrapper.typed_config.execution.residue_dir)
        residue_dir.mkdir(parents=True, exist_ok=True)
        self._s2_local_residue_file = residue_dir / f"s2_residue_{self.current_residue_id}.txt"

        print("Downloading residue file...")
        assert self.current_residue_id is not None
        download_success = self.api_client.download_residue(
            client_id=self.ctx.client_id,
            residue_id=self.current_residue_id,
            output_path=str(self._s2_local_residue_file)
        )

        if not download_success:
            result = FactorResult()
            result.success = False
            result.error_message = "Failed to download residue file"
            return result

        file_size = self._s2_local_residue_file.stat().st_size
        print(f"Downloaded {file_size} bytes")

        self._s2_residue_checksum = self._compute_file_checksum(self._s2_local_residue_file)

        print(f"Running stage 2 with {self._workers} workers...")
        executor = self.Stage2Executor(
            self.wrapper,
            self._s2_local_residue_file,
            work['b1'],
            self._s2_b2,
            self._s2_k,
            self._workers,
            self.args.verbose,
            pin_threads=self._pin_threads
        )

        factor, all_factors, curves, exec_time, sigma = executor.execute(
            early_termination=True,
            progress_interval=self._progress_interval
        )

        result = FactorResult()
        result.success = True
        result.curves_run = curves
        result.execution_time = exec_time

        if all_factors:
            for f in all_factors:
                result.add_factor(f, sigma)

        self._s2_factor = factor
        self._s2_sigma = sigma
        self._s2_curves_completed = curves
        self._s2_found_factor = bool(all_factors)
        self._s2_raw_output = executor.raw_output

        return result

    def _execute_ecm(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']
        current_t = work.get('current_t_level', 0.0) or 0.0

        # Get optimal B1 and expected curve count for the current t-level step
        b1, curves = get_optimal_b1_for_tlevel(max(20, current_t))

        print(f"Mode: multiprocess ECM (B1={b1}, curves={curves}, workers={self._workers})")

        mp_config = MultiprocessConfig(
            composite=composite,
            b1=b1,
            total_curves=curves,
            num_processes=self._workers,
            parametrization=1,  # CPU Montgomery
            method='ecm',
            verbose=self.args.verbose,
            progress_interval=self._progress_interval,
            pin_threads=self._pin_threads
        )
        result = self.wrapper.run_multiprocess_v2(mp_config)

        self._ecm_results_dict = result.to_dict(composite, 'ecm')
        self._ecm_results_dict['b1'] = b1
        self._ecm_results_dict['b2'] = None  # GMP-ECM default
        self._ecm_results_dict['curves_requested'] = curves
        self._ecm_results_dict['parametrization'] = 1
        self._ecm_results_dict['work_id'] = self.current_work_id

        return result

    # --- Submission ---

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        if self._current_mode == 'stage2':
            return self._submit_stage2(work, result)
        else:
            return self._submit_ecm(work, result)

    def _submit_stage2(self, work: Dict[str, Any], result: FactorResult) -> bool:
        success, attempt_id, primary_failed = self._submit_stage2_results(
            work, result, self._s2_b2, self._s2_factor, self._s2_sigma,
            self._s2_raw_output, self.current_residue_id, self._s2_residue_checksum
        )
        if success:
            self._s2_stage2_attempt_id = attempt_id
            self._s2_primary_submission_failed = primary_failed
        return success

    def _submit_ecm(self, work: Dict[str, Any], result: FactorResult) -> bool:
        assert self._ecm_results_dict is not None
        return self._submit_ecm_results(self._ecm_results_dict, 'gmp-ecm-ecm')

    # --- Completion ---

    def complete_work(self, work: Dict[str, Any]) -> None:
        if self._current_mode == 'stage2':
            self._complete_stage2(work)
        else:
            # ECM mode uses standard work_id completion from base class
            super().complete_work(work)

    def _complete_stage2(self, work: Dict[str, Any]) -> None:
        assert self.current_residue_id is not None

        if self._s2_primary_submission_failed:
            self.logger.warning(
                f"Skipping complete_residue for residue {self.current_residue_id} - "
                "primary endpoint submission failed"
            )
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
            self._cleanup_s2_residue()
            return

        # Server bundled residue completion into the submission - nothing left to do
        if self._residue_completed_in_submit:
            if self._submit_new_t_level is not None:
                print(f"T-level updated to {self._submit_new_t_level:.2f}")
            print("Residue completed with submission.")
            self._cleanup_s2_residue()
            return

        # Check completion ratio
        completion_ratio = self._s2_curves_completed / self._s2_expected_curves if self._s2_expected_curves > 0 else 0

        if not self._s2_found_factor and completion_ratio < 0.75:
            print(f"Abandoning residue (only {self._s2_curves_completed}/{self._s2_expected_curves} curves = {completion_ratio:.1%})")
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
        else:
            print("Completing residue work...")
            assert self._s2_stage2_attempt_id is not None  # Set in _submit_stage2 when primary succeeds
            try:
                complete_result = self.api_client.complete_residue(
                    client_id=self.ctx.client_id,
                    residue_id=self.current_residue_id,
                    stage2_attempt_id=self._s2_stage2_attempt_id
                )
                if complete_result:
                    new_t_level = complete_result.get('new_t_level')
                    if new_t_level is not None:
                        print(f"T-level updated to {new_t_level:.2f}")
                else:
                    self.logger.warning("Failed to complete residue - queuing for retry")
                    self.wrapper.submission_queue.enqueue_residue_completion(
                        residue_id=self.current_residue_id,
                        client_id=self.ctx.client_id,
                        stage2_attempt_id=self._s2_stage2_attempt_id
                    )
            except ResourceNotFoundError:
                self.logger.warning(f"Residue {self.current_residue_id} already expired/completed")

        self._cleanup_s2_residue()

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        if self._current_mode == 'stage2':
            self.current_residue_id = None
            self._s2_local_residue_file = None
            self._s2_primary_submission_failed = False
        self._current_mode = None
        self._ecm_results_dict = None
        super().on_work_completed(work, result)

    # --- Cleanup ---

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        if self._current_mode == 'stage2':
            if self.current_residue_id:
                queue = self.wrapper.submission_queue
                if queue.has_pending_result_for_residue(self.current_residue_id):
                    # Same reasoning as Stage2ConsumerMode: hours of stage 2 ran
                    # and only the submission failed, so the queue holds the
                    # result with a complete_residue chain. Releasing the claim
                    # would hand the residue to another client to redo, and the
                    # chained completion would then fire against a residue this
                    # client no longer holds.
                    self.logger.info(
                        f"Holding residue {self.current_residue_id} claim - "
                        "completed result is queued for retry"
                    )
                elif not self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id):
                    queue.enqueue_residue_abandonment(
                        self.current_residue_id, self.ctx.client_id
                    )
                self.current_residue_id = None
            self._cleanup_s2_residue()
        else:
            super().cleanup_on_failure(work, error)

    def cleanup_on_shutdown(self) -> None:
        self._cleanup_s2_residue()
