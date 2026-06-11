#!/usr/bin/env python3
"""
Stage 2 Consumer mode: download residues from server, run CPU stage 2.
"""

from pathlib import Path
from typing import Any, Optional, Dict
import hashlib
import time

from ..ecm_config import FactorResult
from ..work_helpers import print_work_header
from ..arg_parser import get_workers_default, resolve_pin_threads, resolve_stage2_progress_interval
from ..api_client import ResourceNotFoundError
from ..cleanup_helpers import handle_shutdown
from .base import WorkMode, WorkLoopContext


class Stage2ConsumerMode(WorkMode):
    """
    Stage 2 Consumer mode: Download residues from server, CPU processing.

    This mode:
    1. Requests residue work from server
    2. Downloads the residue file
    3. Runs stage 2 processing
    4. Submits results (supersedes stage 1 attempt)
    5. Completes residue work
    """

    mode_name = "Stage 2 Consumer (CPU)"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self._b2: Optional[int] = None
        self._k: Optional[int] = None
        self.local_residue_file: Optional[Path] = None
        self._residue_checksum: Optional[str] = None
        # Track curves for completion validation
        self._expected_curves: int = 0
        self._curves_completed: int = 0
        self._found_factor: bool = False
        self._raw_output: str = ""  # Aggregated ECM output from workers
        self._primary_submission_failed: bool = False
        # Import here to avoid circular dependency
        from ..stage2_executor import Stage2Executor
        self.Stage2Executor = Stage2Executor

    def _cleanup_local_residue(self) -> None:
        """Clean up local residue file if it exists."""
        if self.local_residue_file and self.local_residue_file.exists():
            self.local_residue_file.unlink()
            self.logger.info(f"Deleted local residue file: {self.local_residue_file}")
        self.local_residue_file = None

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA-256 checksum of file for residue verification."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def request_work(self) -> Optional[Dict[str, Any]]:
        # CLI --max-b1 takes priority, then config stage2_max_b1
        max_b1 = self.args.max_b1
        if max_b1 is None:
            max_b1 = self.wrapper.typed_config.programs.gmp_ecm.stage2_max_b1

        residue_work = self.api_client.get_residue_work(
            client_id=self.ctx.client_id,
            min_target_tlevel=self.args.min_target_tlevel,
            max_target_tlevel=self.args.max_target_tlevel,
            min_priority=self.args.priority,
            min_b1=self.args.min_b1,
            max_b1=max_b1,
            claim_timeout_hours=24,
            project=self.args.project
        )

        if not residue_work:
            if self.args.exit_on_no_work:
                self.logger.info("No residue work available. Exiting (--exit-on-no-work).")
                import sys; sys.exit(0)
            self.logger.info("No residue work available, waiting 30 seconds before retry...")
            time.sleep(30)
            return None

        return residue_work

    def on_work_started(self, work: Dict[str, Any]) -> None:
        self.current_residue_id = work['residue_id']
        self.current_work_id = None  # Stage 2 uses residue_id, not work_id

        # Track expected curves for completion validation
        self._expected_curves = work['curve_count']
        self._curves_completed = 0
        self._found_factor = False
        self._raw_output = ""

        b1 = work['b1']

        # Determine B2 and optionally k
        k = 0
        if 'b2_from_dict' in work:
            b2 = work['b2_from_dict']
            if 'k_from_dict' in work:
                k = work['k_from_dict']
        elif self.args.b2 is not None:
            b2 = self.args.b2
        elif self.args.b2_multiplier is not None:
            b2 = int(b1 * self.args.b2_multiplier)
            print(f"Using dynamic B2 = B1 * {self.args.b2_multiplier} = {b2}")
        else:
            b2 = work.get('suggested_b2', b1 * 500)

        self._b2 = b2
        self._k = k if k > 0 else None
        b2_display = "GMP-ECM default" if b2 == -1 else str(b2)

        print_work_header(
            work_id=str(self.current_residue_id) if self.current_residue_id else None,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'B1': b1,
                'B2': b2_display,
                'curves': work['curve_count'],
                'Stage 1 attempt ID': work.get('stage1_attempt_id')
            }
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        # Download residue file
        residue_dir = Path(self.wrapper.typed_config.execution.residue_dir)
        residue_dir.mkdir(parents=True, exist_ok=True)
        self.local_residue_file = residue_dir / f"s2_residue_{self.current_residue_id}.txt"

        print("Downloading residue file...")
        assert self.current_residue_id is not None  # Set in on_work_started
        download_success = self.api_client.download_residue(
            client_id=self.ctx.client_id,
            residue_id=self.current_residue_id,
            output_path=str(self.local_residue_file)
        )

        if not download_success:
            result = FactorResult()
            result.success = False
            result.error_message = "Failed to download residue file"
            return result

        file_size = self.local_residue_file.stat().st_size
        print(f"Downloaded {file_size} bytes")

        # Compute checksum for residue verification
        self._residue_checksum = self._compute_file_checksum(self.local_residue_file)
        self.logger.debug(f"Residue checksum: {self._residue_checksum}")

        # Get workers count
        workers = self.args.workers or get_workers_default(self.wrapper.typed_config)

        # Run stage 2
        print(f"Running stage 2 with {workers} workers...")
        executor = self.Stage2Executor(
            self.wrapper,
            self.local_residue_file,
            work['b1'],
            self._b2,
            self._k,
            workers,
            self.args.verbose,
            pin_threads=resolve_pin_threads(self.args)
        )

        factor, all_factors, curves, exec_time, sigma = executor.execute(
            early_termination=not self.args.continue_after_factor,
            progress_interval=resolve_stage2_progress_interval(self.args)
        )

        # Build FactorResult
        result = FactorResult()
        result.success = True
        result.curves_run = curves
        result.execution_time = exec_time

        if all_factors:
            for f in all_factors:
                result.add_factor(f, sigma)

        # Store for submit_results and complete_work
        self._work = work
        self._factor = factor
        self._sigma = sigma
        self._curves_completed = curves
        self._found_factor = bool(all_factors)
        self._raw_output = executor.raw_output  # Aggregated output from all workers

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        success, attempt_id, primary_failed = self._submit_stage2_results(
            work, result, self._b2, self._factor, self._sigma,
            self._raw_output, self.current_residue_id, self._residue_checksum
        )
        if success:
            self._stage2_attempt_id = attempt_id
            self._primary_submission_failed = primary_failed
        return success

    def complete_work(self, work: Dict[str, Any]) -> None:
        assert self.current_residue_id is not None  # Set in on_work_started

        # If primary endpoint submission failed, we can't complete the residue
        # (the attempt_id would be from a different endpoint)
        if self._primary_submission_failed:
            self.logger.warning(
                f"Skipping complete_residue for residue {self.current_residue_id} - "
                "primary endpoint submission failed, resubmit via resend_failed.py first"
            )
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
            return

        # Server bundled residue completion into the submission - nothing left to do
        if self._residue_completed_in_submit:
            if self._submit_new_t_level is not None:
                print(f"T-level updated to {self._submit_new_t_level:.2f}")
            print("Residue completed with submission.")
            self._cleanup_local_residue()
            return

        # Server requires 75% completion if no factor found
        # If we didn't complete enough curves (e.g., graceful shutdown), abandon instead
        completion_ratio = self._curves_completed / self._expected_curves if self._expected_curves > 0 else 0
        min_completion = 0.75

        if not self._found_factor and completion_ratio < min_completion:
            # Not enough curves completed - abandon to release back to pool
            print(f"Abandoning residue (only {self._curves_completed}/{self._expected_curves} curves = {completion_ratio:.1%}, need {min_completion:.0%})")
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
        else:
            # Completed enough curves or found a factor - mark as complete
            print("Completing residue work...")
            assert self._stage2_attempt_id is not None  # Set in submit_results when primary succeeds
            try:
                complete_result = self.api_client.complete_residue(
                    client_id=self.ctx.client_id,
                    residue_id=self.current_residue_id,
                    stage2_attempt_id=self._stage2_attempt_id
                )

                if complete_result:
                    new_t_level = complete_result.get('new_t_level')
                    if new_t_level is not None:
                        print(f"T-level updated to {new_t_level:.2f}")
                else:
                    self.logger.warning("Failed to complete residue on server - queuing for retry")
                    self.wrapper.submission_queue.enqueue_residue_completion(
                        residue_id=self.current_residue_id,
                        client_id=self.ctx.client_id,
                        stage2_attempt_id=self._stage2_attempt_id
                    )
            except ResourceNotFoundError:
                self.logger.warning(f"Residue {self.current_residue_id} already expired/completed on server, skipping")

        # Clean up local residue file
        self._cleanup_local_residue()

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        self.current_residue_id = None
        self.local_residue_file = None
        self._primary_submission_failed = False
        super().on_work_completed(work, result)

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        if self.current_residue_id:
            queue = self.wrapper.submission_queue
            if queue.has_pending_result_for_residue(self.current_residue_id):
                # Stage 2 finished, but submitting the result failed and the
                # queue now holds it for retry. Hold the residue claim so the
                # 24h timeout (not another client) covers us until the queue
                # drains and chains complete_residue — re-claiming the same
                # residue here would waste hours of CPU re-doing stage 2.
                self.logger.info(
                    f"Holding residue {self.current_residue_id} claim - "
                    "completed result is queued for retry"
                )
            else:
                if not self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id):
                    # Network likely down - queue abandonment so residue gets released on reconnect
                    queue.enqueue_residue_abandonment(
                        self.current_residue_id, self.ctx.client_id
                    )
            self.current_residue_id = None

        self._cleanup_local_residue()

    def cleanup_on_shutdown(self) -> None:
        self._cleanup_local_residue()

    def _handle_keyboard_interrupt(self) -> None:
        """Override to handle residue-specific cleanup."""
        self.cleanup_on_shutdown()
        handle_shutdown(
            wrapper=self.wrapper,
            current_work_id=None,
            current_residue_id=self.current_residue_id,
            mode_name=self.mode_name,
            completed_count=self.completed_count,
            local_residue_file=self.local_residue_file
        )
