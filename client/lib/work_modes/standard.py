#!/usr/bin/env python3
"""
Standard auto-work mode: t-level or B1/B2 based execution.
"""

from typing import Any, Optional, Dict

from ..ecm_config import (
    ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
)
from ..work_helpers import print_work_header, request_ecm_work
from ..arg_parser import (
    resolve_gpu_settings, resolve_worker_count, get_max_batch_default,
    resolve_pin_threads, resolve_stage2_progress_interval,
)
from ..ecm_arg_helpers import parse_sigma_arg, resolve_param
from .base import WorkMode, WorkLoopContext


class StandardAutoWorkMode(WorkMode):
    """
    Standard auto-work mode: T-level or B1/B2 based execution.

    This mode:
    1. Requests ECM work from server
    2. Executes using t-level mode (default) or B1/B2 mode
    3. Submits results (t-level mode submits after each batch)
    4. Completes work assignment
    """

    mode_name = "Auto-work"

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_ecm_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"
            }
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']

        has_b1 = self.args.b1 is not None
        has_client_tlevel = self.args.tlevel is not None

        # Determine execution mode:
        # - Explicit B1 (with or without B2) → B1/B2 mode
        # - Explicit --tlevel or no B1/tlevel → t-level mode
        if has_b1 and not has_client_tlevel:
            return self._execute_b1b2_mode(work, composite)
        else:
            return self._execute_tlevel_mode(work, composite, has_client_tlevel)

    def _execute_tlevel_mode(self, work: Dict[str, Any], composite: str,
                             has_client_tlevel: bool) -> FactorResult:
        """Execute using progressive t-level targeting."""
        server_target = work.get('target_t_level')
        if has_client_tlevel:
            # Caller invariant: has_client_tlevel == (self.args.tlevel is not None)
            assert self.args.tlevel is not None
            if server_target is not None and self.args.tlevel > server_target:
                print(f"Capping --tlevel {self.args.tlevel:.1f} at server target t{server_target:.1f}")
                target_tlevel = server_target
            else:
                target_tlevel = self.args.tlevel
        else:
            target_tlevel = server_target if server_target is not None else 35.0

        if self.args.start_tlevel is not None:
            start_tlevel = self.args.start_tlevel
        else:
            # Server may return current_t_level=None explicitly; `or` covers both
            # the missing-key and explicit-None cases.
            start_tlevel = work.get('current_t_level') or 0.0

        mode_desc = "client t-level" if has_client_tlevel else "server t-level"
        print(f"Mode: {mode_desc} (start: {start_tlevel:.1f}, target: {target_tlevel:.1f})")

        # Handle cases where start is at or very close to target
        if start_tlevel >= target_tlevel:
            # Already past target (e.g. --tlevel 35 but composite at t39.7).
            # Abandon so the server can assign it to a client with a higher cap.
            print(f"Already past target: t{start_tlevel:.2f} >= t{target_tlevel:.1f}, abandoning work")
            if self.current_work_id:
                self.wrapper.abandon_work(self.current_work_id, reason="client_tlevel_exceeded")
                self.current_work_id = None
            self._is_tlevel_mode = True
            self._target_already_met = True
            result = FactorResult(success=True, curves_run=0)
            self._results_dict = result.to_dict(composite, self.args.method)
            return result
        elif target_tlevel - start_tlevel < 0.1:
            # Tiny gap: t-level binary can't calculate curves for very small gaps.
            # Bump target slightly so progressive ECM has room to work,
            # pushing the composite past its real target.
            bumped = start_tlevel + 0.1
            print(f"Small t-level gap ({target_tlevel - start_tlevel:.2f}), "
                  f"bumping target from t{target_tlevel:.1f} to t{bumped:.1f}")
            target_tlevel = bumped

        # Use workers for multiprocess mode OR two-stage mode (for CPU stage 2)
        if self.args.multiprocess or self.args.two_stage:
            workers = resolve_worker_count(self.args, self.wrapper.typed_config)
        else:
            workers = 1

        # Resolve max_batch from args or config
        max_batch = self.args.max_batch or get_max_batch_default(self.wrapper.typed_config)

        # Resolve GPU settings for two-stage mode
        _, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.typed_config)

        tlevel_progress_interval = (
            resolve_stage2_progress_interval(self.args)
            if self.args.two_stage
            else self.args.progress_interval
        )
        config = TLevelConfig(
            composite=composite,
            target_t_level=target_tlevel,
            start_t_level=start_tlevel,
            threads=workers,
            verbose=self.args.verbose,
            progress_interval=tlevel_progress_interval,
            max_batch_curves=max_batch,
            use_two_stage=self.args.two_stage,
            b2_multiplier=self.args.b2_multiplier or 500.0,
            b2_dictionary=getattr(self, '_b2_dictionary', None),
            pin_threads=resolve_pin_threads(self.args),
            project=self.args.project,
            no_submit=False,
            work_id=self.current_work_id,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves
        )

        result = self.wrapper.run_tlevel_v2(config)

        # Store for submit_results - t-level mode submits internally
        self._is_tlevel_mode = True
        self._results_dict = result.to_dict(composite, self.args.method)

        return result

    def _execute_b1b2_mode(self, work: Dict[str, Any], composite: str) -> FactorResult:
        """Execute using explicit B1/B2 parameters."""
        # Caller invariant: execute_work() only routes here when args.b1 is set.
        assert self.args.b1 is not None
        b1: int = self.args.b1
        b2: Optional[int] = 0
        k = 0
        if 'b2_from_dict' in work:
            b2 = work['b2_from_dict']
            if 'k_from_dict' in work:
                k = work['k_from_dict']
        else:
            b2 = self.args.b2
        curves = self.args.curves if self.args.curves else \
                 (1 if self.args.two_stage else self.wrapper.typed_config.programs.gmp_ecm.default_curves)

        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.typed_config)
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        self._is_tlevel_mode = False
        result: FactorResult

        if self.args.two_stage and self.args.method == 'ecm':
            workers = resolve_worker_count(self.args, self.wrapper.typed_config)
            print(f"Mode: two-stage GPU+CPU (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

            two_stage_config = TwoStageConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                stage1_curves=curves,
                stage1_device="GPU" if use_gpu else "CPU",
                stage2_device="CPU",
                stage1_parametrization=param if param else 3,
                threads=workers,
                verbose=self.args.verbose,
                progress_interval=resolve_stage2_progress_interval(self.args),
                pin_threads=resolve_pin_threads(self.args),
                gpu_device=gpu_device,
                gpu_curves=gpu_curves
            )
            result = self.wrapper.run_two_stage_v2(two_stage_config)

        elif self.args.multiprocess:
            workers = resolve_worker_count(self.args, self.wrapper.typed_config)
            print(f"Mode: multiprocess (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

            mp_config = MultiprocessConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                total_curves=curves,
                num_processes=workers,
                parametrization=param if param else 3,
                method=self.args.method,
                verbose=self.args.verbose,
                progress_interval=self.args.progress_interval,
                pin_threads=resolve_pin_threads(self.args)
            )
            result = self.wrapper.run_multiprocess_v2(mp_config)

        else:
            print(f"Mode: standard (B1={b1}, B2={b2}, curves={curves})")

            # sigma can be str or int from parse_sigma_arg, ECMConfig accepts both
            ecm_config = ECMConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                curves=curves,
                sigma=int(sigma) if sigma and str(sigma).isdigit() else None,
                parametrization=param if param else 3,
                method=self.args.method,
                verbose=self.args.verbose,
                progress_interval=self.args.progress_interval,
                maxmem=self.args.maxmem,
            )
            result = self.wrapper.run_ecm_v2(ecm_config)

        self._results_dict = result.to_dict(composite, self.args.method)

        # Add ECM parameters that aren't in FactorResult
        self._results_dict['b1'] = self.args.b1
        self._results_dict['b2'] = b2
        self._results_dict['curves_requested'] = self.args.curves
        use_gpu, _, _ = resolve_gpu_settings(self.args, self.wrapper.typed_config)
        self._results_dict['parametrization'] = self.args.param or (3 if use_gpu else 1)

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        # T-level mode submits internally after each batch
        if self._is_tlevel_mode:
            if result.curves_run == 0:
                if getattr(self, '_target_already_met', False):
                    return True
                self.logger.error("T-level mode ran zero curves, execution may have failed")
                return False
            if result.submission_failed:
                # At least one B1 batch is sitting in the submission queue. The
                # curves ran, so this is a submission failure, not an execution
                # one: reporting success here would complete the assignment with
                # no t-level progress recorded and /ecm-work would immediately
                # hand the same composite to another client.
                self.logger.warning(
                    "T-level batch submission(s) failed and were queued for retry"
                )
                return False
            return True

        # B1/B2 modes need to submit here
        self._results_dict['work_id'] = self.current_work_id
        program_name = f"gmp-ecm-{self._results_dict.get('method', 'ecm')}"
        return self._submit_ecm_results(self._results_dict, program_name)

    # complete_work() inherited from WorkMode base class
