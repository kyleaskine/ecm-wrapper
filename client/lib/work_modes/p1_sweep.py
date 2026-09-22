#!/usr/bin/env python3
"""
P-1 / P+1 sweep mode: run PM1 and/or PP1 across composites from server.
"""

from typing import Any, Optional, Dict

from ..ecm_config import ECMConfig, FactorResult
from ..work_helpers import print_work_header, request_p1_work
from .base import WorkMode, WorkLoopContext


class P1WorkMode(WorkMode):
    """
    P-1/P+1 sweep mode: Run PM1 and/or PP1 across composites from server.

    Uses the /p1-work endpoint, calculates B1 one step
    above the composite's target t-level, and sweeps across composites (one
    composite per work assignment).

    Supports three flags (mutually exclusive):
    - --pm1: Run P-1 only (1 curve per composite)
    - --pp1: Run P+1 only (N curves per composite, default 3)
    - --p1:  Run P-1 (1 curve) + P+1 (N curves) per composite

    B1 calculation: One step above the target t-level in the optimal B1 table,
    capped by config pm1_b1/pp1_b1.
    B2: Omitted (let GMP-ECM use its default ratio).
    """

    mode_name = "P-1/P+1 Sweep"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)

        # Determine which methods to run
        self._run_pm1 = self.args.pm1 or self.args.p1
        self._run_pp1 = self.args.pp1 or self.args.p1
        self._pp1_curves = self.args.pp1_curves

        # Set human-readable mode name
        if self._run_pm1 and self._run_pp1:
            self.mode_name = "P-1/P+1 Sweep"
        elif self._run_pm1:
            self.mode_name = "P-1 Sweep"
        else:
            self.mode_name = "P+1 Sweep"

        # Per-assignment state
        self._pm1_result: Optional[FactorResult] = None
        self._pp1_result: Optional[FactorResult] = None
        self._pm1_b1: int = 0
        self._pp1_b1: int = 0

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_p1_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        # Reset per-assignment state
        self._pm1_result = None
        self._pp1_result = None

        # Use B1 from server response directly (server already computed the appropriate B1)
        # Don't cap here — the server's filter checks for attempts at this exact B1,
        # so capping would cause the submitted attempt to never satisfy the filter,
        # resulting in the same composite being assigned repeatedly.
        # To limit B1, use --max-target-tlevel to avoid high-B1 composites.
        server_pm1_b1 = work.get('pm1_b1') or 0
        server_pp1_b1 = work.get('pp1_b1') or 0

        self._pm1_b1 = server_pm1_b1 if self._run_pm1 else 0
        self._pp1_b1 = server_pp1_b1 if self._run_pp1 else 0

        # Build params display
        params: Dict[str, Any] = {
            'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}",
        }
        methods = []
        if self._run_pm1:
            params['PM1 B1'] = self._pm1_b1
            methods.append("P-1 (1 curve)")
        if self._run_pp1:
            params['PP1 B1'] = self._pp1_b1
            methods.append(f"P+1 ({self._pp1_curves} curves)")
        params['Methods'] = ' + '.join(methods)

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params=params
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']
        combined_result = FactorResult()
        combined_result.success = True
        factor_found = False

        # Run PM1 if applicable
        if self._run_pm1 and not factor_found:
            print(f"Running P-1 (B1={self._pm1_b1}, B2=GMP-ECM default, 1 curve)...")
            pm1_config = ECMConfig(
                composite=composite,
                b1=self._pm1_b1,
                b2=None,  # Let GMP-ECM use default
                curves=1,
                method='pm1',
                parametrization=1,
                verbose=self.args.verbose,
                progress_interval=self.args.progress_interval or 0,
            )
            self._pm1_result = self.wrapper.run_ecm_v2(pm1_config)
            combined_result.curves_run += self._pm1_result.curves_run
            combined_result.execution_time += self._pm1_result.execution_time

            if self._pm1_result.factors:
                for f, s in self._pm1_result.factor_sigma_pairs:
                    combined_result.add_factor(f, s)
                factor_found = True
                print(f"Factor found by P-1: {self._pm1_result.factors[0]}")

        # Run PP1 if applicable and no factor found yet
        if self._run_pp1 and not factor_found:
            print(f"Running P+1 (B1={self._pp1_b1}, B2=GMP-ECM default, {self._pp1_curves} curves)...")
            pp1_config = ECMConfig(
                composite=composite,
                b1=self._pp1_b1,
                b2=None,  # Let GMP-ECM use default
                curves=self._pp1_curves,
                method='pp1',
                parametrization=1,
                verbose=self.args.verbose,
                progress_interval=self.args.progress_interval or 0,
            )
            self._pp1_result = self.wrapper.run_ecm_v2(pp1_config)
            combined_result.curves_run += self._pp1_result.curves_run
            combined_result.execution_time += self._pp1_result.execution_time

            if self._pp1_result.factors:
                for f, s in self._pp1_result.factor_sigma_pairs:
                    combined_result.add_factor(f, s)
                print(f"Factor found by P+1: {self._pp1_result.factors[0]}")

        return combined_result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        composite = work['composite']

        # Check that at least one method actually ran curves
        pm1_curves = self._pm1_result.curves_run if self._pm1_result else 0
        pp1_curves = self._pp1_result.curves_run if self._pp1_result else 0
        if pm1_curves == 0 and pp1_curves == 0:
            self.logger.error("Zero curves completed for P-1/P+1, execution may have failed (check ECM binary path)")
            return False

        success = True

        # Submit PM1 results if we ran PM1
        if self._pm1_result and self._pm1_result.curves_run > 0:
            pm1_dict = self._pm1_result.to_dict(composite, 'pm1')
            pm1_dict['b1'] = self._pm1_b1
            pm1_dict['b2'] = None  # Used GMP-ECM default
            pm1_dict['curves_requested'] = 1
            pm1_dict['parametrization'] = 1
            pm1_dict['work_id'] = self.current_work_id

            submit_response = self.wrapper.submit_result(
                pm1_dict, self.args.project, 'gmp-ecm-pm1'
            )
            if not submit_response:
                self.logger.error("Failed to submit PM1 results")
                success = False

        # Submit PP1 results if we ran PP1
        if self._pp1_result and self._pp1_result.curves_run > 0:
            pp1_dict = self._pp1_result.to_dict(composite, 'pp1')
            pp1_dict['b1'] = self._pp1_b1
            pp1_dict['b2'] = None  # Used GMP-ECM default
            pp1_dict['curves_requested'] = self._pp1_curves
            pp1_dict['parametrization'] = 1
            pp1_dict['work_id'] = self.current_work_id

            submit_response = self.wrapper.submit_result(
                pp1_dict, self.args.project, 'gmp-ecm-pp1'
            )
            if not submit_response:
                self.logger.error("Failed to submit PP1 results")
                success = False

        return success

    # complete_work() inherited from WorkMode base class
