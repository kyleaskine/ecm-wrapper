#!/usr/bin/env python3
"""
Tests for aliquot_wrapper's hybrid factorization routing.

Regression coverage for the threshold fix (hybrid_threshold gates the ECM
pretest loop; siqs_threshold only selects the final method) and for the
--small-method ecm mode that finishes small cofactors with GMP-ECM alone
(platforms without YAFU, e.g. macOS).
"""
import logging
import sys
from pathlib import Path
from typing import Any, List, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ecm_config import FactorResult, TLevelConfig
from lib.ecm_math import calculate_target_tlevel
from lib.typed_config import TypedConfigLoader
from aliquot_wrapper import AliquotWrapper

# Primes just above the 10^7 trial-division bound
P = 10000019
Q = 10000079
R = 10000103


def _result(factors: List[str], t_achieved: float) -> FactorResult:
    result = FactorResult(factors=list(factors),
                          sigmas=[None] * len(factors),
                          success=True)
    result.t_level_achieved = t_achieved
    return result


class FakeECM:
    """run_tlevel_v2 stub replaying a scripted list of FactorResults."""

    def __init__(self, script: List[FactorResult]):
        self.script = list(script)
        self.calls: List[TLevelConfig] = []

    def run_tlevel_v2(self, config: TLevelConfig) -> FactorResult:
        self.calls.append(config)
        return self.script.pop(0)


class ForbiddenTool:
    """Stands in for YAFU/CADO when the test expects them to stay untouched."""

    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, attr):
        raise AssertionError(f"{self.name}.{attr} must not be called in this test")


def _make_wrapper(fake_ecm: FakeECM, hybrid: int = 100, siqs: int = 100,
                  small_method: str = 'ecm') -> AliquotWrapper:
    """AliquotWrapper with just the state _factor_hybrid needs (bypasses
    __init__ so no config file or binaries are required)."""
    wrapper = AliquotWrapper.__new__(AliquotWrapper)
    wrapper.logger = logging.getLogger('test_aliquot_hybrid_routing')
    wrapper.typed_config = TypedConfigLoader()._parse_config({})
    wrapper.ecm_program = 'gmp-ecm'
    wrapper.hybrid_threshold = hybrid
    wrapper.siqs_threshold = siqs
    wrapper.small_method = small_method
    wrapper.threads = None
    wrapper.verbose = False
    wrapper.use_two_stage = False
    wrapper.max_batch_curves = None
    wrapper.progress_interval = 0
    wrapper.use_tracker = False
    wrapper.submit_ecm_results = False
    wrapper.tracker = None
    wrapper.tracker_start = None
    wrapper._ecm = cast(Any, fake_ecm)
    wrapper._yafu = cast(Any, ForbiddenTool('yafu'))
    wrapper._cado = cast(Any, ForbiddenTool('cado'))
    return wrapper


# ==================== threshold routing (bug fix) ====================


def test_below_hybrid_threshold_skips_ecm_pretest():
    # C16 cofactor < hybrid_threshold: the pretest loop must exit without
    # running ECM; --small-method ecm then finishes it via ECM-to-completion
    ecm = FakeECM([_result([str(P)], t_achieved=10.0)])
    wrapper = _make_wrapper(ecm, hybrid=100, siqs=100, small_method='ecm')

    success, factorization, _ = wrapper._factor_hybrid(P * Q, 16)
    assert success
    assert factorization == {P: 1, Q: 1}
    # Exactly one ECM call, from the completion loop: its target is the
    # half-digits bound, NOT the pretest's 4/13 target
    assert len(ecm.calls) == 1
    digits = len(str(P * Q))
    assert ecm.calls[0].target_t_level == float((digits + 1) // 2 + 2)


def test_at_or_above_hybrid_threshold_runs_ecm_pretest():
    # With hybrid_threshold below the cofactor size the pretest loop runs,
    # targeting 4/13 of the digit length (regression: this used to be gated
    # on siqs_threshold, leaving hybrid_threshold entirely unused)
    ecm = FakeECM([_result([str(P)], t_achieved=5.0)])
    wrapper = _make_wrapper(ecm, hybrid=10, siqs=100, small_method='ecm')

    success, factorization, _ = wrapper._factor_hybrid(P * Q, 16)
    assert success
    assert factorization == {P: 1, Q: 1}
    assert len(ecm.calls) == 1
    digits = len(str(P * Q))
    assert ecm.calls[0].target_t_level == calculate_target_tlevel(digits)


# ==================== ECM-to-completion (--small-method ecm) ====================


def test_ecm_completion_escalates_target_until_factor_found():
    digits = len(str(P * Q))
    first_target = float((digits + 1) // 2 + 2)
    ecm = FakeECM([
        _result([], t_achieved=first_target),          # pass 1: miss
        _result([str(P)], t_achieved=first_target + 5),  # pass 2: hit
    ])
    wrapper = _make_wrapper(ecm)

    factors = wrapper._ecm_factor_completely(P * Q, 0.0, [])
    assert sorted(factors) == sorted([str(P), str(Q)])
    assert len(ecm.calls) == 2
    assert ecm.calls[1].target_t_level > ecm.calls[0].target_t_level
    # Second pass continues from the achieved t-level instead of restarting
    assert ecm.calls[1].start_t_level == first_target


def test_ecm_completion_records_full_multiplicity():
    # ECM reports p once; p^2 divides the cofactor, so p must appear twice
    ecm = FakeECM([_result([str(P)], t_achieved=10.0)])
    wrapper = _make_wrapper(ecm)

    factors = wrapper._ecm_factor_completely(P * P * Q, 0.0, [])
    assert sorted(factors) == sorted([str(P), str(P), str(Q)])


def test_ecm_completion_interrupt_propagates():
    interrupted = _result([], t_achieved=1.0)
    interrupted.interrupted = True
    wrapper = _make_wrapper(FakeECM([interrupted]))

    import pytest
    with pytest.raises(KeyboardInterrupt):
        wrapper._ecm_factor_completely(P * Q, 0.0, [])


def test_small_method_siqs_below_threshold_does_not_use_completion():
    # Default small_method='siqs' must route small cofactors to YAFU - the
    # ForbiddenTool stub makes that path raise, proving it was reached
    ecm = FakeECM([])
    wrapper = _make_wrapper(ecm, hybrid=100, siqs=100, small_method='siqs')

    import pytest
    with pytest.raises(AssertionError, match='yafu'):
        wrapper._factor_hybrid(P * Q, 16)
