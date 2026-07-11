#!/usr/bin/env python3
"""
Unit tests for lib/ecm_modes.py - extracted mode handlers.

Primary focus: run_tlevel_mode() progressive loop logic.
Secondary: verify config construction + wrapper calls for other modes.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lib.ecm_config import FactorResult, TLevelConfig, ECMConfig, MultiprocessConfig, TwoStageConfig
from lib.ecm_modes import (
    ResolvedParams,
    run_stage2_only_mode,
    run_stage1_only_mode,
    run_tlevel_mode,
    run_multiprocess_mode,
    run_two_stage_mode,
    run_standard_mode,
    submit_ecm_result,
)


def _default_params(**overrides) -> ResolvedParams:
    """Create ResolvedParams with sensible defaults."""
    defaults = dict(
        b1=50000,
        method='ecm',
        use_gpu=False,
        gpu_device=None,
        gpu_curves=None,
        workers=4,
        max_batch=None,
        b2_dictionary=None,
    )
    defaults.update(overrides)
    return ResolvedParams(**defaults)


def _default_args(**overrides) -> SimpleNamespace:
    """Create args namespace with sensible defaults for t-level mode."""
    defaults = dict(
        composite="123456789012345678901234567890123456789012345678901",  # 51 digits
        tlevel=-1.0,  # progressive mode sentinel
        start_tlevel=0.0,
        param=None,
        two_stage=False,
        workers=None,
        verbose=False,
        progress_interval=None,
        b2_multiplier=None,
        b2=None,
        submit=False,
        project=None,
        curves=None,
        sigma=None,
        method=None,
        config='client.yaml',
        stage1_only=False,
        stage2_only=None,
        multiprocess=False,
        gpu=False,
        save_residues=None,
        upload=False,
        b1=None,
        b2_dictionary=None,
        max_batch=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_factor_result(factors=None, curves_run=100, execution_time=10.0,
                        t_level_achieved=0.0, interrupted=False,
                        curve_summary=None) -> FactorResult:
    """Create a FactorResult with specified values."""
    result = FactorResult()
    if factors:
        for f in factors:
            result.add_factor(f, None)
    result.curves_run = curves_run
    result.execution_time = execution_time
    result.t_level_achieved = t_level_achieved
    result.interrupted = interrupted
    result.curve_summary = curve_summary or []
    return result


class TestTLevelModeProgressive:
    """Tests for run_tlevel_mode() progressive factorization loop."""

    @patch('lib.ecm_modes.calculate_target_tlevel', return_value=50.0)
    def test_factor_found_prime_cofactor_stops(self, mock_calc):
        """Factor found + prime cofactor -> stops, both factors in result."""
        # 7 * 13 = 91
        composite = "91"
        args = _default_args(composite=composite, tlevel=-1.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        # Wrapper returns a result with factor 7
        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            factors=["7"],
            curves_run=50,
            execution_time=5.0,
            t_level_achieved=25.0,
            curve_summary=[{"b1": 11000, "curves": 50}],
        )

        result = run_tlevel_mode(wrapper, args, output, params)

        assert "7" in result.factors
        # The prime cofactor 13 should also be added
        assert "13" in result.factors
        assert result.curves_run == 50
        assert result.execution_time == 5.0
        # Should only call wrapper once since cofactor is prime
        assert wrapper.run_tlevel_v2.call_count == 1

    @patch('lib.ecm_modes.calculate_target_tlevel', return_value=50.0)
    def test_factor_found_composite_cofactor_loops(self, mock_calc):
        """Factor found + composite cofactor -> loops, calls wrapper again."""
        # 3 * 7 * 11 = 231
        composite = "231"
        args = _default_args(composite=composite, tlevel=-1.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        # First call: finds factor 3, cofactor 77 = 7*11 (composite)
        result1 = _make_factor_result(
            factors=["3"],
            curves_run=50,
            execution_time=5.0,
            t_level_achieved=20.0,
            curve_summary=[{"b1": 11000, "curves": 50}],
        )
        # Second call: finds factor 7, cofactor 11 is prime
        result2 = _make_factor_result(
            factors=["7"],
            curves_run=30,
            execution_time=3.0,
            t_level_achieved=25.0,
            curve_summary=[{"b1": 50000, "curves": 30}],
        )
        wrapper.run_tlevel_v2.side_effect = [result1, result2]

        result = run_tlevel_mode(wrapper, args, output, params)

        assert wrapper.run_tlevel_v2.call_count == 2
        assert "3" in result.factors
        assert "7" in result.factors
        assert "11" in result.factors  # prime cofactor appended
        assert result.curves_run == 80  # 50 + 30
        assert result.execution_time == 8.0  # 5.0 + 3.0
        assert len(result.curve_summary) == 2

    def test_no_factor_found_stops(self):
        """No factor found -> stops after one iteration."""
        args = _default_args()
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            factors=None,
            curves_run=200,
            execution_time=20.0,
            t_level_achieved=30.0,
        )

        result = run_tlevel_mode(wrapper, args, output, params)

        assert wrapper.run_tlevel_v2.call_count == 1
        assert result.factors == []
        assert result.success is False
        assert result.curves_run == 200

    def test_interrupted_stops(self):
        """Interrupted -> stops, reports partial progress."""
        args = _default_args()
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            factors=None,
            curves_run=50,
            execution_time=5.0,
            t_level_achieved=15.0,
            interrupted=True,
        )

        result = run_tlevel_mode(wrapper, args, output, params)

        assert wrapper.run_tlevel_v2.call_count == 1
        assert result.curves_run == 50
        output.warning.assert_called()

    def test_explicit_tlevel_stops_after_factor(self):
        """Explicit t-level: stops after finding factor (doesn't continue with cofactor)."""
        # 3 * 7 * 11 = 231
        composite = "231"
        args = _default_args(composite=composite, tlevel=25.0)  # explicit, not progressive
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            factors=["3"],
            curves_run=50,
            execution_time=5.0,
            t_level_achieved=25.0,
        )

        result = run_tlevel_mode(wrapper, args, output, params)

        # Should NOT continue with cofactor in explicit mode
        assert wrapper.run_tlevel_v2.call_count == 1
        assert result.factors == ["3"]
        # Cofactor 77 should NOT be appended (it's composite, and we don't continue)
        assert "77" not in result.factors

    def test_already_at_target_skips(self):
        """Already at target -> skips entirely, wrapper never called."""
        args = _default_args(tlevel=25.0, start_tlevel=30.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        result = run_tlevel_mode(wrapper, args, output, params)

        wrapper.run_tlevel_v2.assert_not_called()
        assert result.factors == []
        assert result.curves_run == 0

    @patch('lib.ecm_modes.calculate_target_tlevel', return_value=50.0)
    def test_fully_factored_cofactor_is_1(self, mock_calc):
        """Cofactor = 1 after dividing out factors -> stops."""
        # composite = 2 * 3 = 6, factor list returns both
        composite = "6"
        args = _default_args(composite=composite, tlevel=-1.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            factors=["2", "3"],
            curves_run=10,
            execution_time=1.0,
            t_level_achieved=20.0,
        )

        result = run_tlevel_mode(wrapper, args, output, params)

        assert wrapper.run_tlevel_v2.call_count == 1
        assert "2" in result.factors
        assert "3" in result.factors
        output.success.assert_any_call("Fully factored!")

    @patch('lib.ecm_modes.calculate_target_tlevel', return_value=50.0)
    def test_aggregates_across_iterations(self, mock_calc):
        """Curves, time, and curve_summary accumulate across iterations."""
        # 2 * 3 * 5 * 7 = 210
        composite = "210"
        args = _default_args(composite=composite, tlevel=-1.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        # Iteration 1: finds 2, cofactor 105 = 3*5*7
        r1 = _make_factor_result(
            factors=["2"],
            curves_run=40,
            execution_time=4.0,
            t_level_achieved=20.0,
            curve_summary=[{"b1": 11000, "curves": 40}],
        )
        # Iteration 2: finds 3, cofactor 35 = 5*7
        r2 = _make_factor_result(
            factors=["3"],
            curves_run=30,
            execution_time=3.0,
            t_level_achieved=20.0,
            curve_summary=[{"b1": 11000, "curves": 30}],
        )
        # Iteration 3: finds 5, cofactor 7 is prime
        r3 = _make_factor_result(
            factors=["5"],
            curves_run=20,
            execution_time=2.0,
            t_level_achieved=20.0,
            curve_summary=[{"b1": 11000, "curves": 20}],
        )
        wrapper.run_tlevel_v2.side_effect = [r1, r2, r3]

        result = run_tlevel_mode(wrapper, args, output, params)

        assert result.curves_run == 90  # 40 + 30 + 20
        assert result.execution_time == 9.0  # 4 + 3 + 2
        assert len(result.curve_summary) == 3
        assert set(result.factors) == {"2", "3", "5", "7"}  # 7 is prime cofactor

    def test_tlevel_config_passed_correctly(self):
        """TLevelConfig passed to wrapper has correct field values."""
        args = _default_args(
            tlevel=35.0,
            start_tlevel=10.0,
            param=3,
            two_stage=True,
            workers=8,
            verbose=True,
            progress_interval=50,
            b2_multiplier=1000.0,
            submit=True,
            project="test-project",
        )
        params = _default_params(max_batch=500, b2_dictionary={50000: 25000000}, gpu_device=0, gpu_curves=2048)
        output = Mock()
        wrapper = Mock()

        wrapper.run_tlevel_v2.return_value = _make_factor_result(
            t_level_achieved=35.0, curves_run=100
        )

        run_tlevel_mode(wrapper, args, output, params)

        config = wrapper.run_tlevel_v2.call_args[0][0]
        assert isinstance(config, TLevelConfig)
        assert config.target_t_level == 35.0
        assert config.start_t_level == 10.0
        assert config.parametrization == 3
        assert config.use_two_stage is True
        assert config.workers == 8
        assert config.verbose is True
        assert config.progress_interval == 50
        assert config.max_batch_curves == 500
        assert config.b2_multiplier == 1000.0
        assert config.b2_dictionary == {50000: 25000000}
        assert config.project == "test-project"
        assert config.no_submit is False
        assert config.gpu_device == 0
        assert config.gpu_curves == 2048


class TestStandardMode:
    """Tests for run_standard_mode()."""

    def test_constructs_config_and_calls_wrapper(self):
        """Verifies ECMConfig is constructed with correct params."""
        args = _default_args(composite="12345", b2=5000000, curves=100, param=1)
        params = _default_params(b1=50000, method='ecm')
        output = Mock()
        wrapper = Mock()

        wrapper.run_ecm_v2.return_value = _make_factor_result(curves_run=100)

        result = run_standard_mode(wrapper, args, output, params)

        config = wrapper.run_ecm_v2.call_args[0][0]
        assert isinstance(config, ECMConfig)
        assert config.composite == "12345"
        assert config.b1 == 50000
        assert config.b2 == 5000000
        assert config.curves == 100
        assert result.curves_run == 100

    def test_gpu_parametrization(self):
        """GPU mode sets parametrization=3 when no explicit param."""
        args = _default_args(composite="12345", param=None)
        params = _default_params(use_gpu=True)
        output = Mock()
        wrapper = Mock()

        wrapper.run_ecm_v2.return_value = _make_factor_result()

        run_standard_mode(wrapper, args, output, params)

        config = wrapper.run_ecm_v2.call_args[0][0]
        assert config.parametrization == 3


class TestMultiprocessMode:
    """Tests for run_multiprocess_mode()."""

    def test_constructs_config_and_calls_wrapper(self):
        """Verifies MultiprocessConfig is constructed correctly."""
        args = _default_args(composite="12345", curves=500, workers=4, param=1)
        params = _default_params(b1=250000)
        output = Mock()
        wrapper = Mock()

        wrapper.run_multiprocess_v2.return_value = _make_factor_result(curves_run=500)

        result = run_multiprocess_mode(wrapper, args, output, params)

        config = wrapper.run_multiprocess_v2.call_args[0][0]
        assert isinstance(config, MultiprocessConfig)
        assert config.composite == "12345"
        assert config.b1 == 250000
        assert config.total_curves == 500
        assert result.curves_run == 500

    def test_default_curves_when_none(self):
        """Uses 1000 curves when args.curves is None."""
        args = _default_args(composite="12345", curves=None)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        wrapper.run_multiprocess_v2.return_value = _make_factor_result()

        run_multiprocess_mode(wrapper, args, output, params)

        config = wrapper.run_multiprocess_v2.call_args[0][0]
        assert config.total_curves == 1000


class TestTwoStageMode:
    """Tests for run_two_stage_mode()."""

    def test_explicit_b2(self):
        """Explicit --b2 is used directly."""
        args = _default_args(composite="12345", b2=9999999, curves=200)
        params = _default_params(b1=110000000)
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert isinstance(config, TwoStageConfig)
        assert config.b2 == 9999999

    def test_b2_from_multiplier(self):
        """B2 calculated from multiplier when --b2 not specified."""
        args = _default_args(composite="12345", b2=None, b2_multiplier=1000.0)
        params = _default_params(b1=100000)
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert config.b2 == 100000000  # 100000 * 1000

    def test_b2_default_when_no_multiplier(self):
        """B2 is None (GMP-ECM default) when neither --b2 nor multiplier given."""
        args = _default_args(composite="12345", b2=None)
        # Remove b2_multiplier attr to simulate it not being set
        del args.b2_multiplier
        params = _default_params(b1=110000000)
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert config.b2 is None

    def test_b2_from_dictionary(self):
        """B2 looked up from dictionary when --b2 not specified."""
        args = _default_args(composite="12345", b2=None)
        params = _default_params(b1=110000000,
                                 b2_dictionary={110000000: 11000000000000})
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert config.b2 == 11000000000000

    def test_explicit_b2_overrides_dictionary(self):
        """Explicit --b2 wins over a dictionary entry."""
        args = _default_args(composite="12345", b2=9999999)
        params = _default_params(b1=110000000,
                                 b2_dictionary={110000000: 11000000000000})
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert config.b2 == 9999999

    def test_dictionary_miss_falls_back_to_multiplier(self):
        """B1 not in dictionary falls back to --b2-multiplier when given."""
        args = _default_args(composite="12345", b2=None, b2_multiplier=1000.0)
        params = _default_params(b1=100000, b2_dictionary={110000000: 11000000000000})
        output = Mock()
        wrapper = Mock()

        wrapper.run_two_stage_v2.return_value = _make_factor_result()

        run_two_stage_mode(wrapper, args, output, params)

        config = wrapper.run_two_stage_v2.call_args[0][0]
        assert config.b2 == 100000000  # 100000 * 1000

    def test_dictionary_miss_without_fallback_exits(self):
        """B1 not in dictionary with no other B2 source is a hard error."""
        args = _default_args(composite="12345", b2=None)
        params = _default_params(b1=100000, b2_dictionary={110000000: 11000000000000})
        output = Mock()
        wrapper = Mock()

        with pytest.raises(SystemExit):
            run_two_stage_mode(wrapper, args, output, params)

        output.error.assert_called_once()
        wrapper.run_two_stage_v2.assert_not_called()


class TestSubmitEcmResult:
    """Tests for submit_ecm_result()."""

    def test_submits_when_factors_found(self):
        """Submits results when factors are found and submit=True."""
        args = _default_args(submit=True, tlevel=None, curves=100, project="proj")
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        result = _make_factor_result(factors=["12345"], curves_run=50)

        submit_ecm_result(wrapper, args, output, params, result, "999999999")

        wrapper.submit_result.assert_called_once()
        call_args = wrapper.submit_result.call_args
        results_dict = call_args[0][0]
        assert results_dict['composite'] == "999999999"
        assert results_dict['b1'] == 50000
        assert results_dict['project'] == "proj"

    def test_skips_when_tlevel_mode(self):
        """Skips submission for t-level mode (handles its own)."""
        args = _default_args(submit=True, tlevel=35.0)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        result = _make_factor_result(factors=["12345"])

        submit_ecm_result(wrapper, args, output, params, result, "999999999")

        wrapper.submit_result.assert_not_called()

    def test_submits_curves_only_when_no_factors(self):
        """Submits curve count when no factors but curves were run."""
        args = _default_args(submit=True, tlevel=None, curves=100)
        params = _default_params()
        output = Mock()
        wrapper = Mock()

        result = _make_factor_result(factors=None, curves_run=100)

        submit_ecm_result(wrapper, args, output, params, result, "999999999")

        wrapper.submit_result.assert_called_once()
        output.info.assert_any_call("\nSubmitting 100 curves (no factors) to API...")
