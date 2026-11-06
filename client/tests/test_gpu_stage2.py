"""
GPU stage 2 (GMP-ECM GPU stage 2 builds) and GPU-for-both-stages routing.

Output samples below are verbatim from the GPU stage 2 build of GMP-ECM
(gpu-stage1-dynamic branch) on an RTX 5070 Ti.
"""
import argparse
import sys
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import ecm_command
from lib.arg_parser import resolve_gpu_both, resolve_gpu_settings
from lib.ecm_command import gpu_stage2_supported
from lib.ecm_config import ECMConfig, FactorResult, TLevelConfig
from lib.ecm_executor import ECMWrapper
from lib.ecm_math import MIN_BATCH_B1, b1_for_current_tlevel, get_optimal_b1_for_tlevel
from lib.ecm_modes import ResolvedParams, submit_ecm_result
from lib.work_helpers import request_ecm_work
import ecm_client
from lib.parsing_utils import count_completed_curves, count_gpu_stage1_curves, parse_b2_used
from lib.stage2_executor import GPUStage2Executor, Stage2Executor, create_stage2_executor
from lib.work_args import WorkArgs
from lib.work_modes import Stage2ConsumerMode, StandardAutoWorkMode, WorkLoopContext

GPU_RESUME_NO_FACTOR = """\
GMP-ECM 7.0.6 [configured with GMP 6.3.0, --enable-asm-redc, --enable-gpu] [ECM]
Resuming ECM residue saved by kylea@MSI with GMP-ECM 7.0.6 on Thu Nov  5 22:38:00 2026
Input number is 4640802841 (10 digits)
Using B1=3000000-3000000, B2=2141548510, sigma=3:2398233522-2398235569 (2048 curves)
Computing 2048 Step 2 took 1558ms of CPU time / 8267ms of GPU time
"""

GPU_RESUME_FACTOR = """\
Using B1=150-150, B2=23766, sigma=3:3336705341-3336705852 (512 curves)
GPU: factor 916011935471 found in Step 2 with curve 55 (-sigma 3:3336705396)
GPU: factor 916011935471 found in Step 2 with curve 352 (-sigma 3:3336705693)
Computing 512 Step 2 took 40ms of CPU time / 147ms of GPU time
********** Factor found in step 2: 916011935471
Found prime factor of 12 digits: 916011935471
"""

GPU_FULL_HEAD = """\
Using B1=3000000, B2=5706890290, sigma=3:100-3171 (3072 curves)
Computing 3072 Step 1 took 900ms of CPU time / 41000ms of GPU time
"""

GPU_FULL = GPU_FULL_HEAD + "Computing 3072 Step 2 took 1500ms of CPU time / 9000ms of GPU time\n"

CPU_RESUME = """\
Using B1=3000000-3000000, B2=1500000000, polynomial Dickson(6), sigma=3:2398233522
Step 2 took 664ms
Using B1=3000000-3000000, B2=1500000000, polynomial Dickson(6), sigma=3:2398233523
Step 2 took 671ms
"""


class TestCountCompletedCurves:
    def test_gpu_batch_counts_every_curve(self):
        assert count_completed_curves(GPU_RESUME_NO_FACTOR) == 2048

    def test_multiple_gpu_batches_sum(self):
        assert count_completed_curves(GPU_RESUME_NO_FACTOR * 2) == 4096

    def test_cpu_curves_count_one_each(self):
        assert count_completed_curves(CPU_RESUME) == 2

    def test_mixed_gpu_and_cpu_fallback(self):
        # A GPU run hands non -param 3 residues to the CPU, one line per curve
        assert count_completed_curves(GPU_RESUME_NO_FACTOR + CPU_RESUME) == 2050

    def test_batch_line_not_double_counted_as_cpu_curve(self):
        assert count_completed_curves("Computing 512 Step 2 took 40ms of CPU time\n") == 512

    def test_gpu_batch_with_cpu_stage2(self):
        """A GPU batch whose stage 2 runs on the CPU (B2 > 2^62, no kernel)."""
        out = (GPU_FULL_HEAD
               + "GPU: stage 2 on the CPU (B2 above 2^62 is too large for GPU stage 2)\n"
               + "Curve 1 Step 2 took 664ms\nCurve 2 Step 2 took 671ms\n"
               + "Computing 3072 Step 2 on CPU took 2039000ms\n")
        # per-curve lines are part of the batch summary, not extra curves
        assert count_completed_curves(out) == 3072

    def test_cpu_stage2_in_gpu_batch_stopped_part_way(self):
        out = GPU_FULL_HEAD + "Curve 1 Step 2 took 664ms\nCurve 2 Step 2 took 671ms\n"
        assert count_completed_curves(out) == 2

    def test_killed_in_stage1_counts_nothing(self):
        """The Using line's "(N curves)" is printed before any work."""
        assert count_completed_curves(GPU_FULL_HEAD.splitlines()[0] + "\n") == 0

    def test_killed_in_stage2_counts_nothing(self):
        assert count_completed_curves(GPU_FULL_HEAD) == 0

    def test_stage1_factor_batch_not_credited(self):
        """GMP-ECM skips stage 2 for the whole batch after a stage 1 factor,
        so none of its curves reached B2; its stage 1 still counts."""
        out = ("Using B1=50000, B2=12746592, sigma=3:100-3171 (3072 curves)\n"
               "GPU: factor 1234567 found in Step 1 with curve 5 (-sigma 3:105)\n"
               "Computing 3072 Step 1 took 900ms of CPU time / 1800ms of GPU time\n")
        assert count_completed_curves(out) == 0
        assert count_gpu_stage1_curves(out) == 3072

    def test_second_batch_killed(self):
        out = GPU_FULL + GPU_FULL_HEAD
        assert count_completed_curves(out) == 3072

    def test_stage1_only_count(self):
        assert count_gpu_stage1_curves(GPU_FULL_HEAD * 2) == 6144


class TestParseB2Used:
    def test_rounded_explicit_b2(self):
        assert parse_b2_used(GPU_RESUME_NO_FACTOR) == 2141548510

    def test_b2min_range(self):
        assert parse_b2_used("Using B1=3000000, B2=1000000-2141548510, sigma=3:1\n") == 2141548510

    def test_smallest_of_several_batches(self):
        out = (GPU_FULL_HEAD.replace("B2=5706890290", "B2=300000000") + GPU_FULL_HEAD)
        assert parse_b2_used(out) == 300000000

    def test_none_without_using_line(self):
        assert parse_b2_used("Computing 512 Step 2 took 40ms\n") is None


RESIDUE_LINE = "METHOD=ECM; PARAM=3; SIGMA=3336705396; B1=150; N=4640802841; X=0x1;\n"


def _fake_ecm(tmp_path: Path, output: str, exit_code: int = 0, help_text: str = "") -> str:
    """A stand-in GMP-ECM: -h prints help_text, anything else prints output."""
    out = tmp_path / "out.txt"
    out.write_text(output)
    hlp = tmp_path / "help.txt"
    hlp.write_text(help_text)
    script = tmp_path / "ecm"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"if '-h' in sys.argv: print(open({str(hlp)!r}).read()); sys.exit(0)\n"
        f"open({str(tmp_path / 'argv.txt')!r}, 'w').write(' '.join(sys.argv[1:]))\n"
        f"sys.stdout.write(open({str(out)!r}).read())\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(0o755)
    # the residue file the executor tests resume (-param 3, as GPU stage 1 makes)
    (tmp_path / "r.txt").write_text(RESIDUE_LINE)
    return str(script)


def _wrapper(ecm_path: str = "/usr/bin/ecm", b1: int = 3_000_000, curve_count: int = 2048) -> Mock:
    wrapper = Mock()
    wrapper.typed_config.programs.gmp_ecm.path = ecm_path
    wrapper.stop_event = threading.Event()
    wrapper.graceful_shutdown_requested = False
    wrapper._parse_residue_file.return_value = {
        'composite': '4640802841', 'b1': b1, 'curve_count': curve_count,
    }
    return wrapper


class TestGPUStage2Command:
    def test_command_shape(self):
        ex = GPUStage2Executor(_wrapper(), Path("r.txt"), 3_000_000, 1_500_000_000,
                               gpu_device=0, gpu_curves=4096)
        cmd = ex.build_command(3_000_000, early_termination=True)
        assert cmd[0] == "/usr/bin/ecm"
        assert cmd[1:4] == ["-gpu", "-gpudevice", "0"]
        assert cmd[cmd.index("-gpucurves") + 1] == "4096"
        assert cmd[cmd.index("-resume") + 1] == "r.txt"
        assert "-one" in cmd
        assert cmd[-2:] == ["3000000-3000000", "1500000000"]

    def test_no_k_and_no_curve_count(self):
        """k is tuned for CPU memory; -c would be meaningless with -resume."""
        cmd = GPUStage2Executor(_wrapper(), Path("r.txt"), 3_000_000, 1_500_000_000) \
            .build_command(3_000_000, early_termination=False)
        assert "-k" not in cmd and "-c" not in cmd and "-one" not in cmd
        assert "-gpucurves" not in cmd  # default: GMP-ECM batches the whole file

    def test_default_b2_omitted(self):
        cmd = GPUStage2Executor(_wrapper(), Path("r.txt"), 3_000_000, -1) \
            .build_command(3_000_000, early_termination=True)
        assert cmd[-1] == "3000000-3000000"


class TestGPUStage2Execute:
    def test_no_factor(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR, exit_code=0)
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, 1_500_000_000)
        factor, factors, curves, _, sigma = ex.execute()
        assert (factor, factors, curves, sigma) == (None, [], 2048, None)
        assert "Computing 2048 Step 2" in ex.raw_output

    def test_factor_found(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_FACTOR, exit_code=14)
        ex = GPUStage2Executor(_wrapper(ecm, b1=150, curve_count=512),
                               tmp_path / "r.txt", 150, 15000)
        factor, factors, curves, _, sigma = ex.execute()
        assert factor == "916011935471"
        assert factors == ["916011935471"]
        assert curves == 512
        assert sigma == "3:3336705396"

    def test_b1_taken_from_residue_file(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR)
        GPUStage2Executor(_wrapper(ecm, b1=11_000_000), tmp_path / "r.txt",
                          3_000_000, 1_500_000_000).execute()
        argv = (tmp_path / "argv.txt").read_text().split()
        assert "11000000-11000000" in argv

    def test_error_exit_without_fallback_flags_failure(self, tmp_path):
        ecm = _fake_ecm(tmp_path, "GPU: Error, cannot fit stage 2 in memory\n", exit_code=1)
        wrapper = _wrapper(ecm)
        ex = GPUStage2Executor(wrapper, tmp_path / "r.txt", 3_000_000, 1_500_000_000)
        _, _, curves, _, _ = ex.execute()
        assert curves == 0
        assert ex.failed is True and "exit code 1" in (ex.error_message or "")
        assert wrapper.logger.error.called

    def test_error_exit_falls_back_to_cpu_workers(self, tmp_path):
        ecm = _fake_ecm(tmp_path, "GPU: Error, cannot fit stage 2 in memory\n", exit_code=1)
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, 1_500_000_000,
                               fallback_k=4, fallback_workers=8)
        cpu = Mock()
        cpu.execute.return_value = ("97", ["97"], 2048, 5.0, "3:42")
        cpu.raw_output = "cpu output"
        cpu.factor_sigmas = [("97", "3:42")]
        cpu.b2_used = 1_500_000_000
        with patch("lib.stage2_executor.Stage2Executor", return_value=cpu) as cls:
            factor, factors, curves, _, sigma = ex.execute(progress_interval=50)
        assert (factor, factors, curves, sigma) == ("97", ["97"], 2048, "3:42")
        assert cls.call_args.args[4:6] == (4, 8)  # k, workers
        assert cpu.execute.call_args.kwargs['progress_interval'] == 50
        assert ex.failed is False
        assert ex.b2_used == 1_500_000_000
        assert "cpu output" in ex.raw_output
        assert ex.factor_sigmas == [("97", "3:42")]

    def test_cpu_fallback_runs_fallback_b2(self, tmp_path):
        """GPU asked for GMP-ECM's default B2: the CPU rerun uses fallback_b2."""
        ecm = _fake_ecm(tmp_path, "GPU: Error, cannot fit stage 2 in memory\n", exit_code=1)
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, -1,
                               fallback_k=4, fallback_workers=8, fallback_b2=1_500_000_000)
        cpu = Mock()
        cpu.execute.return_value = (None, [], 2048, 5.0, None)
        cpu.raw_output, cpu.factor_sigmas, cpu.b2_used = "", [], 1_500_000_000
        with patch("lib.stage2_executor.Stage2Executor", return_value=cpu) as cls:
            ex.execute()
        assert cls.call_args.args[3] == 1_500_000_000
        assert ex.b2_used == 1_500_000_000

    def test_reports_b2_actually_run(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR)
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, 1_500_000_000)
        ex.execute()
        assert ex.b2_used == 2141548510

    def test_each_factor_keeps_its_sigma(self, tmp_path):
        out = ("Using B1=150-150, B2=23766, sigma=3:1000-1511 (512 curves)\n"
               "GPU: factor 97 found in Step 2 with curve 3 (-sigma 3:1003)\n"
               "GPU: factor 101 found in Step 2 with curve 9 (-sigma 3:1009)\n"
               "Computing 512 Step 2 took 40ms of CPU time / 147ms of GPU time\n")
        ex = GPUStage2Executor(_wrapper(_fake_ecm(tmp_path, out, exit_code=2), b1=150),
                               tmp_path / "r.txt", 150, 15000)
        _, factors, _, _, _ = ex.execute()
        assert sorted(factors) == ["101", "97"]
        assert dict(ex.factor_sigmas) == {"97": "3:1003", "101": "3:1009"}

    def test_stop_requested_before_start_does_not_run(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR)
        wrapper = _wrapper(ecm)
        wrapper.graceful_shutdown_requested = True
        wrapper.stop_event.set()
        assert GPUStage2Executor(wrapper, tmp_path / "r.txt", 3_000_000,
                                 1_500_000_000).execute()[2] == 0
        assert not (tmp_path / "argv.txt").exists()
        assert wrapper.stop_event.is_set()  # not swallowed

    def test_non_param3_residues_run_on_cpu_workers(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR)
        (tmp_path / "r.txt").write_text(RESIDUE_LINE.replace("PARAM=3", "PARAM=1"))
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, 1_500_000_000,
                               fallback_k=4, fallback_workers=8)
        cpu = Mock()
        cpu.execute.return_value = (None, [], 2048, 5.0, None)
        cpu.raw_output = "cpu output"
        cpu.factor_sigmas = []
        cpu.b2_used = 1_500_000_000
        with patch("lib.stage2_executor.Stage2Executor", return_value=cpu):
            assert ex.execute()[2] == 2048
        assert not (tmp_path / "argv.txt").exists()  # GPU process never started
        assert ex.raw_output == "cpu output"

    def test_non_param3_residues_without_workers_fail(self, tmp_path):
        ecm = _fake_ecm(tmp_path, GPU_RESUME_NO_FACTOR)
        (tmp_path / "r.txt").write_text(RESIDUE_LINE.replace("PARAM=3", "PARAM=1"))
        ex = GPUStage2Executor(_wrapper(ecm), tmp_path / "r.txt", 3_000_000, 1_500_000_000)
        ex.execute()
        assert ex.failed is True and "-param 1" in (ex.error_message or "")


class TestCapabilityProbe:
    def setup_method(self):
        ecm_command._gpu_stage2_cache.clear()

    def teardown_method(self):
        ecm_command._gpu_stage2_cache.clear()

    def test_new_build_detected(self, tmp_path):
        ecm = _fake_ecm(tmp_path, "", help_text="  -cpustage2   With -gpu, do stage 2 on the CPU")
        assert gpu_stage2_supported(ecm) is True

    def test_old_build_rejected(self, tmp_path):
        ecm = _fake_ecm(tmp_path, "", help_text="  -gpu         Use CGBN for computations stage 1.")
        assert gpu_stage2_supported(ecm) is False

    def test_missing_binary(self, tmp_path):
        assert gpu_stage2_supported(str(tmp_path / "nope")) is False

    def test_failed_probe_not_cached(self, tmp_path):
        """A binary missing for a moment (mid-rebuild) is retried later."""
        path = str(tmp_path / "ecm")
        assert gpu_stage2_supported(path) is False
        _fake_ecm(tmp_path, "", help_text="  -cpustage2   With -gpu, ...")
        assert gpu_stage2_supported(path) is True


class TestFactory:
    def test_cpu_by_default(self):
        ex = create_stage2_executor(_wrapper(), Path("r.txt"), 1, 2, None, 8)
        assert isinstance(ex, Stage2Executor)

    def test_gpu_when_supported(self):
        with patch("lib.stage2_executor.gpu_stage2_supported", return_value=True):
            ex = create_stage2_executor(_wrapper(), Path("r.txt"), 1, 2, None, 8,
                                        use_gpu=True, gpu_curves=4096)
        assert isinstance(ex, GPUStage2Executor)
        assert ex.gpu_curves == 4096
        assert ex.fallback_workers == 8  # CPU pool if the GPU run fails

    def test_falls_back_to_cpu_on_old_binary(self):
        wrapper = _wrapper()
        with patch("lib.stage2_executor.gpu_stage2_supported", return_value=False):
            ex = create_stage2_executor(wrapper, Path("r.txt"), 1, 2, 4, 8, use_gpu=True)
        assert isinstance(ex, Stage2Executor)
        assert ex.k == 4 and ex.workers == 8
        assert wrapper.logger.warning.called

    def test_cpu_b2_replaces_default_on_cpu(self):
        with patch("lib.stage2_executor.gpu_stage2_supported", return_value=False):
            ex = create_stage2_executor(_wrapper(), Path("r.txt"), 1, -1, None, 8,
                                        use_gpu=True, cpu_b2=500)
        assert isinstance(ex, Stage2Executor)
        assert ex.b2 == 500 and ex.b2_used == 500

    def test_cpu_reports_explicit_b2_only(self):
        assert create_stage2_executor(_wrapper(), Path("r.txt"), 1, 2, None, 8).b2_used == 2
        assert create_stage2_executor(_wrapper(), Path("r.txt"), 1, -1, None, 8).b2_used is None


class TestStage2ConsumerGpuSelection:
    def _mode(self, wrapper: Mock | None = None, **kw) -> Stage2ConsumerMode:
        wrapper = wrapper or Mock()
        wrapper._get_api_client.return_value = wrapper.api_client
        wrapper.typed_config.programs.gmp_ecm.gpu_enabled = True
        return Stage2ConsumerMode(WorkLoopContext(
            wrapper=wrapper, client_id='c', args=WorkArgs(stage2_only=True, **kw)))

    def test_config_gpu_enabled_alone_stays_on_cpu(self):
        """gpu_enabled must not move stage 2 onto a card a producer is using."""
        mode = self._mode()
        assert mode._use_gpu is False
        assert "CPU" in mode.mode_name

    def test_explicit_gpu_flag(self):
        mode = self._mode(gpu=True)
        assert mode._use_gpu is True
        assert "GPU" in mode.mode_name

    def test_no_gpu_wins(self):
        assert self._mode(gpu=True, no_gpu=True)._use_gpu is False

    def _run(self, tmp_path, executor, **work_extra):
        wrapper = Mock()
        wrapper.typed_config.execution.residue_dir = str(tmp_path)
        wrapper.typed_config.programs.gmp_ecm.gpu_device = 0
        wrapper.typed_config.programs.gmp_ecm.gpu_curves = None

        def download(**kwargs):
            Path(kwargs['output_path']).write_text('residue\n')
            return True
        wrapper.api_client.download_residue.side_effect = download
        mode = self._mode(wrapper, gpu=True, gpu_curves=4096)

        work = {'residue_id': 7, 'composite': '4640802841', 'digit_length': 10,
                'b1': 3_000_000, 'curve_count': 2048, 'suggested_b2': 1_500_000_000,
                'parametrization': 3, **work_extra}
        with patch("lib.work_modes.stage2_consumer.create_stage2_executor",
                   return_value=executor) as factory:
            mode.on_work_started(work)
            result = mode.execute_work(work)
        self.factory_args = factory.call_args.args
        return mode, result, factory.call_args.kwargs

    @staticmethod
    def _gpu_executor(curves=2048, failed=False, b2_used: 'int | None' = 2141548510,
                      factor_sigmas=()):
        executor = Mock(spec=GPUStage2Executor)
        factors = [f for f, _ in factor_sigmas]
        executor.execute.return_value = (factors[0] if factors else None, factors,
                                         curves, 1.0, None)
        executor.raw_output = GPU_RESUME_NO_FACTOR
        executor.failed = failed
        executor.error_message = "GPU stage 2 failed (exit code 1)" if failed else None
        executor.b2_used = b2_used
        executor.factor_sigmas = list(factor_sigmas)
        return executor

    def test_execute_uses_gpu_executor(self, tmp_path):
        mode, result, kwargs = self._run(tmp_path, self._gpu_executor())
        assert result.success is True
        assert result.curves_run == 2048
        assert kwargs['use_gpu'] is True
        assert kwargs['gpu_curves'] == 4096
        assert kwargs['gpu_device'] == 0
        # the rounded-up B2 the GPU ran is what gets reported
        assert mode._b2 == 2141548510

    def test_gpu_defaults_to_gmp_ecm_b2(self, tmp_path):
        """No dictionary/--b2/--b2-multiplier: GMP-ECM's default, not suggested_b2;
        a run that lands on CPU workers keeps suggested_b2."""
        mode, _, kwargs = self._run(tmp_path, self._gpu_executor())
        assert self.factory_args[3] == -1
        assert kwargs['cpu_b2'] == 1_500_000_000
        assert mode._b2 == 2141548510  # what ran

    def test_explicit_b2_has_no_cpu_override(self, tmp_path):
        _, _, kwargs = self._run(tmp_path, self._gpu_executor(), b2_from_dict=9_000_000_000)
        assert self.factory_args[3] == 9_000_000_000
        assert kwargs['cpu_b2'] is None

    def test_unreported_default_b2_is_not_success(self, tmp_path):
        """Submitting no B2 would let the server credit its own default."""
        _, result, _ = self._run(tmp_path, self._gpu_executor(b2_used=None))
        assert result.success is False
        assert "B2" in (result.error_message or "")

    def test_unreported_default_b2_with_factor_submits_suggestion(self, tmp_path):
        ex = self._gpu_executor(b2_used=None, factor_sigmas=[("97", "3:1003")])
        mode, result, _ = self._run(tmp_path, ex)
        assert result.success is True
        assert mode._b2 == 1_500_000_000

    def test_non_param3_residue_goes_to_cpu(self, tmp_path):
        cpu = Mock(spec=Stage2Executor)
        cpu.execute.return_value = (None, [], 2048, 1.0, None)
        cpu.raw_output = ""
        cpu.failed, cpu.error_message, cpu.b2_used, cpu.factor_sigmas = False, None, None, []
        mode, result, kwargs = self._run(tmp_path, cpu, parametrization=1)
        assert kwargs['use_gpu'] is False
        assert mode._b2 == 1_500_000_000  # CPU stage 2 keeps the server's B2

    def test_failed_gpu_run_is_not_success(self, tmp_path):
        _, result, _ = self._run(tmp_path, self._gpu_executor(curves=0, failed=True,
                                                             b2_used=None))
        assert result.success is False
        assert "exit code 1" in (result.error_message or "")

    def test_zero_curves_is_not_success(self, tmp_path):
        """Stopped before stage 2 started: no empty attempt gets submitted."""
        _, result, _ = self._run(tmp_path, self._gpu_executor(curves=0, b2_used=None))
        assert result.success is False

    def test_each_factor_submitted_with_its_sigma(self, tmp_path):
        ex = self._gpu_executor(factor_sigmas=[("97", "3:1003"), ("101", "3:1009")])
        _, result, _ = self._run(tmp_path, ex)
        assert list(zip(result.factors, result.sigmas)) == [("97", "3:1003"), ("101", "3:1009")]


class TestResolveGpuSettings:
    @staticmethod
    def _config(gpu_enabled=True):
        config = Mock()
        config.programs.gmp_ecm.gpu_enabled = gpu_enabled
        config.programs.gmp_ecm.gpu_device = 0
        config.programs.gmp_ecm.gpu_curves = None
        return config

    @pytest.mark.parametrize("kw, expected", [
        ({}, False),                                   # gpu_enabled alone never
        ({'standard': True}, False),
        ({'stage2_only': True}, False),
        ({'multiprocess': True}, False),
        ({'gpu': True}, True),
        ({'stage1_only': True}, True),                 # implied for stage 1
        ({'two_stage': True}, True),
        ({'stage1_only': True, 'no_gpu': True}, False),
        ({'two_stage': True, 'no_gpu': True}, False),
        ({'stage1_only': True, 'method': 'pm1'}, False),
    ])
    def test_use_gpu(self, kw, expected):
        assert resolve_gpu_settings(WorkArgs(**kw), self._config())[0] is expected

    @pytest.mark.parametrize("kw, expected", [
        ({}, True),
        ({'two_stage': True}, False),
        ({'multiprocess': True}, False),
        ({'stage1_only': True}, False),
        ({'stage2_only': True}, False),
        ({'method': 'pm1'}, False),
    ])
    def test_gpu_both(self, kw, expected):
        assert resolve_gpu_both(WorkArgs(**kw), True) is expected
        assert resolve_gpu_both(WorkArgs(**kw), False) is False


class TestTLevelGpuRouting:
    def _fake_self(self) -> Mock:
        fake = Mock()
        fake.interrupted = False
        fake.shutdown_level = 0
        fake.typed_config.programs.gmp_ecm.path = "/usr/bin/ecm"
        found = FactorResult(curves_run=3072, success=True)
        found.add_factor("1234567", "3:42")
        fake.run_ecm_v2.return_value = found
        return fake

    def test_use_gpu_sets_param_3(self):
        assert TLevelConfig(composite="123", target_t_level=20, use_gpu=True).parametrization == 3

    def test_gpu_step_runs_single_gpu_process(self):
        fake = self._fake_self()
        config = TLevelConfig(composite="1234567891011", target_t_level=20,
                              use_gpu=True, gpu_device=1, gpu_curves=2048,
                              threads=8, no_submit=True)
        with patch("lib.ecm_command.gpu_stage2_supported", return_value=True), \
             patch("lib.ecm_math.calculate_curves_to_target_direct", return_value=100):
            result = ECMWrapper.run_tlevel_v2(fake, config)

        # threads=8 would select multiprocess; GPU must take precedence
        fake.run_multiprocess_v2.assert_not_called()
        fake.run_two_stage_v2.assert_not_called()
        fake._run_tlevel_pipelined.assert_not_called()
        step: ECMConfig = fake.run_ecm_v2.call_args.args[0]
        assert step.use_gpu is True
        assert step.parametrization == 3
        assert step.b2 is None  # GMP-ECM's default B2
        assert (step.gpu_device, step.gpu_curves) == (1, 2048)
        assert result.curves_run == 3072  # actual GPU curves, not requested

    def test_cpu_tlevel_unchanged(self):
        fake = self._fake_self()
        config = TLevelConfig(composite="1234567891011", target_t_level=20, no_submit=True)
        with patch("lib.ecm_math.calculate_curves_to_target_direct", return_value=100):
            ECMWrapper.run_tlevel_v2(fake, config)
        step: ECMConfig = fake.run_ecm_v2.call_args.args[0]
        assert step.use_gpu is False
        assert step.parametrization == 1


class TestTLevelGpuCredit:
    """GPU steps are credited with the curves and B2 that actually ran."""

    @staticmethod
    def _run(step: FactorResult, no_submit=False, target=20.0):
        fake = Mock()
        fake.interrupted = False
        fake.shutdown_level = 0
        fake.graceful_shutdown_requested = False
        fake.typed_config.programs.gmp_ecm.path = "/usr/bin/ecm"
        fake.run_ecm_v2.return_value = step
        config = TLevelConfig(composite="1234567891011", target_t_level=target,
                              use_gpu=True, no_submit=no_submit)
        with patch("lib.ecm_command.gpu_stage2_supported", return_value=True), \
             patch("lib.ecm_math.calculate_curves_to_target_direct", return_value=100), \
             patch("lib.ecm_math.calculate_tlevel", return_value=21.0) as tlevel:
            result = ECMWrapper.run_tlevel_v2(fake, config)
        return fake, result, tlevel

    def test_bsgs_b2_goes_into_history_and_submission(self):
        out = ("Using B1=11000, B2=1100000, sigma=3:100-3171 (3072 curves)\n"
               "Computing 3072 Step 1 took 1ms of CPU time / 2ms of GPU time\n"
               "Computing 3072 Step 2 took 1ms of CPU time / 2ms of GPU time\n")
        fake, result, tlevel = self._run(
            FactorResult(curves_run=3072, success=True, exit_code=0, raw_output=out,
                         b2_used=1100000))
        history = tlevel.call_args.args[0]
        assert history[-1].endswith(",1100000,p=3")
        assert history[-1].startswith("3072@")
        submitted = fake.submit_result.call_args.args[0]
        assert submitted['b2'] == 1100000

    def test_failed_step_is_not_credited_or_submitted(self):
        fake, result, tlevel = self._run(FactorResult(
            curves_run=0, success=False, exit_code=1,
            raw_output="GPU: Error, cannot fit stage 2 in the GPU memory\n"))
        fake.submit_result.assert_not_called()
        tlevel.assert_not_called()
        assert result.curves_run == 0
        assert fake.run_ecm_v2.call_count == 1  # stops instead of trying more steps

    def test_interrupted_step_ends_the_run(self):
        """Auto-work's second Ctrl+C kills the batch; no further steps start."""
        fake, result, _ = self._run(FactorResult(curves_run=0, exit_code=-2, interrupted=True),
                                    target=30.0)
        assert fake.run_ecm_v2.call_count == 1
        fake.submit_result.assert_not_called()

    def test_stage1_factor_step_is_submitted(self):
        """0 curves credited (stage 2 skipped), but the factor is submitted."""
        fake, result, tlevel = self._run(FactorResult(
            factors=["97"], sigmas=["3:5"], curves_run=0, success=True, exit_code=8))
        assert fake.submit_result.call_count == 1
        assert fake.submit_result.call_args.args[0]['curves_completed'] == 0
        tlevel.assert_not_called()


class TestGpuPrimitiveCurveCount:
    @staticmethod
    def _primitive(lines, exit_code, graceful=False, **kw):
        fake = Mock()
        fake.interrupted = False
        fake.graceful_shutdown_requested = graceful
        fake.typed_config.programs.gmp_ecm.path = "/usr/bin/ecm"
        fake._stream_subprocess_output.return_value = (Mock(returncode=exit_code), lines)
        return ECMWrapper._execute_ecm_primitive(
            fake, composite="1234567891011", b1=11000, curves=100, use_gpu=True, **kw)

    def test_failed_run_counts_zero_not_requested(self):
        r = self._primitive(["Using B1=11000, B2=1100000, sigma=3:1-3072 (3072 curves)",
                             "GPU: Error, cannot fit stage 2 in the GPU memory"], 1)
        assert r['curves_completed'] == 0
        assert r['success'] is False

    def test_ctrl_c_level2_is_interrupted(self):
        r = self._primitive(["Using B1=11000, B2=1100000, sigma=3:1-3072 (3072 curves)"],
                            -2, graceful=True)
        assert r['interrupted'] is True
        assert r['curves_completed'] == 0

    def test_sigint_exit_is_interrupted(self):
        assert self._primitive([], -2)['interrupted'] is True

    def test_completed_batches_counted(self):
        r = self._primitive(GPU_FULL.splitlines() * 2, 0)
        assert r['curves_completed'] == 6144

    def test_stage1_only_counts_stage1_batches(self):
        r = self._primitive(GPU_FULL_HEAD.splitlines(), 0, residue_save=Path("r.txt"))
        assert r['curves_completed'] == 3072
        assert r['b2_used'] is None

    def test_b2_used_reported(self):
        r = self._primitive(GPU_FULL.splitlines(), 0)
        assert r['b2_used'] == parse_b2_used(GPU_FULL) is not None

    def test_pm1_with_gpu_flag_counts_as_cpu(self):
        """-gpu is only passed for ECM: P-1 output is not GPU output."""
        r = self._primitive(["Step 1 took 5ms"], 0, method='pm1')
        assert r['curves_completed'] == 100
        assert r['b2_used'] is None


class TestEcmWrapperSubmission:
    """ecm_wrapper.py submits the B2 and -param that actually ran."""

    @staticmethod
    def _submit(result: FactorResult, use_gpu=True, **kw):
        args = argparse.Namespace(**{
            'tlevel': None, 'submit': True, 'b2': None, 'curves': 3000, 'param': None,
            'project': None, 'stage1_only': False, 'stage2_only': None,
            'multiprocess': False, 'two_stage': False, 'gpu': False, **kw})
        params = ResolvedParams(b1=11000, method='ecm', use_gpu=use_gpu, gpu_device=None,
                                gpu_curves=None, workers=1, max_batch=None,
                                b2_dictionary=None)
        wrapper = Mock()
        submit_ecm_result(wrapper, args, Mock(), params, result, "1234567891011")
        return wrapper.submit_result.call_args.args[0]

    def test_gpu_b2_used_submitted(self):
        sent = self._submit(FactorResult(curves_run=3072, b2_used=1100000), gpu=True)
        assert sent['b2'] == 1100000
        assert sent['parametrization'] == 3

    def test_stage1_only_without_gpu_flag_is_param3(self):
        sent = self._submit(FactorResult(curves_run=3072), stage1_only=True)
        assert (sent['b2'], sent['parametrization']) == (0, 3)

    def test_stage2_only_uses_residue_param(self, tmp_path):
        res = tmp_path / "r.txt"
        res.write_text(RESIDUE_LINE.replace("PARAM=3", "PARAM=1"))
        sent = self._submit(FactorResult(curves_run=2048), stage2_only=str(res), b2=10**9)
        assert (sent['b2'], sent['parametrization']) == (10**9, 1)


class TestGpuRunEcmSplitsFactors:
    def test_composite_factor_split_with_sigma(self):
        fake = Mock()
        fake._execute_ecm_primitive.return_value = {
            'success': True, 'factors': ["9797"], 'sigmas': ["3:55"],
            'curves_completed': 3072, 'raw_output': '', 'parametrization': 3,
            'exit_code': 2, 'interrupted': False, 'b2_used': None,
        }
        fake._fully_factor_found_result.return_value = ["97", "101"]
        result = ECMWrapper.run_ecm_v2(fake, ECMConfig(
            composite="1234567891011", b1=11000, curves=100, use_gpu=True))
        assert result.factors == ["97", "101"]
        assert result.sigmas == ["3:55", "3:55"]


class TestOneBatch:
    """--one-batch: one GPU batch (both stages) per assignment, then the next composite."""

    @staticmethod
    def _mode(run_result: FactorResult, **kw):
        wrapper = Mock()
        wrapper.typed_config.programs.gmp_ecm.gpu.curves_per_batch = 1000
        wrapper.typed_config.programs.gmp_ecm.gpu_device = 0
        wrapper.typed_config.programs.gmp_ecm.gpu_curves = None
        wrapper.run_ecm_v2.return_value = run_result
        args = WorkArgs(gpu=True, one_batch=True, **kw)
        mode = StandardAutoWorkMode(WorkLoopContext(wrapper=wrapper, client_id='c', args=args))
        return mode, wrapper

    @staticmethod
    def _work(current_t):
        return {'composite': '1234567891011', 'digit_length': 13,
                'current_t_level': current_t, 'target_t_level': 60.0}

    def test_b1_for_current_tlevel(self):
        assert b1_for_current_tlevel(54.7) == get_optimal_b1_for_tlevel(55)[0]
        assert b1_for_current_tlevel(0.0) == get_optimal_b1_for_tlevel(20)[0]
        assert b1_for_current_tlevel(0.0, MIN_BATCH_B1) == MIN_BATCH_B1

    def test_runs_one_gpu_batch_at_current_tlevel_b1(self):
        mode, wrapper = self._mode(FactorResult(curves_run=2304, success=True,
                                                b2_used=777_000_000_000))
        mode._target_already_met = True  # left over from an earlier assignment
        mode.execute_work(self._work(54.7))
        assert mode._target_already_met is False
        config: ECMConfig = wrapper.run_ecm_v2.call_args.args[0]
        assert config.b1 == get_optimal_b1_for_tlevel(55)[0]
        assert (config.curves, config.use_gpu, config.b2) == (1000, True, None)
        assert wrapper.run_ecm_v2.call_count == 1  # no t-level climb
        sent = mode._results_dict
        assert (sent['b1'], sent['b2'], sent['parametrization']) == (
            config.b1, 777_000_000_000, 3)
        assert mode._is_tlevel_mode is False  # submitted by submit_results

    def test_low_tlevel_uses_minimum_b1(self):
        mode, wrapper = self._mode(FactorResult(curves_run=2304, success=True))
        mode.execute_work(self._work(12.0))
        assert wrapper.run_ecm_v2.call_args.args[0].b1 == MIN_BATCH_B1

    def test_explicit_b1_is_floored(self):
        mode, wrapper = self._mode(FactorResult(curves_run=2304, success=True), b1=11000)
        mode.execute_work(self._work(54.7))
        assert wrapper.run_ecm_v2.call_args.args[0].b1 == MIN_BATCH_B1

    def test_zero_curves_rejected(self):
        mode, _ = self._mode(FactorResult(), curves=0)
        with pytest.raises(ValueError):
            mode.execute_work(self._work(54.7))

    def test_b2_dictionary_uses_chosen_b1(self):
        mode, wrapper = self._mode(FactorResult(curves_run=2304, success=True))
        b1 = get_optimal_b1_for_tlevel(55)[0]
        mode._b2_dictionary, mode._k_dictionary = {b1: 123_000_000_000}, {b1: 4}
        mode.execute_work(self._work(54.7))
        config = wrapper.run_ecm_v2.call_args.args[0]
        assert config.b2 == 123_000_000_000

    def test_stage1_factor_with_zero_curves_is_submitted(self):
        """A stage 1 factor makes GMP-ECM skip stage 2: 0 curves, but a factor."""
        mode, wrapper = self._mode(FactorResult(curves_run=0, success=True,
                                                factors=["97"], sigmas=["3:5"]))
        work = self._work(54.7)
        mode.execute_work(work)
        assert mode.submit_results(work, FactorResult()) is True
        assert wrapper.submit_result.call_args.args[0]['factors_found'] == ["97"]

    def test_explicit_b1_and_curves_win(self):
        mode, wrapper = self._mode(FactorResult(curves_run=4096, success=True),
                                   b1=3_000_000, curves=4096)
        mode.execute_work(self._work(54.7))
        config = wrapper.run_ecm_v2.call_args.args[0]
        assert (config.b1, config.curves) == (3_000_000, 4096)

    @pytest.mark.parametrize("kw, ok", [
        ({'gpu': True}, True),
        ({}, False),                                  # needs --gpu
        ({'gpu': True, 'tlevel': 35.0}, False),
        ({'gpu': True, 'two_stage': True}, False),
        ({'gpu': True, 'stage1_only': True}, False),
        ({'gpu': True, 'pm1': True}, False),
    ])
    def test_validation(self, kw, ok):
        assert (ecm_client._one_batch_error(WorkArgs(one_batch=True, **kw)) is None) is ok

    def test_claim_sized_like_stage1(self):
        api = Mock()
        api.get_ecm_work.return_value = {'composite': '1'}
        request_ecm_work(api, 'c', WorkArgs(gpu=True, one_batch=True), Mock())
        assert api.get_ecm_work.call_args.kwargs['stage1_only'] is True
