#!/usr/bin/env python3
"""
Unit tests for ecm_command.py - GMP-ECM command-line builder.

build_ecm_command() is a pure function with no I/O, making it ideal for
thorough unit testing of all flag combinations and B2 edge cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.ecm_command import build_ecm_command


class TestBasicCommand:
    """Test minimal command construction."""

    def test_minimal_ecm_command(self):
        """Bare minimum: just path and B1."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000)
        assert cmd == ["/usr/bin/ecm", "50000"]

    def test_b1_is_last_positional(self):
        """B1 should always be the last element when no B2."""
        cmd = build_ecm_command("/usr/bin/ecm", 1000000)
        assert cmd[-1] == "1000000"


class TestMethodFlags:
    """Test method selection (ecm/pm1/pp1)."""

    def test_default_method_is_ecm(self):
        """Default method (ecm) should not add any method flag."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000)
        assert "-pm1" not in cmd
        assert "-pp1" not in cmd

    def test_pm1_method(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pm1")
        assert cmd[1] == "-pm1"

    def test_pp1_method(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pp1")
        assert cmd[1] == "-pp1"

    def test_ecm_method_explicit(self):
        """Explicit ecm method should not add a method flag."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="ecm")
        assert "-pm1" not in cmd
        assert "-pp1" not in cmd


class TestGPUFlags:
    """Test GPU-related flags."""

    def test_gpu_flag(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=True)
        assert "-gpu" in cmd

    def test_gpu_with_device(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=True, gpu_device=1)
        idx = cmd.index("-gpudevice")
        assert cmd[idx + 1] == "1"

    def test_gpu_with_curves(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=True, gpu_curves=2048)
        idx = cmd.index("-gpucurves")
        assert cmd[idx + 1] == "2048"

    def test_gpu_ignored_for_pm1(self):
        """GPU flags should be ignored for non-ECM methods."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pm1", use_gpu=True, gpu_device=0)
        assert "-gpu" not in cmd
        assert "-gpudevice" not in cmd

    def test_gpu_ignored_for_pp1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pp1", use_gpu=True)
        assert "-gpu" not in cmd


class TestResidueFlags:
    """Test residue save/load flags."""

    def test_save_residue(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, residue_save=Path("/tmp/res.txt"))
        idx = cmd.index("-save")
        assert cmd[idx + 1] == "/tmp/res.txt"

    def test_load_residue(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, residue_load=Path("/tmp/res.txt"))
        idx = cmd.index("-resume")
        assert cmd[idx + 1] == "/tmp/res.txt"

    def test_save_and_load_together(self):
        cmd = build_ecm_command(
            "/usr/bin/ecm", 50000,
            residue_save=Path("/tmp/save.txt"),
            residue_load=Path("/tmp/load.txt"),
        )
        assert "-save" in cmd
        assert "-resume" in cmd
        # save should come before resume (both in residue operations section)
        assert cmd.index("-save") < cmd.index("-resume")


class TestVerboseFlag:

    def test_verbose_on(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, verbose=True)
        assert "-v" in cmd

    def test_verbose_off(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, verbose=False)
        assert "-v" not in cmd

    def test_gpu_ecm_always_verbose(self):
        """GPU stage 1 progress is OUTPUT_VERBOSE-gated upstream, so without -v
        a multi-hour batch prints nothing at all."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=True, verbose=False)
        assert "-v" in cmd

    def test_gpu_ecm_verbose_not_duplicated(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=True, verbose=True)
        assert cmd.count("-v") == 1

    def test_cpu_ecm_still_respects_verbose_off(self):
        """CPU ECM prints per-curve detail under -v, so it must stay opt-in."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, use_gpu=False, verbose=False)
        assert "-v" not in cmd

    def test_gpu_flag_ignored_for_pm1_does_not_force_verbose(self):
        cmd = build_ecm_command(
            "/usr/bin/ecm", 50000, method="pm1", use_gpu=True, verbose=False
        )
        assert "-gpu" not in cmd
        assert "-v" not in cmd


class TestParametrization:
    """Test -param flag (ECM only)."""

    def test_param_ecm(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, param=3)
        idx = cmd.index("-param")
        assert cmd[idx + 1] == "3"

    def test_param_zero(self):
        """Parametrization 0 is valid (Brent-Suyama for large numbers)."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, param=0)
        idx = cmd.index("-param")
        assert cmd[idx + 1] == "0"

    def test_param_ignored_for_pm1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pm1", param=1)
        assert "-param" not in cmd

    def test_param_ignored_for_pp1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pp1", param=3)
        assert "-param" not in cmd


class TestSigma:
    """Test -sigma flag (ECM only)."""

    def test_sigma_integer(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, sigma=12345)
        idx = cmd.index("-sigma")
        assert cmd[idx + 1] == "12345"

    def test_sigma_string_format(self):
        """Sigma can be '3:12345' format."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, sigma="3:12345")
        idx = cmd.index("-sigma")
        assert cmd[idx + 1] == "3:12345"

    def test_sigma_ignored_for_pm1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, method="pm1", sigma=999)
        assert "-sigma" not in cmd

    def test_sigma_none_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, sigma=None)
        assert "-sigma" not in cmd


class TestOneFlag:

    def test_one_flag(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, one=True)
        assert "-one" in cmd

    def test_one_flag_off(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, one=False)
        assert "-one" not in cmd


class TestCurves:

    def test_curves(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, curves=100)
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "100"

    def test_curves_none_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, curves=None)
        assert "-c" not in cmd


class TestKFlag:

    def test_k_flag(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, k=4)
        idx = cmd.index("-k")
        assert cmd[idx + 1] == "4"

    def test_k_none_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, k=None)
        assert "-k" not in cmd


class TestMaxmem:

    def test_maxmem(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, maxmem=2048)
        idx = cmd.index("-maxmem")
        assert cmd[idx + 1] == "2048"

    def test_maxmem_none_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, maxmem=None)
        assert "-maxmem" not in cmd


class TestB1Done:
    """Test B1done prefix for stage 2 resume."""

    def test_b1done_format(self):
        """B1 should be formatted as 'b1done-b1'."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b1done=50000)
        assert "50000-50000" in cmd

    def test_b1done_different_values(self):
        cmd = build_ecm_command("/usr/bin/ecm", 110000000, b1done=50000)
        assert "50000-110000000" in cmd

    def test_b1done_none_uses_plain_b1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b1done=None)
        assert "50000" in cmd
        assert "-" not in cmd[-1] if cmd[-1] == "50000" else True


class TestB2Handling:
    """Test B2 rules: None/-1 omit, 0 includes as '0', >0 includes."""

    def test_b2_none_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b2=None)
        # B1 should be last element
        assert cmd[-1] == "50000"

    def test_b2_negative_one_omitted(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b2=-1)
        assert cmd[-1] == "50000"

    def test_b2_zero_included(self):
        """B2=0 means stage 1 only."""
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b2=0)
        assert cmd[-1] == "0"
        assert cmd[-2] == "50000"

    def test_b2_positive(self):
        cmd = build_ecm_command("/usr/bin/ecm", 50000, b2=5000000)
        assert cmd[-1] == "5000000"
        assert cmd[-2] == "50000"

    def test_b2_large_value(self):
        cmd = build_ecm_command("/usr/bin/ecm", 110000000, b2=11000000000000)
        assert cmd[-1] == "11000000000000"


class TestFlagOrdering:
    """Test that flags appear in the correct order per GMP-ECM requirements.

    Expected order: method -> GPU -> residue ops -> verbose -> param -> sigma -> -one -> curves -> k -> maxmem -> B1 -> B2
    """

    def test_full_command_ordering(self):
        """Test a command with many flags to verify ordering."""
        cmd = build_ecm_command(
            "/usr/bin/ecm",
            50000,
            b2=5000000,
            curves=100,
            method="ecm",
            use_gpu=True,
            gpu_device=0,
            residue_save=Path("/tmp/res.txt"),
            verbose=True,
            param=3,
            sigma="3:12345",
            one=True,
            k=4,
            maxmem=2048,
        )

        # Verify ordering by checking indices
        gpu_idx = cmd.index("-gpu")
        save_idx = cmd.index("-save")
        v_idx = cmd.index("-v")
        param_idx = cmd.index("-param")
        sigma_idx = cmd.index("-sigma")
        one_idx = cmd.index("-one")
        c_idx = cmd.index("-c")
        k_idx = cmd.index("-k")
        maxmem_idx = cmd.index("-maxmem")
        b1_idx = cmd.index("50000")
        b2_idx = cmd.index("5000000")

        assert gpu_idx < save_idx < v_idx < param_idx < sigma_idx < one_idx < c_idx
        assert c_idx < k_idx < maxmem_idx < b1_idx < b2_idx

    def test_pm1_with_curves_and_b2(self):
        """PM1 command should have method flag first, no GPU/param/sigma."""
        cmd = build_ecm_command(
            "/usr/bin/ecm",
            1000000,
            b2=100000000,
            curves=1,
            method="pm1",
            verbose=True,
        )
        assert cmd[0] == "/usr/bin/ecm"
        assert cmd[1] == "-pm1"
        assert "-gpu" not in cmd
        assert "-param" not in cmd
        assert "-sigma" not in cmd
        assert cmd[-1] == "100000000"  # B2
        assert cmd[-2] == "1000000"    # B1


class TestEdgeCases:

    def test_all_defaults(self):
        """Only required args, all optionals default."""
        cmd = build_ecm_command("/usr/bin/ecm", 11000)
        assert cmd == ["/usr/bin/ecm", "11000"]

    def test_b1_zero(self):
        """B1=0 is technically valid."""
        cmd = build_ecm_command("/usr/bin/ecm", 0)
        assert "0" in cmd

    def test_very_large_b1(self):
        cmd = build_ecm_command("/usr/bin/ecm", 25000000000)
        assert "25000000000" in cmd

    def test_gpu_all_options(self):
        cmd = build_ecm_command(
            "/usr/bin/ecm", 110000000,
            use_gpu=True, gpu_device=2, gpu_curves=4096,
            param=3, b2=0,
        )
        assert "-gpu" in cmd
        assert "-gpudevice" in cmd
        assert "-gpucurves" in cmd
        assert "-param" in cmd
        assert cmd[-1] == "0"  # B2=0 for stage 1 only
