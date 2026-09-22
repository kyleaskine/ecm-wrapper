#!/usr/bin/env python3
"""
Tests for stage 2 progress-interval resolution and reporting.

Two separate contracts that used to disagree:
  - arg_parser.resolve_stage2_progress_interval picks the interval
  - stage2_executor honours 0 as "no progress reporting"

Their disagreement (-v resolved to 0, and 0 meant "every curve" rather than
"off") produced one log line per curve per worker.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lib.arg_parser import resolve_stage2_progress_interval
from lib.stage2_executor import Stage2Executor


def args(progress_interval=None, verbose=False):
    """None is 'flag not given' - the argparse default is None, not 0, so that
    an explicit --progress-interval 0 stays distinguishable from unset."""
    return SimpleNamespace(progress_interval=progress_interval, verbose=verbose)


class TestResolveStage2ProgressInterval:

    def test_defaults_to_50(self):
        assert resolve_stage2_progress_interval(args()) == 50  # type: ignore[arg-type]

    def test_explicit_value_wins(self):
        assert resolve_stage2_progress_interval(args(progress_interval=200)) == 200  # type: ignore[arg-type]

    def test_explicit_zero_disables(self):
        """--progress-interval 0 must reach stage 2 as 0; the help says
        '0 = disabled', and with a 0 argparse default it never could."""
        assert resolve_stage2_progress_interval(args(progress_interval=0)) == 0  # type: ignore[arg-type]

    def test_negative_is_clamped_to_disabled(self):
        assert resolve_stage2_progress_interval(args(progress_interval=-5)) == 0  # type: ignore[arg-type]

    def test_verbose_does_not_silence_progress(self):
        """-v used to resolve to 0, which meant a line per curve downstream."""
        assert resolve_stage2_progress_interval(args(verbose=True)) == 50  # type: ignore[arg-type]

    def test_verbose_with_explicit_value_still_uses_value(self):
        opts = args(progress_interval=1, verbose=True)
        assert resolve_stage2_progress_interval(opts) == 1  # type: ignore[arg-type]

    def test_missing_attributes_fall_back_to_default(self):
        assert resolve_stage2_progress_interval(SimpleNamespace()) == 50  # type: ignore[arg-type]


class FakeStdout:
    """Feeds _stream_worker_output a fixed sequence of GMP-ECM lines."""

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def read(self):
        return ""


def make_executor(logger):
    executor = Stage2Executor.__new__(Stage2Executor)
    executor.logger = logger
    executor.stop_event = Mock()
    executor.stop_event.is_set.return_value = False
    executor.process_lock = Mock()
    executor.running_processes = []
    return executor


def stream(executor, curves, progress_interval):
    process = Mock()
    process.stdout = FakeStdout([f"Step 2 took {i}ms\n" for i in range(curves)])
    process.wait.return_value = 0
    return executor._stream_worker_output(
        process, worker_id=1, total_lines=curves,
        progress_interval=progress_interval, early_termination=False,
    )


def progress_lines(logger):
    return [
        c for c in logger.info.call_args_list
        if "curves" in str(c) and "Worker" in str(c)
    ]


class TestStage2ProgressReporting:

    def test_zero_interval_disables_reporting(self):
        """0 means off, as the docstring and --progress-interval help both say."""
        logger = Mock()
        stream(make_executor(logger), curves=256, progress_interval=0)
        assert progress_lines(logger) == []

    def test_negative_interval_disables_reporting(self):
        logger = Mock()
        stream(make_executor(logger), curves=64, progress_interval=-1)
        assert progress_lines(logger) == []

    def test_interval_of_50_reports_periodically(self):
        logger = Mock()
        stream(make_executor(logger), curves=256, progress_interval=50)
        assert len(progress_lines(logger)) == 5  # 50,100,150,200,250

    def test_interval_of_one_reports_every_curve(self):
        """The old -v behaviour is still reachable, just explicitly."""
        logger = Mock()
        stream(make_executor(logger), curves=20, progress_interval=1)
        assert len(progress_lines(logger)) == 20

    def test_all_curves_still_counted_when_reporting_disabled(self):
        logger = Mock()
        output = stream(make_executor(logger), curves=256, progress_interval=0)
        # Curve accounting reads "Step 2 took" (OUTPUT_NORMAL upstream), so it
        # is unaffected by progress reporting or by -v.
        assert output.count("Step 2 took") == 256
