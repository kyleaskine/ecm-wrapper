#!/usr/bin/env python3
"""
Tests for stage 1 residue handling in pipelined (GPU+CPU) t-level execution.

A stage 1 batch is hours of GPU work. If a run ends between stage 1 finishing
and stage 2 consuming the residue, that work must survive on disk so it can be
finished later with --stage2-only. Only a batch whose stage 2 actually ran may
have its residue deleted.

Covers each way a batch can end up unprocessed:
  - interrupted after stage 1, before the CPU thread picks it up
  - still sitting in the queue when the threads stop
  - stage 2 raised
"""
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lib.ecm_config import ExecutionBatch
from lib.execution_engine import CompositeExecutionEngine


RESIDUE_BODY = "METHOD=ECM; PARAM=3; SIGMA=3:1000; B1=110000000; N=12345; X=0x1;\n"
B2_PLANNED = 55000000000


class LogWatcher(logging.Handler):
    """Fires an Event when a given substring is logged.

    Lets a test synchronise with the CPU thread's exit instead of sleeping,
    which keeps the threaded assertions deterministic.

    Never assert on this from inside a fake stage 1: that runs on the GPU
    thread, whose handler deliberately swallows BaseException so a dead
    producer can't strand the consumer. Record the result and assert on the
    main thread instead, or a failed handshake turns into a silent 10s stall.
    """

    def __init__(self, needle: str):
        super().__init__()
        self.needle = needle
        self.seen = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if self.needle in record.getMessage():
            self.seen.set()


def make_wrapper(residue_dir: Path, logger=None) -> SimpleNamespace:
    """Minimal ECMWrapper stand-in for driving run_pipelined()."""
    return SimpleNamespace(
        logger=logger or Mock(),
        stop_event=threading.Event(),
        interrupted=False,
        _active_stage2_executor=None,
        typed_config=SimpleNamespace(
            programs=SimpleNamespace(gmp_ecm=SimpleNamespace(path="ecm")),
            execution=SimpleNamespace(residue_dir=str(residue_dir)),
        ),
        submit_result=Mock(return_value=True),
        _terminate_all_subprocesses=Mock(),
        _signal_subprocesses_interrupt=Mock(),
        _parse_residue_file=Mock(
            return_value={"b1": 110000000, "curve_count": 3072, "composite": "12345"}
        ),
    )


class StubBatchProducer:
    """Yields a fixed list of batch sizes, then None."""

    def __init__(self, batch_sizes):
        self._sizes = list(batch_sizes)
        self.projected_t_level = 52.8

    def next_batch(self):
        if not self._sizes:
            return None
        batch = ExecutionBatch(composite="12345", b1=110000000, curves=self._sizes.pop(0))
        return (batch, 55.0, B2_PLANNED)

    def update_projected(self, actual_curves, b1, b2):
        self.projected_t_level += 0.1


def run(engine, batch_sizes=(3072,), workers=4):
    return engine.run_pipelined(
        batch_producer=StubBatchProducer(batch_sizes),
        composite="12345",
        stage2_workers=workers,
        no_submit=True,
        start_t_level=52.8,
    )


class TestInterruptBeforeStage2:
    """Ctrl+C after stage 1 must not throw the batch away."""

    def test_residue_survives_and_is_reported(self, tmp_path):
        # Real logger so the test can wait for the CPU thread to stand down
        # before stage 1 returns - that is the window the bug lived in.
        logger = logging.getLogger("test_interrupt_before_stage2")
        logger.setLevel(logging.INFO)
        watcher = LogWatcher("[CPU Thread] Shutdown detected while waiting")
        logger.addHandler(watcher)

        wrapper = make_wrapper(tmp_path, logger=logger)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        handshake: dict = {}

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            # User hits Ctrl+C while stage 1 is wrapping up.
            wrapper.interrupted = True
            # Let the CPU consumer exit first, so this batch has no consumer.
            handshake["stood_down"] = watcher.seen.wait(timeout=10)
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1
        try:
            result = run(engine)
        finally:
            logger.removeHandler(watcher)

        assert handshake.get("stood_down"), "CPU thread never stood down"
        kept = result.preserved_residues
        assert len(kept) == 1, "completed stage 1 work was discarded on interrupt"

        residue_path, b2_planned = kept[0]
        assert residue_path.exists(), f"residue was deleted: {residue_path}"
        assert residue_path.read_text() == RESIDUE_BODY
        # B2 rides along so the printed --stage2-only command is complete.
        assert b2_planned == B2_PLANNED

    def test_residue_lives_in_residue_dir_not_tmp(self, tmp_path):
        logger = logging.getLogger("test_residue_dir")
        logger.setLevel(logging.INFO)
        watcher = LogWatcher("[CPU Thread] Shutdown detected while waiting")
        logger.addHandler(watcher)

        wrapper = make_wrapper(tmp_path, logger=logger)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        handshake: dict = {}

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            wrapper.interrupted = True
            handshake["stood_down"] = watcher.seen.wait(timeout=10)
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1
        try:
            result = run(engine)
        finally:
            logger.removeHandler(watcher)

        assert handshake.get("stood_down"), "CPU thread never stood down"
        residue_path = result.preserved_residues[0][0]
        assert tmp_path in residue_path.parents, (
            f"residue landed outside the configured residue_dir: {residue_path}"
        )


class TestStage2Raised:
    """A stage 2 crash leaves the stage 1 work recoverable."""

    def test_residue_preserved_when_stage2_raises(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        with patch(
            "lib.stage2_executor.Stage2Executor", side_effect=RuntimeError("boom")
        ):
            result = run(engine)

        assert len(written) == 1
        assert written[0].exists(), "residue was lost when stage 2 raised"
        assert [p for p, _ in result.preserved_residues] == written


class TestCompletedBatchCleanedUp:
    """A batch whose stage 2 ran is still cleaned up - no disk leak."""

    def test_processed_residue_is_deleted(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()
        # (factor, all_factors, curves_completed, stage2_time, sigma)
        stage2.execute.return_value = (None, [], 3072, 1.0, None)

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine)

        assert stage2.execute.called, "stage 2 never ran"
        assert result.preserved_residues == []
        assert len(written) == 1
        assert not written[0].exists(), "processed residue should be cleaned up"
        assert result.curves_run == 3072

    def test_multiple_batches_all_cleaned_up(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()
        stage2.execute.return_value = (None, [], 3072, 1.0, None)

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine, batch_sizes=(3072, 3072, 3072))

        # Chunked batches must get distinct paths, not collide on the
        # one-second timestamp.
        assert len(set(written)) == 3, "batch residue paths collided"
        assert result.preserved_residues == []
        assert not any(p.exists() for p in written)
        assert result.curves_run == 3 * 3072


class TestStage2Interrupted:
    """Ctrl+C mid stage 2 leaves unprocessed curves in the residue."""

    def test_partially_processed_residue_is_kept(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()

        def interrupted_execute(**kwargs):
            # Ctrl+C terminates the stage 2 workers partway through.
            wrapper.interrupted = True
            return (None, [], 400, 1.0, None)  # 400 of 3072 curves

        stage2.execute.side_effect = interrupted_execute

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine)

        assert [p for p, _ in result.preserved_residues] == written, (
            "discarded a residue holding 2672 curves that never ran stage 2"
        )
        assert written[0].exists()

    def test_fully_processed_residue_is_still_deleted(self, tmp_path):
        """A complete stage 2 pass spends the residue even if we then stop."""
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()

        def complete_then_stop(**kwargs):
            wrapper.interrupted = True
            return (None, [], 3072, 1.0, None)  # all curves done

        stage2.execute.side_effect = complete_then_stop

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine)

        assert result.preserved_residues == []
        assert not written[0].exists()


class TestFactorEndsTheComposite:
    """A factor makes queued residues spent, not recoverable."""

    def test_queued_residues_discarded_when_factor_found(self, tmp_path):
        """They describe a composite the server has just marked factored, so
        telling the operator to --submit stage 2 for them is wrong."""
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            # Second batch finds a factor in stage 1.
            if len(written) == 2:
                return {"success": True, "factors": ["104729"],
                        "sigmas": ["3:1000"], "raw_output": ""}
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()
        stage2.execute.return_value = (None, [], 3072, 1.0, None)

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine, batch_sizes=(3072, 3072, 3072))

        assert result.factors == ["104729"]
        assert result.preserved_residues == [], (
            "kept residues for an already-factored composite"
        )
        assert not any(p.exists() for p in written)


    def test_queued_residues_discarded_when_stage2_finds_factor(self, tmp_path):
        """The batch that finds the factor is not the only one in flight.

        Stage 1 keeps producing while stage 2 runs, so when stage 2 of the
        first batch hits a factor there are later batches sitting in the
        queue. Those describe the now-factored composite too.
        """
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []
        third_batch_started = threading.Event()

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text(RESIDUE_BODY)
            written.append(Path(residue_file))
            if len(written) == 3:
                # Batch 2 is already queued by the time this fires.
                third_batch_started.set()
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1

        stage2 = Mock()

        def find_factor(**kwargs):
            # Hold batch 1 until later batches have queued up behind it.
            third_batch_started.wait(timeout=10)
            return ("104729", ["104729"], 3072, 1.0, "3:1000")

        stage2.execute.side_effect = find_factor

        with patch("lib.stage2_executor.Stage2Executor", return_value=stage2):
            result = run(engine, batch_sizes=(3072, 3072, 3072))

        assert result.factors == ["104729"]
        assert result.preserved_residues == [], (
            "kept queued residues for an already-factored composite"
        )
        assert not any(p.exists() for p in written), "residue files leaked"


class TestUnusableResidueNotAdvertised:
    """A truncated -save file is not recoverable work."""

    def test_empty_residue_is_discarded_not_preserved(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        # A batch killed mid-write parses as zero curves.
        wrapper._parse_residue_file = Mock(
            return_value={"b1": 110000000, "curve_count": 0, "composite": "12345"}
        )
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        written: list = []

        def fake_stage1(composite, b1, curves, residue_file, **kwargs):
            Path(residue_file).write_text("")  # zero bytes
            written.append(Path(residue_file))
            wrapper.interrupted = True
            return {"success": True, "factors": [], "sigmas": [], "raw_output": ""}

        wrapper._run_stage1_primitive = fake_stage1
        result = run(engine)

        # Whichever path reaches it - GPU preserve branch or CPU drain - an
        # unusable residue must never be advertised for --stage2-only.
        assert result.preserved_residues == [], (
            "advertised an unusable residue for --stage2-only"
        )
        assert not written[0].exists()


class TestProducerAlwaysSignalsConsumer:
    """A dead producer must not strand the consumer (and the join) forever."""

    def test_producer_exception_does_not_hang(self, tmp_path):
        wrapper = make_wrapper(tmp_path)
        engine = CompositeExecutionEngine(wrapper)  # type: ignore[arg-type]

        class ExplodingProducer(StubBatchProducer):
            def next_batch(self):
                raise RuntimeError("t-level binary missing")

        wrapper._run_stage1_primitive = Mock()

        finished = threading.Event()
        box: dict = {}

        def go():
            try:
                box["result"] = engine.run_pipelined(
                    batch_producer=ExplodingProducer([3072]),  # type: ignore[arg-type]
                    composite="12345", stage2_workers=4,
                    no_submit=True, start_t_level=52.8,
                )
            finally:
                finished.set()

        threading.Thread(target=go, daemon=True).start()
        assert finished.wait(timeout=30), (
            "run_pipelined hung after the producer raised"
        )
        assert box["result"].curves_run == 0


class TestResiduePathSuffix:
    """Batch paths must not collide within the same one-second timestamp."""

    def test_suffix_disambiguates_same_second_batches(self, tmp_path):
        engine = CompositeExecutionEngine(make_wrapper(tmp_path))  # type: ignore[arg-type]
        paths = {
            engine._create_residue_path_for_two_stage("12345", None, suffix=f"_b{i}")
            for i in range(5)
        }
        assert len(paths) == 5, "batches produced in the same second collided"

    def test_suffix_defaults_to_empty(self, tmp_path):
        engine = CompositeExecutionEngine(make_wrapper(tmp_path))  # type: ignore[arg-type]
        path = engine._create_residue_path_for_two_stage("12345", None)
        assert path.name.startswith("stage1_")
        assert path.name.endswith(".txt")

    def test_explicit_save_path_wins(self, tmp_path):
        engine = CompositeExecutionEngine(make_wrapper(tmp_path))  # type: ignore[arg-type]
        explicit = str(tmp_path / "chosen.txt")
        assert engine._create_residue_path_for_two_stage(
            "12345", explicit, suffix="_b3"
        ) == Path(explicit)
