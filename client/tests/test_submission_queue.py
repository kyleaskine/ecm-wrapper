"""Tests for the SubmissionQueue persistent retry mechanism."""
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import pytest

from lib.submission_queue import SubmissionQueue


@pytest.fixture
def queue_dir():
    """Create a temporary directory for queue testing."""
    tmpdir = tempfile.mkdtemp(prefix="ecm_queue_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def queue(queue_dir):
    """Create a SubmissionQueue with a temp directory."""
    return SubmissionQueue(queue_dir=queue_dir)


@pytest.fixture
def mock_api_client():
    """Create a mock APIClient."""
    client = Mock()
    client.submit_result.return_value = {"status": "ok", "attempt_id": 1}
    client.complete_work.return_value = True
    client.complete_residue.return_value = {"new_t_level": 35.0}
    client.upload_residue.return_value = {"residue_id": 42, "curve_count": 100}
    return client


class TestEnqueueResult:
    def test_enqueue_creates_file(self, queue):
        payload = {"composite": "123", "method": "ecm"}
        filepath = queue.enqueue_result(payload)
        assert filepath is not None
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["type"] == "result"
        assert data["payload"] == payload
        assert data["attempts"] == 0
        assert "completion_chain" not in data

    def test_enqueue_with_context(self, queue):
        payload = {"composite": "123456789" * 10}
        context = {"composite": "123456789" * 10, "b1": 50000}
        filepath = queue.enqueue_result(payload, results_context=context)
        data = json.loads(filepath.read_text())
        assert "composite_preview" in data
        assert len(data["composite_preview"]) <= 50

    def test_enqueue_with_completion_chain(self, queue):
        payload = {"composite": "123", "method": "ecm"}
        chain = {"residue_id": 32225, "client_id": "worker-7"}
        filepath = queue.enqueue_result(payload, completion_chain=chain)
        data = json.loads(filepath.read_text())
        assert data["completion_chain"] == chain


class TestEnqueueResidueUpload:
    def test_enqueue_preserves_file(self, queue, queue_dir):
        # Create a fake residue file
        residue = Path(queue_dir) / "original_residue.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:12345\n")

        filepath = queue.enqueue_residue_upload(
            residue_file=residue,
            client_id="test-client",
            stage1_attempt_id=42
        )
        assert filepath is not None

        data = json.loads(filepath.read_text())
        assert data["type"] == "residue_upload"
        assert data["payload"]["client_id"] == "test-client"
        assert data["payload"]["stage1_attempt_id"] == 42

        # Verify the preserved residue file exists
        preserved_path = Path(data["residue_file"])
        assert preserved_path.exists()
        assert preserved_path.read_text() == "METHOD=ECM; SIGMA=3:12345\n"

    def test_enqueue_missing_file(self, queue):
        filepath = queue.enqueue_residue_upload(
            residue_file=Path("/nonexistent/residue.txt"),
            client_id="test"
        )
        assert filepath is None


class TestEnqueueCompletions:
    def test_enqueue_work_completion(self, queue):
        filepath = queue.enqueue_work_completion("work-123", "test-client")
        assert filepath is not None
        data = json.loads(filepath.read_text())
        assert data["type"] == "work_complete"
        assert data["payload"]["work_id"] == "work-123"
        assert data["payload"]["client_id"] == "test-client"

    def test_enqueue_residue_completion(self, queue):
        filepath = queue.enqueue_residue_completion(
            residue_id=42, client_id="test-client", stage2_attempt_id=99
        )
        assert filepath is not None
        data = json.loads(filepath.read_text())
        assert data["type"] == "residue_complete"
        assert data["payload"]["residue_id"] == 42
        assert data["payload"]["stage2_attempt_id"] == 99


class TestCount:
    def test_empty_queue(self, queue):
        assert queue.count() == 0

    def test_count_after_enqueue(self, queue):
        queue.enqueue_result({"composite": "123"})
        queue.enqueue_work_completion("w1", "c1")
        assert queue.count() == 2


class TestDrain:
    def test_drain_empty_queue(self, queue, mock_api_client):
        success, fail = queue.drain(mock_api_client)
        assert success == 0
        assert fail == 0

    def test_drain_result_success(self, queue, mock_api_client):
        payload = {"composite": "123", "method": "ecm"}
        filepath = queue.enqueue_result(payload)
        assert queue.count() == 1

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        assert queue.count() == 0

        # Verify the API was called with the right payload
        mock_api_client.submit_result.assert_called_once_with(
            payload=payload, save_on_failure=False
        )

    def test_drain_result_failure_stays_in_queue(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = None  # Simulate failure

        queue.enqueue_result({"composite": "123"})
        success, fail = queue.drain(mock_api_client)
        assert success == 0
        assert fail == 1
        assert queue.count() == 1  # Item stays in queue

    def test_drain_work_completion_success(self, queue, mock_api_client):
        queue.enqueue_work_completion("work-1", "client-1")
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        mock_api_client.complete_work.assert_called_once_with(
            work_id="work-1", client_id="client-1"
        )

    def test_drain_residue_completion_success(self, queue, mock_api_client):
        queue.enqueue_residue_completion(42, "client-1", 99)
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        mock_api_client.complete_residue.assert_called_once_with(
            client_id="client-1", residue_id=42, stage2_attempt_id=99
        )

    def test_drain_residue_upload_success(self, queue, mock_api_client, queue_dir):
        # Create a fake residue file
        residue = Path(queue_dir) / "test_residue.txt"
        residue.write_text("residue data\n")

        queue.enqueue_residue_upload(residue, "client-1", stage1_attempt_id=10)
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0

        # Verify upload was called
        mock_api_client.upload_residue.assert_called_once()
        call_kwargs = mock_api_client.upload_residue.call_args
        assert call_kwargs[1]["client_id"] == "client-1"
        assert call_kwargs[1]["stage1_attempt_id"] == 10

    def test_drain_residue_upload_cleans_preserved_file(self, queue, mock_api_client, queue_dir):
        residue = Path(queue_dir) / "test_residue.txt"
        residue.write_text("residue data\n")

        filepath = queue.enqueue_residue_upload(residue, "client-1")
        data = json.loads(filepath.read_text())
        preserved_path = Path(data["residue_file"])
        assert preserved_path.exists()

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        # Preserved file should be cleaned up after successful upload
        assert not preserved_path.exists()

    def test_drain_multiple_items_oldest_first(self, queue, mock_api_client):
        """Items should be processed oldest-first."""
        import time

        queue.enqueue_result({"composite": "first"})
        time.sleep(0.01)  # Ensure different mtime
        queue.enqueue_result({"composite": "second"})

        call_order = []
        def track_calls(**kwargs):
            payload = kwargs.get("payload", {})
            call_order.append(payload.get("composite"))
            return {"status": "ok"}

        mock_api_client.submit_result.side_effect = lambda **kwargs: track_calls(**kwargs)

        success, fail = queue.drain(mock_api_client)
        assert success == 2
        assert call_order == ["first", "second"]

    def test_drain_increments_attempt_count(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = None  # Always fail

        filepath = queue.enqueue_result({"composite": "123"})

        queue.drain(mock_api_client)
        data = json.loads(filepath.read_text())
        assert data["attempts"] == 1

        queue.drain(mock_api_client)
        data = json.loads(filepath.read_text())
        assert data["attempts"] == 2

    def test_drain_mixed_success_and_failure(self, queue, mock_api_client):
        queue.enqueue_result({"composite": "will_succeed"})
        queue.enqueue_work_completion("will_fail", "client")

        # First call succeeds, second fails
        mock_api_client.submit_result.return_value = {"status": "ok"}
        mock_api_client.complete_work.return_value = False

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 1
        assert queue.count() == 1  # One item remains


class TestPendingResultLookup:
    def test_empty_queue_returns_false(self, queue):
        assert queue.has_pending_result_for_residue(32225) is False

    def test_finds_pending_result_by_residue_id(self, queue):
        queue.enqueue_result(
            {"composite": "123"},
            completion_chain={"residue_id": 32225, "client_id": "c"},
        )
        assert queue.has_pending_result_for_residue(32225) is True
        assert queue.has_pending_result_for_residue(99999) is False

    def test_ignores_results_without_chain(self, queue):
        queue.enqueue_result({"composite": "123"})
        assert queue.has_pending_result_for_residue(32225) is False


class TestChainedResidueCompletion:
    def test_chains_complete_residue_on_drain_success(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 555}
        queue.enqueue_result(
            {"composite": "123"},
            completion_chain={"residue_id": 32225, "client_id": "worker-7"},
        )

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        mock_api_client.complete_residue.assert_called_once_with(
            client_id="worker-7", residue_id=32225, stage2_attempt_id=555,
        )
        assert queue.count() == 0  # Result removed, completion succeeded inline

    def test_falls_back_to_queued_completion_on_chain_failure(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 555}
        mock_api_client.complete_residue.return_value = None  # Network blip

        queue.enqueue_result(
            {"composite": "123"},
            completion_chain={"residue_id": 32225, "client_id": "worker-7"},
        )

        success, fail = queue.drain(mock_api_client)
        # Result item was submitted; chained completion failed and was queued.
        assert success == 1
        assert fail == 0
        assert queue.count() == 1  # The new residue_complete item

        # The new item is a residue_complete carrying the attempt_id
        files = list(queue.completions_dir.glob("residue_complete_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["type"] == "residue_complete"
        assert data["payload"]["residue_id"] == 32225
        assert data["payload"]["stage2_attempt_id"] == 555
        assert data["payload"]["client_id"] == "worker-7"

    def test_no_chain_means_no_complete_residue_call(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 555}
        queue.enqueue_result({"composite": "123"})  # No chain

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        mock_api_client.complete_residue.assert_not_called()


class TestAttachWorkCompletion:
    """The work loop holds an assignment when the queue already has its result."""

    def test_empty_queue_returns_false(self, queue):
        assert queue.attach_work_completion("work-123", "worker-7") is False

    def test_attaches_chain_to_matching_result(self, queue):
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        assert queue.attach_work_completion("work-123", "worker-7") is True

        data = json.loads(filepath.read_text())
        assert data["completion_chain"] == {
            "action": "work_complete",
            "work_id": "work-123",
            "client_id": "worker-7",
        }

    def test_ignores_other_work_ids(self, queue):
        queue.enqueue_result({"composite": "123", "work_id": "other-work"})
        assert queue.attach_work_completion("work-123", "worker-7") is False

    def test_attaches_to_newest_result_for_the_work(self, queue):
        # A 'p1' assignment submits pm1 and pp1 separately; completing on the
        # last one keeps the assignment alive until both results have landed.
        # Ordering comes from the microsecond-stamped filename, so no mtime
        # fixup is needed for this to be deterministic.
        first = queue.enqueue_result({"composite": "123", "work_id": "work-123", "method": "pm1"})
        second = queue.enqueue_result({"composite": "123", "work_id": "work-123", "method": "pp1"})
        assert first.name < second.name

        assert queue.attach_work_completion("work-123", "worker-7") is True

        assert "completion_chain" not in json.loads(first.read_text())
        assert json.loads(second.read_text())["completion_chain"]["action"] == "work_complete"

    def test_does_not_displace_an_existing_chain(self, queue, queue_dir):
        # A stage-1 result already carries a residue_upload chain; the residue is
        # the more valuable follow-up and that path abandons the work on purpose.
        residue = Path(queue_dir) / "residue.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:1\n")
        filepath = queue.enqueue_result(
            {"composite": "123", "work_id": "work-123"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "worker-7",
            },
        )

        assert queue.attach_work_completion("work-123", "worker-7") is False
        assert json.loads(filepath.read_text())["completion_chain"]["action"] == "residue_upload"

    def test_declines_when_any_result_of_the_work_has_a_chain(self, queue, queue_dir):
        # The chain-less sibling must not be used as a back door into holding an
        # assignment the residue-upload path means to release.
        residue = Path(queue_dir) / "residue.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:1\n")
        plain = queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.enqueue_result(
            {"composite": "123", "work_id": "work-123"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "worker-7",
            },
        )

        assert queue.attach_work_completion("work-123", "worker-7") is False
        assert "completion_chain" not in json.loads(plain.read_text())

    def test_corrupt_item_does_not_raise(self, queue):
        (queue.results_dir).mkdir(parents=True, exist_ok=True)
        (queue.results_dir / "result_broken.json").write_text("{ truncated")
        target = queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        assert queue.attach_work_completion("work-123", "worker-7") is True
        assert json.loads(target.read_text())["completion_chain"]["action"] == "work_complete"

    def test_attach_is_atomic(self, queue):
        # A failed rewrite must leave the original result intact, not a truncated
        # file that every future drain skips - it is the only copy of the work.
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        original = filepath.read_text()

        with patch("lib.submission_queue.os.replace", side_effect=OSError("disk full")):
            assert queue.attach_work_completion("work-123", "worker-7") is False

        assert filepath.read_text() == original
        assert list(queue.results_dir.glob("*.tmp")) == []


class TestChainedWorkCompletion:
    def test_completes_work_after_result_submits(self, queue, mock_api_client):
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        success, fail = queue.drain(mock_api_client)

        assert (success, fail) == (1, 0)
        mock_api_client.complete_work.assert_called_once_with(
            work_id="work-123", client_id="worker-7",
        )
        assert queue.count() == 0

    def test_queues_completion_when_chain_call_fails(self, queue, mock_api_client):
        mock_api_client.complete_work.return_value = False  # Network blip again
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        success, fail = queue.drain(mock_api_client)

        # Result landed; only the completion is left for a later drain.
        assert (success, fail) == (1, 0)
        files = list(queue.completions_dir.glob("work_complete_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["payload"] == {
            "work_id": "work-123", "client_id": "worker-7",
        }

    def test_expired_assignment_is_not_requeued(self, queue, mock_api_client):
        from lib.api_client import ResourceNotFoundError
        mock_api_client.complete_work.side_effect = ResourceNotFoundError("gone")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        success, fail = queue.drain(mock_api_client)

        # The result is what mattered; a 404 on the assignment is harmless.
        assert (success, fail) == (1, 0)
        assert queue.count() == 0

    def test_permanent_rejection_releases_the_held_assignment(self, queue, mock_api_client):
        # The chain is the only thing that would ever release a held assignment.
        # If the result is discarded, the work must be abandoned instead - the
        # result will never land, so completing it would be a lie.
        from lib.api_client import ResourceNotFoundError
        mock_api_client.submit_result.side_effect = ResourceNotFoundError("unknown composite")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        queue.drain(mock_api_client)

        files = list(queue.completions_dir.glob("work_abandon_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["payload"] == {
            "work_id": "work-123", "client_id": "worker-7",
        }

    def test_age_cap_releases_the_held_assignment(self, queue, mock_api_client):
        import datetime
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")
        item = json.loads(filepath.read_text())
        aged = datetime.datetime.now() - datetime.timedelta(days=8)
        item["created_at"] = aged.isoformat()
        filepath.write_text(json.dumps(item))

        queue.drain(mock_api_client)

        assert not filepath.exists()
        assert len(list(queue.completions_dir.glob("work_abandon_*.json"))) == 1

    def test_attempt_count_alone_does_not_discard_a_held_result(self, queue, mock_api_client):
        """
        Attempts are consumed by the 30-second no-work poll, not by elapsed
        time: a two-hour outage must not destroy a multi-hour result or release
        the assignment held for it.
        """
        mock_api_client.submit_result.return_value = None  # Server still down
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")
        item = json.loads(filepath.read_text())
        item["attempts"] = 500
        filepath.write_text(json.dumps(item))

        queue.drain(mock_api_client)

        assert filepath.exists()
        assert json.loads(filepath.read_text())["attempts"] == 501
        assert list(queue.completions_dir.glob("work_abandon_*.json")) == []

    def test_discarded_result_without_chain_releases_nothing(self, queue, mock_api_client):
        from lib.api_client import ResourceNotFoundError
        mock_api_client.submit_result.side_effect = ResourceNotFoundError("unknown composite")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})  # No chain

        queue.drain(mock_api_client)

        assert list(queue.completions_dir.glob("work_abandon_*.json")) == []

    def test_unexpected_error_does_not_resubmit_result(self, queue, mock_api_client):
        mock_api_client.complete_work.side_effect = RuntimeError("boom")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        success, fail = queue.drain(mock_api_client)

        # The result was accepted, so it must leave the queue - otherwise every
        # future drain would re-POST it as a duplicate attempt.
        assert (success, fail) == (1, 0)
        assert list(queue.results_dir.glob("result_*.json")) == []
        assert len(list(queue.completions_dir.glob("work_complete_*.json"))) == 1


class TestChainedResidueUpload:
    """Stage-1 residue-upload chain: a queued result carries the residue so a
    long GPU batch is not lost when the server is down at submit time."""

    def _make_residue(self, queue_dir) -> Path:
        residue = Path(queue_dir) / "stage1_batch.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:123; B1=110000000; N=99\n")
        return residue

    def test_enqueue_preserves_residue_and_rewrites_path(self, queue, queue_dir):
        residue = self._make_residue(queue_dir)
        chain = {
            "action": "residue_upload",
            "residue_file": str(residue),
            "client_id": "gpu-1",
            "expiry_days": 7,
        }
        filepath = queue.enqueue_result({"composite": "99"}, completion_chain=chain)

        data = json.loads(filepath.read_text())
        preserved = Path(data["completion_chain"]["residue_file"])
        # Path was rewritten into the queue's residues dir and the copy exists,
        # so deleting the original local residue afterward is safe.
        assert preserved != residue
        assert preserved.parent == queue.residues_dir
        assert preserved.exists()
        residue.unlink()
        assert preserved.exists()

    def test_drain_uploads_residue_with_returned_attempt_id(self, queue, queue_dir, mock_api_client):
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 777}
        residue = self._make_residue(queue_dir)
        queue.enqueue_result(
            {"composite": "99"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "gpu-1",
                "expiry_days": 7,
            },
        )

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        call_kwargs = mock_api_client.upload_residue.call_args[1]
        assert call_kwargs["stage1_attempt_id"] == 777
        assert call_kwargs["client_id"] == "gpu-1"
        # Preserved copy cleaned up after a successful upload.
        assert not list(queue.residues_dir.glob("residue_*"))
        assert queue.count() == 0

    def test_transient_upload_failure_requeues_standalone(self, queue, queue_dir, mock_api_client):
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 777}
        mock_api_client.upload_residue.return_value = None  # 5xx / network
        residue = self._make_residue(queue_dir)
        queue.enqueue_result(
            {"composite": "99"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "gpu-1",
                "expiry_days": 7,
            },
        )

        success, fail = queue.drain(mock_api_client)
        # Result submitted; chained upload failed and was re-queued standalone.
        assert success == 1
        assert queue.count() == 1
        files = list(queue.residues_dir.glob("residue_upload_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["type"] == "residue_upload"
        assert data["payload"]["stage1_attempt_id"] == 777

    def test_unexpected_upload_error_does_not_resubmit_result(self, queue, queue_dir, mock_api_client):
        # An accepted result must not be re-POSTed forever if the chained upload
        # throws an unexpected error. The exception must be swallowed, the result
        # file removed, and the preserved residue kept for recovery.
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 777}
        mock_api_client.upload_residue.side_effect = OSError("disk gone")
        residue = self._make_residue(queue_dir)
        queue.enqueue_result(
            {"composite": "99"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "gpu-1",
                "expiry_days": 7,
            },
        )

        queue.drain(mock_api_client)
        queue.drain(mock_api_client)  # second cycle would re-POST if not removed
        # Result submitted exactly once across both drains (no runaway re-submit).
        assert mock_api_client.submit_result.call_count == 1
        assert queue.count() == 0
        # Preserved residue kept (data not lost).
        assert list(queue.residues_dir.glob("residue_*"))

    def test_requeue_failure_keeps_preserved_residue(self, queue, queue_dir, mock_api_client):
        # If the transient re-queue itself fails, the preserved copy is the only
        # remaining copy and must not be deleted.
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 777}
        mock_api_client.upload_residue.return_value = None  # transient
        residue = self._make_residue(queue_dir)
        queue.enqueue_result(
            {"composite": "99"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "gpu-1",
                "expiry_days": 7,
            },
        )
        preserved_before = {p.name for p in queue.residues_dir.glob("residue_*")}

        with patch.object(queue, "enqueue_residue_upload", return_value=None):
            queue.drain(mock_api_client)

        # No new queue item created, but the preserved copy is still on disk.
        preserved_after = {p.name for p in queue.residues_dir.glob("residue_*")}
        assert preserved_before <= preserved_after
        assert preserved_before  # sanity: something was preserved

    def test_permanent_upload_rejection_is_dropped(self, queue, queue_dir, mock_api_client):
        from lib.api_client import PermanentUploadError
        mock_api_client.submit_result.return_value = {"status": "ok", "attempt_id": 777}
        mock_api_client.upload_residue.side_effect = PermanentUploadError("composite factored")
        residue = self._make_residue(queue_dir)
        queue.enqueue_result(
            {"composite": "99"},
            completion_chain={
                "action": "residue_upload",
                "residue_file": str(residue),
                "client_id": "gpu-1",
                "expiry_days": 7,
            },
        )

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert queue.count() == 0  # Nothing left to retry
        assert not list(queue.residues_dir.glob("residue_*"))


class TestDiscardedChainCleanup:
    """
    What a queued result's chain must do when the result is dropped instead of
    sent: release whatever was held for it, and take its residue copy with it.
    """

    def _residue_chain_result(self, queue, queue_dir):
        residue = Path(queue_dir) / "stage1_batch.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:123; B1=110000000; N=99\n")
        chain = {
            "action": "residue_upload",
            "residue_file": str(residue),
            "client_id": "gpu-1",
            "expiry_days": 7,
            "work_id": "work-123",
        }
        filepath = queue.enqueue_result({"composite": "99", "work_id": "work-123"},
                                        completion_chain=chain)
        preserved = Path(json.loads(filepath.read_text())["completion_chain"]["residue_file"])
        assert preserved.exists()
        return filepath, preserved

    def test_discarded_stage2_result_releases_the_residue_claim(self, queue, mock_api_client):
        """
        Stage 2 holds its claim while the result is queued. If the result is
        dropped, nothing else releases it - the residue would stay 'claimed'
        for 24h, and /ecm-work skips composites with claimed residues.
        """
        from lib.api_client import ResourceNotFoundError
        mock_api_client.submit_result.side_effect = ResourceNotFoundError("unknown composite")
        queue.enqueue_result(
            {"composite": "123"},
            completion_chain={"residue_id": 32225, "client_id": "worker-7"},
        )

        queue.drain(mock_api_client)

        files = list(queue.completions_dir.glob("residue_abandon_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["payload"] == {
            "residue_id": 32225, "client_id": "worker-7",
        }

    def test_discarded_stage1_result_drops_the_preserved_residue(self, queue, queue_dir, mock_api_client):
        """The preserved copy is multi-hundred-MB and nothing else refers to it."""
        from lib.api_client import ResourceNotFoundError
        mock_api_client.submit_result.side_effect = ResourceNotFoundError("unknown composite")
        filepath, preserved = self._residue_chain_result(queue, queue_dir)

        queue.drain(mock_api_client)

        assert not filepath.exists()
        assert not preserved.exists()

    def test_discarded_stage1_result_releases_the_assignment(self, queue, queue_dir, mock_api_client):
        from lib.api_client import ResourceNotFoundError
        mock_api_client.submit_result.side_effect = ResourceNotFoundError("unknown composite")
        self._residue_chain_result(queue, queue_dir)

        queue.drain(mock_api_client)

        files = list(queue.completions_dir.glob("work_abandon_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["payload"] == {
            "work_id": "work-123", "client_id": "gpu-1",
        }

    def test_duplicate_rejection_completes_instead_of_abandoning(self, queue, mock_api_client):
        """
        'Duplicate' means the server already has this result: the assignment is
        fully accounted for, so abandoning would push a recorded composite back
        into the pool for someone to redo.
        """
        mock_api_client.submit_result.side_effect = RuntimeError("Duplicate submission")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        queue.drain(mock_api_client)

        assert list(queue.completions_dir.glob("work_abandon_*.json")) == []
        files = list(queue.completions_dir.glob("work_complete_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["payload"] == {
            "work_id": "work-123", "client_id": "worker-7",
        }

    def test_missing_resource_rejection_still_abandons(self, queue, mock_api_client):
        mock_api_client.submit_result.side_effect = RuntimeError("composite not found")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        queue.attach_work_completion("work-123", "worker-7")

        queue.drain(mock_api_client)

        assert len(list(queue.completions_dir.glob("work_abandon_*.json"))) == 1
        assert list(queue.completions_dir.glob("work_complete_*.json")) == []


class TestStage1ChainCompletesWork:
    """
    Stage 1's assignment is held while its result is queued, and its chain is
    the residue upload - so that chain has to close the assignment out too.
    """

    def _enqueue(self, queue, queue_dir, work_id: Optional[str] = "work-123"):
        residue = Path(queue_dir) / "stage1_batch.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:123; B1=110000000; N=99\n")
        chain = {
            "action": "residue_upload",
            "residue_file": str(residue),
            "client_id": "gpu-1",
            "expiry_days": 7,
        }
        if work_id:
            chain["work_id"] = work_id
        return queue.enqueue_result({"composite": "99", "work_id": work_id},
                                    completion_chain=chain)

    def test_completes_work_after_residue_upload(self, queue, queue_dir, mock_api_client):
        self._enqueue(queue, queue_dir)

        success, fail = queue.drain(mock_api_client)

        assert (success, fail) == (1, 0)
        mock_api_client.upload_residue.assert_called_once()
        mock_api_client.complete_work.assert_called_once_with(
            work_id="work-123", client_id="gpu-1",
        )

    def test_manual_stage1_without_work_id_completes_nothing(self, queue, queue_dir, mock_api_client):
        self._enqueue(queue, queue_dir, work_id=None)

        queue.drain(mock_api_client)

        mock_api_client.complete_work.assert_not_called()

    def test_has_pending_result_for_work(self, queue):
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})
        assert queue.has_pending_result_for_work("work-123") is True
        assert queue.has_pending_result_for_work("work-999") is False


class TestQueueFileRobustness:
    """Malformed or half-written files must never escape into the work loop."""

    def test_non_object_json_is_skipped(self, queue, mock_api_client):
        queue._ensure_dirs()
        (queue.results_dir / "result_bad.json").write_text("[]")
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        # Both the scan and the drain have to tolerate it.
        assert queue.attach_work_completion("work-123", "worker-7") is True
        assert queue.has_pending_result_for_work("work-123") is True
        success, fail = queue.drain(mock_api_client)
        assert success == 1

    def test_invalid_utf8_is_skipped(self, queue, mock_api_client):
        queue._ensure_dirs()
        (queue.results_dir / "result_bad.json").write_bytes(b'{"type": "result", "p": "\xff\xfe')
        queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        assert queue.attach_work_completion("work-123", "worker-7") is True
        queue.drain(mock_api_client)

    def test_rewrite_of_unserializable_item_fails_cleanly(self, queue):
        filepath = queue.enqueue_result({"composite": "123"})
        original = filepath.read_text()

        assert queue._rewrite_item(filepath, {"bad": object()}) is False

        assert filepath.read_text() == original  # Untouched
        assert list(queue.results_dir.glob("*.tmp")) == []

    def test_stale_tmp_sidecars_are_cleaned_up(self, queue, mock_api_client):
        import os as _os
        import time as _time
        queue._ensure_dirs()
        stale = queue.results_dir / "result_20260101_000000_000000.json.tmp"
        stale.write_text("{}")
        old = _time.time() - 7200
        _os.utime(stale, (old, old))
        fresh = queue.results_dir / "result_20260101_000001_000000.json.tmp"
        fresh.write_text("{}")
        queue.enqueue_result({"composite": "123"})

        queue.drain(mock_api_client)

        assert not stale.exists()
        assert fresh.exists()  # A live rewrite in another process may own it

    def test_failed_drain_persists_attempts_atomically(self, queue, mock_api_client):
        """
        The attempt bump runs on every drain of a queued result - up to
        hundreds of times during an outage. A truncating in-place write that
        died midway would leave the only copy of a multi-hour result as
        unparseable JSON, so it goes through the same tmp+replace path the
        chain attach uses.
        """
        mock_api_client.submit_result.return_value = None
        filepath = queue.enqueue_result({"composite": "123"})

        with patch.object(queue, "_rewrite_item", wraps=queue._rewrite_item) as rewrite:
            queue.drain(mock_api_client)

        rewrite.assert_called_once()
        assert rewrite.call_args[0][0] == filepath
        assert json.loads(filepath.read_text())["attempts"] == 1
        assert list(queue.results_dir.glob("*.tmp")) == []


class TestConcurrentDrainSafety:
    """
    Two clients (the documented stage1/stage2 split) share data/queue with no
    other synchronization.
    """

    def test_attach_declines_when_another_process_holds_the_item(self, queue):
        import fcntl
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        fd = os.open(filepath, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # A concurrent drain owns this file and may be about to unlink it.
            assert queue.attach_work_completion("work-123", "worker-7") is False
        finally:
            os.close(fd)

        assert "completion_chain" not in json.loads(filepath.read_text())

    def test_attach_does_not_resurrect_a_submitted_result(self, queue):
        """An item unlinked by another process's drain must not come back."""
        filepath = queue.enqueue_result({"composite": "123", "work_id": "work-123"})

        real_iter = queue._iter_result_items

        def unlink_then_iter(*args, **kwargs):
            items = list(real_iter(*args, **kwargs))
            filepath.unlink()  # The other process's drain accepted and removed it
            return iter(items)

        with patch.object(queue, "_iter_result_items", side_effect=unlink_then_iter):
            assert queue.attach_work_completion("work-123", "worker-7") is False

        assert not filepath.exists()
