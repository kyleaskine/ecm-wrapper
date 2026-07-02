"""Tests for the SubmissionQueue persistent retry mechanism."""
import json
import tempfile
import shutil
from pathlib import Path
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
