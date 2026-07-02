"""
Persistent submission queue for automatic retry of failed API operations.

When the server is down, submissions (results, residue uploads, completions)
are saved to disk and retried automatically on the next server interaction.

Queue directory structure:
    data/queue/results/     - Failed result submissions (JSON payloads)
    data/queue/residues/    - Preserved residue files + metadata for failed uploads
    data/queue/completions/ - Failed work/residue completion calls
"""
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .api_client import ResourceNotFoundError, PermanentUploadError

if TYPE_CHECKING:
    from .api_client import APIClient

logger = logging.getLogger(__name__)


class SubmissionQueue:
    """
    Persistent queue for retrying failed API operations.

    Items are stored as JSON files in subdirectories under the queue root.
    Each item contains the operation type, payload, and metadata needed to
    replay the operation.

    Usage:
        queue = SubmissionQueue("data/queue")

        # Enqueue a failed result submission
        queue.enqueue_result(payload, results_context)

        # Enqueue a failed residue upload
        queue.enqueue_residue_upload(residue_file, client_id, stage1_attempt_id)

        # Enqueue a failed work completion
        queue.enqueue_work_completion(work_id, client_id)

        # Drain all queued items (retry them)
        success, fail = queue.drain(api_client)
    """

    def __init__(self, queue_dir: str = "data/queue"):
        self.queue_dir = Path(queue_dir)
        self.results_dir = self.queue_dir / "results"
        self.residues_dir = self.queue_dir / "residues"
        self.completions_dir = self.queue_dir / "completions"
        self.logger = logging.getLogger(f"{__name__}.SubmissionQueue")

    def _ensure_dirs(self) -> None:
        """Create queue directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.residues_dir.mkdir(parents=True, exist_ok=True)
        self.completions_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str) -> str:
        """Generate a unique timestamped filename."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}_{ts}.json"

    # ==================== Enqueue Methods ====================

    def enqueue_result(
        self,
        payload: Dict[str, Any],
        results_context: Optional[Dict[str, Any]] = None,
        completion_chain: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Enqueue a failed result submission for later retry.

        Args:
            payload: The API submission payload (ready to POST)
            results_context: Optional original results dict for debugging
            completion_chain: Optional follow-up call to make after a successful
                resubmit. Two shapes are supported:
                - Stage 2: {residue_id, client_id} -> complete_residue with the
                  attempt_id returned by the resubmitted result.
                - Stage 1: {action: "residue_upload", residue_file, client_id,
                  expiry_days} -> upload the preserved residue with the returned
                  attempt_id. The residue file is copied into the queue here so a
                  long GPU batch's output survives even if the caller cleans up
                  the original.

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()

        # A stage-1 residue-upload chain references a residue file that the
        # caller is about to delete. Preserve it now (at failure time) so the
        # chained upload can run once the result finally submits.
        if completion_chain and completion_chain.get("action") == "residue_upload":
            completion_chain = self._preserve_chain_residue(completion_chain)

        item: Dict[str, Any] = {
            "type": "result",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": payload,
        }
        if results_context:
            # Store composite for logging, but don't duplicate the full context
            item["composite_preview"] = str(results_context.get("composite", ""))[:50]
        if completion_chain:
            item["completion_chain"] = completion_chain

        filename = self._generate_filename("result")
        filepath = self.results_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued failed result submission: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue result submission: {e}")
            return None

    def enqueue_residue_upload(
        self,
        residue_file: Path,
        client_id: str,
        stage1_attempt_id: Optional[int] = None,
        expiry_days: int = 7
    ) -> Optional[Path]:
        """
        Preserve a residue file and enqueue its upload for later retry.

        The residue file is COPIED to the queue directory to prevent loss
        when the original is cleaned up.

        Args:
            residue_file: Path to the original residue file
            client_id: Client identifier for the upload
            stage1_attempt_id: Stage 1 attempt ID to link
            expiry_days: Expiry days for the upload request

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()

        if not residue_file.exists():
            self.logger.error(f"Cannot queue residue upload: file not found: {residue_file}")
            return None

        # Copy residue file to queue directory to preserve it
        preserved_name = f"residue_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{residue_file.name}"
        preserved_path = self.residues_dir / preserved_name

        try:
            shutil.copy2(residue_file, preserved_path)
            self.logger.info(f"Preserved residue file: {preserved_path}")
        except Exception as e:
            self.logger.error(f"Failed to preserve residue file: {e}")
            return None

        # Create queue metadata
        item = {
            "type": "residue_upload",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": {
                "client_id": client_id,
                "stage1_attempt_id": stage1_attempt_id,
                "expiry_days": expiry_days,
            },
            "residue_file": str(preserved_path),
        }

        filename = self._generate_filename("residue_upload")
        filepath = self.residues_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued residue upload: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue residue upload: {e}")
            # Clean up preserved file on metadata save failure
            if preserved_path.exists():
                preserved_path.unlink()
            return None

    def _preserve_chain_residue(
        self,
        completion_chain: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Copy a residue-upload chain's residue file into the queue directory.

        Returns a new chain dict whose ``residue_file`` points at the preserved
        copy. If the source file is missing or the copy fails, returns None so
        the result is still queued (just without the residue-upload follow-up).
        """
        residue_path = completion_chain.get("residue_file")
        src = Path(residue_path) if residue_path else None
        if not src or not src.exists():
            self.logger.error(
                f"Cannot preserve residue for chained upload: file not found: {residue_path}"
            )
            return None

        preserved_name = (
            f"residue_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{src.name}"
        )
        preserved_path = self.residues_dir / preserved_name
        try:
            shutil.copy2(src, preserved_path)
            self.logger.info(f"Preserved residue for chained upload: {preserved_path}")
        except Exception as e:
            self.logger.error(f"Failed to preserve residue for chained upload: {e}")
            return None

        new_chain = dict(completion_chain)
        new_chain["residue_file"] = str(preserved_path)
        return new_chain

    def enqueue_work_completion(
        self,
        work_id: str,
        client_id: str
    ) -> Optional[Path]:
        """
        Enqueue a failed work completion call for later retry.

        Args:
            work_id: Work assignment ID to complete
            client_id: Client ID completing the work

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()
        item = {
            "type": "work_complete",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": {
                "work_id": work_id,
                "client_id": client_id,
            },
        }

        filename = self._generate_filename("work_complete")
        filepath = self.completions_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued work completion: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue work completion: {e}")
            return None

    def enqueue_work_abandonment(
        self,
        work_id: str,
        client_id: str
    ) -> Optional[Path]:
        """
        Enqueue a failed work abandonment call for later retry.

        Used when cleanup_on_failure can't reach the server to release
        a work assignment back to the pool.

        Args:
            work_id: Work assignment ID to abandon
            client_id: Client identifier

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()
        item = {
            "type": "work_abandon",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": {
                "work_id": work_id,
                "client_id": client_id,
            },
        }

        filename = self._generate_filename("work_abandon")
        filepath = self.completions_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued work abandonment: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue work abandonment: {e}")
            return None

    def enqueue_residue_abandonment(
        self,
        residue_id: int,
        client_id: str
    ) -> Optional[Path]:
        """
        Enqueue a failed residue abandonment call for later retry.

        Used when cleanup_on_failure can't reach the server to release
        a claimed residue back to the available pool.

        Args:
            residue_id: Residue ID to abandon
            client_id: Client identifier

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()
        item = {
            "type": "residue_abandon",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": {
                "residue_id": residue_id,
                "client_id": client_id,
            },
        }

        filename = self._generate_filename("residue_abandon")
        filepath = self.completions_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued residue abandonment: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue residue abandonment: {e}")
            return None

    def enqueue_residue_completion(
        self,
        residue_id: int,
        client_id: str,
        stage2_attempt_id: int
    ) -> Optional[Path]:
        """
        Enqueue a failed residue completion call for later retry.

        Args:
            residue_id: Residue ID to complete
            client_id: Client identifier
            stage2_attempt_id: Stage 2 attempt ID

        Returns:
            Path to the queued item file, or None on error
        """
        self._ensure_dirs()
        item = {
            "type": "residue_complete",
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "payload": {
                "residue_id": residue_id,
                "client_id": client_id,
                "stage2_attempt_id": stage2_attempt_id,
            },
        }

        filename = self._generate_filename("residue_complete")
        filepath = self.completions_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(item, f, indent=2)
            self.logger.info(f"Queued residue completion: {filepath.name}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to queue residue completion: {e}")
            return None

    # ==================== Drain (Retry) Methods ====================

    def count(self) -> int:
        """Count total pending items across all queue directories."""
        total = 0
        for subdir in [self.results_dir, self.residues_dir, self.completions_dir]:
            if subdir.exists():
                total += sum(1 for f in subdir.glob("*.json")
                             if not f.name.startswith("residue_") or f.suffix == ".json")
        return total

    def has_pending_result_for_residue(self, residue_id: int) -> bool:
        """
        True if a queued result item carries a completion_chain for this residue_id.

        Used by stage 2 to decide whether to abandon a residue claim after a
        failed submission: if the queue already holds the computed result, the
        claim should be kept so the queue can finalize it on retry instead of
        the work being re-executed by another client.
        """
        if not self.results_dir.exists():
            return False
        for f in self.results_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            chain = data.get("completion_chain") or {}
            if chain.get("residue_id") == residue_id:
                return True
        return False

    def _get_queue_files(self) -> List[Path]:
        """Get all queue item files sorted oldest-first."""
        files: List[Path] = []
        for subdir in [self.results_dir, self.residues_dir, self.completions_dir]:
            if subdir.exists():
                for f in subdir.glob("*.json"):
                    # Skip residue data files (only process metadata JSONs)
                    try:
                        with open(f, 'r') as fh:
                            data = json.load(fh)
                        if "type" in data:  # Only process queue item files
                            files.append(f)
                    except (json.JSONDecodeError, OSError):
                        continue
        # Sort by creation time (oldest first)
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    def drain(self, api_client: 'APIClient') -> tuple:
        """
        Attempt to submit all queued items.

        Processes items oldest-first. Successfully submitted items are removed.
        Failed items remain in the queue for the next drain cycle.

        Args:
            api_client: APIClient instance to use for submissions

        Returns:
            Tuple of (success_count, fail_count)
        """
        files = self._get_queue_files()
        if not files:
            return (0, 0)

        self.logger.info(f"Draining submission queue: {len(files)} item(s) pending")
        success_count = 0
        fail_count = 0

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    item = json.load(f)

                item_type = item.get("type", "unknown")
                item["attempts"] = item.get("attempts", 0) + 1

                max_attempts = 200
                if item["attempts"] > max_attempts:
                    self.logger.warning(
                        f"Discarding {item_type} after {item['attempts']} failed attempts: {filepath.name}"
                    )
                    filepath.unlink(missing_ok=True)
                    if item_type == "residue_upload":
                        residue_path = item.get("residue_file")
                        if residue_path:
                            Path(residue_path).unlink(missing_ok=True)
                    fail_count += 1
                    continue

                ok = self._retry_item(api_client, item)

                if ok:
                    success_count += 1
                    self.logger.info(f"Queue drain: {item_type} succeeded ({filepath.name})")
                    # Remove the queue file
                    filepath.unlink(missing_ok=True)
                    # If residue upload, also remove the preserved residue file
                    if item_type == "residue_upload":
                        residue_path = item.get("residue_file")
                        if residue_path:
                            Path(residue_path).unlink(missing_ok=True)
                else:
                    fail_count += 1
                    self.logger.warning(
                        f"Queue drain: {item_type} failed (attempt {item['attempts']}, {filepath.name})"
                    )
                    # Update attempt count in file
                    with open(filepath, 'w') as f:
                        json.dump(item, f, indent=2)

            except Exception as e:
                fail_count += 1
                self.logger.error(f"Queue drain error for {filepath.name}: {e}")

        if success_count > 0 or fail_count > 0:
            self.logger.info(f"Queue drain complete: {success_count} succeeded, {fail_count} failed")
            if success_count > 0:
                print(f"Retried {success_count} queued submission(s) successfully")
            if fail_count > 0:
                print(f"{fail_count} queued submission(s) still pending (server may be down)")

        return (success_count, fail_count)

    def _run_completion_chain(
        self,
        api_client: 'APIClient',
        item: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        """
        Dispatch a queued result's completion chain after a successful resubmit.

        - ``action == "residue_upload"`` (stage 1): upload the preserved residue,
          linked to the attempt_id the resubmitted result returned.
        - otherwise (stage 2): complete the residue with that attempt_id.
        """
        chain = item.get("completion_chain") or {}
        if not chain:
            return
        if chain.get("action") == "residue_upload":
            self._chain_residue_upload(api_client, item, response)
        else:
            self._chain_residue_complete(api_client, item, response)

    def _chain_residue_upload(
        self,
        api_client: 'APIClient',
        item: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        """
        After a queued stage-1 result re-submits successfully, upload the
        preserved residue file linked to the attempt_id from the response.

        The residue upload needs the stage-1 attempt_id, which only exists once
        the result is accepted server-side - hence it is chained here rather than
        queued independently. On a transient upload failure the residue is handed
        off to a standalone ``residue_upload`` queue item (now carrying the
        attempt_id) for later retry; a 4xx rejection drops it.
        """
        chain = item.get("completion_chain") or {}
        residue_path = chain.get("residue_file")
        client_id = chain.get("client_id")
        expiry_days = chain.get("expiry_days", 7)
        preserved = Path(residue_path) if residue_path else None

        if not preserved or not preserved.exists():
            self.logger.warning(
                f"Chained residue upload skipped: preserved file missing ({residue_path})"
            )
            return

        if not client_id:
            self.logger.warning("Chained residue upload skipped: missing client_id")
            preserved.unlink(missing_ok=True)
            return

        attempt_id = response.get("attempt_id")
        if not attempt_id:
            self.logger.warning(
                "Queued stage-1 result returned no attempt_id; cannot link residue "
                "upload. Dropping preserved residue."
            )
            preserved.unlink(missing_ok=True)
            return

        # This runs as a best-effort follow-up *after* the result already
        # submitted successfully. Nothing here may propagate: an escaping
        # exception would leave the accepted result in the queue, and every
        # future drain would re-POST it (duplicate stage-1 attempts) since
        # _retry_item never gets to return True. Mirror the stage-2 sibling and
        # swallow everything. The preserved copy is only removed once its data
        # is safely elsewhere (uploaded, re-queued, or permanently rejected).
        try:
            result = api_client.upload_residue(
                client_id=client_id,
                residue_file_path=str(preserved),
                stage1_attempt_id=attempt_id,
                expiry_days=expiry_days,
            )
            if result is not None:
                self.logger.info(
                    f"Chained residue upload succeeded (stage1 attempt {attempt_id})"
                )
                preserved.unlink(missing_ok=True)
                return

            # Transient failure: re-queue as a standalone upload now that we know
            # the attempt_id. enqueue_residue_upload makes its own copy, so only
            # drop this preserved copy once that copy exists.
            self.logger.warning(
                "Chained residue upload failed transiently; re-queuing for retry"
            )
            if self.enqueue_residue_upload(
                residue_file=preserved,
                client_id=client_id,
                stage1_attempt_id=attempt_id,
                expiry_days=expiry_days,
            ):
                preserved.unlink(missing_ok=True)
            else:
                self.logger.error(
                    "Failed to re-queue chained residue upload; keeping preserved "
                    f"copy for manual recovery: {preserved}"
                )
        except PermanentUploadError as e:
            self.logger.warning(
                f"Chained residue upload permanently rejected, dropping: {e}"
            )
            preserved.unlink(missing_ok=True)
        except Exception as e:
            # Unexpected error (I/O, malformed response, ...). Keep the preserved
            # copy so the GPU batch isn't lost, and do NOT re-raise - the result
            # is already accepted server-side.
            self.logger.error(
                f"Chained residue upload errored unexpectedly, keeping preserved "
                f"copy for manual recovery ({preserved}): {e}"
            )

    def _chain_residue_complete(
        self,
        api_client: 'APIClient',
        item: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        """
        After a queued stage-2 result re-submits successfully, finalize the
        residue with the attempt_id from the response.

        If the chained complete_residue call fails (network blip again, etc.),
        enqueue a residue_complete item so a later drain can retry it. A 404
        from the server (residue expired) is treated as already-final and
        silently dropped.
        """
        chain = item.get("completion_chain") or {}
        residue_id = chain.get("residue_id")
        client_id = chain.get("client_id")
        if residue_id is None or client_id is None:
            return

        attempt_id = response.get("attempt_id")
        if not attempt_id:
            self.logger.warning(
                f"Queued result for residue {residue_id} returned no attempt_id; "
                "cannot chain complete_residue"
            )
            return

        if response.get("residue_completed"):
            self.logger.info(
                f"Residue {residue_id} completed server-side with the queued "
                "submission; chained completion not needed"
            )
            return

        try:
            complete_result = api_client.complete_residue(
                client_id=client_id,
                residue_id=residue_id,
                stage2_attempt_id=attempt_id,
            )
            if complete_result is None:
                self.enqueue_residue_completion(
                    residue_id=residue_id,
                    client_id=client_id,
                    stage2_attempt_id=attempt_id,
                )
        except ResourceNotFoundError:
            self.logger.warning(
                f"Residue {residue_id} already expired/completed on server; "
                "chained completion skipped"
            )
        except Exception as e:
            self.logger.warning(
                f"Chained complete_residue failed for residue {residue_id}: {e}; "
                "queuing for later retry"
            )
            self.enqueue_residue_completion(
                residue_id=residue_id,
                client_id=client_id,
                stage2_attempt_id=attempt_id,
            )

    def _retry_item(self, api_client: 'APIClient', item: Dict[str, Any]) -> bool:
        """
        Retry a single queued item.

        Args:
            api_client: APIClient for making API calls
            item: Queue item dict with type and payload

        Returns:
            True if successful (or permanently failed and should be discarded),
            False if transient failure (should retry later)
        """
        item_type = item.get("type")
        payload = item.get("payload", {})

        try:
            if item_type == "result":
                response = api_client.submit_result(
                    payload=payload,
                    save_on_failure=False  # Don't re-save on failure (already in queue)
                )
                if response is None:
                    return False
                self._run_completion_chain(api_client, item, response)
                return True

            elif item_type == "residue_upload":
                residue_path = item.get("residue_file")
                if not residue_path or not Path(residue_path).exists():
                    self.logger.error(f"Residue file missing for queued upload: {residue_path}")
                    return False
                result = api_client.upload_residue(
                    client_id=payload["client_id"],
                    residue_file_path=residue_path,
                    stage1_attempt_id=payload.get("stage1_attempt_id"),
                    expiry_days=payload.get("expiry_days", 7)
                )
                return result is not None

            elif item_type == "work_complete":
                return api_client.complete_work(
                    work_id=payload["work_id"],
                    client_id=payload["client_id"]
                )

            elif item_type == "work_abandon":
                return api_client.abandon_work(
                    work_id=payload["work_id"],
                    client_id=payload["client_id"]
                )

            elif item_type == "residue_complete":
                result = api_client.complete_residue(
                    client_id=payload["client_id"],
                    residue_id=payload["residue_id"],
                    stage2_attempt_id=payload["stage2_attempt_id"]
                )
                return result is not None

            elif item_type == "residue_abandon":
                return api_client.abandon_residue(
                    client_id=payload["client_id"],
                    residue_id=payload["residue_id"]
                )

            else:
                self.logger.error(f"Unknown queue item type: {item_type}")
                return False

        except ResourceNotFoundError:
            # 404 = resource permanently gone (expired, already completed, etc.)
            self.logger.warning(
                f"Discarding {item_type} from queue: resource no longer exists on server "
                f"(likely expired or already completed)"
            )
            return True  # Treat as success to remove from queue

        except PermanentUploadError as e:
            # 4xx residue upload rejection: composite factored, stage-1 attempt
            # gone, invalid file, etc. Retrying can never succeed, so discard.
            self.logger.warning(f"Discarding {item_type} from queue: {e}")
            return True  # Treat as success to remove from queue

        except Exception as e:
            error_str = str(e)
            # Detect permanent failures that will never succeed on retry
            if any(phrase in error_str for phrase in [
                "Duplicate", "already exists", "checksum matches",
                "not claimed by client", "not found"
            ]):
                self.logger.warning(
                    f"Discarding {item_type} from queue: permanent error: {error_str}"
                )
                return True  # Remove from queue
            raise  # Re-raise for transient errors
