"""
Persistent submission queue for automatic retry of failed API operations.

When the server is down, submissions (results, residue uploads, completions)
are saved to disk and retried automatically on the next server interaction.

Queue directory structure:
    data/queue/results/     - Failed result submissions (JSON payloads)
    data/queue/residues/    - Preserved residue files + metadata for failed uploads
    data/queue/completions/ - Failed work/residue completion calls
"""
import contextlib
import datetime
import errno
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from .api_client import ResourceNotFoundError, PermanentUploadError

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platform
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .api_client import APIClient

logger = logging.getLogger(__name__)

# An item is discarded once it has been unsubmittable for this long. This used
# to be a flat 200-attempt cap, but attempts are consumed by the work loop's
# 30-second no-work poll rather than by elapsed time, so a server outage burned
# through them in about two hours - far short of the server's 1-day assignment
# expiry that holding a queued result is meant to ride out.
MAX_QUEUE_ITEM_AGE = datetime.timedelta(days=7)

# Fallback cap for items whose created_at is missing or unparseable (~2 days of
# 30-second polling), so nothing can sit in the queue forever.
MAX_QUEUE_ATTEMPTS = 5000

# Rewrite sidecars older than this are leftovers from a rewrite that was killed
# midway; a live one exists for microseconds.
STALE_TMP_AGE_SECONDS = 3600


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
        for _, data in self._iter_result_items():
            chain = data.get("completion_chain") or {}
            if chain.get("residue_id") == residue_id:
                return True
        return False

    def has_pending_result_for_work(self, work_id: str) -> bool:
        """
        True if a queued result item belongs to this work assignment.

        Stage 1 needs this rather than attach_work_completion: its results
        already carry a residue_upload chain, which now completes the
        assignment itself once the result and residue land.
        """
        for _, data in self._iter_result_items():
            if (data.get("payload") or {}).get("work_id") == work_id:
                return True
        return False

    def _iter_result_items(
        self,
        newest_first: bool = False,
    ) -> Iterator[Tuple[Path, Dict[str, Any]]]:
        """
        Yield (path, item) for every readable result item, oldest-first.

        Ordering is by filename, which carries a microsecond timestamp from
        _generate_filename - lexicographic order is chronological, it needs no
        stat() (which would race a concurrent drain unlinking the file), and it
        cannot tie the way a coarse st_mtime can. Unreadable or half-written
        files are skipped rather than raising: callers run inside failure
        handling, where an exception would escape into the work loop. That
        covers a truncated multi-byte read (UnicodeDecodeError) and a file whose
        top-level JSON is not an object, both of which reach this loop from a
        half-written or hand-edited file.
        """
        if not self.results_dir.exists():
            return
        for f in sorted(self.results_dir.glob("*.json"), reverse=newest_first):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
            except (ValueError, OSError):
                continue
            if not isinstance(data, dict) or data.get("type") != "result":
                continue
            yield f, data

    def attach_work_completion(self, work_id: str, client_id: str) -> bool:
        """
        Attach a work_complete chain to the newest queued result for work_id.

        Used by the work loop to decide whether to abandon an assignment after a
        failed submission: if the queue already holds the computed result, the
        assignment should be kept so a later drain can complete it, instead of
        the composite going back in the pool for another client to re-run.

        The chain hangs off the *newest* matching result because one assignment
        can produce several submissions (a 'p1' assignment submits pm1 and pp1
        separately); completing on the last one keeps the assignment alive until
        every result has landed.

        Returns:
            True if a queued result was found and now carries the chain.
        """
        newest: Optional[Tuple[Path, Dict[str, Any]]] = None
        for path, data in self._iter_result_items(newest_first=True):
            if (data.get("payload") or {}).get("work_id") != work_id:
                continue
            if data.get("completion_chain"):
                # A follow-up already owns this assignment's results (stage 1's
                # residue upload). Attaching to a different result of the same
                # assignment would hold work that path releases deliberately, so
                # decline outright rather than skipping to an older item.
                return False
            if newest is None:
                newest = (path, data)

        if newest is None:
            return False

        path = newest[0]
        # Re-read and rewrite under the item's lock: the scan above is
        # unsynchronized, and a drain in another client process (the decoupled
        # two-stage setup shares data/queue) may have submitted and unlinked
        # this file since. Writing anyway would resurrect an accepted result for
        # every future drain to re-POST.
        with self._locked_item(path, blocking=False) as item:
            if item is None:
                # Gone, unreadable, or a concurrent drain owns it. Declining
                # leaves the caller to abandon the assignment, which is safe.
                self.logger.info(
                    f"Could not attach work completion for {work_id}: "
                    f"{path.name} is gone or locked by another process"
                )
                return False
            if item.get("completion_chain"):
                return False
            item["completion_chain"] = {
                "action": "work_complete",
                "work_id": work_id,
                "client_id": client_id,
            }
            # Rewrite atomically: a truncating in-place write that dies midway
            # would leave the only copy of a multi-hour result as unparseable
            # JSON, which _iter_result_items and drain() both skip forever.
            if not self._rewrite_item(path, item):
                return False

        self.logger.info(
            f"Attached work completion for {work_id} to queued result {path.name}"
        )
        return True

    @staticmethod
    def _lock_fd(fd: int, blocking: bool) -> bool:
        """
        Take an exclusive flock. False means someone else holds it.

        A filesystem that does not support locking (or a platform without
        fcntl) reports success: an unlocked queue is how this worked before,
        and stalling every drain there would be worse than the race.
        """
        if fcntl is None:
            return True
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            return True

    @contextlib.contextmanager
    def _locked_item(
        self,
        path: Path,
        blocking: bool = True,
    ) -> Iterator[Optional[Dict[str, Any]]]:
        """
        Yield one parsed queue item with an exclusive lock held on its file.

        Yields None (holding nothing) when the file is missing, unreadable, not
        a queue item, or - with blocking=False - already locked elsewhere.

        The lock matters because the documented decoupled two-stage setup runs
        two clients from the same directory, sharing data/queue with no other
        synchronization. Every read-modify-write here has to exclude a drain in
        the other process: it can unlink the file the instant its POST is
        accepted, and an os.replace landing after that would recreate an
        already-submitted result. Checking st_nlink after acquiring the lock
        catches the case where the unlink happened while we waited.

        Callers may unlink the file inside the block; the lock goes away with
        the descriptor either way.
        """
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            yield None
            return
        try:
            if not self._lock_fd(fd, blocking=blocking):
                yield None
                return
            if os.fstat(fd).st_nlink == 0:
                # Deleted while we waited for the lock - nothing left to act on.
                yield None
                return
            try:
                with open(fd, 'r', closefd=False) as fh:
                    data = json.load(fh)
            except (ValueError, OSError) as e:
                self.logger.warning(
                    f"Skipping unreadable queue item {path.name}: {e}"
                )
                yield None
                return
            if not isinstance(data, dict) or "type" not in data:
                yield None
                return
            yield data
        finally:
            os.close(fd)

    def _rewrite_item(self, path: Path, item: Dict[str, Any]) -> bool:
        """
        Replace a queue item file atomically. Returns False on any write error.

        Callers hold the item's lock (see _locked_item), so the os.replace
        cannot land on a file another process already submitted and unlinked.
        """
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, 'w') as fh:
                json.dump(item, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except Exception as e:
            # Not just OSError: a json.dump TypeError from an unserializable
            # payload would otherwise escape into the work loop through
            # cleanup_on_failure and leave the sidecar behind.
            self.logger.error(f"Failed to rewrite queue item {path.name}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _cleanup_stale_tmp_files(self) -> None:
        """
        Remove *.json.tmp sidecars from a rewrite that was killed midway.

        No glob in the queue matches them, so they would accumulate forever.
        Only files older than STALE_TMP_AGE_SECONDS are touched: deleting a
        live one would just make another process's os.replace fail.
        """
        cutoff = time.time() - STALE_TMP_AGE_SECONDS
        for subdir in (self.results_dir, self.residues_dir, self.completions_dir):
            if not subdir.exists():
                continue
            for tmp in subdir.glob("*.json.tmp"):
                try:
                    if tmp.stat().st_mtime < cutoff:
                        tmp.unlink(missing_ok=True)
                        self.logger.info(f"Removed stale queue temp file: {tmp.name}")
                except OSError:
                    continue

    def _expiry_reason(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Why this item should be discarded, or None to keep retrying it.

        Age, not attempt count: the work loop polls for work every 30 seconds
        while the server is down, and each poll drains the queue, so an attempt
        counter measures polling frequency rather than how long the outage has
        lasted. MAX_QUEUE_ITEM_AGE is deliberately longer than the server's
        1-day assignment expiry, which is the backstop for held work.
        """
        created_raw = item.get("created_at")
        created: Optional[datetime.datetime] = None
        if isinstance(created_raw, str):
            try:
                created = datetime.datetime.fromisoformat(created_raw)
            except ValueError:
                created = None

        if created is not None:
            age = datetime.datetime.now() - created
            if age > MAX_QUEUE_ITEM_AGE:
                return f"{age.days} day(s) of failed retries"
            return None

        attempts = item.get("attempts", 0)
        if isinstance(attempts, int) and attempts > MAX_QUEUE_ATTEMPTS:
            return f"{attempts} failed attempts (item has no usable created_at)"
        return None

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
                        # isinstance guard: a half-written or hand-edited file
                        # whose top-level JSON is not an object must be skipped,
                        # not crash the scan.
                        if isinstance(data, dict) and "type" in data:
                            files.append(f)
                    except (ValueError, OSError):
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
        self._cleanup_stale_tmp_files()
        files = self._get_queue_files()
        if not files:
            return (0, 0)

        self.logger.info(f"Draining submission queue: {len(files)} item(s) pending")
        success_count = 0
        fail_count = 0

        for filepath in files:
            try:
                # The lock is held for the whole read-retry-write cycle so a
                # drain in another client process cannot act on the same item.
                with self._locked_item(filepath) as item:
                    if item is None:
                        continue

                    item_type = item.get("type", "unknown")
                    item["attempts"] = item.get("attempts", 0) + 1

                    expiry_reason = self._expiry_reason(item)
                    if expiry_reason:
                        self.logger.warning(
                            f"Discarding {item_type} after {expiry_reason}: {filepath.name}"
                        )
                        filepath.unlink(missing_ok=True)
                        if item_type == "residue_upload":
                            residue_path = item.get("residue_file")
                            if residue_path:
                                Path(residue_path).unlink(missing_ok=True)
                        self._discard_chained_followups(item)
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
                        # Persist the attempt count through the same atomic
                        # rewrite the chain attach uses: a truncating in-place
                        # write here runs on every drain of a multi-hour result,
                        # and dying midway would destroy the only copy of it.
                        self._rewrite_item(filepath, item)

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
          linked to the attempt_id the resubmitted result returned, then close
          out the assignment that was held while the result sat in the queue
          (nothing else can complete it - this chain owns the result).
        - ``action == "work_complete"``: complete the work assignment that was
          held open while the result sat in the queue.
        - otherwise (stage 2): complete the residue with that attempt_id.
        """
        chain = item.get("completion_chain") or {}
        if not chain:
            return
        action = chain.get("action")
        if action == "residue_upload":
            self._chain_residue_upload(api_client, item, response)
            if chain.get("work_id"):
                self._chain_work_complete(api_client, item)
        elif action == "work_complete":
            self._chain_work_complete(api_client, item)
        else:
            self._chain_residue_complete(api_client, item, response)

    def _chain_work_complete(
        self,
        api_client: 'APIClient',
        item: Dict[str, Any],
    ) -> None:
        """
        After a queued result re-submits successfully, complete the work
        assignment that was held open for it.

        The assignment was deliberately not abandoned when the submission failed,
        so the composite stayed reserved for this client (rather than being handed
        to someone else to re-run) until the result finally landed. A transient
        failure here queues a standalone work_complete; a 404 means the server
        already expired the assignment, which is harmless - the result is in.

        Errors are swallowed: an exception escaping into _retry_item would leave
        the already-accepted result in the queue to be re-POSTed on every future
        drain (a duplicate attempt, double-counted curves). As with the two
        residue chains, a KeyboardInterrupt landing inside the call still gets
        through - Ctrl+C during a drain must stay interruptible.
        """
        chain = item.get("completion_chain") or {}
        work_id = chain.get("work_id")
        client_id = chain.get("client_id")
        if not work_id or not client_id:
            return

        try:
            if not api_client.complete_work(work_id=work_id, client_id=client_id):
                self.enqueue_work_completion(work_id=work_id, client_id=client_id)
        except ResourceNotFoundError:
            self.logger.info(
                f"Work {work_id} already expired/completed on server; "
                "chained completion skipped"
            )
        except Exception as e:
            self.logger.warning(
                f"Chained complete_work failed for {work_id}: {e}; "
                "queuing for later retry"
            )
            self.enqueue_work_completion(work_id=work_id, client_id=client_id)

    def _discard_chained_followups(
        self,
        item: Dict[str, Any],
        result_landed: bool = False,
    ) -> None:
        """
        Clean up after a queued result that is being dropped instead of sent.

        A result's completion_chain is the only thing that ever releases what
        was held for it: a work assignment (WorkMode.cleanup_on_failure) or a
        residue claim (Stage2ConsumerMode.cleanup_on_failure). When the result
        is discarded - a permanent rejection, or the age cap - that chain never
        runs, so the claim sits until the server expires it: a day of the
        client's active-work quota for an assignment, or 24h during which
        /ecm-work skips the whole composite because one of its residues is
        still 'claimed'.

        ``result_landed`` means the rejection itself says the server already
        has this result (a duplicate). The assignment is then completed rather
        than abandoned - abandoning would push a fully-recorded composite back
        into the pool for someone to redo.

        A stage-1 chain owns a preserved residue copy that nothing else
        references. It is dropped here too, or it stays on disk (hundreds of MB
        per GPU batch) with no metadata pointing at it.
        """
        chain = item.get("completion_chain") or {}
        if not chain:
            return

        residue_path = chain.get("residue_file")
        if residue_path:
            Path(residue_path).unlink(missing_ok=True)
            self.logger.info(
                f"Dropped preserved residue for discarded result: {residue_path}"
            )

        action = chain.get("action")
        client_id = chain.get("client_id")

        # Both chain shapes that hold an assignment carry work_id: the plain
        # work_complete chain, and stage 1's residue_upload chain.
        work_id = chain.get("work_id")
        if work_id and client_id:
            if result_landed:
                self.logger.warning(
                    f"Queued result for work {work_id} was rejected as already "
                    "submitted; completing the assignment held for it"
                )
                self.enqueue_work_completion(work_id=work_id, client_id=client_id)
            else:
                self.logger.warning(
                    f"Queued result for work {work_id} was discarded; releasing the "
                    "assignment that was held for it"
                )
                self.enqueue_work_abandonment(work_id=work_id, client_id=client_id)
            return

        if action in ("residue_upload", "work_complete"):
            # Manual stage 1 (no assignment), or a chain missing its ids.
            return

        # Stage-2 shape: {residue_id, client_id}. There is no stage-2 attempt_id
        # to complete the residue with once the result is gone, so release the
        # claim - another client redoing stage 2 costs CPU, but leaving the
        # residue 'claimed' locks the composite out of ECM work for 24h.
        residue_id = chain.get("residue_id")
        if residue_id is None or not client_id:
            return
        self.logger.warning(
            f"Queued result for residue {residue_id} was discarded; releasing "
            "the residue claim that was held for it"
        )
        self.enqueue_residue_abandonment(residue_id=residue_id, client_id=client_id)

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
            self._discard_chained_followups(item)
            return True  # Treat as success to remove from queue

        except PermanentUploadError as e:
            # 4xx residue upload rejection: composite factored, stage-1 attempt
            # gone, invalid file, etc. Retrying can never succeed, so discard.
            self.logger.warning(f"Discarding {item_type} from queue: {e}")
            self._discard_chained_followups(item)
            return True  # Treat as success to remove from queue

        except Exception as e:
            error_str = str(e)
            # These rejections mean the server already has this submission: the
            # work is recorded even though the queued copy is being dropped, so
            # anything held for it should be completed, not abandoned.
            landed_phrases = ["Duplicate", "already exists", "checksum matches"]
            # These mean it never landed and never will.
            missing_phrases = ["not claimed by client", "not found"]
            if any(phrase in error_str for phrase in landed_phrases + missing_phrases):
                landed = (
                    item_type == "result"
                    and any(phrase in error_str for phrase in landed_phrases)
                )
                self.logger.warning(
                    f"Discarding {item_type} from queue: permanent error: {error_str}"
                )
                self._discard_chained_followups(item, result_landed=landed)
                return True  # Remove from queue
            raise  # Re-raise for transient errors
