"""Deferred residue-file deletion tied to transaction commit.

A residue file must be unlinked only AFTER the transaction that marks the
residue completed/expired actually commits. Deleting inline (before commit)
leaves the file gone if the transaction later rolls back - the DB row reverts
to 'claimed'/'available' while the file it points at no longer exists (the
poison-loop state).

Callers stage the path with stage_residue_file_deletion(); an after_commit
listener drains and unlinks them, and an after_rollback listener discards the
staging without deleting anything. Deletion is best-effort: a leftover file
is harmless (a later cleanup sweep removes it), whereas an early delete is
not, so the ordering is strictly delete-after-commit.
"""
import logging
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Key under which pending deletions are staged on Session.info (per-session,
# so concurrent requests on different sessions don't see each other's paths).
_STAGED_KEY = "residue_files_to_delete"


def stage_residue_file_deletion(db: Session, storage_path: str) -> None:
    """Queue a residue file for deletion after this session's next commit."""
    db.info.setdefault(_STAGED_KEY, []).append(storage_path)


@event.listens_for(Session, "after_commit")
def _delete_staged_files(session: Session) -> None:
    paths = session.info.pop(_STAGED_KEY, None)
    if not paths:
        return
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            logger.info(f"Deleted residue file after commit: {path}")
        except OSError as e:
            # Idempotent fallback: cleanup_orphaned / cleanup_factored sweeps
            # remove anything left behind, so a failed unlink is not fatal.
            logger.error(f"Failed to delete residue file {path} after commit: {e}")


@event.listens_for(Session, "after_rollback")
def _discard_staged_files(session: Session) -> None:
    # The transaction rolled back: the residue rows reverted to their prior
    # status, so their files must survive. Drop the staging without deleting.
    session.info.pop(_STAGED_KEY, None)
