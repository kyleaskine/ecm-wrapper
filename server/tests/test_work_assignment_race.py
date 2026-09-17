"""
Tests for duplicate work-assignment prevention.

Covers the fix for the production race where two /ecm-work requests 16 ms
apart were both assigned the same composite: under READ COMMITTED the
NOT EXISTS exclusion is evaluated against the statement snapshot, so a
concurrent request committing mid-query is invisible while its row lock is
already released. Defense is two layers:

1. pick_and_lock_composite re-checks the exclusions with fresh statements
   while holding the composite row lock.
2. A partial unique index allows at most one active assignment per
   composite at the database level.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from app.services.work_assignment import pick_and_lock_composite
from app.models.composites import Composite
from app.models.residues import ECMResidue
from app.models.work_assignments import WorkAssignment
from conftest import create_composite, create_work_assignment


def _make_assignment(composite_id: int, work_id: str, status: str = "assigned") -> WorkAssignment:
    return WorkAssignment(
        id=work_id,
        composite_id=composite_id,
        client_id="test-client",
        method="ecm",
        b1=50000,
        b2=5000000,
        curves_requested=100,
        expires_at=datetime.utcnow() + timedelta(days=1),
        status=status,
    )


def _add_residue(db, composite_id: int, status: str) -> None:
    db.add(ECMResidue(
        composite_id=composite_id,
        client_id="gpu-client",
        b1=110000000,
        parametrization=3,
        curve_count=5376,
        storage_path=f"/tmp/residue-test-{composite_id}.txt",
        file_size_bytes=1234,
        checksum=f"{composite_id:064x}",
        status=status,
    ))
    db.commit()


class TestUniqueActiveAssignmentIndex:
    """The partial unique index is the database-level backstop."""

    def test_second_active_assignment_rejected(self, db_session):
        composite = create_composite("1" * 60 + "3")
        db_session.add(_make_assignment(composite["id"], "wa-1", status="assigned"))
        db_session.commit()

        db_session.add(_make_assignment(composite["id"], "wa-2", status="assigned"))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_all_active_statuses_conflict(self, db_session):
        composite = create_composite("1" * 60 + "3")
        db_session.add(_make_assignment(composite["id"], "wa-1", status="running"))
        db_session.commit()

        db_session.add(_make_assignment(composite["id"], "wa-2", status="claimed"))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_inactive_assignments_do_not_conflict(self, db_session):
        """History rows (completed/timeout/failed) must not block new work."""
        composite = create_composite("1" * 60 + "3")
        db_session.add(_make_assignment(composite["id"], "wa-1", status="completed"))
        db_session.add(_make_assignment(composite["id"], "wa-2", status="timeout"))
        db_session.add(_make_assignment(composite["id"], "wa-3", status="failed"))
        db_session.add(_make_assignment(composite["id"], "wa-4", status="assigned"))
        db_session.commit()

        count = db_session.query(WorkAssignment).filter(
            WorkAssignment.composite_id == composite["id"]
        ).count()
        assert count == 4


class TestPickAndLockRecheck:
    """
    pick_and_lock_composite must reject a candidate that is busy even when
    the caller's query filters missed it (simulating the stale snapshot).
    """

    def _ordered_query(self, db):
        # Deliberately NO busy-composite exclusion: mimics a stale NOT EXISTS
        return db.query(Composite).order_by(Composite.target_t_level.asc())

    def test_busy_candidate_skipped_for_next_free_one(self, db_session):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        db_session.add(_make_assignment(busy["id"], "wa-busy"))
        db_session.commit()

        picked = pick_and_lock_composite(
            db_session, self._ordered_query(db_session), check_residues=False
        )
        assert picked is not None
        assert picked.id == free["id"]

    def test_returns_none_when_all_candidates_busy(self, db_session):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        db_session.add(_make_assignment(busy["id"], "wa-busy"))
        db_session.commit()

        picked = pick_and_lock_composite(
            db_session, self._ordered_query(db_session), check_residues=False
        )
        assert picked is None

    def test_pending_residue_rejected_when_checked(self, db_session):
        has_residue = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        _add_residue(db_session, has_residue["id"], "available")

        # /ecm-work and /p1-work: pending residue marks the composite busy
        picked = pick_and_lock_composite(
            db_session, self._ordered_query(db_session), check_residues=True
        )
        assert picked is not None
        assert picked.id == free["id"]

        # Legacy /work service does not check residues
        picked = pick_and_lock_composite(
            db_session, self._ordered_query(db_session), check_residues=False
        )
        assert picked is not None
        assert picked.id == has_residue["id"]


class TestEcmWorkEndpoint:
    """Endpoint-level regression: busy composites are never handed out."""

    def test_assigns_free_composite_and_skips_busy(self, client):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        create_work_assignment(busy["id"], "other-client", work_id="wa-busy")

        response = client.get("/api/v1/ecm-work", params={"client_id": "test-gpu"})
        assert response.status_code == 200
        data = response.json()
        assert data["work_id"] is not None
        assert data["composite_id"] == free["id"]

    def test_no_work_when_only_busy_composites(self, client):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        create_work_assignment(busy["id"], "other-client", work_id="wa-busy")

        response = client.get("/api/v1/ecm-work", params={"client_id": "test-gpu"})
        assert response.status_code == 200
        data = response.json()
        assert data["work_id"] is None

    def test_skips_composite_with_pending_residue(self, client, db_session):
        pending = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        _add_residue(db_session, pending["id"], "available")

        data = client.get("/api/v1/ecm-work", params={"client_id": "test-gpu"}).json()
        assert data["work_id"] is not None
        assert data["composite_id"] == free["id"]

    def test_two_sequential_requests_get_different_composites(self, client):
        create_composite("1" * 60 + "3", target_t_level=20.0)
        create_composite("2" * 60 + "3", target_t_level=25.0)

        first = client.get("/api/v1/ecm-work", params={"client_id": "gpu-1"}).json()
        second = client.get("/api/v1/ecm-work", params={"client_id": "gpu-2"}).json()

        assert first["work_id"] is not None
        assert second["work_id"] is not None
        assert first["composite_id"] != second["composite_id"]


class TestP1WorkEndpoint:
    """/p1-work honors the same claims as /ecm-work."""

    def test_skips_composite_with_active_assignment(self, client):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        create_work_assignment(busy["id"], "other-client", work_id="wa-busy")

        data = client.get("/api/v1/p1-work", params={"client_id": "p1-client"}).json()
        assert data["work_id"] is not None
        assert data["composite_id"] == free["id"]

    @pytest.mark.parametrize("residue_status", ["available", "claimed"])
    def test_skips_composite_with_pending_residue(self, client, db_session, residue_status):
        pending = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        _add_residue(db_session, pending["id"], residue_status)

        data = client.get("/api/v1/p1-work", params={"client_id": "p1-client"}).json()
        assert data["work_id"] is not None
        assert data["composite_id"] == free["id"]

    def test_finished_residue_does_not_block(self, client, db_session):
        done = create_composite("1" * 60 + "3", target_t_level=20.0)
        _add_residue(db_session, done["id"], "completed")

        data = client.get("/api/v1/p1-work", params={"client_id": "p1-client"}).json()
        assert data["composite_id"] == done["id"]

    def test_no_work_when_only_residue_pending(self, client, db_session):
        pending = create_composite("1" * 60 + "3", target_t_level=20.0)
        _add_residue(db_session, pending["id"], "available")

        data = client.get("/api/v1/p1-work", params={"client_id": "p1-client"}).json()
        assert data["work_id"] is None


class TestStaleFilterFallsBackToLock:
    """
    The busy-composite filter is a snapshot read; correctness rests on the
    recheck under the row lock. Patching the filter out simulates a
    competitor committing mid-query: both endpoints must still refuse the
    busy composite, which only holds while they pass check_residues=True.
    """

    @pytest.fixture(autouse=True)
    def _stale_filter(self, monkeypatch):
        monkeypatch.setattr(
            "app.api.v1.ecm_work._exclude_busy_composites",
            lambda db, query: query,
        )

    @pytest.mark.parametrize("endpoint", ["/api/v1/ecm-work", "/api/v1/p1-work"])
    def test_pending_residue_still_skipped(self, client, db_session, endpoint):
        pending = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        _add_residue(db_session, pending["id"], "available")

        data = client.get(endpoint, params={"client_id": "late-client"}).json()
        assert data["composite_id"] == free["id"]

    @pytest.mark.parametrize("endpoint", ["/api/v1/ecm-work", "/api/v1/p1-work"])
    def test_active_assignment_still_skipped(self, client, endpoint):
        busy = create_composite("1" * 60 + "3", target_t_level=20.0)
        free = create_composite("2" * 60 + "3", target_t_level=25.0)
        create_work_assignment(busy["id"], "other-client", work_id="wa-busy")

        data = client.get(endpoint, params={"client_id": "late-client"}).json()
        assert data["composite_id"] == free["id"]
