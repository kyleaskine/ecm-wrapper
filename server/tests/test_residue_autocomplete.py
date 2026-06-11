"""
Tests for server-side residue completion bundled into /submit_result.

Covers the fix for the lost-completion workflow bug: a stage 2 result could
be accepted while the separate /residues/{id}/complete call failed, leaving
the residue claimed/available and eventually re-served (a "poison loop" when
a factor had been divided out of the composite).

Scenarios:
- /submit_result auto-completes a claimed residue in the same transaction
- Partial runs (<75% curves) do NOT auto-complete
- Submissions from a different client do NOT auto-complete
- Old clients calling /residues/{id}/complete afterwards get an idempotent OK
- Resubmission after a lost response supersedes the duplicate attempt
- /residues/work auto-completes stale residues instead of re-serving them
- /admin/residues/reconcile repairs stuck residues with targeted recalcs
"""
from datetime import datetime, timedelta

from conftest import create_composite, get_test_engine

from app.models.attempts import ECMAttempt
from app.models.residues import ECMResidue


COMPOSITE = "1234567890123456789012345678901234567891"
CHECKSUM = "a" * 64
B1 = 50000
B2 = 5_000_000  # exactly 100 * B1, the minimum for a valid no-factor completion
CURVES = 100


def create_stage2_setup(composite_id: int, status: str = "claimed",
                        claimed_by: str = "test-client",
                        checksum: str = CHECKSUM) -> dict:
    """Create a stage 1 attempt and a residue linked to it."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        stage1 = ECMAttempt(
            composite_id=composite_id,
            client_id="gpu-producer",
            method="ecm",
            b1=B1,
            b2=0,  # stage 1 only
            parametrization=3,
            curves_requested=CURVES,
            curves_completed=CURVES,
            program="gmp-ecm",
        )
        db.add(stage1)
        db.flush()

        residue = ECMResidue(
            composite_id=composite_id,
            client_id="gpu-producer",
            stage1_attempt_id=stage1.id,
            b1=B1,
            parametrization=3,
            curve_count=CURVES,
            storage_path=f"/nonexistent/residue_{checksum[:8]}.txt",
            file_size_bytes=1234,
            checksum=checksum,
            status=status,
            claimed_by=claimed_by if status == "claimed" else None,
            claimed_at=datetime.utcnow() if status == "claimed" else None,
            expires_at=(datetime.utcnow() + timedelta(hours=24)) if status == "claimed" else None,
        )
        db.add(residue)
        db.commit()
        return {"stage1_attempt_id": stage1.id, "residue_id": residue.id}
    finally:
        db.close()


def get_residue(residue_id: int) -> dict:
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).one()
        return {"status": residue.status, "claimed_by": residue.claimed_by}
    finally:
        db.close()


def get_attempt(attempt_id: int) -> dict:
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        attempt = db.query(ECMAttempt).filter(ECMAttempt.id == attempt_id).one()
        return {"superseded_by": attempt.superseded_by,
                "residue_checksum": attempt.residue_checksum}
    finally:
        db.close()


def create_stage2_attempt(composite_id: int, curves_completed: int = CURVES,
                          checksum: str = CHECKSUM,
                          client_id: str = "old-client") -> int:
    """Create a bare stage 2 attempt row (historical/duplicate data)."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        attempt = ECMAttempt(
            composite_id=composite_id,
            client_id=client_id,
            method="ecm",
            b1=B1,
            b2=B2,
            parametrization=3,
            curves_requested=CURVES,
            curves_completed=curves_completed,
            program="gmp-ecm",
            residue_checksum=checksum,
        )
        db.add(attempt)
        db.commit()
        return attempt.id
    finally:
        db.close()


def set_superseded(attempt_id: int, by_id: int) -> None:
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        attempt = db.query(ECMAttempt).filter(ECMAttempt.id == attempt_id).one()
        attempt.superseded_by = by_id
        db.commit()
    finally:
        db.close()


def submit_stage2_result(client, curves_completed: int = CURVES,
                         client_id: str = "test-client",
                         checksum: str = CHECKSUM):
    return client.post(
        "/api/v1/submit_result",
        json={
            "composite": COMPOSITE,
            "client_id": client_id,
            "method": "ecm",
            "program": "gmp-ecm",
            "parameters": {"b1": B1, "b2": B2, "curves": CURVES},
            "results": {
                "curves_completed": curves_completed,
                "execution_time": 100.0,
            },
            "residue_checksum": checksum,
        },
    )


class TestSubmitAutoCompletesResidue:
    def test_full_run_completes_claimed_residue(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        response = submit_stage2_result(client)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["residue_completed"] is True

        # Residue closed out in the same call
        assert get_residue(setup["residue_id"])["status"] == "completed"
        # Stage 1 superseded by the new stage 2 attempt
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == data["attempt_id"]
        # Attempt linked to the residue
        assert get_attempt(data["attempt_id"])["residue_checksum"] == CHECKSUM

    def test_partial_run_does_not_complete(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        response = submit_stage2_result(client, curves_completed=50)

        assert response.status_code == 200
        data = response.json()
        assert data["residue_completed"] is False

        # Residue untouched: still claimed, client will abandon it
        residue = get_residue(setup["residue_id"])
        assert residue["status"] == "claimed"
        assert residue["claimed_by"] == "test-client"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] is None

    def test_other_clients_submission_does_not_complete(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], claimed_by="someone-else")

        response = submit_stage2_result(client, client_id="test-client")

        assert response.status_code == 200
        assert response.json()["residue_completed"] is False
        assert get_residue(setup["residue_id"])["status"] == "claimed"

    def test_old_client_completion_call_is_idempotent(self, client):
        """Old clients still call /residues/{id}/complete after submitting."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        data = submit_stage2_result(client).json()
        assert data["residue_completed"] is True

        # The follow-up call an old client makes must succeed, not 400/release
        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": data["attempt_id"]},
            headers={"X-Client-ID": "test-client"},
        )
        assert response.status_code == 200
        assert get_residue(setup["residue_id"])["status"] == "completed"

    def test_resubmission_after_lost_response_is_superseded(self, client):
        """A queued retry whose first submission actually landed creates a
        duplicate attempt; it must be superseded, not double-counted."""
        composite = create_composite(COMPOSITE)
        create_stage2_setup(composite["id"])

        first = submit_stage2_result(client).json()
        assert first["residue_completed"] is True

        # No-factor submissions get a unique work hash, so this records a
        # second attempt rather than hitting duplicate detection
        second = submit_stage2_result(client).json()
        assert second["residue_completed"] is True
        assert second["attempt_id"] != first["attempt_id"]

        # The duplicate is superseded by the original winner
        assert get_attempt(second["attempt_id"])["superseded_by"] == first["attempt_id"]
        assert get_attempt(first["attempt_id"])["superseded_by"] is None


class TestResidueWorkClaimGuard:
    def test_stale_residue_is_completed_not_served(self, client):
        """An available residue that already has a qualifying stage 2 attempt
        (its completion call was lost) is finalized instead of re-served."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")

        # Orphaned state built directly: a qualifying attempt exists but the
        # residue was never closed (predates submit-time auto-completion)
        attempt_id = create_stage2_attempt(composite["id"])
        assert get_residue(setup["residue_id"])["status"] == "available"

        response = client.get(
            "/api/v1/residues/work",
            headers={"X-Client-ID": "another-client"},
        )

        assert response.status_code == 200
        # Not served - it was the only residue, so no work is available
        assert response.json().get("residue_id") is None
        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == attempt_id

    def test_fresh_residue_is_served_normally(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")

        response = client.get(
            "/api/v1/residues/work",
            headers={"X-Client-ID": "cpu-worker"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["residue_id"] == setup["residue_id"]
        assert get_residue(setup["residue_id"])["status"] == "claimed"


class TestReconcileEndpoint:
    def test_reconcile_completes_stuck_residues(self, client):
        from app.main import app
        from app.dependencies import verify_admin_key

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")
        attempt_id = create_stage2_attempt(composite["id"])

        app.dependency_overrides[verify_admin_key] = lambda: True
        try:
            response = client.post("/api/v1/admin/residues/reconcile")
        finally:
            del app.dependency_overrides[verify_admin_key]

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["residues_completed"]) == 1
        assert data["residues_completed"][0]["residue_id"] == setup["residue_id"]
        assert data["residues_completed"][0]["attempt_id"] == attempt_id
        # Targeted recalc: exactly the one affected composite
        assert len(data["composites_recalculated"]) == 1
        assert data["composites_recalculated"][0]["composite_id"] == composite["id"]

        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == attempt_id

    def test_reconcile_supersedes_leftover_duplicates(self, client):
        """Duplicate unsuperseded attempts on a completed residue (history
        predating orphan handling) are superseded with one targeted recalc."""
        composite = create_composite(COMPOSITE)
        create_stage2_setup(composite["id"], status="completed")

        # Two unsuperseded stage 2 attempts sharing the checksum
        _, TestingSessionLocal = get_test_engine()
        db = TestingSessionLocal()
        try:
            ids = []
            for curves in (CURVES, 50):
                attempt = ECMAttempt(
                    composite_id=composite["id"],
                    client_id="old-client",
                    method="ecm",
                    b1=B1,
                    b2=B2,
                    parametrization=3,
                    curves_requested=CURVES,
                    curves_completed=curves,
                    program="gmp-ecm",
                    residue_checksum=CHECKSUM,
                )
                db.add(attempt)
                db.flush()
                ids.append(attempt.id)
            db.commit()
            full_id, partial_id = ids
        finally:
            db.close()

        from app.main import app
        from app.dependencies import verify_admin_key
        app.dependency_overrides[verify_admin_key] = lambda: True
        try:
            response = client.post("/api/v1/admin/residues/reconcile")
        finally:
            del app.dependency_overrides[verify_admin_key]

        assert response.status_code == 200
        data = response.json()
        assert len(data["attempts_superseded"]) == 1
        assert data["attempts_superseded"][0]["attempt_id"] == partial_id
        assert data["attempts_superseded"][0]["superseded_by"] == full_id
        assert get_attempt(partial_id)["superseded_by"] == full_id
        assert get_attempt(full_id)["superseded_by"] is None


class TestLapsedClaimCompletion:
    """Regression tests for the released-claim variant of the two-request
    failure (P2): expiry cleanup clears claimed_by between execution and
    submission, which previously blocked both submit-time auto-completion
    (required an active claim) and the fallback /complete call (403 on the
    cleared claim) - leaving the accepted attempt pending indefinitely.
    """

    def test_submit_with_lapsed_claim_autocompletes(self, client):
        """A residue whose claim was released (status back to 'available')
        is still completed by a qualifying submission - the checksum match
        proves the client had the file."""
        composite = create_composite(COMPOSITE)
        # Post-release state: available, no claim fields
        setup = create_stage2_setup(composite["id"], status="available")

        data = submit_stage2_result(client).json()

        assert data["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == data["attempt_id"]

    def test_partial_submit_with_lapsed_claim_does_not_complete(self, client):
        """Widening to 'available' must not bypass the 75% validation."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")

        data = submit_stage2_result(client, curves_completed=50).json()

        assert data["residue_completed"] is False
        assert get_residue(setup["residue_id"])["status"] == "available"

    def test_old_client_follow_up_without_claim_succeeds(self, client):
        """Old client flow with a lapsed claim: submit (auto-completes), then
        the follow-up /complete call - previously a 403 because claimed_by
        was cleared. The client's own matching attempt now authorizes it."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")
        data = submit_stage2_result(client).json()
        assert data["residue_completed"] is True

        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": data["attempt_id"]},
            headers={"X-Client-ID": "test-client"},
        )

        assert response.status_code == 200
        assert get_residue(setup["residue_id"])["status"] == "completed"

    def test_delayed_completion_after_reconcile_succeeds(self, client):
        """Reconcile finalizes a stuck residue (no claim on it); the original
        worker's queued completion retry must then succeed, not 403 forever."""
        from app.main import app
        from app.dependencies import verify_admin_key

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")
        attempt_id = create_stage2_attempt(composite["id"], client_id="cpu-worker")

        app.dependency_overrides[verify_admin_key] = lambda: True
        try:
            assert client.post("/api/v1/admin/residues/reconcile").status_code == 200
        finally:
            del app.dependency_overrides[verify_admin_key]
        assert get_residue(setup["residue_id"])["status"] == "completed"

        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": attempt_id},
            headers={"X-Client-ID": "cpu-worker"},
        )

        assert response.status_code == 200
        assert get_residue(setup["residue_id"])["status"] == "completed"

    def test_unrelated_client_is_still_forbidden(self, client):
        """The attempt-based authorization must not open the endpoint up:
        a client that doesn't own the attempt still gets 403."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], status="available")
        data = submit_stage2_result(client).json()  # attempt owned by test-client

        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": data["attempt_id"]},
            headers={"X-Client-ID": "intruder"},
        )

        assert response.status_code == 403


class TestSupersessionCycleGuard:
    """Regression tests for the reconcile/delayed-completion cycle (P1).

    Reconciliation must not dethrone the attempt stage 1 points at: the
    idempotent completion path resolves the winner via stage1.superseded_by,
    and superseding that attempt without re-pointing stage 1 lets a delayed
    old-client completion create a cycle (A->B, B->A) that excludes both
    attempts from the t-level.
    """

    def _reconcile(self, client):
        from app.main import app
        from app.dependencies import verify_admin_key
        app.dependency_overrides[verify_admin_key] = lambda: True
        try:
            return client.post("/api/v1/admin/residues/reconcile")
        finally:
            del app.dependency_overrides[verify_admin_key]

    def test_delayed_completion_after_reconcile_does_not_cycle(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        # Attempt A completes the residue (80 curves - qualifies at >=75%)
        a_id = submit_stage2_result(client, curves_completed=80).json()["attempt_id"]
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == a_id

        # Historical leftover B out-ranks A on curve count - without the fix,
        # reconcile picks B as winner and supersedes the designated winner A
        b_id = create_stage2_attempt(composite["id"], curves_completed=CURVES)

        response = self._reconcile(client)
        assert response.status_code == 200

        # The stage-1-designated winner must be preserved
        assert get_attempt(a_id)["superseded_by"] is None
        assert get_attempt(b_id)["superseded_by"] == a_id

        # Delayed old-client completion for B must not flip anything
        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": b_id},
            headers={"X-Client-ID": "test-client"},
        )
        assert response.status_code == 200

        # No cycle: A is still the unsuperseded winner
        assert get_attempt(a_id)["superseded_by"] is None
        assert get_attempt(b_id)["superseded_by"] == a_id

    def test_delayed_completion_resolves_terminal_winner(self, client):
        """If the designated winner was itself superseded later, a delayed
        completion must point its duplicate at the terminal of the chain."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        a_id = submit_stage2_result(client).json()["attempt_id"]

        # Simulate a later dethroning: stage1 -> A, but A -> B
        b_id = create_stage2_attempt(composite["id"])
        set_superseded(a_id, b_id)

        # Delayed completion for a third duplicate attempt C
        c_id = create_stage2_attempt(composite["id"])
        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": c_id},
            headers={"X-Client-ID": "test-client"},
        )
        assert response.status_code == 200

        # C is superseded by the terminal winner B, not the stale designee A
        assert get_attempt(c_id)["superseded_by"] == b_id
        assert get_attempt(b_id)["superseded_by"] is None

    def test_delayed_completion_survives_existing_cycle(self, client):
        """Pre-existing cycle in the chain: bail out without extending it."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        a_id = submit_stage2_result(client).json()["attempt_id"]
        b_id = create_stage2_attempt(composite["id"])
        # Corrupt state: A <-> B cycle
        set_superseded(a_id, b_id)
        set_superseded(b_id, a_id)

        c_id = create_stage2_attempt(composite["id"])
        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": c_id},
            headers={"X-Client-ID": "test-client"},
        )

        # Succeeds without joining the cycle
        assert response.status_code == 200
        assert get_attempt(c_id)["superseded_by"] is None
