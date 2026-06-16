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
- Retries that hit duplicate detection still complete the residue
- /residues/work auto-completes stale residues instead of re-serving them
- /admin/residues/reconcile repairs stuck residues with targeted recalcs
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

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


def get_composite_state(composite_id: int) -> dict:
    from app.models.composites import Composite
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        c = db.query(Composite).filter(Composite.id == composite_id).one()
        return {"current_composite": c.current_composite,
                "is_fully_factored": c.is_fully_factored}
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
                         checksum: str = CHECKSUM,
                         composite: str = COMPOSITE,
                         factor: str = None,
                         sigma: str = None):
    results = {
        "curves_completed": curves_completed,
        "execution_time": 100.0,
    }
    if factor is not None:
        results["factor_found"] = factor
        results["factors_found"] = [{"factor": factor, "sigma": None}]
    parameters = {"b1": B1, "b2": B2, "curves": CURVES}
    if sigma is not None:
        # A fixed sigma makes the work hash deterministic, so an identical
        # resubmission hits duplicate detection (sigma-less hashes embed a
        # timestamp and are always unique)
        parameters["sigma"] = sigma
    return client.post(
        "/api/v1/submit_result",
        json={
            "composite": composite,
            "client_id": client_id,
            "method": "ecm",
            "program": "gmp-ecm",
            "parameters": parameters,
            "results": results,
            "residue_checksum": checksum,
        },
    )


def release_claim(residue_id: int) -> None:
    """Simulate expiry cleanup releasing a lapsed claim."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).one()
        residue.status = "available"
        residue.claimed_by = None
        residue.claimed_at = None
        residue.expires_at = None
        db.commit()
    finally:
        db.close()


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


class TestDuplicateRetryCompletesResidue:
    """Regression tests for the duplicate-retry gap: a submission whose
    attempt was recorded but whose bundled residue completion did not happen
    (claim held elsewhere at the time, transient failure) used to return
    factor_status='duplicate' on retry WITHOUT attempting completion,
    leaving the residue claimed until expiry."""

    SIGMA = "3:123456"  # deterministic work hash -> retry hits duplicate detection

    def test_duplicate_retry_completes_lapsed_residue(self, client):
        composite = create_composite(COMPOSITE)
        # Claim held by another client at first-submission time, so the
        # bundled completion is (correctly) skipped...
        setup = create_stage2_setup(composite["id"], claimed_by="someone-else")

        first = submit_stage2_result(client, sigma=self.SIGMA).json()
        assert first["residue_completed"] is False
        assert get_residue(setup["residue_id"])["status"] == "claimed"

        # ...then expiry cleanup releases the claim
        release_claim(setup["residue_id"])

        # The client's queued retry hits duplicate detection - it must
        # complete the residue with the existing attempt, not return early
        second = submit_stage2_result(client, sigma=self.SIGMA).json()
        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]
        assert second["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == first["attempt_id"]

    def test_duplicate_retry_leaves_foreign_claim_alone(self, client):
        """The retry path must not bypass the claim ownership check."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"], claimed_by="someone-else")

        submit_stage2_result(client, sigma=self.SIGMA)
        second = submit_stage2_result(client, sigma=self.SIGMA).json()

        assert second["factor_status"] == "duplicate"
        assert second["residue_completed"] is False
        residue = get_residue(setup["residue_id"])
        assert residue["status"] == "claimed"
        assert residue["claimed_by"] == "someone-else"

    def test_duplicate_retry_after_completion_is_idempotent(self, client):
        """A retry after a fully successful first submission must not
        disturb the completed residue or supersede the winning attempt."""
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])

        first = submit_stage2_result(client, sigma=self.SIGMA).json()
        assert first["residue_completed"] is True

        second = submit_stage2_result(client, sigma=self.SIGMA).json()
        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]
        assert second["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"
        # The original attempt remains the unsuperseded winner
        assert get_attempt(first["attempt_id"])["superseded_by"] is None


class TestInvalidChecksumNormalization:
    """An unknown residue_checksum is normalized to None BEFORE the work
    hash is generated, and the normalized value is used for hashing, the
    duplicate lookup, and storage. Hashing the raw value while storing None
    meant an identical retry hit a unique-hash violation (500), and a
    different bogus checksum minted a fresh hash - duplicate t-level credit
    for the same work."""

    SIGMA = "3:424242"

    def test_identical_retry_with_unknown_checksum_is_duplicate(self, client):
        create_composite(COMPOSITE)  # no residue exists for the checksum

        first = submit_stage2_result(client, checksum="x" * 64, sigma=self.SIGMA)
        assert first.status_code == 200
        first_data = first.json()
        # Stored unlinked: the bogus checksum is not on the attempt
        assert get_attempt(first_data["attempt_id"])["residue_checksum"] is None

        second = submit_stage2_result(client, checksum="x" * 64, sigma=self.SIGMA)

        assert second.status_code == 200  # was a 500 unique-hash violation
        second_data = second.json()
        assert second_data["factor_status"] == "duplicate"
        assert second_data["attempt_id"] == first_data["attempt_id"]

    def test_rotating_unknown_checksums_still_duplicate(self, client):
        """A different bogus checksum must not mint a fresh work identity
        (and with it duplicate t-level credit)."""
        create_composite(COMPOSITE)

        first = submit_stage2_result(client, checksum="x" * 64, sigma=self.SIGMA).json()
        second = submit_stage2_result(client, checksum="y" * 64, sigma=self.SIGMA).json()

        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]

    def test_unknown_checksum_matches_no_checksum_submission(self, client):
        """The same work with no checksum and with a bogus one is one
        identity once normalized."""
        create_composite(COMPOSITE)

        first = submit_stage2_result(client, checksum=None, sigma=self.SIGMA).json()
        second = submit_stage2_result(client, checksum="z" * 64, sigma=self.SIGMA).json()

        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]


class TestCrossCompositeDuplicateGuard:
    """Regression tests for the cross-composite duplicate bug: the work hash
    used to cover only the submitted string, and the duplicate lookup was
    global, so a checksum-pinned submission for composite A could hit
    composite B's attempt (same string + sigma + parameters) and complete
    A's residue with it - superseding A's stage 1 with a foreign attempt."""

    P = "1000003"
    PQR = "1000073001431003663"  # P * 1000033 * 1000037
    QR = "1000070001221"         # cofactor after P is divided out
    SIGMA = "3:777777"

    def test_checksum_submission_does_not_match_other_composites_attempt(self, client):
        # Composite A owns the residue; its current state is still P*Q*R
        composite_a = create_composite(self.PQR)
        setup = create_stage2_setup(composite_a["id"], checksum="n" * 64)

        # Composite B's current value happens to equal Q*R; a plain
        # submission of Q*R with a fixed sigma records B's attempt
        composite_b = create_composite(
            "900000000000000000077", current_composite=self.QR
        )
        b_attempt = submit_stage2_result(
            client, checksum=None, composite=self.QR, sigma=self.SIGMA
        ).json()
        assert b_attempt["composite_id"] == composite_b["id"]

        # P lands on A from another worker, reducing A's current to Q*R
        assert client.post(
            "/api/v1/submit_result",
            json={
                "composite": self.PQR,
                "client_id": "other-worker",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {"b1": 11000, "b2": 1100000, "curves": 50},
                "results": {
                    "factor_found": self.P,
                    "curves_completed": 7,
                    "execution_time": 3.0,
                },
            },
        ).status_code == 200

        # A's stage 2 worker submits Q*R with A's residue checksum and the
        # SAME sigma/parameters as B's attempt. The identity-free hash made
        # this a "duplicate" of B's attempt, which then completed A's
        # residue and superseded A's stage 1 with a foreign attempt.
        response = submit_stage2_result(
            client, checksum="n" * 64, composite=self.QR, sigma=self.SIGMA
        )

        assert response.status_code == 200
        data = response.json()
        assert data["composite_id"] == composite_a["id"]
        assert data["factor_status"] != "duplicate"
        assert data["attempt_id"] != b_attempt["attempt_id"]
        assert data["residue_completed"] is True
        # A's residue completed by A's own attempt
        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == data["attempt_id"]
        # B's attempt untouched (not superseded by the orphan sweep either)
        assert get_attempt(b_attempt["attempt_id"])["superseded_by"] is None

    def test_complete_endpoint_rejects_attempt_from_other_composite(self, client):
        """The claim holder authorizes via the claim, but the attempt it
        passes must still belong to the residue's composite - completing
        with a foreign attempt corrupts both composites' t-levels."""
        composite_a = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite_a["id"])  # claimed by test-client
        composite_b = create_composite("900000000000000000077")
        b_attempt = create_stage2_attempt(
            composite_b["id"], client_id="test-client", checksum="q" * 64
        )

        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": b_attempt},
            headers={"X-Client-ID": "test-client"},
        )

        assert response.status_code == 400
        # Caller error, not an invalid completion: the residue must NOT be
        # released back to the pool, and nothing is superseded
        residue = get_residue(setup["residue_id"])
        assert residue["status"] == "claimed"
        assert residue["claimed_by"] == "test-client"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] is None
        assert get_attempt(b_attempt)["superseded_by"] is None

    def test_complete_endpoint_rejects_attempt_from_sibling_residue(self, client):
        """Two residues on the SAME composite: an attempt that consumed
        residue B must not complete residue A, even when the caller holds
        A's claim - each residue's curves are distinct work, and A's stage 1
        must not be superseded by B's stage 2."""
        composite = create_composite(COMPOSITE)
        setup_a = create_stage2_setup(composite["id"], checksum="r" * 64)
        setup_b = create_stage2_setup(composite["id"], checksum="s" * 64)
        b_attempt = create_stage2_attempt(
            composite["id"], client_id="test-client", checksum="s" * 64
        )

        response = client.post(
            f"/api/v1/residues/{setup_a['residue_id']}/complete",
            json={"stage2_attempt_id": b_attempt},
            headers={"X-Client-ID": "test-client"},
        )

        assert response.status_code == 400
        # A untouched: still claimed, stage 1 not superseded
        residue_a = get_residue(setup_a["residue_id"])
        assert residue_a["status"] == "claimed"
        assert get_attempt(setup_a["stage1_attempt_id"])["superseded_by"] is None
        # B's stage 1 untouched too - nothing was completed
        assert get_attempt(setup_b["stage1_attempt_id"])["superseded_by"] is None

        # The same attempt completes its OWN residue fine
        response = client.post(
            f"/api/v1/residues/{setup_b['residue_id']}/complete",
            json={"stage2_attempt_id": b_attempt},
            headers={"X-Client-ID": "test-client"},
        )
        assert response.status_code == 200
        assert get_residue(setup_b["residue_id"])["status"] == "completed"
        assert get_attempt(setup_b["stage1_attempt_id"])["superseded_by"] == b_attempt


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


class TestStaleResidueSubmission:
    """End-to-end coverage for the sibling-residue race: a factor lands while
    another residue of the same composite is mid-stage-2. The stage 2 worker's
    submission then carries a re-found factor (and possibly the pre-factor
    composite string) - it must be accepted AND close the residue, instead of
    being rejected and leaving the residue to be re-served forever.
    """

    P = "1000003"
    PQR = "1000073001431003663"  # P * 1000033 * 1000037
    QR = "1000070001221"         # cofactor after P is divided out

    def _divide_out_p(self, client):
        """A plain submission (no residue) finds P, shrinking the composite."""
        return client.post(
            "/api/v1/submit_result",
            json={
                "composite": self.PQR,
                "client_id": "other-worker",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {"b1": 11000, "b2": 1100000, "curves": 50},
                "results": {
                    "factor_found": self.P,
                    "curves_completed": 7,
                    "execution_time": 3.0,
                },
            },
        )

    def test_refound_factor_completes_residue(self, client):
        """Worker claimed AFTER the factor: holds the cofactor string but
        re-finds the old factor from the original-N residue file."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="d" * 64)
        assert self._divide_out_p(client).status_code == 200

        response = submit_stage2_result(
            client, checksum="d" * 64, composite=self.QR, factor=self.P
        )

        assert response.status_code == 200
        data = response.json()
        assert data["factor_status"] == "known_factor"
        assert data["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == data["attempt_id"]

    def test_stale_string_submission_completes_residue(self, client):
        """Worker claimed BEFORE the factor: submits the original number,
        which no longer matches current_composite - resolved via the residue
        checksum instead of being rejected with 404."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="e" * 64)
        assert self._divide_out_p(client).status_code == 200

        response = submit_stage2_result(
            client, checksum="e" * 64, composite=self.PQR, factor=self.P
        )

        assert response.status_code == 200
        data = response.json()
        assert data["factor_status"] == "known_factor"
        assert data["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"

    def test_fabricated_stale_state_rejected(self, client):
        """A submission of current * X (X an unrelated prime, reported as the
        factor) must not resolve as a stale state: the quotient of a genuine
        ancestor decomposes entirely into recorded factors. Without this, X
        would be laundered through as known_factor and a 1%-curves run would
        close the residue. The checksum is authoritative, so the mismatch is
        rejected outright (400) rather than falling through to other lookups."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="f" * 64)
        fabricated = "1000056000189979335937729"  # PQR * 999983

        response = submit_stage2_result(
            client, checksum="f" * 64, composite=fabricated, factor="999983"
        )

        assert response.status_code == 400
        # Nothing was touched: residue still claimed, composite unchanged
        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite["id"])["current_composite"] == self.PQR
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] is None

    def test_zero_factor_rejected(self, client):
        """factor "0" passes integer validation and gcd(0, n) == n, which
        would claim the entire cofactor as a known factor and complete the
        residue after a 1% run. Factors must be > 1."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="h" * 64)

        response = submit_stage2_result(
            client, curves_completed=2, checksum="h" * 64,
            composite=self.PQR, factor="0"
        )

        assert response.status_code == 400
        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_factor_multiple_of_cofactor_rejected(self, client):
        """2 * current_composite divides neither the current cofactor nor the
        submitted state and is not recorded - it must 400, not be laundered
        through the gcd path as 'partially stale'."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="j" * 64)

        response = submit_stage2_result(
            client, curves_completed=2, checksum="j" * 64,
            composite=self.PQR, factor="2000146002862007326"  # 2 * PQR
        )

        assert response.status_code == 400
        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_fabricated_squared_factor_state_rejected(self, client):
        """current * P^2 must not validate off a single recorded P: the exact
        check requires the submitted state to divide the original number."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="k" * 64)
        assert self._divide_out_p(client).status_code == 200  # records P

        response = submit_stage2_result(
            client, curves_completed=2, checksum="k" * 64,
            composite="1000076001650007956010989",  # QR * P^2 = PQR * P
            factor=self.P
        )

        assert response.status_code == 400
        assert get_residue(setup["residue_id"])["status"] == "claimed"

    def test_trivial_factor_values_do_not_complete_residue(self, client):
        """factor '1' and factor == the composite itself are skipped as
        trivial; the raw value must not remain in attempt.factor_found where
        it would satisfy residue completion after a near-zero run."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="l" * 64)

        for trivial in ("1", self.PQR):
            response = submit_stage2_result(
                client, curves_completed=2, checksum="l" * 64,
                composite=self.PQR, factor=trivial
            )
            assert response.status_code == 200
            data = response.json()
            assert data["factor_status"] == "no_factor"
            assert data["residue_completed"] is False

        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_zero_composite_with_checksum_rejected(self, client):
        """Composite "0" on a checksum-pinned submission must be a clean 400
        (state mismatch), not a ZeroDivisionError -> 500 in the ancestry
        arithmetic."""
        composite = create_composite(self.PQR)
        setup = create_stage2_setup(composite["id"], checksum="m" * 64)

        response = submit_stage2_result(
            client, curves_completed=2, checksum="m" * 64,
            composite="0", factor=self.P
        )

        assert response.status_code == 400
        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_checksum_mismatch_rejected_not_misattributed(self, client):
        """A valid checksum whose composite doesn't match the submitted
        string must reject, not fall through to the string lookup - that
        would record the attempt against an unrelated composite while the
        checksum's residue stayed claimed."""
        # Composite A: an unrelated number a submission string could match
        composite_a = create_composite("900000000000000000077")
        # Composite B: owns the residue
        composite_b = create_composite(self.PQR)
        setup = create_stage2_setup(composite_b["id"], checksum="i" * 64)

        # B's checksum, but A's number as the submitted string
        response = submit_stage2_result(
            client, checksum="i" * 64, composite="900000000000000000077"
        )

        assert response.status_code == 400
        # B's residue untouched, no work recorded against A
        assert get_residue(setup["residue_id"])["status"] == "claimed"
        assert get_composite_state(composite_a["id"])["is_fully_factored"] is False

    def test_checksum_resolution_takes_precedence(self, client):
        """current_composite is not unique: if another composite's current
        value equals this work's stale string, a string-first lookup applies
        the attempt and factor to the wrong composite. The residue checksum
        pins the right one."""
        # Composite B: residue claimed, then P divided out (current = QR,
        # stale state = PQR)
        composite_b = create_composite(self.PQR)
        setup = create_stage2_setup(composite_b["id"], checksum="g" * 64)
        assert self._divide_out_p(client).status_code == 200

        # Composite A: a different number whose CURRENT cofactor happens to
        # equal B's stale state
        composite_a = create_composite(
            "900000000000000000077", current_composite=self.PQR
        )

        # B's worker submits with the stale string and B's residue checksum
        response = submit_stage2_result(
            client, checksum="g" * 64, composite=self.PQR, factor=self.P
        )

        assert response.status_code == 200
        data = response.json()
        # Applied to B (via checksum), not to A (via string collision)
        assert data["composite_id"] == composite_b["id"]
        assert data["factor_status"] == "known_factor"
        assert data["residue_completed"] is True
        assert get_residue(setup["residue_id"])["status"] == "completed"
        # A is untouched - without checksum precedence, P would have been
        # divided out of A's current_composite
        state_a = get_composite_state(composite_a["id"])
        assert state_a["current_composite"] == self.PQR


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


class TestStatusWriteRaces:
    """Regression tests for stale identity-map state on status writers.

    with_for_update() alone acquires the row lock but does NOT overwrite
    attributes already loaded in the session, so a checker that read
    'claimed' before blocking on the lock would still see 'claimed' after a
    concurrent completion committed 'completed'. The lock queries must use
    populate_existing(). Simulated here with raw SQL updates, which change
    the database row while leaving the ORM identity map stale.
    """

    def test_release_claim_does_not_resurrect_completed_residue(self, db_session):
        """Releasing on a stale 'claimed' read would overwrite 'completed' -
        recreating an available residue with no file and a superseded
        stage 1 (the poison-loop state)."""
        from sqlalchemy import text
        from app.services.residue_manager import ResidueManager

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])  # claimed by test-client
        manager = ResidueManager()

        # Pull the residue into this session's identity map as 'claimed'
        stale = db_session.query(ECMResidue).filter(
            ECMResidue.id == setup["residue_id"]
        ).one()
        assert stale.status == "claimed"

        # A concurrent completion lands (raw SQL keeps the map stale)
        db_session.execute(
            text("UPDATE ecm_residues SET status='completed' WHERE id=:id"),
            {"id": setup["residue_id"]},
        )

        with pytest.raises(ValueError, match="not claimed"):
            manager.release_claim(db_session, setup["residue_id"], "test-client")

        status = db_session.execute(
            text("SELECT status FROM ecm_residues WHERE id=:id"),
            {"id": setup["residue_id"]},
        ).scalar()
        assert status == "completed"

    def test_complete_residue_sees_concurrent_completion(self, db_session):
        """A completion holding a stale 'claimed' object must take the
        idempotent path once the row is actually 'completed' - redoing the
        supersession is how A->B, B->A cycles form."""
        from sqlalchemy import text
        from app.services.residue_manager import ResidueManager

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])
        attempt_id = create_stage2_attempt(composite["id"], client_id="test-client")
        manager = ResidueManager()

        stale = db_session.query(ECMResidue).filter(
            ECMResidue.id == setup["residue_id"]
        ).one()
        assert stale.status == "claimed"

        db_session.execute(
            text("UPDATE ecm_residues SET status='completed' WHERE id=:id"),
            {"id": setup["residue_id"]},
        )

        manager.complete_residue(
            db_session, setup["residue_id"], attempt_id, recalculate_t_level=False
        )
        db_session.flush()

        # Idempotent path taken: stage 1 was NOT superseded again
        superseded = db_session.execute(
            text("SELECT superseded_by FROM ecm_attempts WHERE id=:id"),
            {"id": setup["stage1_attempt_id"]},
        ).scalar()
        assert superseded is None

    def test_expiry_cleanup_skips_completed_residue(self, db_session):
        """The expiry sweep must not flip a residue that completed after the
        candidate read - the conditional UPDATE re-checks status at write
        time. (Sequential stand-in for the concurrent interleaving.)"""
        from sqlalchemy import text
        from app.services.residue_manager import ResidueManager

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])
        manager = ResidueManager()

        # Expired claim, but the row completed in the meantime (expires_at
        # left behind - completion normally clears it, but the sweep must
        # not depend on that)
        db_session.execute(
            text(
                "UPDATE ecm_residues SET status='completed', "
                "expires_at='2000-01-01 00:00:00' WHERE id=:id"
            ),
            {"id": setup["residue_id"]},
        )

        assert manager.cleanup_expired_claims(db_session) == 0
        status = db_session.execute(
            text("SELECT status FROM ecm_residues WHERE id=:id"),
            {"id": setup["residue_id"]},
        ).scalar()
        assert status == "completed"


class TestConcurrentUploadGuard:
    """The residue checksum is an authoritative identity (submission
    resolution and completion authorization key on it), so two rows must
    never share one. The pre-insert duplicate check is race-prone; the
    unique constraint must catch the losing concurrent upload and its
    written file must be cleaned up."""

    COMPOSITE_INT = 1000073001431003663
    CHKCONST = 4294967291  # GMP-ECM's CHKSUMMOD

    def _residue_content(self, b1: int = 50000, sigma: int = 12345,
                         param: int = 3, x: int = 0x1ABCD) -> bytes:
        """One valid GMP-ECM residue line (checksum per ecm-ecm.h)."""
        chk = b1
        chk *= sigma % self.CHKCONST
        chk *= self.COMPOSITE_INT % self.CHKCONST
        chk *= x % self.CHKCONST
        chk *= param + 1
        chk %= self.CHKCONST
        line = (f"METHOD=ECM; B1={b1}; N={self.COMPOSITE_INT}; "
                f"SIGMA={sigma}; PARAM={param}; X={hex(x)}; CHECKSUM={chk};\n")
        return line.encode()

    def _manager(self, tmp_path):
        from app.services.residue_manager import ResidueManager
        manager = ResidueManager()
        manager.storage_dir = tmp_path
        return manager

    def test_losing_concurrent_upload_rejected_and_file_removed(
            self, db_session, tmp_path, monkeypatch):
        from app.services.residue_manager import ResidueManager

        create_composite(str(self.COMPOSITE_INT))
        manager = self._manager(tmp_path)
        content = self._residue_content()

        first = manager.store_residue_file(db_session, content, "uploader-1")
        db_session.commit()
        first_id, first_checksum = first.id, first.checksum
        first_path = Path(first.storage_path)

        # Simulate the race: the loser's pre-check ran before the winner's
        # insert was visible
        monkeypatch.setattr(
            ResidueManager, "_find_duplicate_residue",
            lambda self, db, checksum: None
        )

        with pytest.raises(ValueError, match="Duplicate residue"):
            manager.store_residue_file(db_session, content, "uploader-2")
        db_session.rollback()

        # Only the winner's row and file remain
        rows = db_session.query(ECMResidue).filter(
            ECMResidue.checksum == first_checksum
        ).all()
        assert [r.id for r in rows] == [first_id]
        files = sorted(tmp_path.rglob("*.txt"))
        assert files == [first_path]

    def test_sequential_duplicate_gets_friendly_error_without_file(
            self, db_session, tmp_path):
        """The pre-check still catches ordinary duplicates with the existing
        residue ID in the message, before any file is written."""
        create_composite(str(self.COMPOSITE_INT))
        manager = self._manager(tmp_path)
        content = self._residue_content()

        first = manager.store_residue_file(db_session, content, "uploader-1")
        db_session.commit()
        first_id = first.id

        with pytest.raises(ValueError, match=f"residue ID {first_id}"):
            manager.store_residue_file(db_session, content, "uploader-1")

        assert len(list(tmp_path.rglob("*.txt"))) == 1

    def test_non_checksum_integrity_error_not_masked(self, db_session, tmp_path):
        """A foreign-key failure (invalid stage1_attempt_id) must surface
        as an IntegrityError, not be misreported as a concurrent duplicate;
        the written file is still cleaned up."""
        from sqlalchemy.exc import IntegrityError

        create_composite(str(self.COMPOSITE_INT))
        manager = self._manager(tmp_path)

        with pytest.raises(IntegrityError):
            manager.store_residue_file(
                db_session, self._residue_content(), "uploader-1",
                stage1_attempt_id=999999
            )
        db_session.rollback()

        assert list(tmp_path.rglob("*.txt")) == []


class TestDeferredFileDeletion:
    """A residue file must be unlinked only AFTER the completing transaction
    commits. Deleting inline would leave the file gone if the transaction
    later rolls back, reverting status to 'claimed' with no file behind it."""

    COMPOSITE_INT = 1000073001431003663

    def _setup(self, db, tmp_path):
        composite = create_composite(str(self.COMPOSITE_INT))
        checksum = "d1" * 32  # 64 hex chars
        f = tmp_path / "residue.txt"
        f.write_text("dummy residue contents")

        stage1 = ECMAttempt(
            composite_id=composite["id"], client_id="gpu", method="ecm",
            b1=B1, b2=0, parametrization=3, curves_requested=CURVES,
            curves_completed=CURVES, program="gmp-ecm",
        )
        db.add(stage1)
        db.flush()

        residue = ECMResidue(
            composite_id=composite["id"], client_id="gpu",
            stage1_attempt_id=stage1.id, b1=B1, parametrization=3,
            curve_count=CURVES, storage_path=str(f), file_size_bytes=22,
            checksum=checksum, status="claimed", claimed_by="cpu",
            claimed_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(residue)
        db.flush()

        stage2 = ECMAttempt(
            composite_id=composite["id"], client_id="cpu", method="ecm",
            b1=B1, b2=B2, parametrization=3, curves_requested=CURVES,
            curves_completed=CURVES, program="gmp-ecm",
            residue_checksum=checksum,
        )
        db.add(stage2)
        db.commit()
        return residue.id, stage2.id, f

    def test_file_deleted_only_after_commit(self, db_session, tmp_path):
        from app.services.residue_manager import ResidueManager
        residue_id, stage2_id, f = self._setup(db_session, tmp_path)
        manager = ResidueManager()

        manager.complete_residue(
            db_session, residue_id, stage2_id, recalculate_t_level=False
        )
        # Still present: the DB change isn't committed yet
        assert f.exists()

        db_session.commit()
        # after_commit listener drained the staged deletion
        assert not f.exists()

    def test_file_survives_rollback(self, db_session, tmp_path):
        from app.services.residue_manager import ResidueManager
        residue_id, stage2_id, f = self._setup(db_session, tmp_path)
        manager = ResidueManager()

        manager.complete_residue(
            db_session, residue_id, stage2_id, recalculate_t_level=False
        )
        db_session.rollback()

        # Status reverted to 'claimed'; the file must still be there
        assert f.exists()
        residue = get_residue(residue_id)
        assert residue["status"] == "claimed"

    def test_admin_delete_defers_file_removal_to_commit(self, client, db_session, tmp_path):
        """Admin delete used to os.remove() before the row delete committed -
        an orphaned file if the delete rolled back. It now stages deletion
        through the same after-commit hook."""
        from app.main import app
        from app.dependencies import verify_admin_key
        residue_id, _, f = self._setup(db_session, tmp_path)
        assert f.exists()

        app.dependency_overrides[verify_admin_key] = lambda: True
        try:
            resp = client.delete(f"/api/v1/admin/residues/{residue_id}")
        finally:
            del app.dependency_overrides[verify_admin_key]

        assert resp.status_code == 200
        # Removed via the post-commit hook, after the row delete committed
        assert not f.exists()
        assert get_test_engine()[1]().query(ECMResidue).filter(
            ECMResidue.id == residue_id
        ).first() is None


class TestRejectedCompletionLeavesClaim:
    """A rejected completion (insufficient curves) must 400 and leave the
    claim in place. The earlier code set status='available' then raised, but
    the raise rolls the transaction back - so the release was undone while
    the message claimed the residue had been released back to the pool."""

    def test_insufficient_curves_rejected_claim_kept(self, client):
        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(composite["id"])  # claimed by test-client

        # Too few curves, no factor -> fails the 75% rule
        attempt_id = create_stage2_attempt(
            composite["id"], curves_completed=10, client_id="test-client"
        )

        response = client.post(
            f"/api/v1/residues/{setup['residue_id']}/complete",
            json={"stage2_attempt_id": attempt_id},
            headers={"X-Client-ID": "test-client"},
        )

        assert response.status_code == 400
        # The message must not falsely claim a release that gets rolled back
        assert "released" not in response.json()["detail"].lower()
        # Claim untouched: it expires later, it is not flipped to available
        residue = get_residue(setup["residue_id"])
        assert residue["status"] == "claimed"
        assert residue["claimed_by"] == "test-client"
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] is None


class TestClaimSelectionContention:
    """get_available_work selects candidates unlocked (the residue lock moved
    into claim_residue for lock-order safety), so concurrent consumers all see
    the same top-priority row. The work loop must walk DISTINCT candidates via
    an exclusion set, or a loser re-picks the contended row every iteration
    and returns a false 'no work available' while other residues sit free."""

    def _two_available(self):
        composite = create_composite(COMPOSITE)
        create_stage2_setup(composite["id"], status="available", checksum="a1" * 32)
        create_stage2_setup(composite["id"], status="available", checksum="a2" * 32)
        return composite

    def test_get_available_work_excludes_tried_ids(self, db_session):
        from app.services.residue_manager import ResidueManager
        self._two_available()
        manager = ResidueManager()

        first = manager.get_available_work(db_session, "consumer")
        assert first is not None
        second = manager.get_available_work(
            db_session, "consumer", exclude_ids={first.id}
        )
        assert second is not None and second.id != first.id
        third = manager.get_available_work(
            db_session, "consumer", exclude_ids={first.id, second.id}
        )
        assert third is None

    def test_loop_advances_past_contended_candidate(self, client, monkeypatch):
        """A claim that loses the race must not strand the consumer on the
        contended row - the loop excludes it and serves a distinct residue."""
        from app.services.residue_manager import ResidueManager
        self._two_available()

        real_claim = ResidueManager.claim_residue
        state = {}

        def flaky_claim(self, db, residue_id, client_id, claim_timeout_hours=72):
            # The top-priority candidate is permanently contended (another
            # consumer holds it). Without the exclusion set the loop re-picks
            # it every iteration and exhausts its budget -> false "no work".
            state.setdefault("contended", residue_id)
            if residue_id == state["contended"]:
                raise ValueError("simulated lost claim race")
            return real_claim(
                self, db, residue_id, client_id,
                claim_timeout_hours=claim_timeout_hours
            )

        monkeypatch.setattr(ResidueManager, "claim_residue", flaky_claim)

        resp = client.get(
            "/api/v1/residues/work", headers={"X-Client-ID": "consumer"}
        )

        assert resp.status_code == 200
        data = resp.json()
        # Served the OTHER residue, not "no work" and not the contended one
        assert data.get("residue_id") is not None
        assert data["residue_id"] != state["contended"]

    def test_stale_completion_race_does_not_500(self, client, monkeypatch):
        """complete_residue in the stale-completion branch can raise if a
        concurrent completion lands first (it reads the attempt unlocked).
        That must be swallowed and the loop moved on, not 500 a client that
        was merely requesting work."""
        from app.services.residue_manager import ResidueManager
        composite = create_composite(COMPOSITE)
        create_stage2_setup(composite["id"], status="available", checksum="c1" * 32)
        # Qualifying attempt so find_completing_attempt returns one
        create_stage2_attempt(composite["id"], checksum="c1" * 32)

        def boom(self, db, residue_id, stage2_attempt_id, recalculate_t_level=True):
            raise ValueError("simulated concurrent completion race")

        monkeypatch.setattr(ResidueManager, "complete_residue", boom)

        resp = client.get(
            "/api/v1/residues/work", headers={"X-Client-ID": "consumer"}
        )

        assert resp.status_code == 200  # was a 500
        # The only residue had a completing attempt; after the swallowed race
        # it's excluded and nothing else is available
        assert resp.json().get("residue_id") is None

    def test_claim_rejects_fully_factored_composite(self, db_session):
        """A composite fully factored between the unlocked select and the
        locked claim must not have its residue served as live work."""
        from app.services.residue_manager import ResidueManager
        from app.models.composites import Composite

        composite = create_composite(COMPOSITE)
        setup = create_stage2_setup(
            composite["id"], status="available", checksum="d1" * 32
        )
        c = db_session.query(Composite).filter(
            Composite.id == composite["id"]
        ).one()
        c.is_fully_factored = True
        db_session.commit()

        manager = ResidueManager()
        with pytest.raises(ValueError, match="fully factored"):
            manager.claim_residue(db_session, setup["residue_id"], "consumer")

    def test_lock_residue_missing_returns_none(self, db_session):
        """A residue gone (concurrent delete) must yield None, not raise
        NoResultFound -> 500."""
        from app.services.residue_manager import ResidueManager
        manager = ResidueManager()
        assert manager.lock_residue(db_session, 999999) is None


class TestDuplicateResponseTLevelReporting:
    """The duplicate path runs update_t_level inside a swallowing except, so a
    failed recalc must report new_t_level=None rather than the pre-update
    value (reporting residue_completed=True with a stale t-level is wrong)."""

    SIGMA = "3:909090"

    def test_swallowed_recalc_reports_no_tlevel(self, client, monkeypatch):
        from app.services.composites import CompositeService

        composite = create_composite(COMPOSITE)
        create_stage2_setup(composite["id"])  # claimed by test-client

        first = submit_stage2_result(client, sigma=self.SIGMA).json()
        assert first["residue_completed"] is True

        # The duplicate retry will re-run update_t_level; make it fail
        def boom(self, db, composite_id):
            raise RuntimeError("simulated transient recalc failure")

        monkeypatch.setattr(CompositeService, "update_t_level", boom)

        second = submit_stage2_result(client, sigma=self.SIGMA).json()

        assert second["factor_status"] == "duplicate"
        assert second["residue_completed"] is True
        # Recalc was swallowed -> must not surface a stale t-level
        assert second["new_t_level"] is None
