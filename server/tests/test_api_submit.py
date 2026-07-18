"""
Tests for the /submit_result API endpoint.

Tests cover:
- Successful result submission
- Factor discovery and validation
- Duplicate detection
- Error handling for invalid submissions
"""
import pytest
from conftest import create_composite, create_work_assignment


class TestSubmitResultBasic:
    """Basic submission tests."""

    def test_submit_result_no_factor(self, client):
        """Test submitting a result with no factor found."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "b2": 5000000,
                    "curves": 100,
                },
                "results": {
                    "curves_completed": 100,
                    "execution_time": 10.5,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["factor_status"] == "no_factor"
        assert data["attempt_id"] is not None
        assert data["composite_id"] == composite["id"]

    def test_submit_result_unknown_composite(self, client):
        """Test that submissions for unknown composites are rejected."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "99999999999999999",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 404
        assert "not found in database" in response.json()["detail"]

    def test_submit_result_duplicate_detection(self, client):
        """Test that duplicate submissions return existing attempt."""
        composite = create_composite("12345678901234567890", digit_length=20)

        payload = {
            "composite": composite["current_composite"],
            "client_id": "test-client",
            "method": "ecm",
            "program": "gmp-ecm",
            "parameters": {
                "b1": 50000,
                "b2": 5000000,
                "curves": 100,
                "sigma": "3:12345",
            },
            "results": {
                "curves_completed": 100,
            },
        }

        # First submission
        response1 = client.post("/api/v1/submit_result", json=payload)
        assert response1.status_code == 200
        attempt_id_1 = response1.json()["attempt_id"]

        # Second identical submission
        response2 = client.post("/api/v1/submit_result", json=payload)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["factor_status"] == "duplicate"
        assert data2["attempt_id"] == attempt_id_1


class TestSubmitResultWithFactor:
    """Tests for factor submission and validation."""

    def test_submit_valid_factor(self, client):
        """Test submitting a valid factor."""
        # Composite is 15 = 3 * 5
        composite = create_composite("15", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "15",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                    "sigma": "3:12345",
                },
                "results": {
                    "curves_completed": 5,
                    "factor_found": "3",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["factor_status"] == "new_factor"

    def test_submit_invalid_factor(self, client):
        """Test that invalid factors are rejected."""
        # 7 does not divide 15
        composite = create_composite("15", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "15",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                },
                "results": {
                    "curves_completed": 5,
                    "factor_found": "7",
                },
            },
        )

        assert response.status_code == 400
        assert "does not divide" in response.json()["detail"]

    def test_submit_multiple_factors(self, client):
        """Test submitting multiple factors in one request."""
        # 30 = 2 * 3 * 5
        composite = create_composite("30", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "30",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                },
                "results": {
                    "curves_completed": 5,
                    "factors_found": [
                        {"factor": "2", "sigma": "3:111"},
                        {"factor": "3", "sigma": "3:222"},
                    ],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["factor_status"] == "new_factor"
        assert data["composite_id"] == composite["id"]


class TestSubmitResultParametrization:
    """Tests for parametrization handling."""

    def test_parametrization_from_sigma_string(self, client):
        """Test that parametrization is extracted from sigma string format."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "sigma": "1:98765",  # Parametrization 1
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_explicit_parametrization(self, client):
        """Test explicit parametrization parameter."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "parametrization": 3,
                    "sigma": "12345",  # No prefix, but explicit param
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_invalid_parametrization(self, client):
        """Test that invalid parametrization values are rejected."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "parametrization": 5,  # Invalid - must be 0-3
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        # Should fail validation (Pydantic validates ge=0, le=3)
        assert response.status_code == 422


class TestSubmitResultValidation:
    """Tests for request validation."""

    def test_missing_required_fields(self, client):
        """Test that missing required fields return 422."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "12345",
                # Missing client_id, method, program, parameters, results
            },
        )

        assert response.status_code == 422

    def test_invalid_method(self, client):
        """Test that invalid method returns 422."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "12345",
                "client_id": "test",
                "method": "invalid_method",
                "program": "test",
                "parameters": {"b1": 1000, "curves": 10},
                "results": {"curves_completed": 10},
            },
        )

        assert response.status_code == 422


class TestStaleFactorSubmissions:
    """Submissions made stale by a concurrent factor discovery.

    Long-running work (hours of stage 2) inherently races with other clients'
    factor submissions. The server must tolerate the two stale outcomes
    instead of rejecting real work: a re-found factor that was already
    divided out (was a 400 - the poison-loop trigger), and a composite
    string referencing the pre-factor state (was a 404).
    """

    P = "1000003"
    Q = "1000033"
    R = "1000037"
    PQR = "1000073001431003663"  # P * Q * R
    QR = "1000070001221"         # Q * R
    BOGUS = "999983"             # prime, does not divide PQR
    PQ = "1000036000099"         # P * Q (composite factor)

    def _submit(self, client, composite, factors, b1=50000, work_id=None,
                client_id="stale-test-client", sigma=None):
        parameters = {"b1": b1, "b2": 5000000, "curves": 100}
        if sigma is not None:
            # A fixed sigma makes the work hash deterministic, so identical
            # submissions dedup; without it the hash embeds a timestamp.
            parameters["sigma"] = sigma
        return client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite,
                "client_id": client_id,
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": parameters,
                "results": {
                    "factor_found": factors[0] if factors else None,
                    "factors_found": [{"factor": f, "sigma": None} for f in factors] or None,
                    "curves_completed": 100,
                    "execution_time": 5.0,
                },
                "work_id": work_id,
            },
        )

    def _get_composite_state(self, composite_id):
        from conftest import get_test_engine
        from app.models.composites import Composite
        _, TestingSessionLocal = get_test_engine()
        db = TestingSessionLocal()
        try:
            c = db.query(Composite).filter(Composite.id == composite_id).one()
            return {"current_composite": c.current_composite,
                    "is_fully_factored": c.is_fully_factored}
        finally:
            db.close()

    def test_refound_known_factor_accepted(self, client):
        """A factor re-found after it was divided out is known_factor, not 400."""
        composite = create_composite(self.PQR)
        assert self._submit(client, self.PQR, [self.P]).json()["factor_status"] == "new_factor"
        assert self._get_composite_state(composite["id"])["current_composite"] == self.QR

        response = self._submit(client, self.QR, [self.P], b1=60000)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "known_factor"
        # Cofactor untouched - the stale factor was not divided out again
        assert self._get_composite_state(composite["id"])["current_composite"] == self.QR

    def test_refound_plus_new_factor_in_batch(self, client):
        """A batch mixing a stale re-found factor with a genuinely new one
        processes the new factor instead of rejecting everything."""
        composite = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])

        response = self._submit(client, self.QR, [self.P, self.Q], b1=60000)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "new_factor"
        assert self._get_composite_state(composite["id"])["current_composite"] == self.R

    def test_stale_composite_string_resolved(self, client):
        """A client assigned the number before a factor landed submits the
        original string - previously rejected with 404."""
        composite = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])

        response = self._submit(client, self.PQR, [self.Q], b1=60000)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "new_factor"
        assert self._get_composite_state(composite["id"])["current_composite"] == self.R

    def test_stale_string_on_fully_factored_composite(self, client):
        """Late submission for a composite that got finished mid-run."""
        composite = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P, self.Q])
        assert self._get_composite_state(composite["id"])["is_fully_factored"] is True

        response = self._submit(client, self.PQR, [self.P], b1=60000)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "known_factor"

    def test_bogus_factor_still_rejected(self, client):
        create_composite(self.PQR)
        response = self._submit(client, self.PQR, [self.BOGUS])
        assert response.status_code == 400

    def test_bogus_factor_rejected_on_stale_string(self, client):
        """Stale tolerance must not weaken validation for garbage factors."""
        create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])

        response = self._submit(client, self.PQR, [self.BOGUS], b1=60000)

        assert response.status_code == 400

    def test_partially_stale_composite_factor(self, client):
        """Stale P*Q reported after P was already divided out: the surviving
        component Q must be divided out, not discarded as fully known."""
        composite = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])

        response = self._submit(client, self.PQR, [self.PQ], b1=60000)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "new_factor"
        # Q was extracted from P*Q via gcd and divided out, leaving R
        assert self._get_composite_state(composite["id"])["current_composite"] == self.R

    def test_intermediate_stale_state_resolved(self, client):
        """A worker assigned an already-reduced cofactor (Q*R) submits after
        yet another factor lands (current now R): the intermediate value is
        neither number nor current_composite, but its ancestry verifies, so
        it must resolve instead of returning 404."""
        composite = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])           # current -> Q*R
        self._submit(client, self.QR, [self.Q], b1=60000)  # current -> R

        # Late submission against the intermediate state, re-finding Q
        response = self._submit(client, self.QR, [self.Q], b1=70000)

        assert response.status_code == 200
        data = response.json()
        assert data["composite_id"] == composite["id"]
        assert data["factor_status"] == "known_factor"
        # Current cofactor unchanged by the stale re-find
        assert self._get_composite_state(composite["id"])["current_composite"] == self.R

    def test_ambiguous_intermediate_state_rejected(self, client):
        """If two composites share the same valid prior state (e.g. a
        cofactor was also registered as its own composite), the work cannot
        be attributed by guessing - it must be rejected, not applied to
        whichever composite has the lower ID."""
        X = "999983"
        QRX = "1000053000030979243"  # Q * R * X

        # Composite A: P*Q*R fully reduced to R (Q*R is a prior state)
        composite_a = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])
        self._submit(client, self.QR, [self.Q], b1=60000)

        # Composite B: Q*R*X reduced to Q*R via the API...
        composite_b = create_composite(QRX)
        self._submit(client, QRX, [X], b1=70000)
        # ...and on to Q directly in the DB: reducing through Q*R via the API
        # would itself (correctly) trip the collision check against A's
        # history, so the shared-prior-state scenario is built as data
        from conftest import get_test_engine
        from app.models.composites import Composite as CompositeModel
        from app.models.factors import Factor as FactorModel
        _, TestingSessionLocal = get_test_engine()
        db = TestingSessionLocal()
        try:
            comp = db.query(CompositeModel).filter(
                CompositeModel.id == composite_b["id"]
            ).one()
            comp.current_composite = self.Q
            comp.digit_length = len(self.Q)
            db.add(FactorModel(composite_id=comp.id, factor=self.R))
            db.commit()
        finally:
            db.close()

        # A late submission against Q*R now matches prior states of BOTH
        response = self._submit(client, self.QR, [self.Q], b1=90000)

        assert response.status_code == 409
        # Neither composite was touched
        assert self._get_composite_state(composite_a["id"])["current_composite"] == self.R
        assert self._get_composite_state(composite_b["id"])["current_composite"] == self.Q

    def test_exact_current_match_wins_over_prior_state(self, client):
        """A value that is composite A's CURRENT state and composite B's
        verified PRIOR state resolves to the exact match without paying for
        the ancestry scan: the factor verifies against A's actual current
        value, so dividing it out of A is arithmetically sound - only B's
        stale-credit attribution is forgone."""
        composite_b = create_composite(self.PQR)
        self._submit(client, self.PQR, [self.P])
        self._submit(client, self.QR, [self.Q], b1=60000)  # B.current -> R

        # Composite A's current value collides with B's prior state Q*R
        composite_a = create_composite(
            "880000000000000000077", current_composite=self.QR
        )

        response = self._submit(client, self.QR, [self.Q], b1=70000)

        assert response.status_code == 200
        data = response.json()
        assert data["composite_id"] == composite_a["id"]
        assert data["factor_status"] == "new_factor"
        # Q divided out of A (whose registered current state IS Q*R)
        assert self._get_composite_state(composite_a["id"])["current_composite"] == self.R
        # B untouched by the collision
        assert self._get_composite_state(composite_b["id"])["current_composite"] == self.R

    def test_factorless_import_stale_number_resolves(self, client):
        """A composite imported already-reduced (current set at import, no
        Factor rows) is invisible to the ancestry scan; stale work against
        its original number must resolve via the direct number lookup."""
        composite_b = create_composite(self.PQR, current_composite=self.QR)

        response = self._submit(client, self.PQR, [self.Q])

        assert response.status_code == 200
        data = response.json()
        assert data["composite_id"] == composite_b["id"]
        assert data["factor_status"] == "new_factor"
        assert self._get_composite_state(composite_b["id"])["current_composite"] == self.R

    def test_factorless_import_collision_is_ambiguous(self, client):
        """Composite B was imported already-reduced and its original number
        also collides with composite A's current value. The number lookup
        must run even though A is an exact current match - silently
        resolving to A would divide B's factor out of A."""
        composite_b = create_composite(self.PQR, current_composite=self.QR)
        composite_a = create_composite(
            "880000000000000000077", current_composite=self.PQR
        )

        response = self._submit(client, self.PQR, [self.P])

        assert response.status_code == 409
        # Neither composite was touched - without the collision check, P
        # would have been divided out of A
        assert self._get_composite_state(composite_a["id"])["current_composite"] == self.PQR
        assert self._get_composite_state(composite_b["id"])["current_composite"] == self.QR

    def test_zero_composite_rejected_cleanly(self, client):
        """Composite "0" passes integer validation; the ancestry arithmetic
        must not divide by it (was a ZeroDivisionError -> 500)."""
        create_composite(self.PQR)

        response = self._submit(client, "0", [self.P])

        assert response.status_code == 404


class TestWorkIdResolution:
    """work_id pins the composite by identity: the server-issued assignment
    UUID resolves the composite without string matching, so clients that
    send it never hit the ambiguity 409s or the legacy ancestry scan."""

    P = TestStaleFactorSubmissions.P
    Q = TestStaleFactorSubmissions.Q
    R = TestStaleFactorSubmissions.R
    PQR = TestStaleFactorSubmissions.PQR
    QR = TestStaleFactorSubmissions.QR
    _submit = TestStaleFactorSubmissions._submit
    _get_composite_state = TestStaleFactorSubmissions._get_composite_state

    def test_work_id_resolves_ambiguous_collision(self, client):
        """The factorless-import collision is a 409 for string resolution;
        a work_id for B disambiguates it."""
        composite_b = create_composite(self.PQR, current_composite=self.QR)
        create_composite("880000000000000000077", current_composite=self.PQR)
        work_id = create_work_assignment(composite_b["id"], "stale-test-client")

        # Sanity: without the work_id this exact submission is ambiguous
        assert self._submit(client, self.PQR, [self.Q]).status_code == 409

        response = self._submit(client, self.PQR, [self.Q], work_id=work_id)

        assert response.status_code == 200
        data = response.json()
        assert data["composite_id"] == composite_b["id"]
        assert data["factor_status"] == "new_factor"
        assert self._get_composite_state(composite_b["id"])["current_composite"] == self.R

    def test_work_id_state_mismatch_rejected(self, client):
        """A submitted number that is not a state of the pinned composite is
        a fabrication or client bug - reject rather than fall through to a
        string lookup that could misattribute it."""
        unrelated = create_composite("880000000000000000077")
        create_composite(self.PQR)
        work_id = create_work_assignment(unrelated["id"], "stale-test-client")

        response = self._submit(client, self.PQR, [self.P], work_id=work_id)

        assert response.status_code == 400
        assert self._get_composite_state(unrelated["id"])["current_composite"] == "880000000000000000077"

    def test_work_id_client_mismatch_rejected(self, client):
        """Another client's work_id fails closed (403): falling through to
        string resolution would let a known-invalid assignment be attributed
        to whatever composite the string happens to collide with."""
        composite = create_composite(self.PQR)
        work_id = create_work_assignment(composite["id"], "someone-else")

        response = self._submit(client, self.PQR, [self.P], work_id=work_id)

        assert response.status_code == 403
        assert self._get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_work_id_method_mismatch_rejected(self, client):
        """A work_id for a different method also fails closed."""
        composite = create_composite(self.PQR)
        work_id = create_work_assignment(
            composite["id"], "stale-test-client", method="pm1"
        )

        response = self._submit(client, self.PQR, [self.P], work_id=work_id)

        assert response.status_code == 403
        assert self._get_composite_state(composite["id"])["current_composite"] == self.PQR

    def test_unknown_work_id_falls_through(self, client):
        """An assignment that was already cleaned up must not fail the
        submission - the string lookups still resolve it."""
        composite = create_composite(self.PQR)

        response = self._submit(client, self.PQR, [self.P], work_id="wa-long-gone")

        assert response.status_code == 200
        assert response.json()["composite_id"] == composite["id"]

    def test_unknown_work_ids_do_not_fragment_duplicate_detection(self, client):
        """Two identical no-factor submissions carrying DIFFERENT nonexistent
        work_ids must dedup to one attempt. An unknown work_id is dropped from
        the work hash (like an unknown checksum); leaving it in let each
        garbage ID mint a fresh attempt and double-count the ECM curves."""
        create_composite(self.PQR)

        first = self._submit(
            client, self.PQR, [], work_id="made-up-1", sigma="3:55555"
        ).json()
        second = self._submit(
            client, self.PQR, [], work_id="made-up-2", sigma="3:55555"
        ).json()

        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]

    def test_unknown_work_id_matches_no_work_id(self, client):
        """A bogus work_id and no work_id are the same identity once the
        unknown ID is normalized away."""
        create_composite(self.PQR)

        first = self._submit(client, self.PQR, [], sigma="3:66666").json()
        second = self._submit(
            client, self.PQR, [], work_id="nonexistent", sigma="3:66666"
        ).json()

        assert second["factor_status"] == "duplicate"
        assert second["attempt_id"] == first["attempt_id"]

    def test_real_work_id_still_distinguishes(self, client):
        """A genuine assignment's work_id remains part of the identity:
        distinct real assignments are distinct work."""
        composite = create_composite(self.PQR)
        # First assignment finished; only one may be ACTIVE per composite
        # (partial unique index), and submit only checks work_id existence
        wid_a = create_work_assignment(
            composite["id"], "stale-test-client", work_id="wa-real-a",
            status="completed"
        )
        wid_b = create_work_assignment(
            composite["id"], "stale-test-client", work_id="wa-real-b"
        )

        first = self._submit(
            client, self.PQR, [], work_id=wid_a, sigma="3:77777"
        ).json()
        second = self._submit(
            client, self.PQR, [], work_id=wid_b, sigma="3:77777"
        ).json()

        assert first["factor_status"] != "duplicate"
        assert second["factor_status"] != "duplicate"
        assert second["attempt_id"] != first["attempt_id"]


class TestConcurrentSubmissionRace:
    """The pre-insert duplicate check can miss a row that another in-flight
    submission commits a moment later; the insert then hits the unique
    work_hash constraint. That race must dedup (200 duplicate), not 500.

    The race is simulated by stubbing the pre-check seam to miss while an
    identical attempt already exists - the StaticPool test DB can't run two
    real concurrent commits."""

    PQR = "1000073001431003663"

    def _submit(self, client, sigma="3:888888"):
        return client.post(
            "/api/v1/submit_result",
            json={
                "composite": self.PQR,
                "client_id": "race-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {"b1": 50000, "b2": 5000000, "curves": 100,
                               "sigma": sigma},
                "results": {"curves_completed": 100, "execution_time": 5.0},
            },
        )

    def test_lost_insert_race_dedups_instead_of_500(self, client, monkeypatch):
        from app.api.v1 import submit as submit_module

        create_composite(self.PQR)

        # Land the winning attempt normally
        first = self._submit(client).json()
        assert first["factor_status"] != "duplicate"

        # Now force the pre-check to miss, so the next identical submission
        # reaches the insert and trips the unique work_hash constraint
        monkeypatch.setattr(
            submit_module, "_find_duplicate_attempt",
            lambda db, work_hash, composite_id, residue_checksum: None
        )

        second = self._submit(client)

        assert second.status_code == 200  # was a 500 unique-violation
        data = second.json()
        assert data["factor_status"] == "duplicate"
        assert data["attempt_id"] == first["attempt_id"]


class TestCompositeLockDeletedRace:
    """A composite deleted (admin delete) between resolution and the
    factor-processing lock must surface as a clean 409, not a NoResultFound
    -> 500 from the lock helper's .one()."""

    def test_lock_composite_row_missing_raises_409(self):
        from fastapi import HTTPException
        from conftest import get_test_engine
        from app.api.v1.submit import _lock_composite_row

        _, TestingSessionLocal = get_test_engine()
        db = TestingSessionLocal()
        try:
            with pytest.raises(HTTPException) as exc_info:
                _lock_composite_row(db, 999999)
            assert exc_info.value.status_code == 409
        finally:
            db.close()
