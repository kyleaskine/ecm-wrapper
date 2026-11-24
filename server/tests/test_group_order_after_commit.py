"""
Group orders are calculated after /submit_result commits (PERFORMANCE_PLAN.md
item 10), from a background task, so PARI/GP no longer runs under the
composite's row lock.

TestClient runs background tasks before it returns, so the stored values can
be checked right after the request.
"""
import shutil
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.models.attempts import ECMAttempt
from app.models.factors import Factor
from app.services.factors import FactorService
from app.services.group_order import GroupOrderCalculator
from conftest import create_composite, get_test_engine

P = "10000000000000012363"  # prime
Q = "1000000007"            # prime
N = str(int(P) * int(Q))


def submit(client, method: str = "ecm", sigma: Optional[str] = "3:12345", factor: str = P,
           factor_sigma: Optional[str] = None):
    results: Dict[str, Any] = {"curves_completed": 10, "execution_time": 1.0,
                               "factor_found": factor,
                               "factors_found": [{"factor": factor, "sigma": factor_sigma or sigma}]}
    parameters: Dict[str, Any] = {"b1": 50000, "curves": 10}
    if method == "ecm":
        parameters.update(b2=5000000, parametrization=3, sigma=sigma)
    return client.post("/api/v1/submit_result", json={
        "composite": N, "client_id": "test-client", "method": method,
        "program": "gmp-ecm", "parameters": parameters, "results": results,
    })


def stored(factor: str = P) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """(group_order, factorization) of the factor's row, or None if no row."""
    _, SessionLocal = get_test_engine()
    with SessionLocal() as db:
        row = db.execute(text(
            "SELECT group_order, group_order_factorization FROM factors WHERE factor = :f"
        ), {"f": factor}).first()
    return None if row is None else (row[0], row[1])


@pytest.fixture
def commits_with_factor():
    """Records each commit of a session holding a Factor."""
    seen: List[str] = []

    def before_commit(session):
        if any(isinstance(obj, Factor) for obj in list(session.identity_map.values()) + list(session.new)):
            seen.append("commit")

    event.listen(Session, "before_commit", before_commit)
    yield seen
    event.remove(Session, "before_commit", before_commit)


class TestGroupOrderRunsAfterCommit:
    def test_ecm_group_order_is_stored_after_the_commit(self, client, commits_with_factor):
        create_composite(N)
        calls = []

        def fake(self, factor, sigma, parametrization):
            calls.append((factor, sigma, parametrization, list(commits_with_factor)))
            return ("1234", "2 * 617")

        with patch.object(GroupOrderCalculator, "calculate_group_order", fake):
            response = submit(client)

        assert response.status_code == 200
        assert response.json()["factor_status"] == "new_factor"
        assert len(calls) == 1
        factor, sigma, parametrization, commits_before_gp = calls[0]
        assert (factor, sigma, parametrization) == (P, "12345", 3)
        assert commits_before_gp, "PARI/GP ran before the factor's transaction committed"
        assert stored() == ("1234", "2 * 617")

    def test_p1_factor_gets_p_minus_1(self, client):
        create_composite(N)
        with patch.object(GroupOrderCalculator, "calculate_p1_order",
                          return_value=("10000000000000012362", "2 * 3 * 7")) as p1:
            response = submit(client, method="pm1", sigma=None)

        assert response.status_code == 200
        p1.assert_called_once_with(P, "pm1")
        assert stored() == ("10000000000000012362", "2 * 3 * 7")

    def test_failed_calculation_leaves_null(self, client):
        create_composite(N)
        with patch.object(GroupOrderCalculator, "calculate_group_order", return_value=None):
            response = submit(client)

        assert response.status_code == 200
        assert stored() == (None, None)

    def test_known_factor_is_not_recalculated(self, client):
        create_composite(N)
        with patch.object(GroupOrderCalculator, "calculate_group_order",
                          return_value=("1234", None)) as calc:
            first = submit(client, sigma="3:1")
            again = submit(client, sigma="3:2")  # same factor again
        assert first.status_code == 200 and first.json()["factor_status"] == "new_factor"
        assert again.status_code == 200 and again.json()["factor_status"] == "known_factor"
        assert calc.call_count == 1

    def test_factor_sigma_prefix_sets_the_curve_type(self, client):
        # The submission says parametrization 3, but this factor's own sigma
        # says its curve was Montgomery (param 1)
        create_composite(N)
        with patch.object(GroupOrderCalculator, "calculate_group_order",
                          return_value=("1234", None)) as calc:
            response = submit(client, sigma="3:12345", factor_sigma="1:777")

        assert response.status_code == 200
        calc.assert_called_once_with(P, "777", 1)

    def test_rolled_back_submission_schedules_nothing(self, client):
        create_composite(N)
        with patch.object(GroupOrderCalculator, "calculate_group_order",
                          return_value=("1234", None)) as calc, \
             patch("app.api.v1.submit.FactorService.verify_factorization",
                   side_effect=RuntimeError("boom")):
            response = submit(client)

        assert response.status_code == 500
        calc.assert_not_called()
        assert stored() is None


@pytest.mark.skipif(shutil.which("gp") is None, reason="PARI/GP not installed")
class TestWithRealGp:
    def test_ecm_group_order_is_a_curve_order_mod_p(self, client):
        create_composite(N)
        assert submit(client).status_code == 200

        row = stored()
        assert row is not None
        order, factorization = row
        assert order is not None and factorization is not None
        p = int(P)
        assert abs(int(order) - (p + 1)) <= 2 * int(p ** 0.5) + 1  # Hasse bound
        product = 1
        for term in factorization.split(" * "):
            base, _, exp = term.partition("^")
            product *= int(base) ** int(exp or 1)
        assert product == int(order)

    def test_p1_order_is_p_minus_1(self, client):
        create_composite(N)
        assert submit(client, method="pm1", sigma=None).status_code == 200
        row = stored()
        assert row is not None and row[0] == str(int(P) - 1)


class TestMissingGroupOrderJob:
    """Inputs for recalculating a stored factor (scripts/backfill_group_orders.py)."""

    def _factor(self, db, method: str, parametrization: Optional[int], sigma: Optional[str],
                group_order: Optional[str] = None) -> Factor:
        created = create_composite(N)
        attempt = ECMAttempt(composite_id=created["id"], client_id="c", method=method, b1=50000,
                             parametrization=parametrization, curves_requested=1,
                             curves_completed=1, program="gmp-ecm")
        db.add(attempt)
        db.flush()
        factor = Factor(composite_id=created["id"], factor=P, found_by_attempt_id=attempt.id,
                        sigma=sigma, group_order=group_order)
        db.add(factor)
        db.commit()
        return factor

    def test_ecm_factor_uses_its_attempts_parametrization(self, db_session):
        factor = self._factor(db_session, "ecm", 1, "777")
        job = FactorService.missing_group_order_job(db_session, factor)
        assert job is not None
        assert (job.factor, job.sigma, job.parametrization) == (P, "777", 1)

    def test_p1_factor_uses_its_attempts_method(self, db_session):
        factor = self._factor(db_session, "pp1", None, None)
        job = FactorService.missing_group_order_job(db_session, factor)
        assert job is not None
        assert (job.sigma, job.method) == (None, "pp1")

    def test_factor_with_a_group_order_needs_nothing(self, db_session):
        factor = self._factor(db_session, "ecm", 3, "777", group_order="1234")
        assert FactorService.missing_group_order_job(db_session, factor) is None

