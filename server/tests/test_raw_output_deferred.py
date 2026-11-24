"""
raw_output is deferred on ECMAttempt (PERFORMANCE_PLAN.md item 5).

It averages ~440 KB per attempt and nothing displays it, so no SELECT issued by
a submission or a dashboard page may name the column. It must still be stored
on insert and cleared by the admin log cleanup.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import event, text

from conftest import admin_auth, create_composite, get_test_engine
from app.models.projects import Project, ProjectComposite
from test_residue_autocomplete import (
    COMPOSITE as RESIDUE_COMPOSITE, CURVES, create_stage2_setup, get_attempt,
    submit_stage2_result,
)

RAW = "Step 1 took 12345ms\n" * 50


@contextmanager
def selects() -> Iterator[List[str]]:
    """Collect the SELECT statements sent while the block runs."""
    engine, _ = get_test_engine()
    seen: List[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


def naming_raw_output(statements: List[str]) -> List[str]:
    # Note: Query.count() on the ECMAttempt entity also names the column (it
    # wraps every column in a subquery); count ECMAttempt.id instead.
    return [s for s in statements if "raw_output" in s]


def submit(client, composite: str, b1: int = 50000, sigma: str = "3:12345",
           factor: Optional[str] = None):
    results: Dict[str, Any] = {"curves_completed": 10, "execution_time": 1.0}
    if factor is not None:
        results["factor_found"] = factor
    return client.post("/api/v1/submit_result", json={
        "composite": composite,
        "client_id": "test-client",
        "method": "ecm",
        "program": "gmp-ecm",
        "parameters": {"b1": b1, "b2": b1 * 100, "curves": 10, "sigma": sigma},
        "results": results,
        "raw_output": RAW,
    })


def stored_raw_output(attempt_id: int):
    _, Session = get_test_engine()
    with Session() as db:
        return db.execute(text("SELECT raw_output FROM ecm_attempts WHERE id = :i"),
                          {"i": attempt_id}).scalar()


class TestSubmitDoesNotLoadRawOutput:
    def test_new_submission(self, client):
        composite = create_composite("12345678901234567890", digit_length=20)
        submit(client, composite["current_composite"], sigma="3:1")  # history to recalculate over
        with selects() as seen:
            response = submit(client, composite["current_composite"], sigma="3:2")
        assert response.status_code == 200
        assert naming_raw_output(seen) == []
        assert stored_raw_output(response.json()["attempt_id"]) == RAW

    def test_factor_submission(self, client):
        composite = create_composite("15", digit_length=2)
        with selects() as seen:
            response = submit(client, composite["current_composite"], factor="3")
        assert response.status_code == 200
        assert response.json()["factor_status"] == "new_factor"
        assert naming_raw_output(seen) == []

    def test_duplicate_submission(self, client):
        composite = create_composite("12345678901234567890", digit_length=20)
        first = submit(client, composite["current_composite"], sigma="3:7")
        with selects() as seen:
            again = submit(client, composite["current_composite"], sigma="3:7")
        assert again.json()["attempt_id"] == first.json()["attempt_id"]
        assert naming_raw_output(seen) == []

    def test_stage2_submission_completing_a_residue(self, client):
        composite = create_composite(RESIDUE_COMPOSITE, digit_length=len(RESIDUE_COMPOSITE))
        setup = create_stage2_setup(composite["id"])
        with selects() as seen:
            response = submit_stage2_result(client, curves_completed=CURVES)
        assert response.status_code == 200
        data = response.json()
        assert data["residue_completed"] is True
        assert get_attempt(setup["stage1_attempt_id"])["superseded_by"] == data["attempt_id"]
        assert naming_raw_output(seen) == []


class TestPagesDoNotLoadRawOutput:
    def test_public_and_admin_pages(self, client):
        composite = create_composite("12345678901234567890", digit_length=20)
        factored = create_composite("15", digit_length=2)
        submit(client, composite["current_composite"], sigma="3:1")
        submit(client, factored["current_composite"], factor="3")
        _, Session = get_test_engine()
        with Session() as db:
            project = Project(name="test-project")
            db.add(project)
            db.flush()
            db.add(ProjectComposite(project_id=project.id, composite_id=composite["id"]))
            db.commit()
            project_id = project.id

        public = [
            "/api/v1/dashboard/",
            # filtered: the attempt count joins composites and projects
            "/api/v1/dashboard/?priority=5",
            f"/api/v1/dashboard/?project={project_id}",
            f"/api/v1/dashboard/?priority=5&project={project_id}",
            "/api/v1/dashboard/curves?group_by_composite=false&client_id=test-client",
            "/api/v1/dashboard/testing-status",
            "/api/v1/dashboard/p1-testing-status",
            "/api/v1/dashboard/curves",
            "/api/v1/dashboard/curves?group_by_composite=false",
            "/api/v1/dashboard/factors",
            "/api/v1/dashboard/leaderboard",
            "/api/v1/dashboard/residue-status",
            f"/api/v1/dashboard/composites/{composite['id']}/details",
            f"/api/v1/dashboard/composites/{composite['id']}/attempts-fragment",
            f"/api/v1/dashboard/composites/{factored['id']}/details",
            f"/api/v1/stats/{composite['current_composite']}",
            "/api/v1/factors",
        ]
        admin = [
            "/api/v1/admin/dashboard",
            "/api/v1/admin/residue-status",
            "/api/v1/admin/recent-composites",
            "/api/v1/admin/outstanding-work",
            f"/api/v1/admin/composites/{composite['id']}/details",
            f"/api/v1/admin/composites/{composite['id']}/attempts-fragment",
        ]
        with selects() as seen:
            for url in public:
                assert client.get(url).status_code == 200, url
            with admin_auth():
                for url in admin:
                    assert client.get(url).status_code == 200, url
        assert seen, "no SELECTs captured"
        assert naming_raw_output(seen) == []


class TestLogCleanup:
    def test_cleanup_clears_old_raw_output_only(self, client):
        composite = create_composite("12345678901234567890", digit_length=20)
        old = submit(client, composite["current_composite"], sigma="3:1").json()["attempt_id"]
        new = submit(client, composite["current_composite"], sigma="3:2").json()["attempt_id"]
        _, Session = get_test_engine()
        with Session() as db:
            forty_days_ago = datetime.utcnow() - timedelta(days=40)
            db.execute(text("UPDATE ecm_attempts SET created_at = :t WHERE id = :i"),
                       {"t": forty_days_ago.strftime("%Y-%m-%d %H:%M:%S.%f"), "i": old})
            db.commit()

        with admin_auth():
            response = client.post("/api/v1/admin/ecm-attempts/cleanup-logs?days=30")
        assert response.status_code == 200
        assert response.json()["rows_cleared"] == 1
        assert stored_raw_output(old) is None
        assert stored_raw_output(new) == RAW
