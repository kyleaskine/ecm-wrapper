"""Stage-1 assignments receive the same B1-based deadlines as residues."""
from datetime import datetime, timedelta

import pytest

from conftest import create_composite
from app.models.work_assignments import WorkAssignment


def make_composite(current_t=0.0):
    return create_composite(
        '1' * 195 + '3', current_t_level=current_t, target_t_level=85.0,
    )


def assert_assignment_duration(response, db_session, days):
    assert response.status_code == 200
    data = response.json()
    assert data['work_id']
    assignment = db_session.get(WorkAssignment, data['work_id'])
    assert assignment is not None
    assert assignment.expires_at - assignment.assigned_at == timedelta(days=days)
    assert datetime.fromisoformat(data['expires_at']) == assignment.expires_at


@pytest.mark.parametrize('current_t, days', [
    (0.0, 1),
    (59.99, 1),
    (60.0, 2),
    (64.99, 2),
    (65.0, 5),
    (69.99, 5),
    (80.0, 5),
])
def test_stage1_uses_current_level_not_server_suggestion(client, db_session, current_t, days):
    make_composite(current_t)
    response = client.get('/api/v1/ecm-work', params={
        'client_id': 'gpu-producer', 'stage1_only': True,
    })
    assert_assignment_duration(response, db_session, days)
    # The server's general suggestion for this input is below 850M even
    # when the stage-1 producer will select 850M or 2.9B from current t-level.
    assignment = db_session.get(WorkAssignment, response.json()['work_id'])
    assert assignment.b1 < 850_000_000


@pytest.mark.parametrize('b1, current_t, days', [
    (849_999_999, 0.0, 1),
    (850_000_000, 0.0, 2),
    (2_899_999_999, 0.0, 2),
    (2_900_000_000, 0.0, 5),
    (7_600_000_000, 0.0, 5),
    (43_000_000, 65.0, 1),
])
def test_explicit_b1_overrides_automatic_selection(client, db_session, b1, current_t, days):
    make_composite(current_t)
    response = client.get('/api/v1/ecm-work', params={
        'client_id': 'gpu-producer', 'stage1_only': True, 'requested_b1': b1,
    })
    assert_assignment_duration(response, db_session, days)


@pytest.mark.parametrize('days', [1, 2, 5, 14])
def test_explicit_timeout_remains_an_exact_override(client, db_session, days):
    make_composite(65.0)
    response = client.get('/api/v1/ecm-work', params={
        'client_id': 'gpu-producer', 'stage1_only': True, 'timeout_days': days,
    })
    assert_assignment_duration(response, db_session, days)


@pytest.mark.parametrize('invalid', [
    {'timeout_days': 0}, {'timeout_days': -1}, {'requested_b1': 0},
])
def test_invalid_request_does_not_assign_work(client, db_session, invalid):
    make_composite()
    response = client.get('/api/v1/ecm-work', params={
        'client_id': 'gpu-producer', 'stage1_only': True, **invalid,
    })
    assert response.status_code == 422
    assert db_session.query(WorkAssignment).count() == 0


def test_regular_ecm_request_still_works_without_stage1_hints(client, db_session):
    make_composite()
    response = client.get('/api/v1/ecm-work', params={'client_id': 'cpu-worker'})
    assert_assignment_duration(response, db_session, 1)
