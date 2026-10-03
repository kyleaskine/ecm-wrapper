"""Claim deadlines follow the selected residue's B1, with explicit overrides."""
from datetime import datetime, timedelta

import pytest

from conftest import admin_auth, create_composite
from app.models.residues import ECMResidue


def make_residue(db_session, b1):
    composite = create_composite('1234567890123456789012345678901234567891')
    residue = ECMResidue(
        composite_id=composite['id'], client_id='producer', b1=b1,
        parametrization=3, curve_count=2304, storage_path='/unused/residue.txt',
        file_size_bytes=1234, checksum='a' * 64, status='available',
    )
    db_session.add(residue)
    db_session.commit()
    return residue.id


@pytest.mark.parametrize('b1, hours', [
    (50_000, 24),
    (849_999_999, 24),
    (850_000_000, 48),
    (2_899_999_999, 48),
    (2_900_000_000, 120),
    (7_600_000_000, 120),
])
def test_unfiltered_claim_uses_actual_b1(client, db_session, b1, hours):
    residue_id = make_residue(db_session, b1)

    # No min/max B1 hints: the selected record determines the duration.
    response = client.get('/api/v1/residues/work', headers={'X-Client-ID': 'consumer'})

    assert response.status_code == 200
    assert response.json()['residue_id'] == residue_id
    db_session.expire_all()
    residue = db_session.get(ECMResidue, residue_id)
    assert residue.claimed_by == 'consumer'
    assert residue.expires_at - residue.claimed_at == timedelta(hours=hours)
    assert datetime.fromisoformat(response.json()['expires_at']) == residue.expires_at


@pytest.mark.parametrize('hours', [1, 48, 336])
def test_explicit_claim_timeout_is_preserved(client, db_session, hours):
    residue_id = make_residue(db_session, 2_900_000_000)
    response = client.get(
        '/api/v1/residues/work', headers={'X-Client-ID': 'consumer'},
        params={'claim_timeout_hours': hours},
    )
    assert response.status_code == 200
    db_session.expire_all()
    residue = db_session.get(ECMResidue, residue_id)
    assert residue.expires_at - residue.claimed_at == timedelta(hours=hours)


@pytest.mark.parametrize('hours', [0, 337])
def test_invalid_override_does_not_claim(client, db_session, hours):
    residue_id = make_residue(db_session, 2_900_000_000)
    response = client.get(
        '/api/v1/residues/work', headers={'X-Client-ID': 'consumer'},
        params={'claim_timeout_hours': hours},
    )
    assert response.status_code == 422
    db_session.expire_all()
    assert db_session.get(ECMResidue, residue_id).status == 'available'


def test_expired_claim_stays_owned_until_admin_cleanup(client, db_session):
    residue_id = make_residue(db_session, 2_900_000_000)
    client.get('/api/v1/residues/work', headers={'X-Client-ID': 'original'})
    db_session.expire_all()
    residue = db_session.get(ECMResidue, residue_id)
    residue.expires_at = datetime.utcnow() - timedelta(minutes=10)
    db_session.commit()

    response = client.get('/api/v1/residues/work', headers={'X-Client-ID': 'other'})
    assert response.status_code == 200
    assert response.json()['residue_id'] is None
    db_session.expire_all()
    assert residue.status == 'claimed'
    assert residue.claimed_by == 'original'

    with admin_auth():
        cleanup = client.post('/api/v1/admin/residues/cleanup')
    assert cleanup.status_code == 200
    assert cleanup.json()['claims_released'] == 1
    db_session.expire_all()
    assert residue.status == 'available'
    assert residue.claimed_by is None

    response = client.get('/api/v1/residues/work', headers={'X-Client-ID': 'other'})
    assert response.json()['residue_id'] == residue_id
    db_session.expire_all()
    assert residue.claimed_by == 'other'
    assert residue.expires_at - residue.claimed_at == timedelta(days=5)
