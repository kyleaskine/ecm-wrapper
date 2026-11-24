"""
The admin bulk t-level recalculations use the same calculation as a
submission, including leftover stage-1 curves from a partial stage 2. They
used to read unsuperseded attempts only, which lowered such composites.
"""
from types import SimpleNamespace
from unittest.mock import patch

from app.api.v1.admin import maintenance
from app.models.attempts import ECMAttempt
from app.models.composites import Composite
from app.services.t_level_calculator import TLevelCalculator
from conftest import create_composite, get_test_engine

B1 = 11000000
B2 = 35133391030


def composite_with_partial_stage2(db, target_t_level=None) -> int:
    """3000 stage-1 curves, of which a stage 2 ran 1000."""
    created = create_composite("1234567890123456789012345", current_t_level=0.0)
    composite = db.get(Composite, created["id"])
    composite.target_t_level = target_t_level
    stage1 = ECMAttempt(composite_id=composite.id, client_id="gpu", method="ecm", b1=B1, b2=0,
                        parametrization=3, curves_requested=3000, curves_completed=3000,
                        program="gmp-ecm")
    stage2 = ECMAttempt(composite_id=composite.id, client_id="cpu", method="ecm", b1=B1, b2=B2,
                        parametrization=3, curves_requested=1000, curves_completed=1000,
                        program="gmp-ecm")
    db.add_all([stage1, stage2])
    db.flush()
    stage1.superseded_by = stage2.id
    db.commit()
    return composite.id


def t_level(*classes) -> float:
    attempts = [SimpleNamespace(id=0, method="ecm", curves_completed=c, b1=B1, b2=b2,
                                parametrization=3) for c, b2 in classes]
    result = TLevelCalculator().get_current_t_level_from_attempts(attempts)
    assert result is not None
    return result


def expected():
    with_leftover = t_level((1000, B2), (2000, 0))
    assert with_leftover > t_level((1000, B2))  # the leftover matters
    return with_leftover


def stored_t_level(composite_id: int) -> float:
    _, SessionLocal = get_test_engine()
    with SessionLocal() as db:
        composite = db.get(Composite, composite_id)
        assert composite is not None
        return composite.current_t_level


def test_calculate_t_levels_recalculate_all_counts_leftovers(admin_client, db_session):
    composite_id = composite_with_partial_stage2(db_session, target_t_level=40.0)

    response = admin_client.post("/api/v1/admin/composites/calculate-t-levels",
                                 params={"recalculate_all": True})

    assert response.status_code == 200
    assert stored_t_level(composite_id) == expected()


def test_calculate_t_levels_for_new_composites_counts_leftovers(admin_client, db_session):
    composite_id = composite_with_partial_stage2(db_session, target_t_level=None)

    response = admin_client.post("/api/v1/admin/composites/calculate-t-levels")

    assert response.status_code == 200
    assert stored_t_level(composite_id) == expected()


def test_background_recalculate_all_counts_leftovers(db_session):
    composite_id = composite_with_partial_stage2(db_session, target_t_level=40.0)
    _, SessionLocal = get_test_engine()

    with patch("app.database.SessionLocal", SessionLocal):
        maintenance._recalculate_all_t_levels_background()

    assert maintenance._recalculation_status["result"]["status"] == "completed"
    assert stored_t_level(composite_id) == expected()
