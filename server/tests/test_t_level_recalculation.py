"""
recalculate_composite_t_level reads a composite's work in two queries
(PERFORMANCE_PLAN.md item 6): curve totals per class, and leftover stage-1
curves from partial supersession.

Expected values come from get_current_t_level_from_attempts on the equivalent
plain attempt list, so these tests pin down which curves are credited, not the
probability model.
"""
from types import SimpleNamespace
from typing import List, Optional

import pytest
from sqlalchemy import event

from app.models.attempts import ECMAttempt
from app.models.composites import Composite
from app.services.t_level_calculator import TLevelCalculator
from conftest import create_composite, get_test_engine

B1 = 11000000
B2 = 35133391030


@pytest.fixture
def calculator():
    calc = TLevelCalculator()
    calc.use_library = True
    calc.t_level_available = True
    return calc


@pytest.fixture
def composite(db_session):
    created = create_composite("1234567890123456789012345")
    return db_session.get(Composite, created["id"])


def add(db, composite_id: int, curves: int, b2: Optional[int] = B2, param: Optional[int] = 3,
        b1: int = B1, method: str = "ecm", superseded_by: Optional[int] = None) -> int:
    attempt = ECMAttempt(composite_id=composite_id, client_id="test", method=method,
                         b1=b1, b2=b2, parametrization=param, curves_requested=curves,
                         curves_completed=curves, program="gmp-ecm",
                         superseded_by=superseded_by)
    db.add(attempt)
    db.flush()
    return attempt.id


def supersede(db, attempt_id: int, by: int) -> None:
    db.get(ECMAttempt, attempt_id).superseded_by = by
    db.flush()


def expected(calculator, *classes) -> float:
    """t-level of plain (curves, b1, b2, param) classes."""
    attempts = [SimpleNamespace(id=0, method="ecm", curves_completed=c, b1=b1, b2=b2,
                                parametrization=p) for c, b1, b2, p in classes]
    result = calculator.get_current_t_level_from_attempts(attempts)
    assert result is not None
    return result


def test_unsuperseded_attempts_are_summed_per_class(db_session, calculator, composite):
    for _ in range(3):
        add(db_session, composite.id, 100)
    add(db_session, composite.id, 50, b2=None, param=1)
    add(db_session, composite.id, 7, method="pm1")  # not ECM
    add(db_session, composite.id, 0)                # no curves

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (300, B1, B2, 3), (50, B1, None, 1))


def test_partial_stage2_leaves_stage1_curves_at_b1_only(db_session, calculator, composite):
    stage1 = add(db_session, composite.id, 3000, b2=0)
    stage2 = add(db_session, composite.id, 1000)
    supersede(db_session, stage1, stage2)

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (1000, B1, B2, 3), (2000, B1, 0, 3))


def test_full_stage2_leaves_nothing(db_session, calculator, composite):
    stage1 = add(db_session, composite.id, 1000, b2=0)
    stage2 = add(db_session, composite.id, 1000)
    supersede(db_session, stage1, stage2)

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (1000, B1, B2, 3))


def test_leftover_merges_with_unsuperseded_stage1_of_same_class(db_session, calculator, composite):
    add(db_session, composite.id, 500, b2=0)  # stage 1 still waiting for its stage 2
    stage1 = add(db_session, composite.id, 3000, b2=0)
    stage2 = add(db_session, composite.id, 1000)
    supersede(db_session, stage1, stage2)

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (2500, B1, 0, 3), (1000, B1, B2, 3))


def test_leftover_keeps_param_0(db_session, calculator, composite):
    stage1 = add(db_session, composite.id, 300, b2=0, param=0)
    stage2 = add(db_session, composite.id, 100, param=0)
    supersede(db_session, stage1, stage2)

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (100, B1, B2, 0), (200, B1, 0, 0))
    assert result != expected(calculator, (100, B1, B2, 0), (200, B1, 0, 3))


def test_superseded_stage2_orphan_earns_no_leftover(db_session, calculator, composite):
    # Residue of 3000 curves: A's stage 2 ran all 3000 but B completed the
    # residue with 1000, and the orphan sweep superseded A with B. A's curves
    # are the stage 1's curves; only the stage 1 earns the 2000 leftover.
    stage1 = add(db_session, composite.id, 3000, b2=0)
    orphan = add(db_session, composite.id, 3000)
    winner = add(db_session, composite.id, 1000)
    supersede(db_session, stage1, winner)
    supersede(db_session, orphan, winner)

    result = calculator.recalculate_composite_t_level(db_session, composite)

    assert result == expected(calculator, (1000, B1, B2, 3), (2000, B1, 0, 3))


def test_statement_count_does_not_grow_with_attempts(db_session, calculator, composite):
    engine, _ = get_test_engine()

    def selects_for_recalculation() -> int:
        seen: List[str] = []

        def record(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                seen.append(statement)

        event.listen(engine, "before_cursor_execute", record)
        try:
            calculator.recalculate_composite_t_level(db_session, composite)
        finally:
            event.remove(engine, "before_cursor_execute", record)
        return len(seen)

    stage1 = add(db_session, composite.id, 300, b2=0)
    supersede(db_session, stage1, add(db_session, composite.id, 100))
    few = selects_for_recalculation()

    for i in range(25):
        stage1 = add(db_session, composite.id, 300, b2=0, b1=B1 + i)
        supersede(db_session, stage1, add(db_session, composite.id, 100, b1=B1 + i))
    many = selects_for_recalculation()

    assert few == many == 2
