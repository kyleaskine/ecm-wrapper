"""
Tests for the in-process t-level calculation (app/services/t_level_library.py)
and for how TLevelCalculator chooses between it and the t-level binary.

The expected values are what the deployed binary (server/bin/t-level, version
0.9.6) printed for the same input. Each case's comment gives that input. The
in-process path must reproduce them exactly, or stored t-levels would shift
when the backend changes.

There is deliberately no skip when the package is missing. It is in
requirements.txt, so if it cannot be imported these tests should fail, not
pass silently.
"""
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from app.models.attempts import ECMAttempt
from app.models.composites import Composite
from app.services import t_level_library as lib
from app.services.t_level_calculator import TLevelCalculator
from conftest import create_composite

SERVER_DIR = Path(__file__).parent.parent


# (label, classes, starting_t_level, expected) - expected is the binary's output
BINARY_CASES = [
    # echo "100@11e6,p=1" | t-level
    ("default B2, table B1", [(100, 11000000, None, 1)], 0.0, 34.044),
    # echo "50@1234567,p=1" | t-level
    ("default B2, computed B1", [(50, 1234567, None, 1)], 0.0, 27.737),
    # echo "3072@11e7,0,p=3" | t-level
    ("stage 1 only", [(3072, 110000000, 0, 3)], 0.0, 41.573),
    # echo "1000@11e6,35133391030,p=1" | t-level
    ("explicit B2", [(1000, 11000000, 35133391030, 1)], 0.0, 40.653),
    # echo "200@3e6,5706890290,p=N" | t-level
    ("param 0", [(200, 3000000, 5706890290, 0)], 0.0, 33.721),
    ("param 1", [(200, 3000000, 5706890290, 1)], 0.0, 33.341),
    ("param 2", [(200, 3000000, 5706890290, 2)], 0.0, 33.721),
    ("param 3", [(200, 3000000, 5706890290, 3)], 0.0, 33.24),
    # echo "5376@11e7,0,p=3;2304@11e7,776601600000,p=3;500@43e6,p=1;3000@11e6,35133391030,p=1" | t-level
    ("mixed classes",
     [(5376, 110000000, 0, 3), (2304, 110000000, 776601600000, 3),
      (500, 43000000, None, 1), (3000, 11000000, 35133391030, 1)],
     0.0, 49.571),
    # echo "100@11e6,35133391030,p=1" | t-level -w 45.0
    ("starting t-level", [(100, 11000000, 35133391030, 1)], 45.0, 45.046),
    # echo "800@3e6,p=3" | t-level -w 37.25
    ("non-integer starting t-level", [(800, 3000000, None, 3)], 37.25, 38.584),
    # echo "10@50000,p=1" | t-level -w 5.0   (and -w 100.0)
    ("starting t-level at lower bound", [(10, 50000, None, 1)], 5.0, 18.02),
    ("starting t-level at upper bound", [(10, 50000, None, 1)], 100.0, 100.0),
]


# --- t_level_library: equivalence with the binary -------------------------


@pytest.mark.parametrize(
    "classes,starting,expected",
    [case[1:] for case in BINARY_CASES],
    ids=[case[0] for case in BINARY_CASES],
)
def test_matches_binary_output(classes, starting, expected):
    assert lib.calculate_t_level(classes, starting) == expected


@pytest.mark.parametrize("starting", [3.2, 4.999, 100.5])
def test_starting_t_level_outside_cli_range_fails(starting):
    # The CLI exits 1 for -w outside 5-100 (binary: "t-level -w 3.2" -> no output).
    # TLevelCalculator handles such priors itself; see the calculator tests.
    assert lib.calculate_t_level([(100, 11000000, None, 1)], starting) is None


@pytest.mark.parametrize("bad_class", [
    (100, 11000000, -1, 1),     # negative B2: fails the CLI's grammar
    (100, 11000000, None, 5),   # param outside the grammar's 0-4
    (100, 11000000, None, 4),   # param 4: grammar accepts it, rho divides by zero
    (100, 0, None, 1),          # B1 must be positive
    (-5, 11000000, None, 1),    # negative curves: fails the grammar
])
def test_input_the_cli_rejects_fails_the_whole_calculation(bad_class):
    good = (100, 11000000, None, 1)
    assert lib.calculate_t_level([good, bad_class]) is None


def test_zero_curve_classes_are_dropped():
    # The CLI's parser drops 0-curve lines rather than rejecting them
    assert lib.calculate_t_level([(0, 3000000, None, 1), (100, 11000000, None, 1)]) == 34.044


def test_numeric_inputs_are_normalised_to_int():
    # Floats with integral values must give the same answer as ints
    float_classes: Any = [(100.0, 11000000.0, 35133391030.0, 1.0)]
    assert lib.calculate_t_level(float_classes) == \
        lib.calculate_t_level([(100, 11000000, 35133391030, 1)])


# --- t_level_library: caching, memory, threads ----------------------------


def test_cached_vectors_are_reusable():
    # The package returns a one-shot map on its table path. If that were cached
    # as-is, the second call would see an exhausted iterator and return t0.
    classes = [(100, 11000000, None, 1)]
    first = lib.calculate_t_level(classes)
    misses = lib.cache_info().misses
    second = lib.calculate_t_level(classes)
    assert first == second == 34.044
    assert lib.cache_info().misses == misses  # served from the cache


def test_package_caches_are_cleared_after_every_calculation():
    import t_level
    from t_level import rho
    # A cold class (no other test uses it), plus a prior so the -w conversion
    # populates t_level's own cache as well as rho's
    lib.calculate_t_level([(77, 1234577, 987654321, 1)], 41.5)
    assert rho.ecmprob.cache_info().currsize == 0
    assert rho.dickmanrho.cache_info().currsize == 0
    assert t_level.get_t_level_curves_string.cache_info().currsize == 0
    # ...and the probability vectors, the one cache that is kept, are still there
    assert lib.cache_info().currsize > 0


def test_cache_intercepts_the_packages_own_calls():
    import t_level
    # The replacement is reached through the name the package's code looks up
    assert t_level.get_failure_probabilities.__globals__["get_ecm_success_probs"] \
        is t_level.get_ecm_success_probs
    misses = lib.cache_info().misses
    lib.calculate_t_level([(5, 7654321, 123456789, 3)])  # cold class
    assert lib.cache_info().misses > misses


def test_concurrent_calculations_match_sequential():
    inputs = [case[1:3] for case in BINARY_CASES]
    expected = [case[3] for case in BINARY_CASES]
    results: List[Optional[float]] = [None] * len(inputs)

    def run(i):
        results[i] = lib.calculate_t_level(*inputs[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == expected


# --- TLevelCalculator -----------------------------------------------------


def _attempt(curves, b1, b2, param, method="ecm"):
    return SimpleNamespace(id=1, method=method, curves_completed=curves, b1=b1, b2=b2,
                           parametrization=param)


@pytest.fixture
def calculator():
    # Pin the backend so the tests do not depend on the local .env
    calc = TLevelCalculator()
    calc.use_library = True
    calc.t_level_available = True
    return calc


def test_calculator_aggregates_and_defaults_param_to_3(calculator):
    attempts = [
        _attempt(2000, 110000000, 0, None),   # missing param means 3
        _attempt(3376, 110000000, 0, 3),
        _attempt(2304, 110000000, 776601600000, 3),
        _attempt(250, 43000000, None, 1),
        _attempt(250, 43000000, None, 1),
        _attempt(3000, 11000000, 35133391030, 1),
        _attempt(99, 11000000, None, 1, method="pm1"),  # not ECM: ignored
        _attempt(0, 11000000, None, 1),                 # no curves: ignored
    ]
    # Aggregates to the "mixed classes" case above
    assert calculator.get_current_t_level_from_attempts(attempts) == 49.571


def test_calculator_passes_starting_t_level(calculator):
    attempts = [_attempt(100, 11000000, 35133391030, 1)]
    assert calculator.get_current_t_level_from_attempts(attempts, 45.0) == 45.046


@pytest.mark.parametrize("prior,expected", [
    (3.2, 34.044),   # below the CLI's 5: the attempts' t-level (100@11e6 alone is t34.044)
    (120.0, 120.0),  # above 100: cannot be expressed as curves; the prior dominates
])
def test_calculator_handles_priors_the_cli_rejects(calculator, prior, expected):
    # The server used to keep only the prior here, discarding all ECM work
    attempts = [_attempt(100, 11000000, None, 1)]
    assert calculator.get_current_t_level_from_attempts(attempts, prior) == expected


def test_calculator_returns_none_for_input_the_cli_rejects(calculator):
    attempts = [_attempt(100, 11000000, -5, 1)]  # negative B2
    assert calculator.get_current_t_level_from_attempts(attempts, 40.0) is None


def test_calculator_does_not_retry_input_errors_with_binary(calculator):
    # The binary runs the same calculation and would fail the same way
    attempts = [_attempt(100, 11000000, None, 1)]
    with patch.object(lib, "calculate_t_level", side_effect=TypeError("bad input")), \
            patch.object(calculator, "_calculate_t_level_binary") as binary:
        assert calculator.get_current_t_level_from_attempts(attempts, 40.0) is None
    binary.assert_not_called()


def test_calculator_falls_back_to_binary_on_package_database_error(calculator):
    attempts = [_attempt(100, 11000000, None, 1)]
    with patch.object(lib, "calculate_t_level", side_effect=sqlite3.OperationalError("disk")), \
            patch.object(calculator, "_binary_is_available", return_value=True), \
            patch.object(calculator, "_calculate_t_level_binary", return_value=12.345) as binary:
        assert calculator.get_current_t_level_from_attempts(attempts, 0.0) == 12.345
    binary.assert_called_once()


def test_calculator_returns_none_on_database_error_without_binary(calculator):
    attempts = [_attempt(100, 11000000, None, 1)]
    with patch.object(lib, "calculate_t_level", side_effect=sqlite3.OperationalError("disk")), \
            patch.object(calculator, "_binary_is_available", return_value=False):
        assert calculator.get_current_t_level_from_attempts(attempts, 40.0) is None


def test_binary_backend_setting_bypasses_library():
    settings = SimpleNamespace(t_level_backend="binary", t_level_binary_path="/nonexistent/t-level")
    with patch("app.config.get_settings", return_value=settings):
        calc = TLevelCalculator()
    assert calc.use_library is False
    assert calc.t_level_available is False  # binary missing: no backend at all
    with patch.object(lib, "calculate_t_level") as library_call:
        assert calc.get_current_t_level_from_attempts([_attempt(100, 11000000, None, 1)], 7.0) is None
    library_call.assert_not_called()


def test_binary_backend_never_imports_the_package():
    # The rollback must avoid the package's import-time work entirely. This
    # needs a fresh interpreter, because this one has already imported it.
    code = (
        "import sys\n"
        "from app.services.t_level_calculator import TLevelCalculator\n"
        "calc = TLevelCalculator()\n"
        "assert calc.use_library is False\n"
        "assert 't_level' not in sys.modules and 'gmpy2' not in sys.modules, 'package imported'\n"
    )
    env = {**os.environ, "T_LEVEL_BACKEND": "binary"}
    result = subprocess.run([sys.executable, "-c", code], cwd=SERVER_DIR, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_library_backend_does_not_probe_binary():
    # check_binary_exists() warns when the binary is missing; with the library
    # in use it must not run on every instantiation
    settings = SimpleNamespace(t_level_backend="library", t_level_binary_path="/nonexistent/t-level")
    with patch("app.config.get_settings", return_value=settings), \
            patch("app.services.t_level_calculator.ExternalProgramExecutor.check_binary_exists") as probe:
        calc = TLevelCalculator()
    assert calc.use_library is True
    assert calc.t_level_available is True
    probe.assert_not_called()


# --- recalculate_composite_t_level: failure keeps the last good value -----


def _composite_with_attempt(db, current_t_level):
    created = create_composite("1234567890123456789012345", current_t_level=current_t_level)
    db.add(ECMAttempt(composite_id=created["id"], client_id="test", method="ecm",
                      b1=11000000, b2=None, parametrization=1, curves_requested=100,
                      curves_completed=100, program="gmp-ecm"))
    db.commit()
    return db.get(Composite, created["id"])


def test_recalculation_updates_current_t_level(db_session, calculator):
    composite = _composite_with_attempt(db_session, current_t_level=12.0)
    assert calculator.recalculate_composite_t_level(db_session, composite) == 34.044
    assert composite.current_t_level == 34.044


def test_recalculation_failure_keeps_current_t_level(db_session, calculator):
    composite = _composite_with_attempt(db_session, current_t_level=45.2)
    with patch.object(calculator, "_calculate_from_classes", return_value=None):
        assert calculator.recalculate_composite_t_level(db_session, composite) == 45.2
    assert composite.current_t_level == 45.2
