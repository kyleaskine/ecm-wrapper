"""
In-process t-level calculation using the ``t-level`` package (PyPI ``t-level``).

The server used to spawn ``bin/t-level`` for every calculation. That file is a
PyInstaller one-file build of an earlier version of this package (0.9.6). Each
spawn unpacked ~28 MB and started an interpreter (200-400 ms) before doing any
work, inside the submit transaction while the composite row lock was held.
Calling the package directly removes that start-up cost. Caching its one
expensive function removes the rest for (B1, B2, param) classes it has seen
before.

What is cached: ``t_level.get_ecm_success_probs(b1, b2, param)`` returns the 91
per-digit-size success probabilities of one curve. It depends only on its
arguments, so it is cached for the life of the process (about 3 KB per class;
production has used about 510 classes in total). Combining classes into a
t-level is cheap arithmetic.

Equivalence: ``calculate_t_level`` reproduces what the 0.9.6 binary printed for
the input the server used to pipe to it, including ``-w`` (prior t-level) and
the 3-decimal output, so stored t-levels do not shift. It matched on every
production class and on every composite checked (2026-11).
``tests/test_t_level_library.py`` pins the binary's outputs. The two differ only
on degenerate input that no server path produces: for ``1@2``, 0.9.6 crashes
with ZeroDivisionError, while 1.0.0 guards it and returns t0.000.

Import: the package is imported on first use, not when this module is imported.
``T_LEVEL_BACKEND=binary`` therefore never loads it (or gmpy2), and a rollback
avoids its import-time work entirely.

Thread safety: the package opens one SQLite connection at import and shares a
single cursor (``check_same_thread=False``), and ``rho`` keeps module-level
tables. Routes run on threadpool threads, so every call into the package holds
``_lock``.

Memory: the package decorates its helpers with unbounded ``functools.cache``.
Computing every production class without clearing them grew RSS by about
200 MB, and it kept growing. After every calculation, each of those caches that
holds anything is cleared. The package then adds about 25 MB, flat after the
first ~100 classes. The probability-vector cache above is the only one kept. A
cold class then costs about 55 ms on a fast machine (maximum about 290 ms), paid
once per class per process.
"""
import logging
import threading
import time
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# (curves, b1, b2, param). b2 None is GMP-ECM's default B2; b2 0 is stage 1 only.
CurveClass = Tuple[int, int, Optional[int], int]

# The CLI prints t-levels with 3 decimals, so every stored value has 3.
PRECISION = 3
# The CLI exits 1 for -w outside this range. calculate_t_level returns None for
# it; TLevelCalculator handles such priors itself and never passes them.
MIN_STARTING_T_LEVEL = 5.0
MAX_STARTING_T_LEVEL = 100.0
# Parametrizations the calculation supports. The CLI's grammar also accepts 4,
# but rho has no model for it: it raises ZeroDivisionError and the CLI exits 1.
# Rejecting 4 here gives the same outcome without the traceback.
VALID_PARAMS = frozenset({0, 1, 2, 3})

_VECTOR_CACHE_SIZE = 2048

_lock = threading.Lock()        # serialises every call into the package
_load_lock = threading.Lock()   # guards the one-time import below
_loaded = False
_t_level: Any = None
_import_error: Optional[BaseException] = None
_success_probs: Any = None
_package_caches: List[Any] = []


def _setup() -> None:
    """Import the package, route its expensive function through a cache, and check it."""
    global _t_level, _success_probs, _package_caches
    import t_level as package
    from t_level import rho

    original = package.get_ecm_success_probs

    @lru_cache(maxsize=_VECTOR_CACHE_SIZE)
    def success_probs(b1: int, b2: Optional[int], param: int) -> Tuple[float, ...]:
        # tuple(): on its table-lookup path the package returns a one-shot map
        # object, which a cache would hand out exhausted on the second call.
        return tuple(original(b1, b2, param))

    # The package looks this name up in its module globals at call time
    # (get_failure_probabilities), so replacing it routes every caller,
    # including the -w conversion, through the cache.
    package.get_ecm_success_probs = success_probs

    # Check that the package's own code actually calls through the replaced
    # name. If a future version binds it differently, the cache would be
    # bypassed without any error, so treat the package as unusable instead. The
    # input takes the SQLite-table path, which is one indexed read and no rho.
    with _lock:
        package.get_t_level([(1, 11000000, None, 1)])
    if success_probs.cache_info().misses == 0:
        package.get_ecm_success_probs = original
        raise RuntimeError(
            "t-level package does not call get_ecm_success_probs through its module "
            "globals; the probability cache would be bypassed"
        )

    _package_caches = [
        fn for module in (package, rho) for fn in vars(module).values()
        if fn is not success_probs
        and callable(getattr(fn, "cache_clear", None))
        and callable(getattr(fn, "cache_info", None))
    ]
    _success_probs = success_probs
    _t_level = package


def _load() -> bool:
    """Import and set up the package on first use. True if it is usable."""
    global _loaded, _import_error
    if not _loaded:
        with _load_lock:
            if not _loaded:
                try:
                    _setup()
                except Exception as e:  # ImportError, or the package's own setup failing
                    _import_error = e
                    logger.warning(
                        "t-level package unavailable (%r); t-level calculations will "
                        "use the t-level binary", e
                    )
                _loaded = True
    return _t_level is not None


def is_available() -> bool:
    """True if the t-level package can be used. Imports it on the first call."""
    return _load()


def import_error() -> Optional[BaseException]:
    """The exception that made the package unusable, if any. Imports it on the first call."""
    _load()
    return _import_error


def cache_info() -> Any:
    """lru_cache statistics for the probability-vector cache (None if not loaded)."""
    return _success_probs.cache_info() if _success_probs is not None else None


def _clear_package_caches() -> None:
    for fn in _package_caches:
        if fn.cache_info().currsize:
            fn.cache_clear()


def _starting_work_class(starting_t_level: float) -> CurveClass:
    """The CLI's ``-w``: an equivalent ``curves@B1`` line (param 1, default B2).

    Parsed here rather than with ``t_level.parse_line``, which logs every line at
    INFO on the root logger. Call with ``_lock`` held (the package's shared cursor).
    """
    curve_string = _t_level.get_t_level_curves_string(starting_t_level, PRECISION)
    curves_str, b1_str = curve_string.split("@")
    return (_t_level.sci_int(curves_str), _t_level.sci_int(b1_str), None, 1)


def calculate_t_level(curve_classes: Sequence[CurveClass],
                      starting_t_level: float = 0.0) -> Optional[float]:
    """Return the t-level that ``t-level [-w starting_t_level]`` would print.

    Mirrors the server's former call, ``echo "<classes>" | t-level [-w W]``:

    - classes are combined in the given order, with the ``-w`` line last;
    - ``-w`` applies only when ``starting_t_level > 0``, and must lie in 5-100;
    - classes with 0 curves are dropped, as the CLI's parser drops them;
    - a class the CLI would reject fails the whole input;
    - the result is rounded to 3 decimals, as printed.

    Returns None wherever the CLI would exit non-zero, and the caller decides
    what to do. Raises RuntimeError if the package is unusable. Any other
    exception from the package propagates.
    """
    if not _load():
        raise RuntimeError(f"t-level package unavailable: {_import_error!r}")

    classes: List[CurveClass] = []
    for curves, b1, b2, param in curve_classes:
        n_curves, b1_int, param_int = int(curves), int(b1), int(param)
        b2_int = None if b2 is None else int(b2)
        if n_curves == 0:
            continue
        if (n_curves < 0 or b1_int <= 0 or (b2_int is not None and b2_int < 0)
                or param_int not in VALID_PARAMS):
            logger.error(
                "t-level input rejected (the CLI would exit 1): curves=%s b1=%s b2=%s param=%s",
                curves, b1, b2, param
            )
            return None
        classes.append((n_curves, b1_int, b2_int, param_int))

    use_work = starting_t_level > 0
    if use_work and not (MIN_STARTING_T_LEVEL <= starting_t_level <= MAX_STARTING_T_LEVEL):
        logger.warning(
            "Starting t-level %s is outside %s-%s (the CLI would exit 1)",
            starting_t_level, MIN_STARTING_T_LEVEL, MAX_STARTING_T_LEVEL
        )
        return None

    with _lock:
        misses_before = _success_probs.cache_info().misses
        started = time.perf_counter()
        try:
            if use_work:
                classes.append(_starting_work_class(float(starting_t_level)))
            t_level_value = _t_level.get_t_level(classes)
        finally:
            _clear_package_caches()
            computed = _success_probs.cache_info().misses - misses_before
            if computed:
                logger.info(
                    "t-level: computed %d new probability vector(s) in %.0f ms",
                    computed, (time.perf_counter() - started) * 1000
                )

    # Same rounding as parsing the CLI's f"t{value:.3f}" output.
    return float(f"{t_level_value:.{PRECISION}f}")
