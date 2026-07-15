#!/usr/bin/env python3
"""
Tests for per-term external factor reconciliation (AliquotWrapper.factor_term)
and the no-queue-on-permanent-rejection submission behavior.

FactorDB trial-divides deeper than the wrapper's local 10^7 bound, so a term's
externally-known cofactor is usually smaller than what local TD leaves. In
tracker mode the ECM server's registered composite IS the tracker's cofactor,
so factor_term must align the working cofactor with it before ECM starts.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.base_wrapper import BaseWrapper
from lib.api_client import ResourceNotFoundError
from lib.tracker_client import TrackerSequence, _parse_known_factors, _parse_sequence
from aliquot_wrapper import AliquotWrapper


def _make_wrapper(use_tracker: bool = False, use_factordb: bool = False,
                  tracker: Any = None) -> AliquotWrapper:
    """AliquotWrapper with just the attributes factor_term uses (bypasses
    __init__ so no config file or binaries are needed)."""
    wrapper = AliquotWrapper.__new__(AliquotWrapper)
    wrapper.use_tracker = use_tracker
    wrapper.use_factordb = use_factordb
    wrapper.tracker = tracker
    wrapper.tracker_start = 276 if tracker is not None else None
    wrapper.logger = logging.getLogger('test_factor_term')
    return wrapper


def _state(composite: Optional[int], status: str = 'active', index: int = 5,
           known_factors: Optional[List[Tuple[str, int]]] = None) -> TrackerSequence:
    return TrackerSequence(
        id='seq-1',
        start_number='276',
        current_index=index,
        current_composite=str(composite) if composite is not None else None,
        status=status,
        factordb_status='CF',
        known_factors=known_factors,
    )


class _FailIfCalled:
    """Stub that fails the test if invoked (network/factoring must not run)."""

    def __init__(self, what: str):
        self.what = what

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{self.what} should not have been called")


# ==================== factor_term ====================


def test_factor_term_divides_known_and_factors_cofactor():
    wrapper = _make_wrapper()
    calls: List[int] = []

    def fake_factor_number(n: int):
        calls.append(n)
        return True, {7: 1, 13: 1}, {'success': True}

    wrapper.factor_number = fake_factor_number  # type: ignore[method-assign]
    n = (2 ** 2) * 3 * 7 * 13
    success, factorization, _ = wrapper.factor_term(n, known_factors={2: 2, 3: 1})

    assert success
    assert factorization == {2: 2, 3: 1, 7: 1, 13: 1}
    assert calls == [7 * 13]  # only the reduced cofactor was factored locally


def test_factor_term_prime_cofactor_needs_no_local_factoring():
    wrapper = _make_wrapper()
    wrapper.factor_number = _FailIfCalled('factor_number')  # type: ignore[method-assign]

    n = (2 ** 2) * 3 * 101
    success, factorization, results = wrapper.factor_term(n, known_factors={2: 2, 3: 1})

    assert success
    assert factorization == {2: 2, 3: 1, 101: 1}
    assert results['method'] == 'external_factors+primality'


def test_factor_term_fully_known():
    wrapper = _make_wrapper()
    wrapper.factor_number = _FailIfCalled('factor_number')  # type: ignore[method-assign]

    success, factorization, results = wrapper.factor_term(8, known_factors={2: 3})

    assert success
    assert factorization == {2: 3}
    assert results['method'] == 'external_factors'


def test_factor_term_divides_beyond_reported_exponent():
    # External exponents are advisory: division runs while divisible, so the
    # factorization is derived from n itself and can't be corrupted by a
    # wrong or stale report
    wrapper = _make_wrapper()
    wrapper.factor_number = _FailIfCalled('factor_number')  # type: ignore[method-assign]

    n = (2 ** 5) * 101
    success, factorization, _ = wrapper.factor_term(n, known_factors={2: 2})

    assert success
    assert factorization == {2: 5, 101: 1}


def test_factor_term_no_known_factors_delegates():
    wrapper = _make_wrapper()
    calls: List[int] = []

    def fake_factor_number(n: int):
        calls.append(n)
        return True, {3: 1, 7: 1}, {'success': True}

    wrapper.factor_number = fake_factor_number  # type: ignore[method-assign]
    success, factorization, _ = wrapper.factor_term(21, known_factors={})

    assert success
    assert factorization == {3: 1, 7: 1}
    assert calls == [21]


def test_factor_term_ignores_nondividing_report():
    # A factor that doesn't divide n (wrong-term data) must be skipped, and
    # with nothing divided out the whole term goes to local factoring
    wrapper = _make_wrapper()
    calls: List[int] = []

    def fake_factor_number(n: int):
        calls.append(n)
        return True, {3: 1, 7: 1}, {'success': True}

    wrapper.factor_number = fake_factor_number  # type: ignore[method-assign]
    success, factorization, _ = wrapper.factor_term(21, known_factors={5: 1})

    assert success
    assert factorization == {3: 1, 7: 1}
    assert calls == [21]


def test_factor_term_propagates_cofactor_failure():
    wrapper = _make_wrapper()
    wrapper.factor_number = lambda n: (False, {}, {'success': False})  # type: ignore[method-assign]

    success, factorization, _ = wrapper.factor_term(4 * 7 * 13, known_factors={2: 2})

    assert not success
    assert factorization == {}


# ==================== _external_known_factors source selection ====================


def test_external_lookup_skipped_for_small_terms():
    wrapper = _make_wrapper(use_tracker=True, tracker=object())
    wrapper._tracker_known_factors = _FailIfCalled('tracker lookup')  # type: ignore[method-assign]
    wrapper._factordb_known_factors = _FailIfCalled('FactorDB lookup')  # type: ignore[method-assign]

    assert wrapper._external_known_factors(12345) == {}


def test_external_lookup_skipped_without_submission_destination():
    wrapper = _make_wrapper()
    wrapper._tracker_known_factors = _FailIfCalled('tracker lookup')  # type: ignore[method-assign]
    wrapper._factordb_known_factors = _FailIfCalled('FactorDB lookup')  # type: ignore[method-assign]

    assert wrapper._external_known_factors(10 ** 60 + 7) == {}


def test_external_lookup_uses_tracker_in_tracker_mode():
    wrapper = _make_wrapper(use_tracker=True, tracker=object())
    wrapper._tracker_known_factors = lambda n: {2: 2, 28025087: 1}  # type: ignore[method-assign]
    wrapper._factordb_known_factors = _FailIfCalled('FactorDB lookup')  # type: ignore[method-assign]

    assert wrapper._external_known_factors(10 ** 60 + 7) == {2: 2, 28025087: 1}


def test_external_lookup_falls_back_to_factordb_when_tracker_cannot_answer():
    wrapper = _make_wrapper(use_tracker=True, tracker=object())
    wrapper._tracker_known_factors = lambda n: None  # type: ignore[method-assign]
    wrapper._factordb_known_factors = lambda n: {3: 1}  # type: ignore[method-assign]

    assert wrapper._external_known_factors(10 ** 60 + 7) == {3: 1}


def test_external_lookup_uses_factordb_in_factordb_mode():
    wrapper = _make_wrapper(use_factordb=True)
    wrapper._factordb_known_factors = lambda n: {7: 2}  # type: ignore[method-assign]

    assert wrapper._external_known_factors(10 ** 60 + 7) == {7: 2}


def test_external_lookup_treats_failed_factordb_query_as_nothing_known():
    wrapper = _make_wrapper(use_factordb=True)
    wrapper._factordb_known_factors = lambda n: None  # type: ignore[method-assign]

    assert wrapper._external_known_factors(10 ** 60 + 7) == {}


# ==================== _tracker_known_factors ====================


class _FakeTracker:
    def __init__(self, state: Optional[TrackerSequence]):
        self.state = state

    def get_sequence(self, start: int) -> Optional[TrackerSequence]:
        return self.state


def test_tracker_known_factors_happy_path():
    # Term = 3^2 * 91 with tracker cofactor 91; knownFactors mirrors
    # FactorDB's array, which includes the composite cofactor - only the
    # settled prime factors must come back
    state = _state(composite=91, known_factors=[('3', 2), ('91', 1)])
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(state))

    assert wrapper._tracker_known_factors(9 * 91) == {3: 2}


def test_tracker_known_factors_rejects_different_term():
    state = _state(composite=17 * 19, known_factors=[('2', 1)])
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(state))

    assert wrapper._tracker_known_factors(3 * 7) is None


def test_tracker_known_factors_requires_active_sequence():
    state = _state(composite=91, status='terminated', known_factors=[('3', 2)])
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(state))

    assert wrapper._tracker_known_factors(9 * 91) is None


def test_tracker_known_factors_unreachable_tracker():
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(None))

    assert wrapper._tracker_known_factors(9 * 91) is None


def test_tracker_known_factors_missing_payload():
    state = _state(composite=91, known_factors=None)
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(state))

    assert wrapper._tracker_known_factors(9 * 91) is None


def test_tracker_known_factors_survives_overlong_value():
    # Passes the parser's isdecimal check but exceeds CPython's int-conversion
    # length limit: must reject the payload, never raise
    huge = '1' * 5000
    state = _state(composite=91, known_factors=[(huge, 1), ('3', 2)])
    wrapper = _make_wrapper(use_tracker=True, tracker=_FakeTracker(state))

    assert wrapper._tracker_known_factors(9 * 91) is None


# ==================== _factordb_known_factors ====================


class _FakeResponse:
    def __init__(self, payload: Any = None, error: Optional[Exception] = None):
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _factordb_wrapper(monkeypatch: Any, response: _FakeResponse,
                      raises: Optional[Exception] = None) -> AliquotWrapper:
    import requests

    wrapper = _make_wrapper(use_factordb=True)
    wrapper._factordb_cookies = lambda: {}  # type: ignore[method-assign]

    def fake_get(url: str, cookies: Any = None, timeout: Any = None) -> _FakeResponse:
        assert url.startswith('https://factordb.com/api?query=')
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(requests, 'get', fake_get)
    return wrapper


def test_factordb_known_factors_keeps_only_settled_primes(monkeypatch):
    # n = 2^2 * 3 * 91: the factors array carries the composite cofactor (91)
    # and malformed entries; only the prime factors below n come back
    n = 4 * 3 * 91
    payload = {'status': 'CF',
               'factors': [['2', 2], ['3', 1], ['91', 1], ['x', 1], ['7']]}
    wrapper = _factordb_wrapper(monkeypatch, _FakeResponse(payload=payload))

    assert wrapper._factordb_known_factors(n) == {2: 2, 3: 1}


def test_factordb_known_factors_excludes_n_itself(monkeypatch):
    # Status C/P responses list n itself as the only "factor"
    payload = {'status': 'P', 'factors': [['101', 1]]}
    wrapper = _factordb_wrapper(monkeypatch, _FakeResponse(payload=payload))

    assert wrapper._factordb_known_factors(101) == {}


def test_factordb_known_factors_network_failure_returns_none(monkeypatch):
    import requests

    wrapper = _factordb_wrapper(monkeypatch, _FakeResponse(),
                                raises=requests.ConnectionError('boom'))

    assert wrapper._factordb_known_factors(1092) is None


def test_factordb_known_factors_http_error_returns_none(monkeypatch):
    import requests

    response = _FakeResponse(error=requests.HTTPError('503'))
    wrapper = _factordb_wrapper(monkeypatch, response)

    assert wrapper._factordb_known_factors(1092) is None


def test_factordb_known_factors_bad_json_returns_none(monkeypatch):
    wrapper = _factordb_wrapper(monkeypatch, _FakeResponse(payload=ValueError('not json')))

    assert wrapper._factordb_known_factors(1092) is None


def test_factordb_known_factors_non_dict_body_returns_none(monkeypatch):
    wrapper = _factordb_wrapper(monkeypatch, _FakeResponse(payload=['not', 'a', 'dict']))

    assert wrapper._factordb_known_factors(1092) is None


# ==================== knownFactors payload parsing ====================


def test_parse_known_factors_well_formed():
    assert _parse_known_factors([['2', 3], ['28025087', 1]]) == [('2', 3), ('28025087', 1)]
    assert _parse_known_factors([[2, 1]]) == [('2', 1)]  # ints tolerated
    assert _parse_known_factors([]) == []


def test_parse_known_factors_malformed():
    assert _parse_known_factors(None) is None
    assert _parse_known_factors('2^3') is None
    assert _parse_known_factors([['2']]) is None
    assert _parse_known_factors([['x', 1]]) is None
    assert _parse_known_factors([['2', 0]]) is None
    assert _parse_known_factors([['2', '3']]) is None
    assert _parse_known_factors([['2', True]]) is None
    # isdigit()-true but int()-invalid characters (superscript two)
    assert _parse_known_factors([['²', 1]]) is None


def test_parse_sequence_carries_known_factors():
    seq = _parse_sequence({
        'id': 'seq-1',
        'startNumber': '276',
        'currentIndex': 5,
        'currentComposite': '91',
        'status': 'active',
        'knownFactors': [['3', 2], ['91', 1]],
    })
    assert seq is not None
    assert seq.known_factors == [('3', 2), ('91', 1)]


def test_parse_sequence_tolerates_missing_known_factors():
    seq = _parse_sequence({'id': 'seq-1', 'currentIndex': 5})
    assert seq is not None
    assert seq.known_factors is None


# ==================== permanent rejection is not queued ====================


class _FakeQueue:
    def __init__(self):
        self.enqueued: List[Dict[str, Any]] = []

    def enqueue_result(self, payload, results_context=None, completion_chain=None):
        self.enqueued.append(payload)


class _FakeAPIClient:
    """submit_result stub: raises `error` if set, else returns `response`."""

    def __init__(self, error: Optional[Exception] = None,
                 response: Optional[Dict[str, Any]] = None):
        self.error = error
        self.response = response

    def submit_result(self, payload, save_on_failure=True, results_context=None):
        if self.error is not None:
            raise self.error
        return self.response


def _make_base_wrapper(clients: List[_FakeAPIClient],
                       names: Optional[List[str]] = None
                       ) -> Tuple[BaseWrapper, _FakeQueue]:
    wrapper = BaseWrapper.__new__(BaseWrapper)
    wrapper.logger = logging.getLogger('test_factor_term')
    wrapper.api_clients = [  # type: ignore[assignment]
        {'client': client, 'name': names[i] if names else f'ep{i}', 'url': 'http://test'}
        for i, client in enumerate(clients)
    ]
    queue = _FakeQueue()
    wrapper.submission_queue = queue  # type: ignore[assignment]
    return wrapper, queue


def test_permanent_rejection_not_queued():
    wrapper, queue = _make_base_wrapper(
        [_FakeAPIClient(error=ResourceNotFoundError('composite not found (404)'))])

    result = wrapper.submit_payload_to_endpoints({'composite': '123'})

    assert not result
    assert queue.enqueued == []


def test_transient_failure_still_queued():
    wrapper, queue = _make_base_wrapper([_FakeAPIClient(response=None)])

    result = wrapper.submit_payload_to_endpoints({'composite': '123'})

    assert not result
    assert len(queue.enqueued) == 1


def test_mixed_permanent_and_transient_failures_queued():
    wrapper, queue = _make_base_wrapper([
        _FakeAPIClient(error=ResourceNotFoundError('composite not found (404)')),
        _FakeAPIClient(response=None),
    ])

    result = wrapper.submit_payload_to_endpoints({'composite': '123'})

    assert not result
    assert len(queue.enqueued) == 1


def test_duplicate_endpoint_names_do_not_mask_transient_failure():
    # endpoint_responses is keyed by name, so two endpoints sharing a name
    # collapse to one entry - the transient/permanent classification must
    # count per attempt, not per key
    wrapper, queue = _make_base_wrapper(
        [
            _FakeAPIClient(error=ResourceNotFoundError('composite not found (404)')),
            _FakeAPIClient(response=None),
        ],
        names=['same', 'same'],
    )

    result = wrapper.submit_payload_to_endpoints({'composite': '123'})

    assert not result
    assert len(queue.enqueued) == 1


def test_success_not_queued():
    wrapper, queue = _make_base_wrapper([_FakeAPIClient(response={'status': 'ok'})])

    result = wrapper.submit_payload_to_endpoints({'composite': '123'})

    assert result
    assert queue.enqueued == []
