#!/usr/bin/env python3
"""
Tests for aliquot tracker submission (lib/tracker_client.py + AliquotWrapper).

Covers the one-factor-at-a-time submission loop: factors are validated
against the tracker's CURRENT composite (which shrinks between submissions),
the final prime cofactor is never sent, and tracker failures signal the
caller to fall back to direct FactorDB submission.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.tracker_client import TrackerSequence, TrackerSubmitResult, _parse_sequence
from aliquot_wrapper import AliquotWrapper


def _state(seq_id: str = 'seq-1', index: int = 5, composite: Optional[int] = None,
           status: str = 'active') -> TrackerSequence:
    return TrackerSequence(
        id=seq_id,
        start_number='276',
        current_index=index,
        current_composite=str(composite) if composite is not None else None,
        status=status,
        factordb_status='CF',
    )


class FakeTracker:
    """Scripted tracker: submit_factor divides the composite and advances when
    the remaining cofactor is prime (mirroring FactorDB + auto-advance)."""

    def __init__(self, composite: int, prime_cofactor_advances: bool = True):
        self.composite = composite
        self.index = 5
        self.prime_cofactor_advances = prime_cofactor_advances
        self.submitted: List[int] = []

    def get_sequence(self, start: int) -> Optional[TrackerSequence]:
        return _state(index=self.index, composite=self.composite)

    def submit_factor(self, sequence_id: str, factor: str) -> TrackerSubmitResult:
        f = int(factor)
        if self.composite % f != 0 or f == self.composite:
            return TrackerSubmitResult(accepted=False, permanent=True,
                                       error='Factor does not divide the current composite.')
        self.submitted.append(f)
        # FactorDB divides out the full power of a reported prime
        while self.composite % f == 0:
            self.composite //= f
        from lib.ecm_math import is_probably_prime
        if self.composite == 1 or (self.prime_cofactor_advances
                                   and is_probably_prime(self.composite)):
            self.index += 1
            self.composite = 9999999  # next term (opaque)
            return TrackerSubmitResult(accepted=True, auto_advanced=True,
                                       sequence=_state(index=self.index,
                                                       composite=self.composite))
        return TrackerSubmitResult(accepted=True, auto_advanced=False,
                                   sequence=_state(index=self.index,
                                                   composite=self.composite))


def _make_wrapper(tracker) -> AliquotWrapper:
    """Build an AliquotWrapper with just the attributes the tracker path uses
    (bypasses __init__ so no config file or binaries are needed)."""
    wrapper = AliquotWrapper.__new__(AliquotWrapper)
    wrapper.tracker = tracker
    wrapper.tracker_start = 276
    wrapper.use_tracker = True
    wrapper.logger = logging.getLogger('test_tracker_submission')
    return wrapper


# ==================== submission loop ====================


def test_submit_via_tracker_skips_final_prime_cofactor():
    # term n = 2^2 * 3 * 7 * 101; tracker cofactor = 3 * 7 * 101 (FDB knows 2^2)
    n = 4 * 3 * 7 * 101
    tracker = FakeTracker(composite=3 * 7 * 101)
    wrapper = _make_wrapper(tracker)

    ok = wrapper.submit_via_tracker(n, {2: 2, 3: 1, 7: 1, 101: 1})
    assert ok
    # 3 then 7 submitted; cofactor 101 is prime -> auto-advance, 101 never sent
    assert tracker.submitted == [3, 7]


def test_submit_via_tracker_handles_multiplicity_reduction():
    # tracker composite = 3^2 * 11; FDB divides the full power of 3 at once,
    # so the second 3 in our multiset must be skipped, not resubmitted
    n = 3 * 3 * 11 * 13
    tracker = FakeTracker(composite=3 * 3 * 11)
    wrapper = _make_wrapper(tracker)

    ok = wrapper.submit_via_tracker(n, {3: 2, 11: 1, 13: 1})
    assert ok
    assert tracker.submitted == [3]  # 3^2 divided out; cofactor 11 prime -> advance


def test_submit_via_tracker_rejects_wrong_term():
    # Tracker composite does not divide our term -> different term -> fallback
    tracker = FakeTracker(composite=17 * 19)
    wrapper = _make_wrapper(tracker)

    ok = wrapper.submit_via_tracker(3 * 7, {3: 1, 7: 1})
    assert not ok
    assert tracker.submitted == []


def test_submit_via_tracker_rejects_bad_factorization():
    tracker = FakeTracker(composite=3 * 7)
    wrapper = _make_wrapper(tracker)

    ok = wrapper.submit_via_tracker(3 * 7, {3: 1, 5: 1})  # 15 != 21
    assert not ok
    assert tracker.submitted == []


def test_submit_via_tracker_inactive_sequence_falls_back():
    class InactiveTracker(FakeTracker):
        def get_sequence(self, start):
            return _state(composite=None, status='terminated')

    wrapper = _make_wrapper(InactiveTracker(composite=1))
    assert not wrapper.submit_via_tracker(21, {3: 1, 7: 1})


def test_submit_via_tracker_unreachable_falls_back():
    class DownTracker(FakeTracker):
        def get_sequence(self, start):
            return None

    wrapper = _make_wrapper(DownTracker(composite=1))
    assert not wrapper.submit_via_tracker(21, {3: 1, 7: 1})


def test_transient_submit_failure_signals_fallback():
    class FlakyTracker(FakeTracker):
        def submit_factor(self, sequence_id, factor):
            return TrackerSubmitResult(accepted=False, permanent=False,
                                       error='tracker 500 (FactorDB down)')

    tracker = FlakyTracker(composite=3 * 7 * 101)
    wrapper = _make_wrapper(tracker)

    ok = wrapper.submit_via_tracker(4 * 3 * 7 * 101, {2: 2, 3: 1, 7: 1, 101: 1})
    assert not ok  # caller falls back to direct FactorDB


def test_sync_tracker_factors_is_best_effort():
    class ExplodingTracker(FakeTracker):
        def get_sequence(self, start):
            raise RuntimeError('boom')

    wrapper = _make_wrapper(ExplodingTracker(composite=1))
    # Must not raise - mid-run sync never kills a factorization
    wrapper._sync_tracker_factors(['3', '7'])


def test_sync_tracker_factors_submits_only_dividing_factors():
    tracker = FakeTracker(composite=3 * 7 * 10007 * 10009)
    wrapper = _make_wrapper(tracker)

    # 5 does not divide the tracker composite (already known to FDB)
    wrapper._sync_tracker_factors(['5', '3', '7'])
    assert tracker.submitted == [3, 7]
    # cofactor 10007*10009 is composite -> no advance
    assert tracker.index == 5


# ==================== response parsing ====================


def test_parse_sequence_full_payload():
    seq = _parse_sequence({
        'id': 'abc', 'startNumber': '276', 'currentIndex': 7,
        'currentComposite': '12345', 'status': 'active', 'factordbStatus': 'CF',
    })
    assert seq is not None
    assert seq.id == 'abc'
    assert seq.current_index == 7
    assert seq.current_composite == '12345'


def test_parse_sequence_degraded_payload_returns_none():
    # submit-factor's catch path returns only {sequenceId, autoAdvanced}
    assert _parse_sequence({'sequenceId': 'abc', 'autoAdvanced': False}) is None


# ==================== typed config ====================


def test_aliquot_tracker_config_defaults():
    from lib.typed_config import TypedConfigLoader
    cfg = TypedConfigLoader()._parse_config({})
    assert cfg.aliquot_tracker.url is None
    assert cfg.aliquot_tracker.api_key is None
    assert cfg.aliquot_tracker.submitter is None
    assert cfg.factordb.cookie is None


def test_aliquot_tracker_config_parses_fields():
    from lib.typed_config import TypedConfigLoader
    cfg = TypedConfigLoader()._parse_config({
        'aliquot_tracker': {'url': 'http://localhost:3001/', 'api_key': 'k'},
        'factordb': {'cookie': 'abc123'},
    })
    assert cfg.aliquot_tracker.url == 'http://localhost:3001/'
    assert cfg.aliquot_tracker.api_key == 'k'
    assert cfg.factordb.cookie == 'abc123'


def test_client_normalizes_trailing_slash():
    from lib.tracker_client import AliquotTrackerClient
    client = AliquotTrackerClient('http://localhost:3001/')
    assert client.base_url == 'http://localhost:3001'


# ==================== malformed tracker payloads ====================


def test_submit_via_tracker_survives_zero_composite():
    # A malformed "0" composite must fall back, not ZeroDivisionError a run
    class ZeroTracker(FakeTracker):
        def get_sequence(self, start):
            return _state(composite=0)

    wrapper = _make_wrapper(ZeroTracker(composite=1))
    assert not wrapper.submit_via_tracker(21, {3: 1, 7: 1})


def test_submit_via_tracker_survives_non_numeric_composite():
    class GarbageTracker(FakeTracker):
        def get_sequence(self, start):
            state = _state(composite=None)
            state.current_composite = '<html>bad gateway</html>'
            return state

    wrapper = _make_wrapper(GarbageTracker(composite=1))
    assert not wrapper.submit_via_tracker(21, {3: 1, 7: 1})


# ==================== submit_factors fallback ====================


def _make_fallback_wrapper(tracker):
    """Wrapper whose direct-FactorDB path records the call instead of hitting
    the network."""
    wrapper = _make_wrapper(tracker)
    calls = []

    def fake_fdb(n, factorization):
        calls.append((n, factorization))
        return True

    wrapper.submit_to_factordb = fake_fdb  # type: ignore[method-assign]
    return wrapper, calls


def test_submit_factors_falls_back_to_factordb_on_tracker_failure():
    class DownTracker(FakeTracker):
        def get_sequence(self, start):
            return None

    wrapper, fdb_calls = _make_fallback_wrapper(DownTracker(composite=1))
    assert wrapper.submit_factors(21, {3: 1, 7: 1})
    assert fdb_calls == [(21, {3: 1, 7: 1})]


def test_submit_factors_skips_factordb_when_tracker_succeeds():
    tracker = FakeTracker(composite=3 * 7 * 101)
    wrapper, fdb_calls = _make_fallback_wrapper(tracker)
    assert wrapper.submit_factors(4 * 3 * 7 * 101, {2: 2, 3: 1, 7: 1, 101: 1})
    assert fdb_calls == []
    assert tracker.submitted == [3, 7]


def test_submit_factors_without_tracker_goes_direct():
    wrapper, fdb_calls = _make_fallback_wrapper(FakeTracker(composite=1))
    wrapper.use_tracker = False
    assert wrapper.submit_factors(21, {3: 1, 7: 1})
    assert fdb_calls == [(21, {3: 1, 7: 1})]


# ==================== AliquotTrackerClient HTTP contract ====================


class FakeResponse:
    def __init__(self, status_code: int, body=None, text: str = '',
                 headers: Optional[Dict[str, str]] = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        if body is not None:
            import json as _json
            self.content = _json.dumps(body).encode()
        else:
            self.content = text.encode()

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        import json as _json
        return _json.loads(self.content.decode())


def _client_with_responses(monkeypatch, responses):
    """AliquotTrackerClient whose HTTP layer replays `responses` (a list of
    FakeResponse or Exception) and records sleeps instead of waiting."""
    import lib.tracker_client as tc

    client = tc.AliquotTrackerClient('http://tracker.test', api_key='key123')
    calls = []
    sleeps = []
    queue = list(responses)

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tc.requests, 'request', fake_request)
    monkeypatch.setattr(tc.time, 'sleep', lambda s: sleeps.append(s))
    return client, calls, sleeps


def test_submit_factor_4xx_html_body_is_permanent_and_not_retried(monkeypatch):
    # Reject with a NON-JSON body (proxy error page): must still be treated
    # as a permanent 4xx after exactly one request - the tracker's rate
    # limiter counts failed submit attempts
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(400, text='<html>Bad Request</html>'),
    ])
    result = client.submit_factor('seq-1', '3')
    assert not result.accepted
    assert result.permanent
    assert len(calls) == 1


def test_submit_factor_401_is_permanent(monkeypatch):
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(401, body={'success': False, 'error': 'Invalid or revoked API key'}),
    ])
    result = client.submit_factor('seq-1', '3')
    assert not result.accepted
    assert result.permanent
    assert 'API key' in (result.error or '')
    assert len(calls) == 1


def test_submit_factor_5xx_is_retried_then_transient(monkeypatch):
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(500, body={'success': False, 'error': 'FactorDB submission failed'}),
        FakeResponse(500, body={'success': False, 'error': 'FactorDB submission failed'}),
        FakeResponse(500, body={'success': False, 'error': 'FactorDB submission failed'}),
    ])
    result = client.submit_factor('seq-1', '3')
    assert not result.accepted
    assert not result.permanent  # caller falls back to direct FactorDB
    assert len(calls) == 3


def test_submit_factor_429_honors_retry_after_then_succeeds(monkeypatch):
    client, calls, sleeps = _client_with_responses(monkeypatch, [
        FakeResponse(429, body={'success': False, 'error': 'rate limited'},
                     headers={'Retry-After': '7'}),
        FakeResponse(200, body={'success': True, 'data': {
            'id': 'seq-1', 'currentIndex': 6, 'currentComposite': '35',
            'status': 'active', 'autoAdvanced': False}}),
    ])
    result = client.submit_factor('seq-1', '3')
    assert result.accepted
    assert len(calls) == 2
    assert 7.0 in sleeps


def test_submit_factor_network_errors_exhaust_to_transient(monkeypatch):
    import requests as _requests
    client, calls, _ = _client_with_responses(monkeypatch, [
        _requests.ConnectionError('refused'),
        _requests.ConnectionError('refused'),
        _requests.ConnectionError('refused'),
    ])
    result = client.submit_factor('seq-1', '3')
    assert not result.accepted
    assert not result.permanent
    assert len(calls) == 3


def test_get_sequence_403_is_not_retried(monkeypatch):
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(403, body={'success': False, 'error': 'Forbidden'}),
    ])
    assert client.get_sequence(276) is None
    assert len(calls) == 1


def test_get_sequence_500_then_success_retries(monkeypatch):
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(500, text='oops'),
        FakeResponse(200, body={'success': True, 'data': {
            'id': 'seq-1', 'startNumber': '276', 'currentIndex': 5,
            'currentComposite': '12345', 'status': 'active'}}),
    ])
    seq = client.get_sequence(276)
    assert seq is not None and seq.current_index == 5
    assert len(calls) == 2


def test_submit_factor_sends_api_key_header_not_handle(monkeypatch):
    client, calls, _ = _client_with_responses(monkeypatch, [
        FakeResponse(200, body={'success': True, 'data': {}}),
    ])
    client.submit_factor('seq-1', '3')
    method, url, kwargs = calls[0]
    assert kwargs['headers']['X-Api-Key'] == 'key123'
    assert 'submitterHandle' not in kwargs['json']


def test_retry_after_parses_http_date_without_crashing():
    from typing import Any, cast
    from lib.tracker_client import _retry_after_seconds

    def seconds(resp: FakeResponse, default: float) -> float:
        return _retry_after_seconds(cast(Any, resp), default)

    past = FakeResponse(429, headers={'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT'})
    assert seconds(past, 5.0) == 0.0  # date in the past -> no wait
    garbage = FakeResponse(429, headers={'Retry-After': 'soonish'})
    assert seconds(garbage, 5.0) == 5.0
    missing = FakeResponse(429)
    assert seconds(missing, 5.0) == 5.0
