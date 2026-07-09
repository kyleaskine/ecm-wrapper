"""
HTTP client for the aliquot-tracker web app.

The tracker sits in front of FactorDB: a factor submitted to it is forwarded
to FactorDB first (the tracker rejects it if FactorDB is down), recorded with
attribution, and the sequence is auto-advanced when the term becomes fully
factored. It also keeps the ECM coordination server's aliquot:{start}:i{index}
composites registered in lockstep as sequences advance.

Retry contract (both endpoints): network errors and 5xx are retried with
exponential backoff; 429 is retried honoring Retry-After; any other 4xx is
permanent and returned after a single request - the tracker's per-IP rate
limiter counts failed submit attempts, so hammering rejections could lock
this client out.
"""
import datetime
import logging
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests


@dataclass
class TrackerSequence:
    """Subset of the tracker's sequence state the wrapper needs."""
    id: str
    start_number: str
    current_index: int
    current_composite: Optional[str]
    status: str
    factordb_status: Optional[str]


@dataclass
class TrackerSubmitResult:
    """Outcome of a single factor submission to the tracker."""
    accepted: bool
    permanent: bool = False  # rejection that retrying cannot fix (4xx)
    auto_advanced: bool = False
    sequence: Optional[TrackerSequence] = None
    error: Optional[str] = None


def _parse_sequence(data: Dict[str, Any]) -> Optional[TrackerSequence]:
    """Build a TrackerSequence from a tracker API payload, or None if the
    payload doesn't carry sequence state (e.g. the degraded submit-factor
    response after a failed FactorDB refresh)."""
    if 'currentIndex' not in data or 'id' not in data:
        return None
    return TrackerSequence(
        id=data['id'],
        start_number=str(data.get('startNumber', '')),
        current_index=data['currentIndex'],
        current_composite=data.get('currentComposite'),
        status=data.get('status', ''),
        factordb_status=data.get('factordbStatus'),
    )


def _parse_body(response: requests.Response) -> Dict[str, Any]:
    """Parse the tracker's JSON envelope, tolerating non-JSON bodies (proxy
    error pages etc.) so a 4xx with an HTML body is still treated as a 4xx."""
    if not response.content:
        return {}
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _retry_after_seconds(response: requests.Response, default: float) -> float:
    """Parse Retry-After as delta-seconds or HTTP-date (RFC 7231), falling
    back to `default` when absent or unparseable."""
    value = response.headers.get('Retry-After')
    if not value:
        return default
    try:
        return max(0.0, float(int(value)))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (dt - now).total_seconds())


class AliquotTrackerClient:
    """Client for the aliquot-tracker's public API.

    Authentication: when api_key is set it is sent as X-Api-Key (verified
    attribution - the tracker 401s on an invalid/revoked key rather than
    degrading to anonymous); otherwise submissions are anonymous with a
    submitterHandle.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 submitter: Optional[str] = None, timeout: int = 30,
                 retry_attempts: int = 3,
                 logger: Optional[logging.Logger] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.submitter = submitter
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.logger = logger or logging.getLogger(__name__)

    def _headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['X-Api-Key'] = self.api_key
        return headers

    def _request_with_retries(self, method: str, url: str,
                              json_body: Optional[Dict[str, Any]] = None
                              ) -> Optional[requests.Response]:
        """Issue a request under the module retry contract (see module
        docstring). Returns the final response - any 2xx/3xx, a 4xx after a
        single attempt, or the last 5xx/429 after retries are exhausted - or
        None when every attempt failed at the network level.
        """
        last_response: Optional[requests.Response] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.request(
                    method, url, json=json_body, headers=self._headers(),
                    timeout=self.timeout)
            except requests.RequestException as e:
                self.logger.warning(
                    f"Tracker request failed (attempt {attempt}/{self.retry_attempts}): {e}")
            else:
                last_response = response
                if response.status_code == 429:
                    if attempt < self.retry_attempts:
                        wait = _retry_after_seconds(response, 2 ** attempt)
                        self.logger.warning(
                            f"Tracker rate limit hit, waiting {wait:.0f}s before retry...")
                        time.sleep(wait)
                        continue
                    return response
                if response.status_code >= 500:
                    self.logger.warning(
                        f"Tracker returned HTTP {response.status_code} "
                        f"(attempt {attempt}/{self.retry_attempts})")
                else:
                    # 2xx/3xx, or a permanent 4xx - never retried
                    return response

            if attempt < self.retry_attempts:
                time.sleep(2 ** attempt)
        return last_response

    def _describe_error(self, response: requests.Response,
                        body: Dict[str, Any]) -> str:
        """Human-readable error from the tracker envelope, with a config hint
        for credential rejections."""
        error = body.get('error') or f"HTTP {response.status_code}"
        if response.status_code == 401:
            self.logger.error(
                "Tracker rejected credentials - check aliquot_tracker.api_key "
                "in client.local.yaml")
        return error

    def get_sequence(self, start: int) -> Optional[TrackerSequence]:
        """Fetch a sequence's current state by its start number.

        Returns None if the tracker is unreachable or the sequence isn't
        tracked.
        """
        url = f"{self.base_url}/api/sequences/{start}"
        response = self._request_with_retries('GET', url)
        if response is None:
            self.logger.error(
                f"Tracker sequence fetch failed after {self.retry_attempts} attempts")
            return None
        if response.status_code == 404:
            self.logger.warning(f"Tracker does not track sequence {start}")
            return None
        body = _parse_body(response)
        if not response.ok or not body.get('success'):
            self.logger.warning(
                f"Tracker sequence fetch failed: {self._describe_error(response, body)}")
            return None
        return _parse_sequence(body.get('data', {}))

    def submit_factor(self, sequence_id: str, factor: str) -> TrackerSubmitResult:
        """Submit one factor of the sequence's current composite.

        Transient failures (network, 5xx, 429) are retried and reported with
        permanent=False; 4xx rejections are permanent=True after a single
        attempt (the tracker validated the factor and said no).
        """
        url = f"{self.base_url}/api/submit-factor"
        payload: Dict[str, Any] = {'sequenceId': sequence_id, 'factor': factor}
        if not self.api_key and self.submitter:
            payload['submitterHandle'] = self.submitter

        response = self._request_with_retries('POST', url, payload)
        if response is None:
            return TrackerSubmitResult(
                accepted=False, permanent=False,
                error=f"tracker unreachable after {self.retry_attempts} attempts")

        body = _parse_body(response)
        if response.ok and body.get('success'):
            data = body.get('data', {})
            return TrackerSubmitResult(
                accepted=True,
                auto_advanced=bool(data.get('autoAdvanced')),
                sequence=_parse_sequence(data),
            )

        error = self._describe_error(response, body)
        # 429 means retries were exhausted while rate-limited: transient.
        permanent = 400 <= response.status_code < 500 and response.status_code != 429
        return TrackerSubmitResult(accepted=False, permanent=permanent, error=error)
