"""
API Client Utility

Provides unified API communication with retry logic, error handling,
and failed submission persistence.
"""

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import requests
from .file_utils import save_json

logger = logging.getLogger(__name__)


class ResourceNotFoundError(Exception):
    """Raised when an API resource returns 404 (permanently gone)."""
    pass


class APIClient:
    """
    Handle API communication with retry logic and failure persistence.

    This utility consolidates API submission patterns, handling:
    - HTTP POST requests with retry logic and exponential backoff
    - Failed submission persistence for later retry
    - Response parsing and error handling
    """

    def __init__(self, api_endpoint: str, timeout: int = 30, retry_attempts: int = 3):
        """
        Initialize API client.

        Args:
            api_endpoint: Base API endpoint URL (e.g., 'http://localhost:8000')
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
        """
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.logger = logging.getLogger(f"{__name__}.APIClient")

    def _retry_with_exponential_backoff(
        self,
        operation_name: str,
        api_call_func: Callable[[], requests.Response]
    ) -> Optional[requests.Response]:
        """
        Execute an API call with exponential backoff retry logic.

        Args:
            operation_name: Name of operation for logging
            api_call_func: Function that makes the API call and returns response

        Returns:
            Response object if successful, None if all retries failed
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = api_call_func()
                response.raise_for_status()
                self.logger.info(f"{operation_name} succeeded")
                return response
            except requests.exceptions.HTTPError as e:
                # Don't retry on 404 - resource is permanently gone
                if e.response is not None and e.response.status_code == 404:
                    self.logger.warning(f"{operation_name} failed: resource not found (404)")
                    raise ResourceNotFoundError(f"{operation_name}: resource not found (404)")
                # Fall through to retry logic for other HTTP errors
                if attempt < self.retry_attempts:
                    delay = 2 ** attempt
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt}/{self.retry_attempts}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    try:
                        time.sleep(delay)
                    except KeyboardInterrupt:
                        self.logger.info(f"{operation_name} interrupted during retry wait")
                        return None
                else:
                    self.logger.error(
                        f"{operation_name} failed after {self.retry_attempts} attempts: {e}"
                    )
                    return None
            except requests.RequestException as e:
                if attempt < self.retry_attempts:
                    delay = 2 ** attempt
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt}/{self.retry_attempts}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    try:
                        time.sleep(delay)
                    except KeyboardInterrupt:
                        self.logger.info(f"{operation_name} interrupted during retry wait")
                        return None
                else:
                    self.logger.error(
                        f"{operation_name} failed after {self.retry_attempts} attempts: {e}"
                    )
                    return None
        return None

    def submit_result(
        self, payload: Dict[str, Any], save_on_failure: bool = True,
        results_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Submit result to API with retry logic.

        Args:
            payload: API payload to submit
            save_on_failure: Whether to save failed submissions to disk
            results_context: Optional full results dict for failure persistence

        Returns:
            Response dictionary with keys: status, attempt_id, composite_id, message, factor_status
            Returns None if submission failed
        """
        url = f"{self.api_endpoint}/submit_result"

        # Log submission attempt (only in debug mode)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Submitting to {url}")
            self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Use centralized retry logic
        response = self._retry_with_exponential_backoff(
            "Result submission",
            lambda: requests.post(url, json=payload, timeout=self.timeout)
        )

        if response is not None:
            try:
                response_data = response.json()
                self.logger.info(f"Successfully submitted results: {response_data}")
                return response_data
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse response JSON: {e}")

        # Submission failed - save for later retry if requested
        if save_on_failure and results_context:
            self.save_failed_submission(results_context, payload)

        return None

    def save_failed_submission(
        self, results: Dict[str, Any], payload: Dict[str, Any],
        output_dir: str = "data/results"
    ) -> Optional[str]:
        """
        Save failed submission for later retry.

        Args:
            results: Full results dictionary
            payload: API payload that failed to submit
            output_dir: Directory to save failed submissions

        Returns:
            Path to saved file, or None if save failed
        """
        try:
            # Create filename with timestamp and composite hash
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            composite = results.get('composite', 'unknown')
            composite_hash = hashlib.md5(composite.encode()).hexdigest()[:8]
            filename = f"failed_submission_{timestamp}_{composite_hash}.json"

            # Combine original results with API payload for context
            save_data = {
                **results,
                'api_payload': payload,
                'submitted': False,
                'failed_at': datetime.datetime.now().isoformat(),
                'retry_count': self.retry_attempts
            }

            filepath = Path(output_dir) / filename
            if save_json(filepath, save_data):
                self.logger.info(f"Saved failed submission to: {filepath}")
                return str(filepath)
            return None

        except (OSError, IOError) as e:
            self.logger.error(f"Failed to save submission data - I/O error: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error saving submission data: {e}")
            return None

    def build_submission_payload(
        self,
        composite: str,
        client_id: str,
        method: str,
        program: str,
        program_version: str,
        results: Dict[str, Any],
        project: Optional[str] = None,
        residue_checksum: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build standard API submission payload.

        Args:
            composite: Composite number being factored
            client_id: Client identifier
            method: Factorization method (ecm, pm1, pp1, etc.)
            program: Program name (gmp-ecm, yafu, etc.)
            program_version: Program version string
            results: Results dictionary with execution data
            project: Optional project name
            residue_checksum: Optional SHA-256 checksum of residue file (for stage 2 from residue pool)

        Returns:
            Formatted API payload dictionary
        """
        # Handle different result formats for factor_found (backward compatibility)
        factor_found = None
        if 'factor_found' in results:
            factor_found = results['factor_found']
        elif 'factors_found' in results and results['factors_found']:
            factor_found = results['factors_found'][0]  # Use first factor

        # Build factors_found list - use ecm_found_factors if available (excludes cofactor primes)
        # Otherwise fall back to factors_found for backward compatibility
        factors_to_submit = results.get('ecm_found_factors', results.get('factors_found', []))
        cofactor_primes = results.get('cofactor_primes', [])

        if cofactor_primes:
            self.logger.info(f"Excluding {len(cofactor_primes)} cofactor prime(s) from API submission: {cofactor_primes}")

        factors_found_list = None
        if factors_to_submit and len(factors_to_submit) > 0:
            factor_sigmas = results.get('factor_sigmas', {})
            factors_found_list = []

            self.logger.debug(f"Building factors_found list from {len(factors_to_submit)} ECM-found factors")
            self.logger.debug(f"factor_sigmas available: {list(factor_sigmas.keys()) if factor_sigmas else 'None'}")

            for factor in factors_to_submit:
                # Get sigma for this specific factor, or use main sigma
                factor_sigma = factor_sigmas.get(factor, results.get('sigma'))
                factors_found_list.append({
                    'factor': factor,
                    'sigma': str(factor_sigma) if factor_sigma is not None else None
                })
                self.logger.debug(f"  Added factor: {factor[:20]}... with sigma: {factor_sigma}")

        payload = {
            'composite': composite,
            'project': project,
            'client_id': client_id,
            'method': method,
            'program': program,
            'program_version': program_version,
            'parameters': {
                'b1': results.get('b1'),
                'b2': results.get('b2'),
                'curves': results.get('curves_requested'),
                'parametrization': results.get('parametrization', 3),  # Default to param 3
                'sigma': results.get('sigma')
            },
            'results': {
                'factor_found': factor_found,  # Legacy field for backward compatibility
                'factors_found': factors_found_list,  # New field for multiple factors
                'curves_completed': results.get('curves_completed', 0),
                'execution_time': results.get('execution_time', 0)
            },
            'raw_output': results.get('raw_output', ''),
            'residue_checksum': residue_checksum,  # For stage 2 work from residue pool
            # Pins the submission to its work assignment so the server can
            # resolve the composite by identity instead of string matching
            'work_id': results.get('work_id')
        }

        # Debug logging for multi-factor submissions
        if factors_found_list and len(factors_found_list) > 1:
            self.logger.info(f"Built payload with {len(factors_found_list)} factors: {[f['factor'][:20] + '...' for f in factors_found_list]}")
            self.logger.debug(f"DEBUG: Full factors_found_list structure: {factors_found_list}")

        return payload

    def health_check(self) -> bool:
        """
        Check if API endpoint is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        url = f"{self.api_endpoint}/health"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            self.logger.debug("API health check passed")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API health check failed: {e}")
            return False

    def get_ecm_work(
        self,
        client_id: str,
        min_target_tlevel: Optional[float] = None,
        max_target_tlevel: Optional[float] = None,
        max_current_tlevel: Optional[float] = None,
        priority: Optional[int] = None,
        min_digits: Optional[int] = None,
        max_digits: Optional[int] = None,
        timeout_days: int = 1,
        work_type: str = "standard",
        project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request ECM work assignment from server.

        Args:
            client_id: Client identifier
            min_target_tlevel: Minimum target t-level (filter by difficulty)
            max_target_tlevel: Maximum target t-level (filter by difficulty)
            priority: Minimum priority filter (optional)
            timeout_days: Work assignment expiration in days (default: 1)
            work_type: Work assignment strategy - "standard" (smallest first) or "progressive" (least ECM done first)

        Returns:
            Work assignment dictionary with keys:
                - work_id: Work assignment ID
                - composite_id: Database ID of composite
                - composite: Number to factor
                - digit_length: Number of digits
                - current_t_level: Current t-level progress
                - target_t_level: Target t-level
                - expires_at: Expiration timestamp
            Returns None if no work available or on error
        """
        url = f"{self.api_endpoint}/ecm-work"

        # Build query parameters
        params = {'client_id': client_id, 'timeout_days': timeout_days, 'work_type': work_type}
        if min_target_tlevel is not None:
            params['min_target_tlevel'] = min_target_tlevel
        if max_target_tlevel is not None:
            params['max_target_tlevel'] = max_target_tlevel
        if max_current_tlevel is not None:
            params['max_current_tlevel'] = max_current_tlevel
        if priority is not None:
            params['priority'] = priority
        if min_digits is not None:
            params['min_digits'] = min_digits
        if max_digits is not None:
            params['max_digits'] = max_digits
        if project is not None:
            params['project'] = project

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Check if work was assigned
            if data.get('work_id'):
                self.logger.info(
                    f"Received work assignment: composite {data['composite'][:30]}... "
                    f"({data['digit_length']} digits, work_id={data['work_id']})"
                )
                return data
            else:
                # No work available
                message = data.get('message', 'No work available')
                self.logger.info(f"No work available: {message}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to request work: {e}")
            return None

    def get_pm1_work(
        self,
        client_id: str,
        min_digits: Optional[int] = None,
        max_digits: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Request P-1 factorization work from server.

        Calls the /work endpoint with methods=["pm1"] to request P-1 specific
        work assignments. The server determines optimal B1/B2 parameters.

        Args:
            client_id: Client identifier
            min_digits: Minimum composite digit length filter
            max_digits: Maximum composite digit length filter

        Returns:
            Work assignment dictionary with keys:
                - work_id: Work assignment ID
                - composite: Number to factor
                - method: 'pm1'
                - parameters: {b1, b2, curves=1}
                - expires_at: Expiration timestamp
            Returns None if no work available or on error
        """
        url = f"{self.api_endpoint}/work"

        params: Dict[str, Any] = {
            'client_id': client_id,
            'methods': 'pm1',
        }
        if min_digits is not None:
            params['min_digits'] = min_digits
        if max_digits is not None:
            params['max_digits'] = max_digits

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            if data.get('work_id'):
                composite = data.get('composite', '')
                self.logger.info(
                    f"Received PM1 work assignment: composite {composite[:30]}... "
                    f"(work_id={data['work_id']}, B1={data.get('parameters', {}).get('b1')})"
                )
                return data
            else:
                message = data.get('message', 'No PM1 work available')
                self.logger.info(f"No PM1 work available: {message}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to request PM1 work: {e}")
            return None

    def get_p1_work(
        self,
        client_id: str,
        method: str = "p1",
        min_target_tlevel: Optional[float] = None,
        max_target_tlevel: Optional[float] = None,
        priority: Optional[int] = None,
        min_digits: Optional[int] = None,
        max_digits: Optional[int] = None,
        timeout_days: int = 1,
        work_type: str = "standard",
        project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request P-1/P+1 work assignment from server.

        Calls the /p1-work endpoint which only assigns composites that haven't
        had PM1/PP1 done at the required B1 level.

        Args:
            client_id: Client identifier
            method: "pm1" (P-1 only), "pp1" (P+1 only), or "p1" (both)
            min_target_tlevel: Minimum target t-level filter
            max_target_tlevel: Maximum target t-level filter
            priority: Minimum priority filter
            timeout_days: Work assignment expiration in days
            work_type: "standard" or "progressive"

        Returns:
            Work assignment dictionary with keys:
                - work_id, composite_id, composite, digit_length
                - current_t_level, target_t_level
                - pm1_b1, pp1_b1: Server-computed B1 values
                - expires_at
            Returns None if no work available or on error
        """
        url = f"{self.api_endpoint}/p1-work"

        params: Dict[str, Any] = {
            'client_id': client_id,
            'method': method,
            'timeout_days': timeout_days,
            'work_type': work_type,
        }
        if min_target_tlevel is not None:
            params['min_target_tlevel'] = min_target_tlevel
        if max_target_tlevel is not None:
            params['max_target_tlevel'] = max_target_tlevel
        if priority is not None:
            params['priority'] = priority
        if min_digits is not None:
            params['min_digits'] = min_digits
        if max_digits is not None:
            params['max_digits'] = max_digits
        if project is not None:
            params['project'] = project

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            if data.get('work_id'):
                self.logger.info(
                    f"Received P1 work assignment: composite {data['composite'][:30]}... "
                    f"({data['digit_length']} digits, method={method}, "
                    f"pm1_b1={data.get('pm1_b1')}, pp1_b1={data.get('pp1_b1')}, "
                    f"work_id={data['work_id']})"
                )
                return data
            else:
                message = data.get('message', 'No P1 work available')
                self.logger.info(f"No P1 work available: {message}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to request P1 work: {e}")
            return None

    def complete_work(self, work_id: str, client_id: str) -> bool:
        """
        Mark work assignment as completed.

        Args:
            work_id: Work assignment ID to complete
            client_id: Client ID completing the work

        Returns:
            True if successfully marked complete, False otherwise
        """
        url = f"{self.api_endpoint}/work/{work_id}/complete"

        def api_call():
            return requests.post(url, params={'client_id': client_id}, timeout=self.timeout)

        response = self._retry_with_exponential_backoff(f"Mark work {work_id} as complete", api_call)
        return response is not None

    def abandon_work(self, work_id: str, reason: str = "client_terminated", client_id: Optional[str] = None) -> bool:
        """
        Abandon work assignment (release it back to the pool).

        Args:
            work_id: Work assignment ID to abandon
            reason: Reason for abandoning (optional)
            client_id: Client ID that owns the work (required for server validation)

        Returns:
            True if successfully abandoned, False otherwise
        """
        url = f"{self.api_endpoint}/work/{work_id}"

        # Build query parameters
        params = {'reason': reason}
        if client_id:
            params['client_id'] = client_id

        def api_call():
            return requests.delete(url, params=params, timeout=self.timeout)

        try:
            response = self._retry_with_exponential_backoff(f"Abandon work {work_id}", api_call)
            return response is not None
        except ResourceNotFoundError:
            # 404 = work already gone (expired, completed, etc.) - abandon goal achieved
            self.logger.info(f"Work {work_id} already gone (404), treating abandon as success")
            return True

    # ==================== Residue Management Methods ====================

    def upload_residue(
        self,
        client_id: str,
        residue_file_path: str,
        stage1_attempt_id: Optional[int] = None,
        expiry_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Upload a residue file after completing stage 1 ECM.

        Args:
            client_id: Client identifier
            residue_file_path: Path to the residue file
            stage1_attempt_id: Optional ID of the stage 1 attempt to link
            expiry_days: Days until residue expires if not consumed

        Returns:
            Dictionary with residue info (residue_id, composite, b1, curves, etc.)
            Returns None on error
        """
        url = f"{self.api_endpoint}/residues/upload"

        try:
            with open(residue_file_path, 'rb') as f:
                files = {'file': (Path(residue_file_path).name, f, 'text/plain')}
                headers = {'X-Client-ID': client_id}
                params = {'expiry_days': expiry_days}
                if stage1_attempt_id is not None:
                    params['stage1_attempt_id'] = stage1_attempt_id

                response = requests.post(
                    url,
                    files=files,
                    headers=headers,
                    params=params,
                    timeout=120  # Longer timeout for file upload
                )
                response.raise_for_status()

                data = response.json()
                self.logger.info(
                    f"Uploaded residue: ID {data['residue_id']}, "
                    f"{data['curve_count']} curves, B1={data['b1']}"
                )
                return data

        except FileNotFoundError:
            self.logger.error(f"Residue file not found: {residue_file_path}")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                try:
                    detail = e.response.json().get("detail", "")
                except (ValueError, AttributeError):
                    detail = e.response.text
                if "Duplicate residue" in detail or "checksum matches" in detail:
                    self.logger.info(f"Residue already uploaded (server confirmed duplicate): {detail}")
                    return {"already_uploaded": True, "detail": detail}
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = f" - Response: {e.response.text}"
                except (AttributeError, ValueError, UnicodeDecodeError):
                    pass
            self.logger.error(f"Failed to upload residue: {e}{error_details}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to upload residue: {e}")
            return None

    def get_residue_work(
        self,
        client_id: str,
        min_target_tlevel: Optional[float] = None,
        max_target_tlevel: Optional[float] = None,
        min_priority: Optional[int] = None,
        min_b1: Optional[int] = None,
        max_b1: Optional[int] = None,
        claim_timeout_hours: int = 24,
        project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request stage 2 work (available residue file).

        Args:
            client_id: Client identifier
            min_target_tlevel: Minimum target t-level
            max_target_tlevel: Maximum target t-level
            min_priority: Minimum composite priority
            min_b1: Minimum B1 bound of residue
            max_b1: Maximum B1 bound of residue
            claim_timeout_hours: Hours until claim expires

        Returns:
            Dictionary with residue work info (residue_id, composite, b1, download_url, etc.)
            Returns None if no work available or on error
        """
        url = f"{self.api_endpoint}/residues/work"

        headers = {'X-Client-ID': client_id}
        params: Dict[str, Any] = {'claim_timeout_hours': claim_timeout_hours}
        if min_target_tlevel is not None:
            params['min_target_tlevel'] = min_target_tlevel
        if max_target_tlevel is not None:
            params['max_target_tlevel'] = max_target_tlevel
        if min_priority is not None:
            params['min_priority'] = min_priority
        if min_b1 is not None:
            params['min_b1'] = min_b1
        if max_b1 is not None:
            params['max_b1'] = max_b1
        if project is not None:
            params['project'] = project

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if data.get('residue_id'):
                self.logger.info(
                    f"Received residue work: ID {data['residue_id']}, "
                    f"{data['curve_count']} curves, B1={data['b1']}, "
                    f"composite {data['composite'][:30]}..."
                )
                return data
            else:
                message = data.get('message', 'No residues available')
                self.logger.info(f"No residue work available: {message}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to request residue work: {e}")
            return None

    def download_residue(
        self,
        client_id: str,
        residue_id: int,
        output_path: str
    ) -> bool:
        """
        Download a residue file for stage 2 processing.

        Args:
            client_id: Client identifier (must match claimer)
            residue_id: ID of the residue to download
            output_path: Local path to save the downloaded file

        Returns:
            True if download succeeded, False otherwise
        """
        url = f"{self.api_endpoint}/residues/{residue_id}/download"

        headers = {'X-Client-ID': client_id}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=120,  # Longer timeout for file download
                    stream=True
                )
                response.raise_for_status()

                # Write file in chunks
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                file_size = Path(output_path).stat().st_size
                self.logger.info(
                    f"Downloaded residue {residue_id} to {output_path} ({file_size} bytes)"
                )
                return True

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Download attempt {attempt + 1}/{max_retries} failed for residue {residue_id}: {e}")
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    self.logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return False
            except IOError as e:
                self.logger.error(f"Failed to write residue file: {e}")
                return False
        return False

    def complete_residue(
        self,
        client_id: str,
        residue_id: int,
        stage2_attempt_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Mark residue as completed after stage 2 finishes.

        This supersedes the stage 1 attempt and deletes the residue file on the server.

        Args:
            client_id: Client identifier
            residue_id: ID of the completed residue
            stage2_attempt_id: ID of the stage 2 ECM attempt

        Returns:
            Dictionary with completion info (new_t_level, etc.)
            Returns None on error
        """
        url = f"{self.api_endpoint}/residues/{residue_id}/complete"

        headers = {'X-Client-ID': client_id}
        payload = {'stage2_attempt_id': stage2_attempt_id}

        def api_call():
            return requests.post(url, json=payload, headers=headers, timeout=self.timeout)

        response = self._retry_with_exponential_backoff(f"Complete residue {residue_id}", api_call)
        return response.json() if response else None

    def abandon_residue(self, client_id: str, residue_id: int) -> bool:
        """
        Release a claimed residue back to the available pool.

        Args:
            client_id: Client identifier
            residue_id: ID of the residue to release

        Returns:
            True if successfully released, False otherwise
        """
        url = f"{self.api_endpoint}/residues/{residue_id}/claim"

        headers = {'X-Client-ID': client_id}

        def api_call():
            return requests.delete(url, headers=headers, timeout=self.timeout)

        try:
            response = self._retry_with_exponential_backoff(f"Release residue {residue_id}", api_call)
            return response is not None
        except ResourceNotFoundError:
            # 404 = residue already gone (expired, completed, etc.) - release goal achieved
            self.logger.info(f"Residue {residue_id} already gone (404), treating release as success")
            return True

    # ==================== Composite Status Methods ====================

    def get_composite_status(self, composite: str) -> Optional[Dict[str, Any]]:
        """
        Query a composite's current status from the server.

        Args:
            composite: The composite number to query

        Returns:
            Dictionary with composite info:
                - composite: Original number
                - current_composite: Current unfactored portion
                - digit_length: Number of digits
                - target_t_level: Target t-level
                - current_t_level: Current t-level progress
                - status: 'composite', 'prime', or 'fully_factored'
                - factors_found: List of known factors
                - ecm_work: Summary of ECM work done
            Returns None if composite not found or on error
        """
        url = f"{self.api_endpoint}/stats/{composite}"

        try:
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 404:
                self.logger.info(f"Composite not found in server database: {composite[:30]}...")
                return None

            response.raise_for_status()
            data = response.json()

            self.logger.info(
                f"Got status for composite: {data['digit_length']} digits, "
                f"t-level {data.get('current_t_level', 0):.1f}/{data.get('target_t_level', 0):.1f}"
            )
            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get composite status: {e}")
            return None
