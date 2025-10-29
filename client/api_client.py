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
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)


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

    def submit_result(
        self, payload: Dict[str, Any], save_on_failure: bool = True,
        results_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit result to API with retry logic.

        Args:
            payload: API payload to submit
            save_on_failure: Whether to save failed submissions to disk
            results_context: Optional full results dict for failure persistence

        Returns:
            True if submission succeeded, False otherwise
        """
        url = f"{self.api_endpoint}/submit_result"

        # Log submission attempt (only in debug mode)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Submitting to {url}")
            self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    self.logger.error(
                        f"Server response ({response.status_code}): {response.text}"
                    )

                response.raise_for_status()
                self.logger.info(f"Successfully submitted results: {response.json()}")
                return True

            except requests.exceptions.RequestException as e:
                error_details = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = f" - Response: {e.response.text}"
                    except:
                        pass

                self.logger.error(
                    f"API submission failed (attempt {attempt + 1}): {e}{error_details}"
                )

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    backoff_time = 2 ** attempt
                    self.logger.debug(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)

        # All retry attempts failed
        if save_on_failure and results_context:
            self.save_failed_submission(results_context, payload)

        self.logger.error(f"Failed to submit results after {self.retry_attempts} attempts")
        return False

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
            # Create data directory if it doesn't exist
            data_dir = Path(output_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

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

            filepath = data_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)

            self.logger.info(f"Saved failed submission to: {filepath}")
            return str(filepath)

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
        project: Optional[str] = None
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

        Returns:
            Formatted API payload dictionary
        """
        # Handle different result formats for factor_found (backward compatibility)
        factor_found = None
        if 'factor_found' in results:
            factor_found = results['factor_found']
        elif 'factors_found' in results and results['factors_found']:
            factor_found = results['factors_found'][0]  # Use first factor

        # Build factors_found list if we have multiple factors
        factors_found_list = None
        if 'factors_found' in results and len(results['factors_found']) > 0:
            factor_sigmas = results.get('factor_sigmas', {})
            factors_found_list = []

            self.logger.debug(f"Building factors_found list from {len(results['factors_found'])} factors")
            self.logger.debug(f"factor_sigmas available: {list(factor_sigmas.keys()) if factor_sigmas else 'None'}")

            for factor in results['factors_found']:
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
            'raw_output': results.get('raw_output', '')
        }

        # Debug logging for multi-factor submissions
        if factors_found_list and len(factors_found_list) > 1:
            self.logger.info(f"Built payload with {len(factors_found_list)} factors: {[f['factor'][:20] + '...' for f in factors_found_list]}")
            self.logger.debug(f"DEBUG: Full factors_found_list structure: {factors_found_list}")

        return payload


    def get_work_assignment(
        self,
        client_id: str,
        methods: Optional[str] = None,
        max_digits: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request work assignment from API.

        Args:
            client_id: Client identifier
            methods: Comma-separated factorization methods (e.g., 'ecm,pm1')
            max_digits: Maximum composite digit length to request

        Returns:
            Work assignment dictionary with keys:
                - work_id: Unique work assignment ID
                - composite: Number to factor
                - method: Factorization method (ecm, pm1, etc.)
                - parameters: Dict with b1, b2, curves
                - estimated_time_minutes: Estimated completion time
                - expires_at: Work assignment expiration timestamp
                - message: Optional message if no work available
            Or None if request failed
        """
        url = f"{self.api_endpoint}/work"
        params = {'client_id': client_id}

        if methods:
            params['methods'] = methods
        if max_digits is not None:
            params['max_digits'] = max_digits

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            work_assignment = response.json()

            # Check if we got actual work or just a message
            if 'work_id' in work_assignment:
                composite = work_assignment.get('composite', '')
                self.logger.info(
                    f"Received work assignment: {work_assignment['work_id']} - "
                    f"{work_assignment.get('method')} on {len(composite)} digit number"
                )
            elif 'message' in work_assignment:
                self.logger.info(f"No work available: {work_assignment['message']}")

            return work_assignment

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get work assignment: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    self.logger.error(f"Server error detail: {error_detail}")
                except:
                    self.logger.error(f"Server response: {e.response.text}")
            return None

    def claim_work(self, work_id: str, client_id: str) -> bool:
        """
        Claim work assignment before execution.

        Args:
            work_id: Work assignment ID
            client_id: Client identifier

        Returns:
            True if claim succeeded, False otherwise
        """
        url = f"{self.api_endpoint}/work/{work_id}/claim"
        params = {'client_id': client_id}

        try:
            response = requests.post(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            self.logger.debug(f"Successfully claimed work {work_id}")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to claim work {work_id}: {e}")
            return False

    def start_work(self, work_id: str, client_id: str) -> bool:
        """
        Mark work as started.

        Args:
            work_id: Work assignment ID
            client_id: Client identifier

        Returns:
            True if start succeeded, False otherwise
        """
        url = f"{self.api_endpoint}/work/{work_id}/start"
        params = {'client_id': client_id}

        try:
            response = requests.post(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            self.logger.debug(f"Successfully started work {work_id}")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to start work {work_id}: {e}")
            return False

    def complete_work(
        self,
        work_id: str,
        client_id: str,
        curves_completed: int = 0,
        execution_time: float = 0.0
    ) -> bool:
        """
        Mark work assignment as completed.

        Args:
            work_id: Work assignment ID
            client_id: Client identifier
            curves_completed: Number of curves completed (currently unused by server)
            execution_time: Total execution time in seconds (currently unused by server)

        Returns:
            True if completion succeeded, False otherwise
        """
        url = f"{self.api_endpoint}/work/{work_id}/complete"
        params = {'client_id': client_id}

        try:
            response = requests.post(
                url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            self.logger.info(f"Successfully marked work {work_id} as completed")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to complete work {work_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    self.logger.error(f"Server error detail: {error_detail}")
                except:
                    self.logger.error(f"Server response: {e.response.text}")
            return False

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
