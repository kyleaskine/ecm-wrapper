#!/usr/bin/env python3
"""
Base wrapper class containing shared functionality for ECM and YAFU wrappers.
"""
import yaml
import time
import logging
import requests
import datetime
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import subprocess


class BaseWrapper:
    """Base class for factorization wrappers with common functionality."""
    
    def __init__(self, config_path: str):
        """Initialize wrapper with configuration."""
        self._validate_working_directory()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.setup_logging()
        self.client_id = self.config['client']['id']
        self.api_endpoint = self.config['api']['endpoint']

    def _validate_working_directory(self):
        """Validate that we're running from the correct directory."""
        current_dir = Path.cwd()

        # Check if we're in the client directory by looking for key files
        expected_files = ['ecm-wrapper.py', 'yafu-wrapper.py', 'client.yaml', 'base_wrapper.py']
        missing_files = [f for f in expected_files if not (current_dir / f).exists()]

        if missing_files:
            print("🚨 WARNING: You appear to be running from the wrong directory!")
            print(f"   Current directory: {current_dir}")
            print("   Expected to be in: .../ecm-wrapper/client/")
            print(f"   Missing files: {', '.join(missing_files)}")
            print("   This may cause issues with file paths and data organization.")
            print("   Please run from the client/ directory for proper operation.\n")

            # Also check if we're one level up (in ecm-wrapper root)
            if (current_dir / 'client').exists():
                print("💡 TIP: Try running: cd client && python3 ecm-wrapper.py [args]")
            print()

    def setup_logging(self):
        """Set up logging configuration."""
        log_file = Path(self.config['logging']['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_factor_found(self, composite: str, factor: str, b1: Optional[int],
                        b2: Optional[int], curves: Optional[int],
                        method: str = "ecm", sigma: Optional[str] = None,
                        program: str = "unknown"):
        """Log found factors to a dedicated factors file."""
        factors_file = Path("data/factors_found.txt")
        factors_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(factors_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FACTOR FOUND: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Composite ({len(composite)} digits): {composite}\n")
            f.write(f"Factor: {factor}\n")
            if b1 is not None:
                f.write(f"Parameters: B1={b1}, B2={b2}, Curves={curves}")
                if sigma is not None:
                    f.write(f", Sigma={sigma}")
                f.write("\n")
            f.write(f"Program: {program} ({method.upper()} mode)\n")
            f.write(f"{'='*80}\n\n")
        
        # Also log to console with highlight
        print(f"\n🎉 FACTOR FOUND: {factor}")
        print(f"📋 Logged to: {factors_file}")
    
    def submit_result(self, results: Dict[str, Any], project: Optional[str] = None, 
                     program: str = "unknown") -> bool:
        """Submit results to API with retry logic."""
        # Handle different result formats
        factor_found = None
        if 'factor_found' in results:
            factor_found = results['factor_found']
        elif 'factors_found' in results and results['factors_found']:
            factor_found = results['factors_found'][0]  # Use first factor
        
        payload = {
            'composite': results['composite'],
            'project': project,
            'client_id': self.client_id,
            'method': results.get('method', 'ecm'),
            'program': program,
            'program_version': self.get_program_version(program),
            'parameters': {
                'b1': results.get('b1'),
                'b2': results.get('b2'),
                'curves': results.get('curves_requested'),
                'sigma': results.get('sigma')
            },
            'results': {
                'factor_found': factor_found,
                'curves_completed': results.get('curves_completed', 0),
                'execution_time': results.get('execution_time', 0)
            },
            'raw_output': results.get('raw_output', '')
        }
        
        url = f"{self.api_endpoint}/submit_result"
        retry_count = self.config['api']['retry_attempts']
        
        # Log submission attempt (only in debug mode)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Submitting to {url}")
            self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    url, 
                    json=payload,
                    timeout=self.config['api']['timeout']
                )
                if response.status_code != 200:
                    print(f"❌ Server response ({response.status_code}): {response.text}")
                response.raise_for_status()
                self.logger.info(f"Successfully submitted results: {response.json()}")
                
                # Submit additional factors if present
                if ('factors_found' in results and 
                    len(results['factors_found']) > 1):
                    self._submit_additional_factors(results, project, program)
                
                return True
            except requests.exceptions.RequestException as e:
                error_details = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = f" - Response: {e.response.text}"
                    except:
                        pass
                self.logger.error(f"API submission failed (attempt {attempt + 1}): {e}{error_details}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Save failed submission for later retry
        self._save_failed_submission(results, payload)
        self.logger.error(f"Failed to submit results after {retry_count} attempts")
        
        return False
    
    def _save_failed_submission(self, results: Dict[str, Any], payload: Dict[str, Any]):
        """Save failed submission for later retry."""
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("data/results")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp and composite hash
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            composite_hash = hashlib.md5(results['composite'].encode()).hexdigest()[:8]
            filename = f"failed_submission_{timestamp}_{composite_hash}.json"
            
            # Combine original results with API payload for context
            save_data = {
                **results,
                'api_payload': payload,
                'submitted': False,
                'failed_at': datetime.datetime.now().isoformat(),
                'retry_count': self.config['api']['retry_attempts']
            }
            
            filepath = data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Saved failed submission to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save submission data: {e}")
    
    def _submit_additional_factors(self, results: Dict[str, Any], 
                                  project: Optional[str], program: str):
        """Submit additional factors found in the same run."""
        for factor in results['factors_found'][1:]:
            payload = {
                'composite': results['composite'],
                'project': project,
                'client_id': self.client_id,
                'method': results.get('method', 'ecm'),
                'program': program,
                'program_version': self.get_program_version(program),
                'parameters': {
                    'b1': results.get('b1'),
                    'b2': results.get('b2'),
                    'curves': results.get('curves_requested'),
                    'sigma': results.get('sigma')
                },
                'results': {
                    'factor_found': factor,
                    'curves_completed': 0,  # Additional factor from same run
                    'execution_time': 0
                },
                'raw_output': f"Additional factor from same run: {factor}"
            }
            
            try:
                response = requests.post(
                    f"{self.api_endpoint}/submit_result",
                    json=payload,
                    timeout=self.config['api']['timeout']
                )
                if response.status_code != 200:
                    self.logger.error(f"Additional factor submission failed ({response.status_code}): {response.text}")
                else:
                    result = response.json()
                    self.logger.info(f"Submitted additional factor: {factor} - {result}")
            except Exception as e:
                self.logger.error(f"Failed to submit additional factor {factor}: {e}")
    
    def create_base_results(self, composite: str, method: str = "ecm", **kwargs) -> Dict[str, Any]:
        """Create standardized results dictionary."""
        return {
            'composite': composite,
            'method': method,
            'factor_found': None,
            'factors_found': [],
            'curves_completed': 0,
            'curves_requested': kwargs.get('curves', 0),
            'execution_time': 0,
            'raw_output': '',
            'b1': kwargs.get('b1'),
            'b2': kwargs.get('b2'),
            'sigma': kwargs.get('sigma'),
            **kwargs
        }

    def run_subprocess_with_parsing(self, cmd: List[str], timeout: int,
                                  composite: str, method: str,
                                  parse_function: Callable,
                                  **kwargs) -> Dict[str, Any]:
        """
        Unified subprocess execution with parsing for factorization programs.

        Args:
            cmd: Command list to execute
            timeout: Timeout in seconds
            composite: Number being factored
            method: Method name (ecm, pm1, pp1, etc.)
            parse_function: Function to parse output (factor, sigma)
            **kwargs: Additional parameters for results
        """
        start_time = time.time()
        results = self.create_base_results(composite, method, **kwargs)

        try:
            self.logger.info(f"Running {method.upper()} on {len(composite)}-digit number with {' '.join(cmd[1:3])}")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send input and get output (only if input is specified)
            program_input = kwargs.get('input')
            if program_input:
                stdout, _ = process.communicate(input=program_input, timeout=timeout)
            else:
                stdout, _ = process.communicate(timeout=timeout)
            results['raw_output'] = stdout

            # Parse for factors using provided function
            parse_result = parse_function(stdout)
            if isinstance(parse_result, tuple) and len(parse_result) == 2:
                # Single factor result (factor, sigma) - from ECM parsing
                factor, sigma = parse_result
                if factor:
                    results['factor_found'] = factor
                    results['factors_found'] = [factor]
                    results['sigma'] = sigma
            elif isinstance(parse_result, list):
                # Multiple factors result [(factor, sigma), ...] - from YAFU parsing
                if parse_result:
                    results['factors_found'] = [f[0] for f in parse_result]
                    results['factor_found'] = parse_result[0][0]  # First factor
                    results['sigma'] = parse_result[0][1] if parse_result[0][1] else None

            # Parse curves completed for YAFU methods
            if kwargs.get('track_curves', False):
                from parsing_utils import YAFUPatterns
                curves_match = YAFUPatterns.CURVES_COMPLETED.search(stdout)
                if curves_match:
                    results['curves_completed'] = int(curves_match.group(1))
                else:
                    # Try alternative format
                    progress_match = YAFUPatterns.CURVE_PROGRESS.search(stdout)
                    if progress_match:
                        results['curves_completed'] = int(progress_match.group(1))

            results['success'] = process.returncode == 0

        except subprocess.TimeoutExpired:
            self.logger.error(f"{method.upper()} timed out after {timeout} seconds")
            process.kill()
            results['success'] = False
            results['timeout'] = True
        except Exception as e:
            self.logger.error(f"{method.upper()} execution failed: {e}")
            results['success'] = False
            results['error'] = str(e)

        results['execution_time'] = time.time() - start_time
        return results

    def get_program_version(self, program: str) -> str:
        """Get program version - to be implemented by subclasses."""
        return "unknown"
    
    def save_raw_output(self, results: Dict[str, Any], program: str = "unknown"):
        """Save raw output to file for debugging."""
        output_dir = Path(self.config['execution']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        method = results.get('method', 'unknown')
        curves = results.get('curves_completed', 0)
        filename = output_dir / f"{program}_{method}_{timestamp}_{curves}curves.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Composite: {results['composite']}\n")
            if 'b1' in results:
                f.write(f"B1: {results['b1']}, B2: {results.get('b2')}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Program: {program}\n")
            
            # Handle both single factor and multiple factors
            if 'factor_found' in results:
                f.write(f"Factor found: {results['factor_found']}\n")
            elif 'factors_found' in results:
                f.write(f"Factors found: {results['factors_found']}\n")
                
            f.write(f"Curves completed: {results.get('curves_completed', 0)}\n")
            f.write(f"Execution time: {results.get('execution_time', 0):.2f}s\n\n")
            f.write(results.get('raw_output', ''))