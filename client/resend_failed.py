#!/usr/bin/env python3
"""
Resend Failed Results Script

This script finds ECM attempts that were completed but failed to submit to the API,
and retries the submission.
"""

import json
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import BaseWrapper to reuse config loading logic
from base_wrapper import BaseWrapper

class FailedResultsResender(BaseWrapper):
    """Handles resending of failed results from local logs."""

    def __init__(self, config_path: str = 'client.yaml', dry_run: bool = False):
        """Initialize with configuration."""
        # Use BaseWrapper's config loading (handles client.local.yaml merge)
        super().__init__(config_path)

        self.dry_run = dry_run

    def find_failed_results(self) -> List[Dict[str, Any]]:
        """Find results files that may have failed to submit."""
        failed_results = []

        # Look for result files in data directory
        data_dir = Path("data/results")
        if not data_dir.exists():
            print(f"⚠️  Results directory not found: {data_dir}")
            return failed_results

        # Find JSON result files
        result_files = glob.glob(str(data_dir / "*.json"))

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)

                # Check if this result was successfully submitted
                if not result_data.get('submitted', False):
                    result_data['_file_path'] = result_file
                    failed_results.append(result_data)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Error reading {result_file}: {e}")

        return failed_results

    def parse_log_for_failed_attempts(self, log_file: str = "data/logs/ecm_client.log") -> List[Dict[str, Any]]:
        """Parse log files to find attempts that completed but failed to submit."""
        failed_attempts = []
        log_path = Path(log_file)

        if not log_path.exists():
            print(f"⚠️  Log file not found: {log_file}")
            return failed_attempts

        print(f"📋 Parsing log file: {log_file}")

        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()

            current_attempt = None

            for line in lines:
                line = line.strip()

                # Look for completed attempts
                if "No factor found" in line or "Factor found" in line:
                    # This marks a completed attempt
                    if current_attempt:
                        failed_attempts.append(current_attempt)
                    current_attempt = {"completed": True}

                # Look for submission failures
                elif "API submission failed" in line and current_attempt:
                    current_attempt["submission_failed"] = True

                # Extract result data from log lines
                elif "Successfully submitted results" in line:
                    # This attempt succeeded, don't include it
                    current_attempt = None

            # Add the last attempt if it failed
            if current_attempt and current_attempt.get("submission_failed"):
                failed_attempts.append(current_attempt)

        except FileNotFoundError:
            print(f"⚠️  Could not read log file: {log_file}")

        return failed_attempts

    def create_submission_from_saved_data(self, result_file: str) -> Optional[Dict[str, Any]]:
        """Create API submission from saved result data."""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # If the saved file has the original api_payload, use it directly
            if 'api_payload' in data:
                return data['api_payload']

            # Otherwise reconstruct from saved data (legacy format)
            # Use client_id from BaseWrapper (already constructed as username-cpu_name)

            # Build factors_found list if available (new format)
            factors_found_list = None
            if 'factors_found' in data and data['factors_found']:
                factor_sigmas = data.get('factor_sigmas', {})
                factors_found_list = []
                for factor in data['factors_found']:
                    sigma = factor_sigmas.get(factor, data.get('sigma'))
                    factors_found_list.append({
                        'factor': factor,
                        'sigma': str(sigma) if sigma else None
                    })

            submission = {
                "composite": data.get("composite"),
                "client_id": self.client_id,
                "method": data.get("method", "ecm"),
                "program": data.get("program", "gmp-ecm"),
                "program_version": data.get("program_version"),
                "parameters": {
                    "b1": data.get("b1"),
                    "b2": data.get("b2"),
                    "curves": data.get("curves_requested"),
                    "parametrization": data.get("parametrization", 3),
                    "sigma": data.get("sigma")
                },
                "results": {
                    "factor_found": data.get("factor_found"),
                    "factors_found": factors_found_list,  # Include multiple factors if available
                    "curves_completed": data.get("curves_completed", 0),
                    "execution_time": data.get("execution_time", 0)
                },
                "raw_output": data.get("raw_output", "")
            }

            return submission

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"❌ Error processing {result_file}: {e}")
            return None

    def submit_result(self, submission: Dict[str, Any]) -> bool:
        """Submit a single result to all configured API endpoints using BaseWrapper infrastructure."""
        # Use BaseWrapper's submit_payload_to_endpoints method
        # Note: save_on_failure=False since we're already retrying a failed submission
        return self.submit_payload_to_endpoints(
            payload=submission,
            save_on_failure=False,
            results_context=None
        )

    def mark_as_submitted(self, result_file: str):
        """Mark a result file as successfully submitted."""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            data['submitted'] = True
            data['resubmitted_at'] = datetime.now().isoformat()

            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"⚠️  Could not mark {result_file} as submitted: {e}")

    def resend_failed_results(self) -> Dict[str, int]:
        """Main method to resend all failed results."""
        stats = {"total": 0, "success": 0, "failed": 0, "duplicates": 0}

        print("🔍 Looking for failed results to resend...")

        # Find failed results from saved files
        failed_results = self.find_failed_results()

        if not failed_results:
            print("ℹ️  No failed results found in data directory")
            return stats

        print(f"📤 Found {len(failed_results)} results to resend")

        for result_data in failed_results:
            stats["total"] += 1
            result_file = result_data.pop('_file_path')

            # Create submission
            submission = self.create_submission_from_saved_data(result_file)
            if not submission:
                stats["failed"] += 1
                continue

            print(f"📡 Resubmitting: {submission['composite'][:50]}...")

            # Attempt submission
            if self.submit_result(submission):
                stats["success"] += 1
                if not self.dry_run:
                    self.mark_as_submitted(result_file)
                else:
                    print("   [DRY RUN] File not marked as submitted")
            else:
                stats["failed"] += 1

        return stats

def main():
    """Main entry point."""
    print("📡 ECM Failed Results Resender")
    print("=" * 40)

    # Check for dry-run flag
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    if dry_run:
        print("🔍 DRY RUN MODE - Files will not be marked as submitted")
        print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
Usage: python3 resend_failed.py [options] [config_file]

Resends ECM results that completed but failed to submit to the API server.
The script looks for .json result files in data/results/ directory that
are marked as not submitted.

Options:
  --dry-run, -n  Test submission without marking files as submitted
  -h, --help     Show this help message

Arguments:
  config_file    Path to client.yaml (default: client.yaml)
                 Note: client.local.yaml will be auto-merged if present
        """)
        return

    # Get config file (skip flag arguments) - defaults to client.yaml
    config_file = 'client.yaml'
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            config_file = arg
            break

    try:
        resender = FailedResultsResender(config_file, dry_run=dry_run)

        # Print endpoint info (handle both single and multi-endpoint configs)
        if hasattr(resender, 'api_endpoint'):
            print(f"🌐 API Endpoint: {resender.api_endpoint}")
        else:
            endpoints_str = ', '.join([c['name'] for c in resender.api_clients])
            print(f"🌐 API Endpoints: {endpoints_str}")
        print()

        stats = resender.resend_failed_results()

        print("\n📊 Resend Summary:")
        print(f"   Total attempts: {stats['total']}")
        print(f"   ✅ Successful: {stats['success']}")
        print(f"   ❌ Failed: {stats['failed']}")

        if stats['success'] > 0:
            print(f"\n🎉 Successfully resubmitted {stats['success']} results!")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()