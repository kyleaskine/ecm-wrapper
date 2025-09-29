#!/usr/bin/env python3
import json
import requests
import sys
from pathlib import Path

def resubmit_failed_result(failed_file_path):
    """Resubmit a failed result with corrected method"""

    with open(failed_file_path, 'r') as f:
        failed_data = json.load(f)

    # Extract the original payload
    payload = failed_data['api_payload'].copy()

    # Fix the method - change from "ecm-stage2-only" to "ecm"
    if payload['method'] == 'ecm-stage2-only':
        payload['method'] = 'ecm'
        payload['program'] = 'gmp-ecm-ecm'  # Also fix the program name
        print(f"✓ Corrected method from 'ecm-stage2-only' to 'ecm'")

    # Submit to server
    api_url = "http://localhost:8000/api/v1/submit_result"

    print(f"🔄 Resubmitting to {api_url}")
    print(f"📊 Composite: {payload['composite'][:20]}...{payload['composite'][-20:]} ({len(payload['composite'])} digits)")
    print(f"📈 Curves: {payload['results']['curves_completed']}")
    print(f"⏱️  Execution time: {payload['results']['execution_time']:.1f}s")

    try:
        response = requests.post(api_url, json=payload, timeout=30)

        if response.status_code == 200:
            print(f"✅ Successfully resubmitted!")
            print(f"Server response: {response.json()}")
            return True
        else:
            print(f"❌ Failed to resubmit (HTTP {response.status_code})")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during resubmission: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 resubmit_failed.py <failed_submission_file.json>")
        sys.exit(1)

    failed_file = Path(sys.argv[1])
    if not failed_file.exists():
        print(f"❌ File not found: {failed_file}")
        sys.exit(1)

    success = resubmit_failed_result(failed_file)
    sys.exit(0 if success else 1)