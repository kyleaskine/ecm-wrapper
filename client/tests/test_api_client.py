#!/usr/bin/env python3
"""
Tests for APIClient functionality
"""
import tempfile
import os
from pathlib import Path
from api_client import APIClient


def test_build_submission_payload():
    """Test building API submission payload"""
    api_client = APIClient(
        api_endpoint="http://localhost:8000",
        timeout=30,
        retry_attempts=3
    )

    results = {
        'composite': '123456789',
        'method': 'ecm',
        'b1': 50000,
        'b2': 5000000,
        'curves_requested': 100,
        'curves_completed': 100,
        'execution_time': 45.2,
        'factor_found': '3',
        'parametrization': 3,
        'sigma': '3:12345',
        'raw_output': 'GMP-ECM output here...'
    }

    payload = api_client.build_submission_payload(
        composite='123456789',
        client_id='testuser-testcpu',
        method='ecm',
        program='gmp-ecm',
        program_version='7.0.4',
        results=results,
        project='test-project'
    )

    # Verify payload structure
    assert payload['composite'] == '123456789'
    assert payload['client_id'] == 'testuser-testcpu'
    assert payload['method'] == 'ecm'
    assert payload['program'] == 'gmp-ecm'
    assert payload['project'] == 'test-project'
    assert payload['parameters']['b1'] == 50000
    assert payload['parameters']['b2'] == 5000000
    assert payload['parameters']['curves'] == 100
    assert payload['parameters']['parametrization'] == 3
    assert payload['parameters']['sigma'] == '3:12345'
    assert payload['results']['factor_found'] == '3'
    assert payload['results']['curves_completed'] == 100
    assert payload['results']['execution_time'] == 45.2

    print("✓ test_build_submission_payload passed")
    return True


def test_build_payload_with_multiple_factors():
    """Test payload building with multiple factors"""
    api_client = APIClient("http://localhost:8000")

    results = {
        'composite': '221',
        'method': 'ecm',
        'factors_found': ['13', '17'],  # Multiple factors
        'b1': 11000,
        'curves_completed': 10,
        'execution_time': 5.0,
        'raw_output': 'output'
    }

    payload = api_client.build_submission_payload(
        composite='221',
        client_id='test-client',
        method='ecm',
        program='gmp-ecm',
        program_version='7.0.4',
        results=results
    )

    # Should use first factor
    assert payload['results']['factor_found'] == '13'

    print("✓ test_build_payload_with_multiple_factors passed")
    return True


def test_save_failed_submission():
    """Test saving failed submissions"""
    api_client = APIClient("http://localhost:8000")

    with tempfile.TemporaryDirectory() as tmpdir:
        results = {
            'composite': '123456789',
            'method': 'ecm',
            'b1': 50000,
            'factor_found': None,
            'curves_completed': 100
        }

        payload = {
            'composite': '123456789',
            'client_id': 'test-client',
            'method': 'ecm'
        }

        # Save failed submission
        saved_path = api_client.save_failed_submission(
            results=results,
            payload=payload,
            output_dir=tmpdir
        )

        assert saved_path is not None, "Should return saved file path"
        assert os.path.exists(saved_path), "Saved file should exist"

        # Verify file contents
        import json
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert saved_data['composite'] == '123456789'
        assert saved_data['submitted'] is False
        assert 'api_payload' in saved_data
        assert 'failed_at' in saved_data

        print("✓ test_save_failed_submission passed")
        return True


def test_api_client_initialization():
    """Test APIClient initialization"""
    api_client = APIClient(
        api_endpoint="http://example.com:8000",
        timeout=60,
        retry_attempts=5
    )

    assert api_client.api_endpoint == "http://example.com:8000"
    assert api_client.timeout == 60
    assert api_client.retry_attempts == 5

    print("✓ test_api_client_initialization passed")
    return True


def test_default_parametrization():
    """Test that parametrization defaults to 3"""
    api_client = APIClient("http://localhost:8000")

    results = {
        'composite': '12345',
        'method': 'ecm',
        'curves_completed': 10
        # No parametrization specified
    }

    payload = api_client.build_submission_payload(
        composite='12345',
        client_id='test-client',
        method='ecm',
        program='gmp-ecm',
        program_version='7.0.4',
        results=results
    )

    # Should default to 3
    assert payload['parameters']['parametrization'] == 3

    print("✓ test_default_parametrization passed")
    return True


def test_integration_with_base_wrapper():
    """Test that APIClient integrates with BaseWrapper"""
    try:
        from base_wrapper import BaseWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'client.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("""
client:
  username: testuser
  cpu_name: testcpu

api:
  endpoint: http://localhost:8000
  timeout: 30
  retry_attempts: 3

programs:
  gmp_ecm:
    path: /usr/bin/ecm

execution:
  output_dir: data/outputs

logging:
  level: INFO
  file: data/logs/test.log
""")

            wrapper = BaseWrapper(str(config_file))

            assert hasattr(wrapper, 'api_client'), "Should have api_client attribute"
            assert wrapper.api_client.api_endpoint == "http://localhost:8000"
            assert wrapper.api_client.timeout == 30
            assert wrapper.api_client.retry_attempts == 3

            print("✓ test_integration_with_base_wrapper passed")
            return True

    except Exception as e:
        print(f"✗ test_integration_with_base_wrapper failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running APIClient tests...\n")

    tests = [
        test_api_client_initialization,
        test_build_submission_payload,
        test_build_payload_with_multiple_factors,
        test_save_failed_submission,
        test_default_parametrization,
        test_integration_with_base_wrapper
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
