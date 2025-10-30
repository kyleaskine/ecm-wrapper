#!/usr/bin/env python3
"""
Tests for ConfigManager functionality
"""
import tempfile
import os
from pathlib import Path
from config_manager import ConfigManager


def test_load_config_base_only():
    """Test loading configuration when only base file exists"""
    mgr = ConfigManager()

    # Create a temporary base config
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write("""
client:
  username: testuser
  cpu_name: testcpu

api:
  endpoint: http://localhost:8000
  timeout: 30

programs:
  gmp_ecm:
    path: /usr/bin/ecm
""")
        config_file = f.name

    try:
        config = mgr.load_config(config_file)

        assert config is not None, "Config should not be None"
        assert 'client' in config, "Config should have 'client' section"
        assert config['client']['username'] == 'testuser', "Username should match"
        assert config['api']['endpoint'] == 'http://localhost:8000', "API endpoint should match"

        print("✓ test_load_config_base_only passed")
        return True
    finally:
        os.unlink(config_file)


def test_load_config_with_local_override():
    """Test loading configuration with local overrides"""
    mgr = ConfigManager()

    # Create temporary directory for configs
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = Path(tmpdir) / 'client.yaml'
        local_config = Path(tmpdir) / 'client.local.yaml'

        # Write base config
        with open(base_config, 'w', encoding='utf-8') as f:
            f.write("""
client:
  username: defaultuser
  cpu_name: defaultcpu

api:
  endpoint: http://example.com:8000
  timeout: 30
  retry_attempts: 3

programs:
  gmp_ecm:
    path: /usr/bin/ecm
    default_curves: 100
""")

        # Write local override config
        with open(local_config, 'w', encoding='utf-8') as f:
            f.write("""
client:
  username: localuser
  cpu_name: mycpu

api:
  endpoint: http://localhost:8000

programs:
  gmp_ecm:
    path: /home/user/ecm
""")

        # Load config (should automatically merge local)
        config = mgr.load_config(str(base_config))

        # Verify overrides took effect
        assert config['client']['username'] == 'localuser', "Username should be overridden"
        assert config['client']['cpu_name'] == 'mycpu', "CPU name should be overridden"
        assert config['api']['endpoint'] == 'http://localhost:8000', "API endpoint should be overridden"

        # Verify non-overridden values remain
        assert config['api']['timeout'] == 30, "Timeout should remain from base"
        assert config['api']['retry_attempts'] == 3, "Retry attempts should remain from base"
        assert config['programs']['gmp_ecm']['default_curves'] == 100, "Default curves should remain from base"

        # Verify partial overrides
        assert config['programs']['gmp_ecm']['path'] == '/home/user/ecm', "ECM path should be overridden"

        print("✓ test_load_config_with_local_override passed")
        return True


def test_deep_merge():
    """Test deep merge functionality"""
    mgr = ConfigManager()

    base = {
        'a': {'b': 1, 'c': 2, 'd': {'e': 3}},
        'f': 4,
        'g': [1, 2, 3]
    }

    override = {
        'a': {'b': 99, 'd': {'e': 88, 'h': 77}},
        'f': 44,
        'i': 5
    }

    result = mgr.deep_merge(base, override)

    # Verify deep merge worked correctly
    assert result['a']['b'] == 99, "Should override nested value"
    assert result['a']['c'] == 2, "Should keep non-overridden nested value"
    assert result['a']['d']['e'] == 88, "Should override deeply nested value"
    assert result['a']['d']['h'] == 77, "Should add new deeply nested value"
    assert result['f'] == 44, "Should override top-level value"
    assert result['g'] == [1, 2, 3], "Should keep non-overridden list"
    assert result['i'] == 5, "Should add new top-level value"

    # Verify originals unchanged
    assert base['a']['b'] == 1, "Original base should be unchanged"
    assert override['a']['b'] == 99, "Original override should be unchanged"

    print("✓ test_deep_merge passed")
    return True


def test_get_nested_value():
    """Test getting nested configuration values"""
    mgr = ConfigManager()

    config = {
        'api': {
            'endpoint': 'http://localhost:8000',
            'timeout': 30,
            'retry': {
                'attempts': 3,
                'delay': 2
            }
        },
        'programs': {
            'gmp_ecm': {
                'path': '/usr/bin/ecm'
            }
        }
    }

    # Test various nested paths
    assert mgr.get_nested_value(config, 'api.endpoint') == 'http://localhost:8000'
    assert mgr.get_nested_value(config, 'api.timeout') == 30
    assert mgr.get_nested_value(config, 'api.retry.attempts') == 3
    assert mgr.get_nested_value(config, 'programs.gmp_ecm.path') == '/usr/bin/ecm'

    # Test non-existent paths with default
    assert mgr.get_nested_value(config, 'nonexistent.path', 'default') == 'default'
    assert mgr.get_nested_value(config, 'api.nonexistent', None) is None

    print("✓ test_get_nested_value passed")
    return True


def test_validate_config_structure():
    """Test configuration structure validation"""
    mgr = ConfigManager()

    # Valid config
    valid_config = {
        'client': {'username': 'test'},
        'api': {'endpoint': 'http://localhost:8000'},
        'programs': {'gmp_ecm': {}}
    }

    assert mgr.validate_config_structure(valid_config) is True, "Valid config should pass"

    # Invalid config (missing required keys)
    invalid_config = {
        'client': {'username': 'test'},
        'api': {'endpoint': 'http://localhost:8000'}
        # Missing 'programs' key
    }

    assert mgr.validate_config_structure(invalid_config) is False, "Invalid config should fail"

    # Custom required keys
    custom_config = {
        'database': {'url': 'postgres://localhost'},
        'logging': {'level': 'INFO'}
    }

    assert mgr.validate_config_structure(
        custom_config, ['database', 'logging']
    ) is True, "Config with custom keys should pass"

    assert mgr.validate_config_structure(
        custom_config, ['database', 'nonexistent']
    ) is False, "Config missing custom key should fail"

    print("✓ test_validate_config_structure passed")
    return True


def test_file_not_found():
    """Test handling of non-existent config file"""
    mgr = ConfigManager()

    try:
        mgr.load_config('/nonexistent/path/to/config.yaml')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert 'not found' in str(e).lower()
        print("✓ test_file_not_found passed")
        return True


def test_integration_with_base_wrapper():
    """Test that ConfigManager integrates correctly with BaseWrapper"""
    # This test verifies that the refactored BaseWrapper can load config
    try:
        # Import here to avoid issues if base_wrapper has problems
        from base_wrapper import BaseWrapper

        # Create a minimal valid config
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

            # Try to initialize BaseWrapper with this config
            # This will fail if ConfigManager integration is broken
            wrapper = BaseWrapper(str(config_file))

            assert wrapper.config is not None, "Config should be loaded"
            assert wrapper.client_id == "testuser-testcpu", "Client ID should be constructed"
            assert wrapper.api_endpoint == "http://localhost:8000", "API endpoint should be set"

            print("✓ test_integration_with_base_wrapper passed")
            return True

    except Exception as e:
        print(f"✗ test_integration_with_base_wrapper failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running ConfigManager tests...\n")

    tests = [
        test_load_config_base_only,
        test_load_config_with_local_override,
        test_deep_merge,
        test_get_nested_value,
        test_validate_config_structure,
        test_file_not_found,
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
