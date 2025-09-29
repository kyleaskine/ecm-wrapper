#!/usr/bin/env python3
"""
Quick validation test for optimized wrapper functionality.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all optimized modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from execution_engine import UnifiedExecutionEngine, ExecutionParams
        print("   ✅ execution_engine imported")
    except Exception as e:
        print(f"   ❌ execution_engine failed: {e}")
        return False

    try:
        from result_processor import ConsolidatedResultProcessor, ProcessingResult
        print("   ✅ result_processor imported")
    except Exception as e:
        print(f"   ❌ result_processor failed: {e}")
        return False

    try:
        from optimized_parsing import OptimizedPatterns, StreamingParser
        print("   ✅ optimized_parsing imported")
    except Exception as e:
        print(f"   ❌ optimized_parsing failed: {e}")
        return False

    try:
        from unified_ecm_executor import UnifiedECMExecutor, ECMExecutionParams
        print("   ✅ unified_ecm_executor imported")
    except Exception as e:
        print(f"   ❌ unified_ecm_executor failed: {e}")
        return False

    try:
        from optimized_base_wrapper import OptimizedECMWrapper, OptimizedYAFUWrapper
        print("   ✅ optimized_base_wrapper imported")
    except Exception as e:
        print(f"   ❌ optimized_base_wrapper failed: {e}")
        return False

    print("   🎉 All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality without actual ECM execution."""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test execution params
        from execution_engine import ExecutionParams
        params = ExecutionParams(
            cmd=["echo", "test"],
            timeout=5
        )
        print("   ✅ ExecutionParams creation")

        # Test ECM params
        from unified_ecm_executor import ECMExecutionParams
        ecm_params = ECMExecutionParams(
            composite="123456789",
            b1=1000,
            curves=5
        )
        print("   ✅ ECMExecutionParams creation")

        # Test pattern compilation
        from optimized_parsing import OptimizedPatterns
        # Just access the patterns to ensure they're compiled
        pattern = OptimizedPatterns.ECM_STANDARD_FACTOR
        print("   ✅ Pattern compilation")

        # Test result creation
        from result_processor import ProcessingResult
        result = ProcessingResult(
            composite="123456789",
            method="ecm",
            program="test",
            success=True,
            execution_time=1.0
        )
        print("   ✅ ProcessingResult creation")

        print("   🎉 Basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration loading...")

    config_file = Path("client.yaml")
    if not config_file.exists():
        print("   ⚠️  client.yaml not found - skipping config test")
        return True

    try:
        from optimized_base_wrapper import OptimizedECMWrapper
        wrapper = OptimizedECMWrapper("client.yaml")
        print("   ✅ Configuration loaded successfully")
        print(f"   📋 Client ID: {wrapper.client_id}")
        print(f"   🌐 API Endpoint: {wrapper.api_endpoint}")
        return True

    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False

def test_parsing_performance():
    """Test parsing performance with sample data."""
    print("\n⚡ Testing parsing performance...")

    try:
        from optimized_parsing import parse_ecm_output, parse_yafu_ecm_output
        import time

        # Sample ECM output
        ecm_output = """
        Using B1=50000, B2=5000000, polynomial Dickson(6), sigma=123456
        Step 1 took 1234ms
        Step 2 took 5678ms
        Factor found in step 1: 12345678901234567890
        """

        # Test ECM parsing
        start_time = time.time()
        for _ in range(1000):
            factor, sigma = parse_ecm_output(ecm_output)
        ecm_time = time.time() - start_time

        print(f"   ✅ ECM parsing: {ecm_time:.4f}s for 1000 iterations")
        print(f"   📊 Factor found: {factor}")

        # Sample YAFU output
        yafu_output = """
        factors found:
        P39 = 123456789012345678901234567890123456789
        """

        # Test YAFU parsing
        start_time = time.time()
        for _ in range(1000):
            factors = parse_yafu_ecm_output(yafu_output)
        yafu_time = time.time() - start_time

        print(f"   ✅ YAFU parsing: {yafu_time:.4f}s for 1000 iterations")
        print(f"   📊 Factors found: {len(factors)}")

        return True

    except Exception as e:
        print(f"   ❌ Parsing performance test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Optimized Wrapper Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_basic_functionality,
        test_config_loading,
        test_parsing_performance
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Optimization infrastructure is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)