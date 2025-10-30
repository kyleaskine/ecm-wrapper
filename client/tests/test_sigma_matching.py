#!/usr/bin/env python3
"""
Test case for GPU ECM sigma matching with multiple factors.

This tests the scenario where GPU finds two prime factors and also reports
their product as a "factor". We need to:
1. Extract only the prime factors (not composites)
2. Match each prime to the correct sigma from the curve that found it
"""

import sys
from parsing_utils import parse_ecm_output_multiple

# Simplified test case based on real GPU ECM output
# Scenario: GPU finds two primes (p1 and p2) and reports both plus their product
SAMPLE_GPU_OUTPUT = """
GMP-ECM 7.0.6 [configured with GMP 6.3.0, --enable-asm-redc] [ECM]
Input number is 771740345279535829905655342951 (30 digits)
Using B1=1000000, B2=100000000, polynomial x^1, sigma=3:9999999999

Starting ECM stage 1...
GPU: factor 856395168938929 found in Step 1 with curve 5 (-sigma 3:1111111111)
...more curves...
GPU: factor 901149811757719 found in Step 1 with curve 12 (-sigma 3:2222222222)
...primality testing...
GPU: factor 771740345279535829905655342951 found in Step 1 with curve 20 (-sigma 3:1234567890)
...

********** Factor found in step 1:
Found prime factor of 15 digits: 856395168938929
Found prime factor of 15 digits: 901149811757719

Composite cofactor 771740345279535829905655342951 has 30 digits
"""

def test_sigma_matching():
    """Test that primes get matched to correct sigmas."""
    print("=" * 80)
    print("Testing GPU ECM Sigma Matching")
    print("=" * 80)

    print("\nScenario:")
    print("  - GPU finds prime p1 = 856395168938929 with sigma=3:1111111111")
    print("  - GPU finds prime p2 = 901149811757719 with sigma=3:2222222222")
    print("  - GPU also reports p1×p2 = 771740345279535829905655342951 with sigma=3:1234567890")
    print("\nExpected behavior:")
    print("  ✓ Submit p1 with sigma=3:1111111111 (curve that found it)")
    print("  ✓ Submit p2 with sigma=3:2222222222 (curve that found it)")
    print("  ✗ DON'T submit p1×p2 (composite)")
    print("\n" + "=" * 80)

    # Parse the output with debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    factors = parse_ecm_output_multiple(SAMPLE_GPU_OUTPUT)

    print(f"\nParsing results: Found {len(factors)} factor(s)")
    print("-" * 80)

    for i, (factor, sigma) in enumerate(factors, 1):
        print(f"{i}. Factor: {factor}")
        print(f"   Sigma:  {sigma}")
        print()

    # Verify expected results
    print("=" * 80)
    print("Validation:")
    print("=" * 80)

    success = True

    # Should have exactly 2 factors (the two primes)
    if len(factors) != 2:
        print(f"❌ FAIL: Expected 2 factors, got {len(factors)}")
        success = False
    else:
        print(f"✓ Correct number of factors: {len(factors)}")

    # Check p1
    p1 = "856395168938929"
    p1_sigma = "3:1111111111"
    p1_found = False
    for factor, sigma in factors:
        if factor == p1:
            p1_found = True
            if sigma == p1_sigma:
                print(f"✓ Prime p1 has correct sigma: {p1_sigma}")
            else:
                print(f"❌ FAIL: Prime p1 has wrong sigma: {sigma} (expected {p1_sigma})")
                success = False
            break

    if not p1_found:
        print(f"❌ FAIL: Prime p1 not found in results")
        success = False

    # Check p2
    p2 = "901149811757719"
    p2_sigma = "3:2222222222"
    p2_found = False
    for factor, sigma in factors:
        if factor == p2:
            p2_found = True
            if sigma == p2_sigma:
                print(f"✓ Prime p2 has correct sigma: {p2_sigma}")
            else:
                print(f"❌ FAIL: Prime p2 has wrong sigma: {sigma} (expected {p2_sigma})")
                success = False
            break

    if not p2_found:
        print(f"❌ FAIL: Prime p2 not found in results")
        success = False

    # Check that composite is NOT present
    composite = "771740345279535829905655342951"
    composite_found = False
    for factor, sigma in factors:
        if factor == composite:
            composite_found = True
            print(f"❌ FAIL: Composite {composite} should not be in results")
            success = False
            break

    if not composite_found:
        print(f"✓ Composite correctly filtered out")

    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(test_sigma_matching())
