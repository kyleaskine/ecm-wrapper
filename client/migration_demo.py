#!/usr/bin/env python3
"""
Migration demonstration: Using optimized infrastructure vs legacy wrappers.
Shows performance improvements and simplified code.
"""
import time
import sys

# Legacy imports
import importlib.util

# Import ECM wrapper
ecm_spec = importlib.util.spec_from_file_location("ecm_wrapper", "ecm-wrapper.py")
ecm_module = importlib.util.module_from_spec(ecm_spec)
ecm_spec.loader.exec_module(ecm_module)
ECMWrapper = ecm_module.ECMWrapper

# Import YAFU wrapper
yafu_spec = importlib.util.spec_from_file_location("yafu_wrapper", "yafu-wrapper.py")
yafu_module = importlib.util.module_from_spec(yafu_spec)
yafu_spec.loader.exec_module(yafu_module)
YAFUWrapper = yafu_module.YAFUWrapper

# Optimized imports
from optimized_base_wrapper import OptimizedECMWrapper, OptimizedYAFUWrapper

def compare_ecm_performance():
    """Compare ECM execution performance between legacy and optimized versions."""
    composite = "1452848330851205097904585085040436106411154882691267873125029250271117855878805943654146936940471067989034288416808883136159810949249242967838918383859611639069461760271925231084798197070104618831"

    print("🔬 ECM Performance Comparison")
    print("=" * 50)

    # Legacy ECM wrapper
    print("\n📊 Legacy ECM Wrapper:")
    legacy_wrapper = ECMWrapper("client.yaml")

    start_time = time.time()
    legacy_result = legacy_wrapper.run_ecm(
        composite=composite,
        b1=5000000,
        curves=1,  # Small number for demo
        method="ecm"
    )
    legacy_time = time.time() - start_time

    print(f"   Execution time: {legacy_time:.3f}s")
    print(f"   Factor found: {legacy_result.get('factor_found')}")
    print(f"   Curves completed: {legacy_result.get('curves_completed')}")

    # Optimized ECM wrapper
    print("\n⚡ Optimized ECM Wrapper:")
    optimized_wrapper = OptimizedECMWrapper("client.yaml")

    start_time = time.time()
    optimized_result = optimized_wrapper.run_ecm(
        composite=composite,
        b1=5000000,
        curves=1,  # Small number for demo
        method="ecm"
    )
    optimized_time = time.time() - start_time

    print(f"   Execution time: {optimized_time:.3f}s")
    print(f"   Factor found: {optimized_result.get('factor_found')}")
    print(f"   Curves completed: {optimized_result.get('curves_completed')}")

    # Calculate improvement
    if legacy_time > 0:
        improvement = ((legacy_time - optimized_time) / legacy_time) * 100
        print(f"\n📈 Performance improvement: {improvement:.1f}%")
        print(f"   Overhead reduction: {legacy_time - optimized_time:.3f}s")

def demonstrate_unified_execution():
    """Demonstrate unified execution modes."""
    composite = "2^128+1"  # Small Fermat number for demo

    print("\n🔧 Unified Execution Modes Demo")
    print("=" * 50)

    wrapper = OptimizedECMWrapper("client.yaml")

    # Standard mode
    print("\n1️⃣  Standard ECM:")
    result = wrapper.run_ecm(composite=composite, b1=1000, curves=5)
    print(f"   Method: {result['method']}")
    print(f"   Execution time: {result['execution_time']:.3f}s")

    # Multiprocess mode
    print("\n2️⃣  Multiprocess ECM:")
    result = wrapper.run_ecm_multiprocess(composite=composite, b1=1000, curves=10, workers=2)
    print(f"   Workers: 2")
    print(f"   Execution time: {result['execution_time']:.3f}s")

    # Two-stage mode
    print("\n3️⃣  Two-stage ECM:")
    result = wrapper.run_ecm_two_stage(composite=composite, b1=1000, b2=100000, curves=5)
    print(f"   Two-stage: {result.get('two_stage', 'N/A')}")
    print(f"   Execution time: {result['execution_time']:.3f}s")

def show_code_reduction():
    """Show how much code was reduced."""
    print("\n📉 Code Reduction Summary")
    print("=" * 50)

    # Count lines in legacy files
    legacy_files = [
        "ecm-wrapper.py",
        "yafu-wrapper.py",
        "base_wrapper.py",
        "parsing_utils.py"
    ]

    optimized_files = [
        "optimized_base_wrapper.py",
        "unified_ecm_executor.py",
        "execution_engine.py",
        "result_processor.py",
        "optimized_parsing.py"
    ]

    def count_lines(filename):
        try:
            with open(filename, 'r') as f:
                return len([line for line in f if line.strip() and not line.strip().startswith('#')])
        except:
            return 0

    legacy_lines = sum(count_lines(f) for f in legacy_files)
    optimized_lines = sum(count_lines(f) for f in optimized_files)

    print(f"Legacy code lines: {legacy_lines}")
    print(f"Optimized code lines: {optimized_lines}")

    if legacy_lines > 0:
        reduction = ((legacy_lines - optimized_lines) / legacy_lines) * 100
        print(f"Code reduction: {reduction:.1f}%")
        print(f"Lines saved: {legacy_lines - optimized_lines}")

    print(f"\n✨ Benefits:")
    print(f"   • Unified execution engine")
    print(f"   • Consolidated result processing")
    print(f"   • Optimized parsing with pre-compiled patterns")
    print(f"   • Reduced subprocess overhead")
    print(f"   • Minimal logging overhead")
    print(f"   • Cached program version detection")

def demonstrate_memory_efficiency():
    """Show memory efficiency improvements."""
    print("\n🧠 Memory Efficiency Demo")
    print("=" * 50)

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        print("\n📊 Memory usage comparison:")

        # Measure legacy wrapper memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        legacy_wrapper = ECMWrapper("client.yaml")
        legacy_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Measure optimized wrapper memory
        optimized_wrapper = OptimizedECMWrapper("client.yaml")
        optimized_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Legacy wrapper: {legacy_memory:.1f} MB")
        print(f"   Optimized wrapper: {optimized_memory:.1f} MB")

        legacy_overhead = legacy_memory - initial_memory
        optimized_overhead = optimized_memory - legacy_memory

        print(f"   Legacy overhead: {legacy_overhead:.1f} MB")
        print(f"   Optimized overhead: {optimized_overhead:.1f} MB")

        if legacy_overhead > 0:
            improvement = ((legacy_overhead - optimized_overhead) / legacy_overhead) * 100
            print(f"   Memory improvement: {improvement:.1f}%")

    except ImportError:
        print("   psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"   Memory measurement error: {e}")

def main():
    """Main demonstration."""
    print("🚀 ECM Wrapper Optimization Demo")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--performance":
        compare_ecm_performance()
    elif len(sys.argv) > 1 and sys.argv[1] == "--unified":
        demonstrate_unified_execution()
    elif len(sys.argv) > 1 and sys.argv[1] == "--memory":
        demonstrate_memory_efficiency()
    elif len(sys.argv) > 1 and sys.argv[1] == "--code":
        show_code_reduction()
    else:
        print("\nAvailable demos:")
        print("  --performance  : Compare execution performance")
        print("  --unified      : Show unified execution modes")
        print("  --memory       : Memory efficiency comparison")
        print("  --code         : Code reduction summary")
        print("\nExample: python3 migration_demo.py --performance")

if __name__ == "__main__":
    main()