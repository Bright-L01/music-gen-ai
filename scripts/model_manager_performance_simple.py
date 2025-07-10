#!/usr/bin/env python3
"""
Simplified Performance benchmarking for ModelManager.

This script runs performance tests without complex patching to avoid dependency issues.
"""

import sys
import time
import statistics
import gc
from pathlib import Path
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mock dependencies
sys.modules["torch"] = Mock()
sys.modules["torch.cuda"] = Mock()
sys.modules["psutil"] = Mock()

torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.memory_allocated.return_value = 1024**3
torch_mock.cuda.empty_cache = Mock()
torch_mock.cuda.OutOfMemoryError = Exception

# Import ModelManager directly
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

model_manager_path = project_root / "music_gen" / "core" / "model_manager.py"
model_manager_namespace = {
    "__name__": "music_gen.core.model_manager",
    "__file__": str(model_manager_path),
    "gc": gc,
    "logging": Mock(),
    "Path": Path,
    "torch": torch_mock,
    "FastMusicGenerator": Mock(),  # Mock the FastMusicGenerator directly
}

with open(model_manager_path, "r") as f:
    exec(f.read(), model_manager_namespace)

ModelManager = model_manager_namespace["ModelManager"]


def benchmark_function(func, iterations=100, warmup=10):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stddev": statistics.stdev(times) if len(times) > 1 else 0,
        "iterations": iterations,
    }


def main():
    """Run performance benchmarks."""
    print("🎵 Model Manager Performance Benchmark")
    print("=" * 60)
    print()

    results = {}

    # Test 1: Model Manager Initialization
    print("🔄 Testing initialization...")

    def init_manager():
        ModelManager._instance = None
        return ModelManager()

    results["Initialization"] = benchmark_function(init_manager, iterations=100)

    # Test 2: Cached Model Access
    print("🔄 Testing cached access...")

    ModelManager._instance = None
    manager = ModelManager()

    # Mock the FastMusicGenerator in namespace
    mock_generator = Mock()
    mock_model = Mock()
    mock_generator.return_value = mock_model
    model_manager_namespace["FastMusicGenerator"] = mock_generator

    # Pre-load a model
    manager.get_model("test-model", "optimized")

    def cached_access():
        return manager.get_model("test-model", "optimized")

    results["Cached Access"] = benchmark_function(cached_access, iterations=1000)

    # Test 3: Cache Info Generation
    print("🔄 Testing cache info...")

    def cache_info():
        return manager.get_cache_info()

    results["Cache Info"] = benchmark_function(cache_info, iterations=500)

    # Test 4: Model Listing
    print("🔄 Testing model listing...")

    def list_models():
        return manager.list_loaded_models()

    results["Model Listing"] = benchmark_function(list_models, iterations=500)

    # Test 5: Concurrent Access
    print("🔄 Testing concurrent access...")

    def concurrent_access():
        def access_model():
            return manager.get_model("test-model", "optimized")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_model) for _ in range(50)]
            return [f.result() for f in as_completed(futures)]

    results["Concurrent Access"] = benchmark_function(concurrent_access, iterations=20)

    # Test 6: Model Unloading
    print("🔄 Testing model unloading...")

    def model_unload():
        # Add a model to unload
        manager._models["temp_model_optimized_cuda"] = Mock()
        return manager.unload_model("temp_model")

    results["Model Unloading"] = benchmark_function(model_unload, iterations=100)

    # Test 7: Cache Clearing
    print("🔄 Testing cache clearing...")

    def cache_clear():
        # Save state
        models_backup = manager._models.copy()

        # Clear cache
        manager.clear_cache()

        # Restore state
        manager._models = models_backup

    results["Cache Clearing"] = benchmark_function(cache_clear, iterations=50)

    # Generate Report
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    print()

    for operation, result in results.items():
        mean_ms = result["mean"] * 1000
        median_ms = result["median"] * 1000
        min_ms = result["min"] * 1000
        max_ms = result["max"] * 1000
        stddev_ms = result["stddev"] * 1000

        print(f"📊 {operation}")
        print(f"   Mean:    {mean_ms:.3f}ms")
        print(f"   Median:  {median_ms:.3f}ms")
        print(f"   Min:     {min_ms:.3f}ms")
        print(f"   Max:     {max_ms:.3f}ms")
        print(f"   StdDev:  {stddev_ms:.3f}ms")
        print(f"   Samples: {result['iterations']}")

        # Performance assessment
        if mean_ms < 1:
            print("   Status:  📈 Excellent")
        elif mean_ms < 10:
            print("   Status:  ✅ Good")
        elif mean_ms < 100:
            print("   Status:  ⚠️  Acceptable")
        else:
            print("   Status:  🐌 Needs optimization")

        print()

    # Summary
    print("🎯 SUMMARY")
    print("-" * 30)

    fast_ops = sum(1 for r in results.values() if r["mean"] < 0.001)
    good_ops = sum(1 for r in results.values() if 0.001 <= r["mean"] < 0.01)
    acceptable_ops = sum(1 for r in results.values() if 0.01 <= r["mean"] < 0.1)
    slow_ops = sum(1 for r in results.values() if r["mean"] >= 0.1)

    total_ops = len(results)

    print(f"📈 Excellent: {fast_ops}/{total_ops} operations")
    print(f"✅ Good:      {good_ops}/{total_ops} operations")
    print(f"⚠️  Acceptable: {acceptable_ops}/{total_ops} operations")
    print(f"🐌 Slow:      {slow_ops}/{total_ops} operations")
    print()

    if slow_ops == 0 and acceptable_ops <= 1:
        print("🎉 OVERALL: Excellent performance!")
    elif slow_ops == 0:
        print("✅ OVERALL: Good performance with minor optimizations possible.")
    elif slow_ops <= 1:
        print("⚠️  OVERALL: Acceptable performance with some optimization needed.")
    else:
        print("🔧 OVERALL: Performance needs improvement.")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
