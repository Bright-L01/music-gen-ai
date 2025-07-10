#!/usr/bin/env python3
"""
Performance benchmarking and reporting for ModelManager.

This script runs comprehensive performance tests and generates a detailed report
covering all aspects of model management performance.
"""

import sys
import time
import statistics
import gc
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mock dependencies
sys.modules["torch"] = Mock()
sys.modules["torch.cuda"] = Mock()
sys.modules["psutil"] = Mock()
sys.modules["music_gen.optimization.fast_generator"] = Mock()

torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.memory_allocated.return_value = 1024**3
torch_mock.cuda.empty_cache = Mock()
torch_mock.cuda.OutOfMemoryError = Exception

# Import ModelManager
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
}

with open(model_manager_path, "r") as f:
    exec(f.read(), model_manager_namespace)

ModelManager = model_manager_namespace["ModelManager"]


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        self.results = {}

    def benchmark(self, name, func, iterations=100, warmup=10):
        """Benchmark a function with multiple iterations."""
        # Warmup runs
        for _ in range(warmup):
            func()

        # Benchmark runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        self.results[name] = {
            "times": times,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0,
            "iterations": iterations,
        }

        return self.results[name]

    def report(self):
        """Generate performance report."""
        print("=" * 80)
        print("MODEL MANAGER PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        print()

        for name, result in self.results.items():
            print(f"📊 {name}")
            print(f"   Mean:     {result['mean']*1000:.3f}ms")
            print(f"   Median:   {result['median']*1000:.3f}ms")
            print(f"   Min:      {result['min']*1000:.3f}ms")
            print(f"   Max:      {result['max']*1000:.3f}ms")
            print(f"   Std Dev:  {result['stddev']*1000:.3f}ms")
            print(f"   Samples:  {result['iterations']}")
            print()


def benchmark_model_loading():
    """Benchmark model loading performance."""
    print("🔄 Benchmarking Model Loading...")

    benchmark = PerformanceBenchmark()

    # Test 1: Fresh model loading
    def fresh_model_load():
        ModelManager._instance = None
        manager = ModelManager()

        # Mock in the namespace
        original_generator = model_manager_namespace.get("FastMusicGenerator")
        mock_generator = Mock(return_value=Mock())
        model_manager_namespace["FastMusicGenerator"] = mock_generator

        try:
            manager._models.clear()  # Force fresh load
            result = manager.get_model("test-model", "optimized")
        finally:
            if original_generator:
                model_manager_namespace["FastMusicGenerator"] = original_generator

        return result

    benchmark.benchmark("Fresh Model Loading", fresh_model_load, iterations=50)

    # Test 2: Cached model access
    ModelManager._instance = None
    manager = ModelManager()

    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
        mock.return_value = Mock()
        manager.get_model("cached-model", "optimized")  # Pre-load

        def cached_access():
            return manager.get_model("cached-model", "optimized")

        benchmark.benchmark("Cached Model Access", cached_access, iterations=1000)

    # Test 3: Model switching
    def model_switching():
        manager.unload_model("cached-model")
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock()
            return manager.get_model("new-model", "optimized")

    benchmark.benchmark("Model Switching", model_switching, iterations=30)

    return benchmark


def benchmark_memory_operations():
    """Benchmark memory-related operations."""
    print("🧠 Benchmarking Memory Operations...")

    benchmark = PerformanceBenchmark()

    ModelManager._instance = None
    manager = ModelManager()

    # Pre-load some models
    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
        mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])
        for i in range(10):
            manager.get_model(f"model{i}", "optimized")

    # Test cache info generation
    def cache_info():
        return manager.get_cache_info()

    benchmark.benchmark("Cache Info Generation", cache_info)

    # Test model listing
    def list_models():
        return manager.list_loaded_models()

    benchmark.benchmark("Model Listing", list_models)

    # Test cache clearing
    def cache_clear():
        # Save state
        models_backup = manager._models.copy()

        # Clear cache
        manager.clear_cache()

        # Restore state for next iteration
        manager._models = models_backup

    benchmark.benchmark("Cache Clearing", cache_clear, iterations=50)

    return benchmark


def benchmark_concurrent_access():
    """Benchmark concurrent access patterns."""
    print("🔀 Benchmarking Concurrent Access...")

    benchmark = PerformanceBenchmark()

    ModelManager._instance = None
    manager = ModelManager()

    # Pre-load a model
    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
        mock.return_value = Mock()
        manager.get_model("concurrent-test", "optimized")

    # Test concurrent cached access
    def concurrent_cached_access():
        def access_model():
            return manager.get_model("concurrent-test", "optimized")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_model) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        return len(results)

    benchmark.benchmark(
        "Concurrent Cached Access (50 threads)", concurrent_cached_access, iterations=20
    )

    # Test concurrent mixed operations
    def concurrent_mixed_operations():
        def mixed_ops(thread_id):
            for i in range(5):
                if i % 3 == 0:
                    manager.get_model("concurrent-test", "optimized")
                elif i % 3 == 1:
                    manager.list_loaded_models()
                else:
                    manager.get_cache_info()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mixed_ops, i) for i in range(20)]
            for f in as_completed(futures):
                f.result()

    benchmark.benchmark(
        "Concurrent Mixed Operations (20 threads)", concurrent_mixed_operations, iterations=10
    )

    return benchmark


def benchmark_scalability():
    """Benchmark scalability with many models."""
    print("📈 Benchmarking Scalability...")

    benchmark = PerformanceBenchmark()

    ModelManager._instance = None
    manager = ModelManager()

    # Test loading many models
    def load_many_models():
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            start_time = time.perf_counter()
            for i in range(50):
                manager.get_model(f"scale_model_{i}", "optimized")
            end_time = time.perf_counter()

            # Clear for next iteration
            manager.clear_cache()

            return end_time - start_time

    # Custom benchmark for scalability
    times = []
    for _ in range(10):
        times.append(load_many_models())

    benchmark.results["Loading 50 Models"] = {
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stddev": statistics.stdev(times) if len(times) > 1 else 0,
        "iterations": len(times),
    }

    # Test cache operations with many models
    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
        mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])
        for i in range(100):
            manager.get_model(f"cache_test_{i}", "optimized")

    def cache_ops_with_many():
        manager.list_loaded_models()
        manager.get_cache_info()
        manager.has_loaded_models()

    benchmark.benchmark("Cache Operations (100 models)", cache_ops_with_many)

    return benchmark


def benchmark_error_handling():
    """Benchmark error handling performance."""
    print("⚠️  Benchmarking Error Handling...")

    benchmark = PerformanceBenchmark()

    ModelManager._instance = None
    manager = ModelManager()

    # Test exception handling overhead
    def handle_loading_error():
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = RuntimeError("Model not found")

            try:
                manager.get_model("error-model", "optimized")
            except RuntimeError:
                pass

            # Reset for next iteration
            mock.side_effect = None

    benchmark.benchmark("Error Handling Overhead", handle_loading_error, iterations=100)

    # Test unknown model type handling
    def handle_unknown_type():
        try:
            manager.get_model("test-model", "unknown_type")
        except ValueError:
            pass

    benchmark.benchmark("Unknown Type Error", handle_unknown_type)

    return benchmark


def generate_summary_report(benchmarks):
    """Generate a comprehensive summary report."""
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print()

    all_results = {}
    for bench in benchmarks:
        all_results.update(bench.results)

    # Performance categories
    categories = {
        "🚀 Core Operations": [
            "Fresh Model Loading",
            "Cached Model Access",
            "Model Switching",
        ],
        "🧠 Memory Operations": [
            "Cache Info Generation",
            "Model Listing",
            "Cache Clearing",
        ],
        "🔀 Concurrency": [
            "Concurrent Cached Access (50 threads)",
            "Concurrent Mixed Operations (20 threads)",
        ],
        "📈 Scalability": [
            "Loading 50 Models",
            "Cache Operations (100 models)",
        ],
        "⚠️  Error Handling": [
            "Error Handling Overhead",
            "Unknown Type Error",
        ],
    }

    for category, operations in categories.items():
        print(f"{category}")
        print("-" * 40)

        for op in operations:
            if op in all_results:
                result = all_results[op]
                print(f"  {op}:")
                print(f"    Avg: {result['mean']*1000:.2f}ms")
                if result["mean"] < 0.001:
                    print(f"    📈 Excellent performance")
                elif result["mean"] < 0.01:
                    print(f"    ✅ Good performance")
                elif result["mean"] < 0.1:
                    print(f"    ⚠️  Acceptable performance")
                else:
                    print(f"    🐌 Needs optimization")
        print()

    # Overall assessment
    print("🎯 OVERALL ASSESSMENT")
    print("-" * 40)

    fast_ops = sum(1 for r in all_results.values() if r["mean"] < 0.001)
    good_ops = sum(1 for r in all_results.values() if 0.001 <= r["mean"] < 0.01)
    acceptable_ops = sum(1 for r in all_results.values() if 0.01 <= r["mean"] < 0.1)
    slow_ops = sum(1 for r in all_results.values() if r["mean"] >= 0.1)

    total_ops = len(all_results)

    print(f"📈 Excellent operations: {fast_ops}/{total_ops} ({fast_ops/total_ops*100:.1f}%)")
    print(f"✅ Good operations: {good_ops}/{total_ops} ({good_ops/total_ops*100:.1f}%)")
    print(
        f"⚠️  Acceptable operations: {acceptable_ops}/{total_ops} ({acceptable_ops/total_ops*100:.1f}%)"
    )
    print(f"🐌 Slow operations: {slow_ops}/{total_ops} ({slow_ops/total_ops*100:.1f}%)")
    print()

    if slow_ops == 0 and acceptable_ops <= 2:
        print("🎉 EXCELLENT: ModelManager shows excellent performance across all operations!")
    elif slow_ops == 0:
        print("✅ GOOD: ModelManager performance is good with minor optimization opportunities.")
    elif slow_ops <= 2:
        print(
            "⚠️  ACCEPTABLE: ModelManager performance is acceptable but has some slow operations."
        )
    else:
        print("🔧 NEEDS WORK: ModelManager has several performance bottlenecks that need attention.")


def main():
    """Run all performance benchmarks."""
    print("🎵 Model Manager Performance Benchmark Suite")
    print("=" * 80)
    print()

    benchmarks = []

    # Run all benchmarks
    benchmarks.append(benchmark_model_loading())
    benchmarks.append(benchmark_memory_operations())
    benchmarks.append(benchmark_concurrent_access())
    benchmarks.append(benchmark_scalability())
    benchmarks.append(benchmark_error_handling())

    # Generate individual reports
    for bench in benchmarks:
        bench.report()

    # Generate summary
    generate_summary_report(benchmarks)

    print("✅ Performance benchmarking complete!")


if __name__ == "__main__":
    main()
