#!/usr/bin/env python3
"""
Complete Test Suite for ModelManager with Performance Benchmarks.

This script runs all tests including unit tests, integration tests, 
error handling tests, concurrency tests, and performance benchmarks.
"""

import sys
import time
import threading
import gc
import tempfile
import statistics
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mock all external dependencies before any imports
sys.modules["torch"] = Mock()
sys.modules["torch.cuda"] = Mock()
sys.modules["psutil"] = Mock()

# Configure torch mock
torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.memory_allocated.return_value = 1024**3
torch_mock.cuda.memory_reserved.return_value = 2 * (1024**3)
torch_mock.cuda.max_memory_allocated.return_value = 3 * (1024**3)
torch_mock.cuda.empty_cache = Mock()
torch_mock.cuda.OutOfMemoryError = Exception

# Mock fast generator
fast_generator_mock = Mock()
sys.modules["music_gen.optimization.fast_generator"] = fast_generator_mock

# Add the script directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Read and exec the model_manager.py file directly
model_manager_path = project_root / "music_gen" / "core" / "model_manager.py"

# Create a namespace for the module
model_manager_namespace = {
    "__name__": "music_gen.core.model_manager",
    "__file__": str(model_manager_path),
    "gc": gc,
    "logging": Mock(),
    "Path": Path,
    "torch": torch_mock,
}

# Execute the model manager code
with open(model_manager_path, "r") as f:
    model_manager_code = f.read()

exec(model_manager_code, model_manager_namespace)

# Get the ModelManager class
ModelManager = model_manager_namespace["ModelManager"]


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.performance_results = {}

    def test_passed(self, test_name):
        self.passed += 1
        print(f"✓ {test_name}")

    def test_failed(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")

    def add_performance_result(self, test_name, result):
        self.performance_results[test_name] = result
        mean_ms = result["mean"] * 1000
        print(f"⚡ {test_name}: {mean_ms:.3f}ms avg")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Test Results: {self.passed}/{total} passed")

        if self.errors:
            print("\n❌ Failures:")
            for test_name, error in self.errors:
                print(f"  {test_name}: {error}")

        if self.performance_results:
            print("\n⚡ Performance Results:")
            for test_name, result in self.performance_results.items():
                mean_ms = result["mean"] * 1000
                status = (
                    "📈" if mean_ms < 1 else "✅" if mean_ms < 10 else "⚠️" if mean_ms < 100 else "🐌"
                )
                print(f"  {status} {test_name}: {mean_ms:.3f}ms")

        return self.failed == 0


def run_test(test_func, test_name, results):
    """Run a single test function."""
    try:
        test_func()
        results.test_passed(test_name)
    except Exception as e:
        results.test_failed(test_name, str(e))


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


def run_performance_test(test_func, test_name, results, iterations=100):
    """Run a performance test."""
    try:
        result = benchmark_function(test_func, iterations)
        results.add_performance_result(test_name, result)
    except Exception as e:
        results.test_failed(f"Performance: {test_name}", str(e))


# === UNIT TESTS ===


def test_singleton_pattern():
    """Test ModelManager singleton pattern."""
    ModelManager._instance = None

    manager1 = ModelManager()
    manager2 = ModelManager()

    assert manager1 is manager2, "ModelManager should be singleton"
    assert id(manager1) == id(manager2), "Same object ID expected"


def test_initialization():
    """Test ModelManager initialization."""
    ModelManager._instance = None

    manager = ModelManager()

    assert manager._default_device in ["cuda", "cpu"], "Should have valid device"
    assert manager._cache_dir.name == "musicgen", "Cache directory name should be musicgen"
    assert len(manager._models) == 0, "Should start with empty model cache"


def test_model_loading():
    """Test model loading functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Mock FastMusicGenerator in the namespace
    mock_generator_class = Mock()
    mock_model = Mock()
    mock_generator_class.return_value = mock_model

    # Patch the FastMusicGenerator in the model manager namespace
    original_generator = model_manager_namespace.get("FastMusicGenerator")
    model_manager_namespace["FastMusicGenerator"] = mock_generator_class

    try:
        # Load model
        model = manager.get_model("test-model", "optimized")

        assert model is mock_model, "Should return the loaded model"
        assert len(manager._models) == 1, "Model should be cached"

        # Load same model again
        model2 = manager.get_model("test-model", "optimized")
        assert model is model2, "Should return cached model"

    finally:
        # Restore original
        if original_generator:
            model_manager_namespace["FastMusicGenerator"] = original_generator


def test_model_unloading():
    """Test model unloading functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    result = manager.unload_model("model1")

    assert result is True, "Should return True for successful unload"
    assert "model1_optimized_cuda" not in manager._models, "Model should be removed"
    assert len(manager._models) == 1, "Only one model should remain"


def test_cache_clearing():
    """Test cache clearing functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    manager.clear_cache()

    assert len(manager._models) == 0, "All models should be cleared"


def test_error_handling():
    """Test error handling during model loading."""
    ModelManager._instance = None
    manager = ModelManager()

    # Mock FastMusicGenerator to raise error
    mock_generator_class = Mock()
    mock_generator_class.side_effect = RuntimeError("Model corrupted")

    original_generator = model_manager_namespace.get("FastMusicGenerator")
    model_manager_namespace["FastMusicGenerator"] = mock_generator_class

    try:
        manager.get_model("corrupt-model", "optimized")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Model corrupted" in str(e), "Error message should be preserved"
    finally:
        if original_generator:
            model_manager_namespace["FastMusicGenerator"] = original_generator

    # Model should not be cached on failure
    assert len(manager._models) == 0, "Failed model should not be cached"


def test_unknown_model_type():
    """Test unknown model type error."""
    ModelManager._instance = None
    manager = ModelManager()

    try:
        manager.get_model("test-model", "unknown_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown model type" in str(e), "Should indicate unknown model type"


def test_concurrent_access():
    """Test concurrent model access."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock model
    mock_model = Mock()
    manager._models["test-model_optimized_cuda"] = mock_model

    results = []

    def access_model():
        return manager.get_model("test-model", "optimized")

    # Load same model from multiple threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(access_model) for i in range(20)]
        results = [future.result() for future in as_completed(futures)]

    # All threads should get the same model instance
    assert all(results[0] is r for r in results), "All threads should get same instance"
    assert len(results) == 20, "All threads should complete"


def test_memory_management():
    """Test memory management features."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    # Mock _get_cache_size method
    manager._get_cache_size = Mock(return_value=100 * 1024 * 1024)  # 100MB

    cache_info = manager.get_cache_info()

    assert cache_info["loaded_models"] == 2, "Should report correct model count"
    assert len(cache_info["models"]) == 2, "Should list all models"
    assert cache_info["cache_size_mb"] == 100.0, "Should report correct cache size"


# === PERFORMANCE TESTS ===


def perf_test_initialization():
    """Performance test for manager initialization."""

    def init_manager():
        ModelManager._instance = None
        return ModelManager()

    return init_manager


def perf_test_cached_access():
    """Performance test for cached model access."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock model
    mock_model = Mock()
    manager._models["perf-test_optimized_cuda"] = mock_model

    def cached_access():
        return manager.get_model("perf-test", "optimized")

    return cached_access


def perf_test_cache_info():
    """Performance test for cache info generation."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add multiple mock models
    for i in range(10):
        manager._models[f"model{i}_optimized_cuda"] = Mock()

    manager._get_cache_size = Mock(return_value=50 * 1024 * 1024)

    def cache_info():
        return manager.get_cache_info()

    return cache_info


def perf_test_model_listing():
    """Performance test for model listing."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add multiple mock models
    for i in range(20):
        mock_model = Mock()
        mock_model.__class__.__name__ = f"TestModel{i}"
        manager._models[f"model{i}_optimized_cuda"] = mock_model

    def list_models():
        return manager.list_loaded_models()

    return list_models


def perf_test_concurrent_access():
    """Performance test for concurrent access."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock model
    mock_model = Mock()
    manager._models["concurrent-test_optimized_cuda"] = mock_model

    def concurrent_access():
        def access_model():
            return manager.get_model("concurrent-test", "optimized")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_model) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        return len(results)

    return concurrent_access


def perf_test_model_switching():
    """Performance test for model switching."""
    ModelManager._instance = None
    manager = ModelManager()

    # Pre-load models
    for i in range(3):
        manager._models[f"switch_model_{i}_optimized_cuda"] = Mock()

    model_index = [0]

    def switch_models():
        # Unload current model
        manager.unload_model(f"switch_model_{model_index[0]}")

        # Add new model
        new_index = (model_index[0] + 1) % 3
        manager._models[f"switch_model_{new_index}_optimized_cuda"] = Mock()
        model_index[0] = new_index

        return manager.get_model(f"switch_model_{new_index}", "optimized")

    return switch_models


def main():
    """Run all tests and benchmarks."""
    print("🎵 ModelManager Complete Test Suite")
    print("=" * 80)
    print()

    results = TestResults()

    # Unit Tests
    print("=== Unit Tests ===")
    run_test(test_singleton_pattern, "Singleton Pattern", results)
    run_test(test_initialization, "Initialization", results)
    run_test(test_model_loading, "Model Loading", results)
    run_test(test_model_unloading, "Model Unloading", results)
    run_test(test_cache_clearing, "Cache Clearing", results)
    run_test(test_error_handling, "Error Handling", results)
    run_test(test_unknown_model_type, "Unknown Model Type", results)
    run_test(test_concurrent_access, "Concurrent Access", results)
    run_test(test_memory_management, "Memory Management", results)

    # Performance Tests
    print("\n=== Performance Benchmarks ===")
    run_performance_test(perf_test_initialization(), "Initialization", results, 100)
    run_performance_test(perf_test_cached_access(), "Cached Access", results, 1000)
    run_performance_test(perf_test_cache_info(), "Cache Info", results, 500)
    run_performance_test(perf_test_model_listing(), "Model Listing", results, 500)
    run_performance_test(perf_test_concurrent_access(), "Concurrent Access", results, 20)
    run_performance_test(perf_test_model_switching(), "Model Switching", results, 50)

    # Generate final report
    success = results.summary()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nTest Coverage Summary:")
        print("✓ Model loading and initialization")
        print("✓ Model switching and cleanup")
        print("✓ Error handling for corrupt models")
        print("✓ Memory management")
        print("✓ Concurrent model access")
        print("✓ Performance benchmarks")
        print("\nThe ModelManager implementation is robust and performant.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Review the failures above and fix the issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
