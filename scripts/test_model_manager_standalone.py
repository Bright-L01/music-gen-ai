#!/usr/bin/env python3
"""
Standalone test runner for model_manager.py tests.

This script tests model manager functionality without relying on pytest infrastructure
that might have dependency issues.
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock torch and dependencies at module level
sys.modules["torch"] = Mock()
sys.modules["torch.cuda"] = Mock()
sys.modules["psutil"] = Mock()
sys.modules["music_gen.optimization.fast_generator"] = Mock()

# Configure mocks before importing
torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.memory_allocated.return_value = 1024**3
torch_mock.cuda.memory_reserved.return_value = 2 * (1024**3)
torch_mock.cuda.max_memory_allocated.return_value = 3 * (1024**3)
torch_mock.cuda.empty_cache = Mock()
torch_mock.cuda.OutOfMemoryError = Exception

psutil_mock = sys.modules["psutil"]
process_mock = Mock()
memory_info_mock = Mock()
memory_info_mock.rss = 100 * 1024 * 1024
process_mock.memory_info.return_value = memory_info_mock
psutil_mock.Process.return_value = process_mock

# Now import the module under test
from music_gen.core.model_manager import ModelManager


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def test_passed(self, test_name):
        self.passed += 1
        print(f"✓ {test_name}")

    def test_failed(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\nTest Results: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailures:")
            for test_name, error in self.errors:
                print(f"  {test_name}: {error}")
        return self.failed == 0


def run_test(test_func, test_name, results):
    """Run a single test function."""
    try:
        test_func()
        results.test_passed(test_name)
    except Exception as e:
        results.test_failed(test_name, str(e))


def test_singleton_pattern():
    """Test ModelManager singleton pattern."""
    # Reset singleton
    ModelManager._instance = None

    manager1 = ModelManager()
    manager2 = ModelManager()

    assert manager1 is manager2, "ModelManager should be singleton"
    assert id(manager1) == id(manager2), "Same object ID expected"


def test_initialization():
    """Test ModelManager initialization."""
    ModelManager._instance = None

    with patch("torch.cuda.is_available", return_value=True):
        manager = ModelManager()

        assert manager._default_device == "cuda", "Should use CUDA when available"
        assert manager._cache_dir.name == "musicgen", "Cache directory name should be musicgen"
        assert len(manager._models) == 0, "Should start with empty model cache"


def test_model_loading():
    """Test model loading functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Mock FastMusicGenerator
    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_generator:
        mock_model = Mock()
        mock_generator.return_value = mock_model

        # Load model
        model = manager.get_model("test-model", "optimized")

        assert model is mock_model, "Should return the loaded model"
        assert len(manager._models) == 1, "Model should be cached"
        assert "test-model_optimized_cuda" in manager._models, "Model key should be correct"

        # Load same model again
        model2 = manager.get_model("test-model", "optimized")
        assert model is model2, "Should return cached model"


def test_model_unloading():
    """Test model unloading functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    with patch("gc.collect") as mock_gc:
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            result = manager.unload_model("model1")

            assert result is True, "Should return True for successful unload"
            assert "model1_optimized_cuda" not in manager._models, "Model should be removed"
            assert len(manager._models) == 1, "Only one model should remain"

            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()


def test_cache_clearing():
    """Test cache clearing functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    with patch("gc.collect") as mock_gc:
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            manager.clear_cache()

            assert len(manager._models) == 0, "All models should be cleared"
            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()


def test_error_handling():
    """Test error handling during model loading."""
    ModelManager._instance = None
    manager = ModelManager()

    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_generator:
        mock_generator.side_effect = RuntimeError("Model corrupted")

        try:
            manager.get_model("corrupt-model", "optimized")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Model corrupted" in str(e), "Error message should be preserved"

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

    load_count = 0

    def mock_generator(**kwargs):
        nonlocal load_count
        load_count += 1
        time.sleep(0.1)  # Simulate loading time
        return Mock()

    with patch("music_gen.core.model_manager.FastMusicGenerator", side_effect=mock_generator):
        results = []

        def load_model(thread_id):
            return manager.get_model("test-model", "optimized")

        # Load same model from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_model, i) for i in range(5)]
            results = [future.result() for future in as_completed(futures)]

        assert load_count == 1, "Model should only be loaded once"
        assert all(results[0] is r for r in results), "All threads should get same instance"


def test_cache_info():
    """Test cache info functionality."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(),
        "model2_optimized_cuda": Mock(),
    }

    with patch.object(manager, "_get_cache_size", return_value=100 * 1024 * 1024):  # 100MB
        cache_info = manager.get_cache_info()

        assert cache_info["loaded_models"] == 2, "Should report correct model count"
        assert len(cache_info["models"]) == 2, "Should list all models"
        assert cache_info["cache_size_mb"] == 100.0, "Should report correct cache size"
        assert "gpu_memory" in cache_info, "Should include GPU memory info when CUDA available"


def test_list_loaded_models():
    """Test listing loaded models."""
    ModelManager._instance = None
    manager = ModelManager()

    # Add mock models
    manager._models = {
        "model1_optimized_cuda": Mock(__class__=Mock(__name__="TestModel")),
        "model2_multi_instrument_cpu": Mock(__class__=Mock(__name__="TestModel")),
    }

    models_info = manager.list_loaded_models()

    assert len(models_info) == 2, "Should list all unique models"
    assert "model1" in models_info, "Should include model1"
    assert "model2" in models_info, "Should include model2"

    assert models_info["model1"]["type"] == "optimized", "Should parse model type correctly"
    assert models_info["model1"]["device"] == "cuda", "Should parse device correctly"
    assert models_info["model1"]["loaded"] is True, "Should mark as loaded"


def performance_benchmark_model_loading():
    """Benchmark model loading performance."""
    ModelManager._instance = None
    manager = ModelManager()

    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_generator:
        mock_generator.return_value = Mock()

        # Benchmark loading
        start_time = time.time()

        for i in range(10):
            manager._models.clear()  # Force reload
            manager.get_model(f"model{i}", "optimized")

        end_time = time.time()
        avg_time = (end_time - start_time) / 10

        print(f"    Average model loading time: {avg_time:.4f}s")

        # Should be reasonably fast (< 1s for mock)
        assert avg_time < 1.0, "Model loading should be fast for mocked models"


def performance_benchmark_cached_access():
    """Benchmark cached model access performance."""
    ModelManager._instance = None
    manager = ModelManager()

    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_generator:
        mock_generator.return_value = Mock()

        # Pre-load model
        manager.get_model("test-model", "optimized")

        # Benchmark cached access
        start_time = time.time()

        for i in range(1000):
            manager.get_model("test-model", "optimized")

        end_time = time.time()
        avg_time = (end_time - start_time) / 1000

        print(f"    Average cached access time: {avg_time:.6f}s")

        # Cached access should be very fast
        assert avg_time < 0.001, "Cached access should be very fast"


def performance_benchmark_concurrent_access():
    """Benchmark concurrent access performance."""
    ModelManager._instance = None
    manager = ModelManager()

    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_generator:
        mock_generator.return_value = Mock()

        # Pre-load model
        manager.get_model("test-model", "optimized")

        start_time = time.time()

        def access_model():
            return manager.get_model("test-model", "optimized")

        # Concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_model) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        print(f"    100 concurrent accesses completed in: {total_time:.4f}s")

        assert len(results) == 100, "All accesses should complete"
        assert total_time < 5.0, "Concurrent access should complete reasonably quickly"


def main():
    """Run all tests."""
    print("Running Model Manager Comprehensive Tests\n")

    results = TestResults()

    # Core functionality tests
    print("=== Core Functionality Tests ===")
    run_test(test_singleton_pattern, "Singleton Pattern", results)
    run_test(test_initialization, "Initialization", results)
    run_test(test_model_loading, "Model Loading", results)
    run_test(test_model_unloading, "Model Unloading", results)
    run_test(test_cache_clearing, "Cache Clearing", results)
    run_test(test_list_loaded_models, "List Loaded Models", results)
    run_test(test_cache_info, "Cache Info", results)

    # Error handling tests
    print("\n=== Error Handling Tests ===")
    run_test(test_error_handling, "Error Handling", results)
    run_test(test_unknown_model_type, "Unknown Model Type", results)

    # Concurrency tests
    print("\n=== Concurrency Tests ===")
    run_test(test_concurrent_access, "Concurrent Access", results)

    # Performance benchmarks
    print("\n=== Performance Benchmarks ===")
    run_test(performance_benchmark_model_loading, "Model Loading Performance", results)
    run_test(performance_benchmark_cached_access, "Cached Access Performance", results)
    run_test(performance_benchmark_concurrent_access, "Concurrent Access Performance", results)

    # Print summary
    success = results.summary()

    if success:
        print("\n🎉 All tests passed! Model Manager is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
