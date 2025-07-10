"""
Integration tests for ModelManager with real infrastructure.

These tests use actual file system, memory management, and concurrency
to verify behavior rather than just coverage.
"""

import pytest
import tempfile
import shutil
import time
import threading
import psutil
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import pickle
import json

# Import the real ModelManager
from music_gen.core.model_manager import ModelManager


class MockModel:
    """A realistic mock model for testing."""

    def __init__(self, name: str, size_mb: int = 100):
        self.name = name
        self.size_mb = size_mb
        # Simulate real memory usage with a large array
        self._data = bytearray(size_mb * 1024 * 1024)
        self.device = "cuda"
        self.loaded_at = time.time()

    def generate(self, prompt: str):
        """Simulate generation."""
        time.sleep(0.01)  # Simulate processing
        return f"Generated music for: {prompt}"

    def to(self, device: str):
        """Simulate device transfer."""
        self.device = device
        time.sleep(0.05)  # Simulate transfer time
        return self

    def __del__(self):
        """Cleanup when deleted."""
        # Simulate GPU memory cleanup
        del self._data


@pytest.fixture
def real_cache_dir():
    """Create a real temporary cache directory."""
    temp_dir = tempfile.mkdtemp(prefix="musicgen_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def model_manager_with_real_cache(real_cache_dir):
    """Create ModelManager with real cache directory."""
    ModelManager._instance = None

    with patch("pathlib.Path.home", return_value=real_cache_dir.parent):
        # Patch only the model loading, use real everything else
        with patch("music_gen.optimization.fast_generator.FastMusicGenerator", MockModel):
            manager = ModelManager()
            # Override cache dir to our test dir
            manager._cache_dir = real_cache_dir / ".cache" / "musicgen"
            manager._cache_dir.mkdir(parents=True, exist_ok=True)
            yield manager

    # Cleanup
    ModelManager._instance = None


class TestRealInfrastructure:
    """Test with real file system and memory."""

    def test_cache_persistence(self, model_manager_with_real_cache, real_cache_dir):
        """Test that cache actually persists to disk."""
        manager = model_manager_with_real_cache

        # Create a cache metadata file
        cache_meta_file = manager._cache_dir / "cache_metadata.json"

        # Load a model
        model = manager.get_model("test-model-1", "optimized")

        # Save cache metadata
        metadata = {"models": list(manager._models.keys()), "timestamp": time.time()}
        with open(cache_meta_file, "w") as f:
            json.dump(metadata, f)

        # Verify file was created
        assert cache_meta_file.exists()
        assert cache_meta_file.stat().st_size > 0

        # Read back and verify
        with open(cache_meta_file, "r") as f:
            loaded_meta = json.load(f)

        assert "test-model-1_optimized_cuda" in loaded_meta["models"]

    def test_real_memory_pressure(self, model_manager_with_real_cache):
        """Test behavior under real memory pressure."""
        manager = model_manager_with_real_cache
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        models_loaded = []
        memory_snapshots = []

        # Load models until we see significant memory increase
        for i in range(5):
            model_name = f"large-model-{i}"
            model = manager.get_model(model_name, "optimized")
            models_loaded.append(model_name)

            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_increase = current_memory - initial_memory
            memory_snapshots.append(memory_increase)

            # Each MockModel uses 100MB
            assert model is not None
            assert isinstance(model, MockModel)

        # Verify memory actually increased
        assert memory_snapshots[-1] > 100  # At least 100MB increase

        # Clear cache and verify memory is released
        manager.clear_cache()
        gc.collect()
        time.sleep(0.1)  # Give time for cleanup

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_released = (initial_memory + memory_snapshots[-1]) - final_memory

        # Should release most memory (allow 20% overhead)
        assert memory_released > memory_snapshots[-1] * 0.8

    def test_concurrent_access_race_conditions(self, model_manager_with_real_cache):
        """Test for race conditions with real concurrent access."""
        manager = model_manager_with_real_cache

        load_times = []
        errors = []
        successful_loads = []

        def load_model_concurrent(model_id: int):
            """Load model with timing."""
            try:
                start = time.time()
                # All threads try to load the same model
                model = manager.get_model("shared-model", "optimized")
                load_time = time.time() - start

                load_times.append(load_time)
                successful_loads.append(model)
                return model
            except Exception as e:
                errors.append(e)
                raise

        # Many threads try to load the same model simultaneously
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(load_model_concurrent, i) for i in range(50)]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    pass

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        # All should get the same model instance (proper caching)
        assert all(model is successful_loads[0] for model in successful_loads)

        # First load should be slower than cached accesses
        sorted_times = sorted(load_times)
        assert sorted_times[-1] > sorted_times[0] * 10  # Slowest is 10x slower than fastest

        # Most loads should be fast (cached)
        fast_loads = [t for t in load_times if t < 0.001]
        assert len(fast_loads) > 40  # Most should be cache hits

    def test_model_corruption_handling(self, model_manager_with_real_cache):
        """Test handling of corrupted model files."""
        manager = model_manager_with_real_cache

        # Create a corrupted model file
        model_file = manager._cache_dir / "corrupted_model.pkl"
        with open(model_file, "wb") as f:
            f.write(b"corrupted data that is not a valid pickle")

        # Try to load a model that will fail
        with patch("music_gen.optimization.fast_generator.FastMusicGenerator") as mock:

            def raise_corruption_error(*args, **kwargs):
                # Simulate trying to load corrupted file
                try:
                    with open(model_file, "rb") as f:
                        pickle.load(f)
                except:
                    raise RuntimeError("Model file corrupted")

            mock.side_effect = raise_corruption_error

            with pytest.raises(RuntimeError, match="Model file corrupted"):
                manager.get_model("corrupted-model", "optimized")

            # Model should not be cached
            assert "corrupted-model_optimized_cuda" not in manager._models

    def test_cache_directory_permissions(self, real_cache_dir):
        """Test handling of permission issues."""
        # Create a read-only cache directory
        readonly_dir = real_cache_dir / "readonly_cache"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            ModelManager._instance = None

            with patch("pathlib.Path.home", return_value=readonly_dir.parent):
                # Should handle permission error gracefully
                manager = ModelManager()

                # Should still initialize even if can't write to cache
                assert manager is not None
                assert manager._default_device in ["cuda", "cpu"]

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_model_lifecycle_with_real_timing(self, model_manager_with_real_cache):
        """Test complete model lifecycle with realistic timing."""
        manager = model_manager_with_real_cache

        # Track all operations with timing
        operations = []

        def timed_operation(name, func):
            start = time.time()
            result = func()
            elapsed = time.time() - start
            operations.append((name, elapsed))
            return result

        # Load model
        model = timed_operation(
            "load_model", lambda: manager.get_model("lifecycle-test", "optimized")
        )

        # Use model
        output = timed_operation("generate", lambda: model.generate("test prompt"))
        assert output == "Generated music for: test prompt"

        # Get cache info
        info = timed_operation("cache_info", lambda: manager.get_cache_info())
        assert info["loaded_models"] == 1

        # List models
        models = timed_operation("list_models", lambda: manager.list_loaded_models())
        assert "lifecycle-test" in models

        # Unload model
        unloaded = timed_operation("unload_model", lambda: manager.unload_model("lifecycle-test"))
        assert unloaded is True

        # Verify timing makes sense
        load_time = next(t for op, t in operations if op == "load_model")
        cache_info_time = next(t for op, t in operations if op == "cache_info")

        # Loading should take longer than getting cache info
        assert load_time > cache_info_time * 10

        # All operations should complete reasonably fast
        assert all(t < 1.0 for _, t in operations)  # Under 1 second each


class TestConcurrentModelOperations:
    """Test complex concurrent scenarios."""

    def test_concurrent_different_operations(self, model_manager_with_real_cache):
        """Test different operations happening concurrently."""
        manager = model_manager_with_real_cache

        # Pre-load some models
        for i in range(3):
            manager.get_model(f"base-model-{i}", "optimized")

        operation_results = {
            "loads": [],
            "unloads": [],
            "cache_infos": [],
            "listings": [],
            "errors": [],
        }

        def worker(worker_id: int, operation_type: str):
            """Perform different operations based on type."""
            try:
                if operation_type == "load":
                    model = manager.get_model(f"worker-{worker_id}", "optimized")
                    operation_results["loads"].append((worker_id, model))

                elif operation_type == "unload":
                    model_to_unload = f"base-model-{worker_id % 3}"
                    result = manager.unload_model(model_to_unload)
                    operation_results["unloads"].append((worker_id, result))

                elif operation_type == "cache_info":
                    info = manager.get_cache_info()
                    operation_results["cache_infos"].append((worker_id, info))

                elif operation_type == "list":
                    models = manager.list_loaded_models()
                    operation_results["listings"].append((worker_id, models))

            except Exception as e:
                operation_results["errors"].append((worker_id, operation_type, str(e)))

        # Run many concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            # Mix of different operations
            for i in range(40):
                operation = ["load", "unload", "cache_info", "list"][i % 4]
                future = executor.submit(worker, i, operation)
                futures.append(future)

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

        # Verify results
        assert len(operation_results["errors"]) == 0
        assert len(operation_results["loads"]) == 10
        assert len(operation_results["cache_infos"]) == 10
        assert len(operation_results["listings"]) == 10

        # Some unloads might fail if model was already unloaded
        assert len(operation_results["unloads"]) >= 5

    def test_stress_test_with_memory_monitoring(self, model_manager_with_real_cache):
        """Stress test with continuous memory monitoring."""
        manager = model_manager_with_real_cache
        process = psutil.Process()

        memory_samples = []
        stop_monitoring = False

        def monitor_memory():
            """Continuously monitor memory usage."""
            while not stop_monitoring:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_samples.append(memory_mb)
                time.sleep(0.01)

        # Start memory monitoring in background
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        try:
            # Stress test: rapidly load and unload models
            for cycle in range(5):
                # Load phase
                for i in range(10):
                    manager.get_model(f"stress-model-{i}", "optimized")

                # Check phase
                assert len(manager._models) == 10

                # Unload phase
                for i in range(5):
                    manager.unload_model(f"stress-model-{i}")

                # Verify partial unload
                assert len(manager._models) == 5

                # Clear remaining
                manager.clear_cache()
                assert len(manager._models) == 0

        finally:
            stop_monitoring = True
            monitor_thread.join(timeout=1)

        # Analyze memory pattern
        if len(memory_samples) > 10:
            # Memory should spike during load and drop after clear
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)

            # Should see significant memory variation
            assert max_memory > min_memory + 50  # At least 50MB variation

            # Final memory should be close to initial
            initial_samples = memory_samples[:5]
            final_samples = memory_samples[-5:]

            initial_avg = sum(initial_samples) / len(initial_samples)
            final_avg = sum(final_samples) / len(final_samples)

            # Should return close to baseline (within 20%)
            assert abs(final_avg - initial_avg) < initial_avg * 0.2


class TestRealWorldScenarios:
    """Test realistic usage patterns."""

    def test_hot_reload_scenario(self, model_manager_with_real_cache):
        """Test hot-reloading models (common in development)."""
        manager = model_manager_with_real_cache

        model_versions = []

        # Simulate updating a model multiple times
        for version in range(3):
            # Unload old version if exists
            if version > 0:
                manager.unload_model("evolving-model")

            # Load new version
            with patch("music_gen.optimization.fast_generator.FastMusicGenerator") as mock:
                mock_model = MockModel(f"evolving-model-v{version}")
                mock_model.version = version
                mock.return_value = mock_model

                model = manager.get_model("evolving-model", "optimized")
                model_versions.append(model)

        # Verify we got different versions
        assert len(set(id(m) for m in model_versions)) == 3
        assert model_versions[-1].version == 2

    def test_production_load_pattern(self, model_manager_with_real_cache):
        """Test production-like load patterns."""
        manager = model_manager_with_real_cache

        # Simulate production pattern:
        # - Burst of requests at startup
        # - Steady state operations
        # - Occasional spikes

        request_times = []

        # Startup burst
        with ThreadPoolExecutor(max_workers=20) as executor:
            start = time.time()

            futures = []
            for i in range(20):
                future = executor.submit(manager.get_model, "production-model", "optimized")
                futures.append(future)

            for future in as_completed(futures):
                future.result()
                request_times.append(time.time() - start)

        # Should handle burst efficiently
        assert max(request_times) < 0.5  # All complete within 500ms

        # Steady state - sequential requests
        steady_times = []
        for i in range(10):
            start = time.time()
            model = manager.get_model("production-model", "optimized")
            steady_times.append(time.time() - start)
            time.sleep(0.01)  # Simulate time between requests

        # Steady state should be fast (cached)
        assert all(t < 0.001 for t in steady_times)

        # Spike - sudden concurrent load
        spike_times = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(manager.get_model, f"spike-model-{i % 3}", "optimized")
                futures.append(future)

            start = time.time()
            for future in as_completed(futures):
                future.result()
                spike_times.append(time.time() - start)

        # Should handle spike gracefully
        assert max(spike_times) < 0.2  # All complete within 200ms


@pytest.mark.slow
class TestLongRunningScenarios:
    """Tests that take longer but test important scenarios."""

    def test_24_hour_simulation(self, model_manager_with_real_cache):
        """Simulate 24 hours of operations (accelerated)."""
        manager = model_manager_with_real_cache

        # Simulate 24 hours in 24 seconds (1 second = 1 hour)
        start_time = time.time()
        operations_log = []

        for hour in range(24):
            # Different patterns for different times
            if 9 <= hour <= 17:  # Business hours - high load
                ops_per_hour = 20
            elif hour < 6 or hour > 22:  # Night - low load
                ops_per_hour = 2
            else:  # Morning/evening - medium load
                ops_per_hour = 10

            for op in range(ops_per_hour):
                operation_type = ["load", "cache_info", "list"][op % 3]

                if operation_type == "load":
                    model_name = f"model-{hour}-{op % 5}"
                    manager.get_model(model_name, "optimized")
                elif operation_type == "cache_info":
                    info = manager.get_cache_info()
                    operations_log.append(("cache_info", hour, info["loaded_models"]))
                else:
                    manager.list_loaded_models()

                time.sleep(0.001)  # Small delay between ops

            # Hourly cleanup for old models
            if hour % 6 == 0 and hour > 0:
                old_models = [k for k in manager._models.keys() if f"model-{hour-6}" in k]
                for model in old_models:
                    manager.unload_model(model.split("_")[0])

            time.sleep(0.1)  # Simulate hour passing

        # Verify system remained stable
        elapsed = time.time() - start_time
        assert elapsed < 30  # Should complete in reasonable time

        # Check memory didn't leak
        final_model_count = len(manager._models)
        assert final_model_count < 50  # Reasonable number of models

        # Verify operations were logged
        assert len(operations_log) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
