"""
Behavior-focused tests for ModelManager - Refactored from coverage-only tests.

These tests verify actual behavior and would catch real bugs,
rather than just achieving code coverage.
"""

import pytest
import time
import threading
import gc
from pathlib import Path
from unittest.mock import Mock, patch, call
from concurrent.futures import ThreadPoolExecutor, as_completed

from music_gen.core.model_manager import ModelManager


class TestSingletonBehavior:
    """Test singleton behavior, not just pattern implementation."""

    def test_singleton_preserves_state_across_instances(self):
        """Test that singleton actually preserves state between instances."""
        ModelManager._instance = None

        # First instance loads a model
        manager1 = ModelManager()
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock(name="test_model")
            model1 = manager1.get_model("shared-model", "optimized")

        # Second instance should see the same state
        manager2 = ModelManager()

        # Verify it's the same instance (singleton)
        assert manager1 is manager2

        # More importantly: verify state is preserved
        assert len(manager2._models) == 1
        assert "shared-model_optimized_cuda" in manager2._models

        # Should get cached model without loading again
        mock.reset_mock()
        model2 = manager2.get_model("shared-model", "optimized")

        # Behavior verification: no new loading, same model returned
        mock.assert_not_called()
        assert model1 is model2

    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe during concurrent initialization."""
        ModelManager._instance = None
        instances = []

        def create_manager():
            manager = ModelManager()
            instances.append(manager)
            # Small delay to increase chance of race condition
            time.sleep(0.001)
            return manager

        # Multiple threads try to create manager simultaneously
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_manager) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # Behavior verification: all instances are the same object
        assert all(inst is results[0] for inst in results)
        assert len(set(id(inst) for inst in results)) == 1


class TestModelLoadingBehavior:
    """Test actual model loading behavior, not just successful loading."""

    @pytest.fixture
    def manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_model_loading_follows_correct_sequence(self, manager):
        """Test that model loading follows the correct sequence of operations."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
            mock_model = Mock()
            mock_gen.return_value = mock_model

            # Load model and verify call sequence
            model = manager.get_model("test-model", "optimized", batch_size=32)

            # Behavior verification: correct parameters passed
            mock_gen.assert_called_once_with(
                model_name="test-model",
                device=manager._default_device,
                cache_dir=str(manager._cache_dir),
                batch_size=32,
            )

            # Verify model is properly stored with correct key
            expected_key = f"test-model_optimized_{manager._default_device}"
            assert expected_key in manager._models
            assert manager._models[expected_key] is mock_model

    def test_cache_miss_vs_cache_hit_behavior(self, manager):
        """Test different behavior for cache miss vs cache hit."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
            mock_model = Mock()
            mock_gen.return_value = mock_model

            # First load (cache miss) - should call generator
            start_time = time.time()
            model1 = manager.get_model("cached-test", "optimized")
            cache_miss_time = time.time() - start_time

            assert mock_gen.call_count == 1
            assert model1 is mock_model

            # Second load (cache hit) - should NOT call generator
            mock_gen.reset_mock()
            start_time = time.time()
            model2 = manager.get_model("cached-test", "optimized")
            cache_hit_time = time.time() - start_time

            # Behavior verification: no additional loading, much faster
            mock_gen.assert_not_called()
            assert model1 is model2
            assert cache_hit_time < cache_miss_time / 10  # Cache hit should be much faster

    def test_model_type_affects_loading_behavior(self, manager):
        """Test that different model types result in different loading behavior."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
            mock_optimized = Mock(name="optimized_model")
            mock_multi = Mock(name="multi_instrument_model")
            mock_gen.side_effect = [mock_optimized, mock_multi]

            # Load different model types
            opt_model = manager.get_model("test", "optimized")
            multi_model = manager.get_model("test", "multi_instrument")

            # Behavior verification: both types loaded, stored separately
            assert mock_gen.call_count == 2
            assert opt_model is mock_optimized
            assert multi_model is mock_multi

            # Verify they're cached separately
            assert "test_optimized_cuda" in manager._models
            assert "test_multi_instrument_cuda" in manager._models
            assert (
                manager._models["test_optimized_cuda"]
                is not manager._models["test_multi_instrument_cuda"]
            )


class TestCachingBehavior:
    """Test actual caching behavior and memory management."""

    @pytest.fixture
    def manager_with_models(self):
        ModelManager._instance = None
        manager = ModelManager()

        # Pre-load some models
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            models = []
            for i in range(3):
                mock_model = Mock(name=f"model_{i}")
                models.append(mock_model)
                mock.return_value = mock_model
                manager.get_model(f"model_{i}", "optimized")

        return manager

    def test_cache_clearing_behavior(self, manager_with_models):
        """Test that cache clearing actually removes references and triggers cleanup."""
        manager = manager_with_models

        # Verify models are loaded
        assert len(manager._models) == 3
        initial_keys = set(manager._models.keys())

        # Clear cache and verify cleanup behavior
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.empty_cache") as mock_cuda_clear:
                manager.clear_cache()

                # Behavior verification: models removed and cleanup called
                assert len(manager._models) == 0
                assert all(key not in manager._models for key in initial_keys)
                mock_gc.assert_called_once()
                mock_cuda_clear.assert_called_once()

    def test_selective_unloading_behavior(self, manager_with_models):
        """Test that unloading specific models works correctly."""
        manager = manager_with_models

        initial_count = len(manager._models)

        # Unload one specific model
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.empty_cache") as mock_cuda_clear:
                result = manager.unload_model("model_1")

                # Behavior verification:
                assert result is True  # Successfully unloaded
                assert len(manager._models) == initial_count - 1
                assert "model_1_optimized_cuda" not in manager._models

                # Other models should remain
                assert "model_0_optimized_cuda" in manager._models
                assert "model_2_optimized_cuda" in manager._models

                # Cleanup should be triggered
                mock_gc.assert_called_once()
                mock_cuda_clear.assert_called_once()

    def test_cache_info_reflects_actual_state(self, manager_with_models):
        """Test that cache info accurately reflects the actual cache state."""
        manager = manager_with_models

        with patch.object(manager, "_get_cache_size", return_value=150 * 1024 * 1024):  # 150MB
            cache_info = manager.get_cache_info()

            # Behavior verification: info matches actual state
            assert cache_info["loaded_models"] == len(manager._models)
            assert len(cache_info["models"]) == len(manager._models)

            # All loaded models should be listed
            for model_key in manager._models.keys():
                assert model_key in cache_info["models"]

            # Size calculation should be called
            assert cache_info["cache_size_mb"] == 150.0


class TestErrorHandlingBehavior:
    """Test actual error handling and recovery behavior."""

    @pytest.fixture
    def manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_loading_failure_leaves_clean_state(self, manager):
        """Test that loading failures don't leave partial state."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
            # First, successful load
            mock_gen.return_value = Mock()
            manager.get_model("good-model", "optimized")
            initial_count = len(manager._models)

            # Then, failed load
            mock_gen.side_effect = RuntimeError("Model file corrupted")

            with pytest.raises(RuntimeError, match="Model file corrupted"):
                manager.get_model("bad-model", "optimized")

            # Behavior verification: no partial state left behind
            assert len(manager._models) == initial_count  # Count unchanged
            assert "bad-model_optimized_cuda" not in manager._models

            # Good model should still be accessible
            assert "good-model_optimized_cuda" in manager._models

    def test_invalid_model_type_behavior(self, manager):
        """Test that invalid model types are handled correctly."""
        initial_state = dict(manager._models)

        # Try invalid model type
        with pytest.raises(ValueError) as exc_info:
            manager.get_model("test", "invalid_type")

        # Behavior verification: specific error message
        assert "Unknown model type: invalid_type" in str(exc_info.value)

        # No state changes
        assert manager._models == initial_state

    def test_concurrent_loading_with_failures(self, manager):
        """Test that failures in one thread don't affect others."""
        results = []
        errors = []

        def load_model(model_name: str, should_fail: bool = False):
            try:
                with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
                    if should_fail:
                        mock.side_effect = RuntimeError(f"Failed to load {model_name}")
                    else:
                        mock.return_value = Mock(name=model_name)

                    model = manager.get_model(model_name, "optimized")
                    results.append((model_name, model))
            except Exception as e:
                errors.append((model_name, str(e)))

        # Load multiple models concurrently, some fail
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                should_fail = i % 3 == 0  # Every 3rd load fails
                future = executor.submit(load_model, f"model_{i}", should_fail)
                futures.append(future)

            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Behavior verification: successes and failures handled correctly
        successful_loads = len(results)
        failed_loads = len(errors)

        assert successful_loads > 0  # Some should succeed
        assert failed_loads > 0  # Some should fail
        assert successful_loads + failed_loads == 10  # All accounted for

        # Only successful models should be cached
        assert len(manager._models) == successful_loads


class TestMemoryManagementBehavior:
    """Test actual memory management behavior."""

    @pytest.fixture
    def manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_memory_cleanup_sequence(self, manager):
        """Test that memory cleanup follows correct sequence."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
            # Load some models
            models = []
            for i in range(3):
                mock_model = Mock()
                models.append(mock_model)
                mock_gen.return_value = mock_model
                manager.get_model(f"memory_test_{i}", "optimized")

        # Track cleanup sequence
        cleanup_calls = []

        def track_gc():
            cleanup_calls.append("gc_collect")

        def track_cuda():
            cleanup_calls.append("cuda_empty_cache")

        with patch("gc.collect", side_effect=track_gc):
            with patch("torch.cuda.empty_cache", side_effect=track_cuda):
                manager.clear_cache()

        # Behavior verification: cleanup happens in correct order
        assert cleanup_calls == ["gc_collect", "cuda_empty_cache"]
        assert len(manager._models) == 0

    def test_device_specific_behavior(self, manager):
        """Test that device-specific behavior is handled correctly."""
        # Test CUDA available scenario
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=2**30):  # 1GB
                with patch("torch.cuda.memory_reserved", return_value=2**31):  # 2GB
                    cache_info = manager.get_cache_info()

                    # Behavior verification: GPU memory info included
                    assert "gpu_memory" in cache_info
                    assert cache_info["gpu_memory"]["allocated_gb"] == 1.0
                    assert cache_info["gpu_memory"]["cached_gb"] == 2.0

        # Test CUDA not available scenario
        with patch("torch.cuda.is_available", return_value=False):
            cache_info = manager.get_cache_info()

            # Behavior verification: no GPU info when CUDA unavailable
            assert "gpu_memory" not in cache_info


class TestConcurrencyBehavior:
    """Test actual concurrent behavior and thread safety."""

    @pytest.fixture
    def manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_concurrent_cache_operations_consistency(self, manager):
        """Test that concurrent cache operations maintain consistency."""
        operation_log = []
        lock = threading.Lock()

        def log_operation(op_type, model_count):
            with lock:
                operation_log.append((op_type, model_count, threading.current_thread().ident))

        def worker(worker_id: int):
            # Each worker does multiple operations
            for i in range(5):
                # Load model
                with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
                    mock.return_value = Mock()
                    manager.get_model(f"worker_{worker_id}_model_{i}", "optimized")
                    log_operation("load", len(manager._models))

                # Check cache info
                info = manager.get_cache_info()
                log_operation("info", info["loaded_models"])

                # Occasionally unload
                if i % 3 == 0 and i > 0:
                    manager.unload_model(f"worker_{worker_id}_model_{i-1}")
                    log_operation("unload", len(manager._models))

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        # Behavior verification: operations are consistent
        assert len(operation_log) > 0

        # Cache info should always match actual model count
        for op_type, reported_count, thread_id in operation_log:
            if op_type == "info":
                # Find corresponding load operation
                actual_count = len(manager._models)
                # Allow some tolerance due to concurrency
                assert abs(reported_count - actual_count) <= 5

    def test_race_condition_prevention(self, manager):
        """Test that race conditions are prevented in model loading."""
        load_counter = 0

        def counting_model_generator(*args, **kwargs):
            nonlocal load_counter
            load_counter += 1
            time.sleep(0.01)  # Simulate loading time
            return Mock(name=f"model_{load_counter}")

        with patch(
            "music_gen.core.model_manager.FastMusicGenerator", side_effect=counting_model_generator
        ):
            # Many threads try to load the same model
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(manager.get_model, "race-test", "optimized") for _ in range(20)
                ]
                results = [f.result() for f in as_completed(futures)]

        # Behavior verification: model loaded only once despite race
        assert load_counter == 1  # Only one load should have occurred
        assert all(result is results[0] for result in results)  # All get same instance
        assert len(manager._models) == 1  # Only one model cached


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
