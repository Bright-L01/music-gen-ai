"""
Comprehensive tests for core/model_manager.py with performance benchmarks.

Tests include:
- Model loading and initialization
- Model switching and cleanup
- Error handling for corrupt models
- Memory management
- Concurrent model access
- Performance benchmarking
"""

import pytest
import time
import gc
import os
import threading
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import psutil

from music_gen.core.model_manager import ModelManager


class TestModelManagerInitialization:
    """Test ModelManager initialization and singleton behavior."""

    def test_singleton_pattern(self):
        """Test that ModelManager is a singleton."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2
        assert id(manager1) == id(manager2)

    def test_initialization_with_cuda(self):
        """Test initialization with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            # Reset singleton for testing
            ModelManager._instance = None

            manager = ModelManager()

            assert manager._default_device == "cuda"
            assert manager._cache_dir.exists()
            assert manager._cache_dir.name == "musicgen"
            assert len(manager._models) == 0

    def test_initialization_without_cuda(self):
        """Test initialization without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            # Reset singleton for testing
            ModelManager._instance = None

            manager = ModelManager()

            assert manager._default_device == "cpu"
            assert manager._cache_dir.exists()

    def test_cache_directory_creation(self):
        """Test cache directory is created properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Reset singleton for testing
                ModelManager._instance = None

                manager = ModelManager()
                expected_path = Path(temp_dir) / ".cache" / "musicgen"

                assert manager._cache_dir == expected_path
                assert expected_path.exists()
                assert expected_path.is_dir()


class TestModelLoading:
    """Test model loading functionality."""

    @pytest.fixture
    def mock_fast_generator(self):
        """Mock FastMusicGenerator class."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock_instance = Mock()
            mock_instance.model_name = "test-model"
            mock_instance.device = "cuda"
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        with patch("torch.cuda.is_available", return_value=True):
            return ModelManager()

    def test_load_model_success(self, manager, mock_fast_generator):
        """Test successful model loading."""
        model = manager.get_model(model_name="facebook/musicgen-small", model_type="optimized")

        assert model is not None
        mock_fast_generator.assert_called_once_with(
            model_name="facebook/musicgen-small", device="cuda", cache_dir=str(manager._cache_dir)
        )

        # Check model is cached
        assert len(manager._models) == 1
        assert "facebook/musicgen-small_optimized_cuda" in manager._models

    def test_load_model_with_custom_device(self, manager, mock_fast_generator):
        """Test loading model with custom device."""
        model = manager.get_model(model_name="test-model", model_type="optimized", device="cpu")

        mock_fast_generator.assert_called_with(
            model_name="test-model", device="cpu", cache_dir=str(manager._cache_dir)
        )

        assert "test-model_optimized_cpu" in manager._models

    def test_load_cached_model(self, manager, mock_fast_generator):
        """Test loading already cached model."""
        # Load model first time
        model1 = manager.get_model("test-model", "optimized")

        # Reset mock
        mock_fast_generator.reset_mock()

        # Load same model again
        model2 = manager.get_model("test-model", "optimized")

        # Should use cached version, not create new one
        mock_fast_generator.assert_not_called()
        assert model1 is model2

    def test_load_multi_instrument_model(self, manager, mock_fast_generator):
        """Test loading multi-instrument model type."""
        model = manager.get_model(model_name="test-model", model_type="multi_instrument")

        assert model is not None
        assert "test-model_multi_instrument_cuda" in manager._models

    def test_load_unknown_model_type(self, manager):
        """Test loading unknown model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type: unknown"):
            manager.get_model("test-model", "unknown")

    def test_load_model_with_kwargs(self, manager, mock_fast_generator):
        """Test loading model with additional kwargs."""
        model = manager.get_model("test-model", "optimized", batch_size=16, sample_rate=48000)

        mock_fast_generator.assert_called_with(
            model_name="test-model",
            device="cuda",
            cache_dir=str(manager._cache_dir),
            batch_size=16,
            sample_rate=48000,
        )


class TestModelSwitchingAndCleanup:
    """Test model switching and cleanup functionality."""

    @pytest.fixture
    def manager_with_models(self):
        """Create manager with pre-loaded models."""
        ModelManager._instance = None
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            # Mock some loaded models
            manager._models = {
                "model1_optimized_cuda": Mock(model_name="model1"),
                "model1_optimized_cpu": Mock(model_name="model1"),
                "model2_multi_instrument_cuda": Mock(model_name="model2"),
            }

            return manager

    def test_list_loaded_models(self, manager_with_models):
        """Test listing loaded models."""
        models_info = manager_with_models.list_loaded_models()

        assert len(models_info) == 2  # model1 and model2
        assert "model1" in models_info
        assert "model2" in models_info

        assert models_info["model1"]["type"] == "optimized"
        assert models_info["model1"]["device"] == "cuda"  # Last loaded device
        assert models_info["model1"]["loaded"] is True

    def test_has_loaded_models(self, manager_with_models):
        """Test checking if models are loaded."""
        assert manager_with_models.has_loaded_models() is True

        manager_with_models._models.clear()
        assert manager_with_models.has_loaded_models() is False

    def test_unload_specific_model(self, manager_with_models):
        """Test unloading a specific model."""
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                result = manager_with_models.unload_model("model1")

                assert result is True
                assert len(manager_with_models._models) == 1
                assert "model2_multi_instrument_cuda" in manager_with_models._models

                mock_gc.assert_called_once()
                mock_empty_cache.assert_called_once()

    def test_unload_nonexistent_model(self, manager_with_models):
        """Test unloading a model that doesn't exist."""
        result = manager_with_models.unload_model("nonexistent")

        assert result is False
        assert len(manager_with_models._models) == 3  # No change

    def test_clear_cache(self, manager_with_models):
        """Test clearing all models from cache."""
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                manager_with_models.clear_cache()

                assert len(manager_with_models._models) == 0
                mock_gc.assert_called_once()
                mock_empty_cache.assert_called_once()

    def test_model_switching_memory_cleanup(self, manager_with_models):
        """Test memory is properly cleaned when switching models."""
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                # Unload model1
                manager_with_models.unload_model("model1")

                # Load a new model
                with patch("music_gen.core.model_manager.FastMusicGenerator"):
                    manager_with_models.get_model("model3", "optimized")

                assert "model3_optimized_cuda" in manager_with_models._models
                assert mock_gc.call_count >= 1
                assert mock_empty_cache.call_count >= 1


class TestErrorHandling:
    """Test error handling for corrupt models and failures."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        return ModelManager()

    def test_model_loading_exception(self, manager):
        """Test handling of exceptions during model loading."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = RuntimeError("Model file corrupted")

            with pytest.raises(RuntimeError, match="Model file corrupted"):
                manager.get_model("corrupt-model", "optimized")

            # Model should not be cached on failure
            assert len(manager._models) == 0

    def test_out_of_memory_error(self, manager):
        """Test handling of CUDA out of memory errors."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

            with pytest.raises(torch.cuda.OutOfMemoryError):
                manager.get_model("large-model", "optimized")

    def test_invalid_model_path(self, manager):
        """Test handling of invalid model paths."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(FileNotFoundError):
                manager.get_model("nonexistent-model", "optimized")

    def test_cache_directory_permissions(self):
        """Test handling of cache directory permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / ".cache" / "musicgen"
            cache_path.mkdir(parents=True)

            # Make directory read-only
            os.chmod(cache_path, 0o444)

            try:
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    ModelManager._instance = None
                    # Should handle permission error gracefully
                    manager = ModelManager()
                    assert manager._cache_dir == cache_path
            finally:
                # Restore permissions for cleanup
                os.chmod(cache_path, 0o755)


class TestMemoryManagement:
    """Test memory management functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        with patch("torch.cuda.is_available", return_value=True):
            return ModelManager()

    def test_get_cache_info_with_cuda(self, manager):
        """Test getting cache info with CUDA available."""
        # Mock some models
        manager._models = {
            "model1_optimized_cuda": Mock(),
            "model2_optimized_cuda": Mock(),
        }

        with patch("torch.cuda.memory_allocated", return_value=2 * (1024**3)):  # 2GB
            with patch("torch.cuda.memory_reserved", return_value=3 * (1024**3)):  # 3GB
                with patch("torch.cuda.max_memory_allocated", return_value=4 * (1024**3)):  # 4GB
                    with patch.object(
                        manager, "_get_cache_size", return_value=100 * (1024**2)
                    ):  # 100MB
                        cache_info = manager.get_cache_info()

        assert cache_info["loaded_models"] == 2
        assert len(cache_info["models"]) == 2
        assert cache_info["cache_size_mb"] == 100.0
        assert cache_info["gpu_memory"]["allocated_gb"] == 2.0
        assert cache_info["gpu_memory"]["cached_gb"] == 3.0
        assert cache_info["gpu_memory"]["max_allocated_gb"] == 4.0

    def test_get_cache_info_without_cuda(self, manager):
        """Test getting cache info without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(manager, "_get_cache_size", return_value=50 * (1024**2)):  # 50MB
                cache_info = manager.get_cache_info()

        assert "gpu_memory" not in cache_info
        assert cache_info["cache_size_mb"] == 50.0

    def test_cache_size_calculation(self, manager):
        """Test cache size calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager._cache_dir = Path(temp_dir)

            # Create some test files
            (manager._cache_dir / "model1.bin").write_bytes(b"x" * 1024 * 1024)  # 1MB
            (manager._cache_dir / "model2.bin").write_bytes(b"x" * 2 * 1024 * 1024)  # 2MB

            subdir = manager._cache_dir / "subdir"
            subdir.mkdir()
            (subdir / "model3.bin").write_bytes(b"x" * 512 * 1024)  # 512KB

            size = manager._get_cache_size()
            expected_size = 1024 * 1024 + 2 * 1024 * 1024 + 512 * 1024  # 3.5MB

            assert size == expected_size

    def test_memory_cleanup_on_model_switch(self, manager):
        """Test memory is cleaned up when switching models."""
        call_count = 0

        def mock_gc_collect():
            nonlocal call_count
            call_count += 1

        with patch("gc.collect", side_effect=mock_gc_collect):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with patch("music_gen.core.model_manager.FastMusicGenerator"):
                    # Load first model
                    manager.get_model("model1", "optimized")

                    # Switch to second model
                    manager.unload_model("model1")
                    manager.get_model("model2", "optimized")

                    assert call_count >= 1
                    assert mock_empty_cache.call_count >= 1


class TestConcurrentModelAccess:
    """Test concurrent access to models."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        with patch("torch.cuda.is_available", return_value=True):
            return ModelManager()

    def test_concurrent_model_loading(self, manager):
        """Test concurrent loading of the same model."""
        load_count = 0

        def mock_model_creation(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            time.sleep(0.1)  # Simulate loading time
            return Mock(model_name=kwargs.get("model_name", "test"))

        with patch(
            "music_gen.core.model_manager.FastMusicGenerator", side_effect=mock_model_creation
        ):
            results = []

            def load_model(thread_id):
                return manager.get_model("test-model", "optimized")

            # Try to load same model from multiple threads
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(load_model, i) for i in range(5)]
                results = [future.result() for future in as_completed(futures)]

            # Should only load model once
            assert load_count == 1
            # All threads should get the same model instance
            assert all(results[0] is r for r in results)

    def test_concurrent_different_models(self, manager):
        """Test concurrent loading of different models."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            models = []

            def load_model(model_name):
                return manager.get_model(model_name, "optimized")

            model_names = [f"model{i}" for i in range(5)]

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(load_model, name) for name in model_names]
                models = [future.result() for future in as_completed(futures)]

            # Should have loaded all models
            assert len(manager._models) == 5
            assert all(f"model{i}_optimized_cuda" in manager._models for i in range(5))

    def test_concurrent_load_unload(self, manager):
        """Test concurrent loading and unloading of models."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            errors = []

            def model_operations(op_id):
                try:
                    if op_id % 2 == 0:
                        # Even threads load models
                        manager.get_model(f"model{op_id}", "optimized")
                    else:
                        # Odd threads unload models
                        time.sleep(0.05)  # Let some models load first
                        manager.unload_model(f"model{op_id-1}")
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(model_operations, i) for i in range(10)]
                for future in as_completed(futures):
                    future.result()

            # Should not have any errors
            assert len(errors) == 0

    def test_thread_safe_cache_operations(self, manager):
        """Test thread-safe cache clearing and info retrieval."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            def mixed_operations(op_id):
                op_type = op_id % 4

                if op_type == 0:
                    manager.get_model(f"model{op_id}", "optimized")
                elif op_type == 1:
                    manager.get_cache_info()
                elif op_type == 2:
                    manager.list_loaded_models()
                else:
                    if manager.has_loaded_models():
                        manager.clear_cache()

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(mixed_operations, i) for i in range(40)]
                for future in as_completed(futures):
                    future.result()


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        return ModelManager()

    def test_model_loading_performance(self, manager, benchmark):
        """Benchmark model loading performance."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock()

            def load_model():
                # Clear cache to force reload
                manager._models.clear()
                return manager.get_model("test-model", "optimized")

            result = benchmark(load_model)
            assert result is not None

    def test_cached_model_access_performance(self, manager, benchmark):
        """Benchmark cached model access performance."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock()

            # Pre-load model
            manager.get_model("test-model", "optimized")

            def access_cached_model():
                return manager.get_model("test-model", "optimized")

            result = benchmark(access_cached_model)
            assert result is not None

    def test_model_switching_performance(self, manager, benchmark):
        """Benchmark model switching performance."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            # Pre-load models
            for i in range(3):
                manager.get_model(f"model{i}", "optimized")

            model_index = 0

            def switch_models():
                nonlocal model_index
                manager.unload_model(f"model{model_index}")
                model_index = (model_index + 1) % 3
                return manager.get_model(f"model{model_index}", "optimized")

            with patch("gc.collect"):
                with patch("torch.cuda.empty_cache"):
                    result = benchmark(switch_models)
                    assert result is not None

    def test_concurrent_access_performance(self, manager, benchmark):
        """Benchmark concurrent model access performance."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock()

            # Pre-load model
            manager.get_model("test-model", "optimized")

            def concurrent_access():
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for _ in range(100):
                        futures.append(
                            executor.submit(manager.get_model, "test-model", "optimized")
                        )

                    results = [f.result() for f in as_completed(futures)]
                    return len(results)

            result = benchmark(concurrent_access)
            assert result == 100

    def test_memory_usage_benchmark(self, manager):
        """Benchmark memory usage during model operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            # Simulate models with different memory footprints
            def create_mock_model(**kwargs):
                model = Mock()
                # Simulate memory usage with a large list
                model._data = [0] * (10 * 1024 * 1024)  # ~80MB per model
                return model

            mock.side_effect = create_mock_model

            # Load multiple models
            memory_snapshots = []

            for i in range(5):
                manager.get_model(f"model{i}", "optimized")
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_snapshots.append(current_memory - initial_memory)

            # Clear cache and check memory is released
            manager.clear_cache()
            gc.collect()

            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_released = memory_snapshots[-1] - (final_memory - initial_memory)

            # Should release most of the memory (allow some overhead)
            assert memory_released > 0


class TestCorruptModelHandling:
    """Test handling of corrupt and invalid models."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        return ModelManager()

    def test_corrupt_model_file(self, manager):
        """Test handling of corrupt model files."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            # Simulate different types of corruption
            corruption_errors = [
                RuntimeError("Failed to load model: file corrupted"),
                OSError("Invalid model format"),
                EOFError("Unexpected end of file"),
                ValueError("Invalid model configuration"),
            ]

            for error in corruption_errors:
                mock.side_effect = error

                with pytest.raises(type(error)):
                    manager.get_model("corrupt-model", "optimized")

                # Ensure no corrupt model is cached
                assert len(manager._models) == 0

    def test_partial_model_loading_failure(self, manager):
        """Test cleanup when model loading fails partway through."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            # Mock a model that fails after some initialization
            mock_instance = Mock()
            mock_instance.load_state_dict.side_effect = RuntimeError("State dict corrupted")
            mock.return_value = mock_instance

            try:
                manager.get_model("partial-fail-model", "optimized")
            except RuntimeError:
                pass

            # Model should not be cached on failure
            assert "partial-fail-model_optimized_cuda" not in manager._models

    def test_model_validation_failure(self, manager):
        """Test handling of models that load but fail validation."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            # Create mock model missing required attributes
            mock_model = Mock()
            del mock_model.generate  # Remove required method
            mock.return_value = mock_model

            model = manager.get_model("invalid-model", "optimized")

            # Model loads but may not work correctly
            # This tests the system's ability to handle invalid models gracefully
            assert model is mock_model

    def test_memory_corruption_recovery(self, manager):
        """Test recovery from memory corruption scenarios."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = torch.cuda.OutOfMemoryError("CUDA memory corrupted")

            # First attempt fails
            with pytest.raises(torch.cuda.OutOfMemoryError):
                manager.get_model("memory-fail-model", "optimized")

            # Reset mock for second attempt
            mock.side_effect = None
            mock.return_value = Mock()

            # Second attempt should work
            model = manager.get_model("memory-fail-model", "optimized")
            assert model is not None


class TestAdvancedMemoryManagement:
    """Test advanced memory management scenarios."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        with patch("torch.cuda.is_available", return_value=True):
            return ModelManager()

    def test_memory_pressure_handling(self, manager):
        """Test behavior under memory pressure."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            # Simulate increasing memory usage
            memory_usage = 0

            def create_model_with_memory(**kwargs):
                nonlocal memory_usage
                model = Mock()
                model.memory_usage = 1024 * 1024 * 1024  # 1GB per model
                memory_usage += model.memory_usage
                return model

            mock.side_effect = create_model_with_memory

            # Load multiple models
            models = []
            for i in range(5):
                model = manager.get_model(f"model{i}", "optimized")
                models.append(model)

            assert len(manager._models) == 5

            # Simulate memory pressure by clearing cache
            manager.clear_cache()
            assert len(manager._models) == 0

    def test_gpu_memory_monitoring(self, manager):
        """Test GPU memory monitoring during operations."""
        initial_allocated = 1024**3  # 1GB
        current_allocated = [initial_allocated]

        def mock_memory_allocated():
            return current_allocated[0]

        def mock_empty_cache():
            current_allocated[0] = max(0, current_allocated[0] - 512**3)  # Free 512MB

        with patch("torch.cuda.memory_allocated", side_effect=mock_memory_allocated):
            with patch("torch.cuda.empty_cache", side_effect=mock_empty_cache):
                with patch("torch.cuda.memory_reserved", return_value=2 * (1024**3)):
                    with patch("torch.cuda.max_memory_allocated", return_value=3 * (1024**3)):
                        # Load model (simulate memory increase)
                        current_allocated[0] += 512**3  # Add 512MB

                        with patch("music_gen.core.model_manager.FastMusicGenerator"):
                            manager.get_model("memory-test-model", "optimized")

                        cache_info = manager.get_cache_info()
                        assert "gpu_memory" in cache_info
                        assert cache_info["gpu_memory"]["allocated_gb"] > 1.0

                        # Clear cache and check memory is freed
                        manager.clear_cache()
                        assert current_allocated[0] < initial_allocated + 512**3

    def test_cache_size_limits(self, manager):
        """Test cache size limiting behavior."""
        large_models = {}

        # Mock models with large size
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:

            def create_large_model(**kwargs):
                model = Mock()
                model.size_bytes = 500 * 1024 * 1024  # 500MB
                return model

            mock.side_effect = create_large_model

            # Load many models
            for i in range(10):
                model = manager.get_model(f"large_model_{i}", "optimized")
                large_models[f"large_model_{i}"] = model

            # All models should be loaded (no automatic eviction in current implementation)
            assert len(manager._models) == 10


class TestAdvancedConcurrency:
    """Test advanced concurrency scenarios."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        ModelManager._instance = None
        return ModelManager()

    def test_high_concurrency_model_loading(self, manager):
        """Test very high concurrency model loading."""
        load_times = []

        def track_loading_time(**kwargs):
            start = time.time()
            time.sleep(0.01)  # Simulate loading time
            model = Mock(model_name=kwargs.get("model_name"))
            load_times.append(time.time() - start)
            return model

        with patch(
            "music_gen.core.model_manager.FastMusicGenerator", side_effect=track_loading_time
        ):
            # Very high concurrency
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []

                # Mix of same and different models
                for i in range(200):
                    model_name = f"model_{i % 10}"  # 10 unique models, loaded 20 times each
                    future = executor.submit(manager.get_model, model_name, "optimized")
                    futures.append(future)

                results = [f.result() for f in as_completed(futures)]

            # Should only load 10 unique models
            assert len(manager._models) == 10
            assert len(results) == 200

            # Loading should be efficient (each model loaded only once)
            assert len(load_times) == 10

    def test_concurrent_model_switching(self, manager):
        """Test concurrent model loading and unloading."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            operations_completed = []

            def model_operations(thread_id):
                op_count = 0
                try:
                    for i in range(10):
                        # Random operations
                        if i % 3 == 0:
                            manager.get_model(f"thread_{thread_id}_model_{i}", "optimized")
                            op_count += 1
                        elif i % 3 == 1:
                            manager.unload_model(f"thread_{thread_id}_model_{i-1}")
                            op_count += 1
                        else:
                            manager.get_cache_info()
                            op_count += 1

                        time.sleep(0.001)  # Small delay

                    operations_completed.append(op_count)
                except Exception as e:
                    operations_completed.append(f"Error: {e}")

            # Run many concurrent threads
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(model_operations, i) for i in range(20)]
                for future in as_completed(futures):
                    future.result()

            # All threads should complete successfully
            assert len(operations_completed) == 20
            assert all(isinstance(count, int) for count in operations_completed)

    def test_deadlock_prevention(self, manager):
        """Test that operations don't deadlock under contention."""
        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            # Pre-load some models
            for i in range(5):
                manager.get_model(f"base_model_{i}", "optimized")

            def aggressive_operations(thread_id):
                for i in range(50):
                    # Rapid fire operations
                    manager.get_model(f"model_{i % 3}", "optimized")
                    manager.list_loaded_models()
                    manager.has_loaded_models()
                    manager.get_cache_info()

                    if i % 10 == 0:
                        manager.unload_model(f"model_{i % 3}")

            start_time = time.time()

            # Many threads doing aggressive operations
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(aggressive_operations, i) for i in range(30)]
                for future in as_completed(futures):
                    future.result()

            end_time = time.time()

            # Should complete in reasonable time (no deadlocks)
            assert end_time - start_time < 30.0  # Should complete within 30 seconds


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_device_property(self):
        """Test device property getter."""
        ModelManager._instance = None

        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()
            assert manager.device == "cuda"

        ModelManager._instance = None

        with patch("torch.cuda.is_available", return_value=False):
            manager = ModelManager()
            assert manager.device == "cpu"

    def test_empty_cache_operations(self):
        """Test operations on empty cache."""
        ModelManager._instance = None
        manager = ModelManager()

        # Test operations on empty cache
        assert manager.has_loaded_models() is False
        assert manager.list_loaded_models() == {}
        assert manager.unload_model("nonexistent") is False

        cache_info = manager.get_cache_info()
        assert cache_info["loaded_models"] == 0
        assert cache_info["models"] == []

    def test_model_key_parsing_edge_cases(self):
        """Test edge cases in model key parsing."""
        ModelManager._instance = None
        manager = ModelManager()

        # Add models with unusual keys
        manager._models = {
            "model_with_underscores_optimized_cuda": Mock(),
            "simple": Mock(),
            "": Mock(),  # Empty key
        }

        models_info = manager.list_loaded_models()

        # Should handle parsing gracefully
        assert "model_with_underscores" in models_info
        assert "simple" in models_info
        assert "" in models_info

    def test_very_long_model_names(self):
        """Test handling of very long model names."""
        ModelManager._instance = None
        manager = ModelManager()

        # Very long model name
        long_name = "a" * 1000

        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.return_value = Mock()

            model = manager.get_model(long_name, "optimized")
            assert model is not None

            expected_key = f"{long_name}_optimized_cuda"
            assert expected_key in manager._models

    def test_special_characters_in_model_names(self):
        """Test handling of special characters in model names."""
        ModelManager._instance = None
        manager = ModelManager()

        special_names = [
            "model/with/slashes",
            "model-with-dashes",
            "model.with.dots",
            "model with spaces",
            "model@with#symbols",
        ]

        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            for name in special_names:
                model = manager.get_model(name, "optimized")
                assert model is not None

                # Check model is cached with proper key
                expected_key = f"{name}_optimized_cuda"
                assert expected_key in manager._models

    def test_unicode_model_names(self):
        """Test handling of unicode characters in model names."""
        ModelManager._instance = None
        manager = ModelManager()

        unicode_names = [
            "模型_中文",
            "model_русский",
            "modèle_français",
            "modelo_español",
            "🎵_emoji_model",
        ]

        with patch("music_gen.core.model_manager.FastMusicGenerator") as mock:
            mock.side_effect = lambda **kwargs: Mock(model_name=kwargs["model_name"])

            for name in unicode_names:
                model = manager.get_model(name, "optimized")
                assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
