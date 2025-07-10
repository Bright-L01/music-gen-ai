"""
Unit tests for ModelManager core component.

Tests model lifecycle management, caching, loading, and validation.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from typing import Dict, Any

from music_gen.core.model_manager import (
    ModelManager,
    ModelCache,
    ModelLoader,
    ModelValidator,
    ModelLoadError,
    ModelValidationError,
)
from music_gen.core.config import AppConfig


@pytest.mark.unit
class TestModelManager:
    """Test ModelManager functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AppConfig(model_cache_size=3, model_device="cpu", model_cache_dir="/tmp/test_models")

    @pytest.fixture
    def model_manager(self, config):
        """Create ModelManager instance."""
        return ModelManager(config)

    def test_model_manager_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager is not None
        assert hasattr(model_manager, "_cache")
        assert hasattr(model_manager, "_loader")
        assert hasattr(model_manager, "_validator")

    @pytest.mark.asyncio
    async def test_load_model_success(self, model_manager):
        """Test successful model loading."""
        model_id = "facebook/musicgen-small"

        with patch.object(model_manager._loader, "load_model") as mock_load:
            mock_model = Mock()
            mock_model.config.vocab_size = 2048
            mock_model.config.hidden_size = 1024
            mock_load.return_value = mock_model

            model = await model_manager.load_model(model_id)

            assert model is mock_model
            mock_load.assert_called_once_with(model_id)

            # Model should be cached
            assert model_id in model_manager._cache._models

    @pytest.mark.asyncio
    async def test_load_model_from_cache(self, model_manager):
        """Test loading model from cache."""
        model_id = "cached_model"
        cached_model = Mock()

        # Pre-populate cache
        model_manager._cache._models[model_id] = cached_model

        with patch.object(model_manager._loader, "load_model") as mock_load:
            model = await model_manager.load_model(model_id)

            # Should return cached model without calling loader
            assert model is cached_model
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_failure(self, model_manager):
        """Test model loading failure."""
        model_id = "nonexistent_model"

        with patch.object(model_manager._loader, "load_model") as mock_load:
            mock_load.side_effect = ModelLoadError("Model not found")

            with pytest.raises(ModelLoadError):
                await model_manager.load_model(model_id)

    @pytest.mark.asyncio
    async def test_model_validation(self, model_manager):
        """Test model validation during loading."""
        model_id = "test_model"

        with patch.object(model_manager._loader, "load_model") as mock_load, patch.object(
            model_manager._validator, "validate_model"
        ) as mock_validate:
            mock_model = Mock()
            mock_load.return_value = mock_model
            mock_validate.return_value = True

            model = await model_manager.load_model(model_id)

            assert model is mock_model
            mock_validate.assert_called_once_with(mock_model)

    @pytest.mark.asyncio
    async def test_model_validation_failure(self, model_manager):
        """Test model validation failure."""
        model_id = "invalid_model"

        with patch.object(model_manager._loader, "load_model") as mock_load, patch.object(
            model_manager._validator, "validate_model"
        ) as mock_validate:
            mock_model = Mock()
            mock_load.return_value = mock_model
            mock_validate.side_effect = ModelValidationError("Invalid model structure")

            with pytest.raises(ModelValidationError):
                await model_manager.load_model(model_id)

    def test_list_available_models(self, model_manager):
        """Test listing available models."""
        with patch.object(model_manager._loader, "list_available_models") as mock_list:
            mock_list.return_value = [
                {"id": "facebook/musicgen-small", "type": "pretrained"},
                {"id": "facebook/musicgen-medium", "type": "pretrained"},
                {"id": "custom_model_1", "type": "custom"},
            ]

            models = model_manager.list_available_models()

            assert len(models) == 3
            assert all("id" in model for model in models)
            assert all("type" in model for model in models)

    def test_get_model_info(self, model_manager):
        """Test getting model information."""
        model_id = "facebook/musicgen-small"

        with patch.object(model_manager._loader, "get_model_info") as mock_info:
            mock_info.return_value = {
                "id": model_id,
                "name": "MusicGen Small",
                "size_mb": 300,
                "parameters": "300M",
                "capabilities": ["text_to_music"],
            }

            info = model_manager.get_model_info(model_id)

            assert info["id"] == model_id
            assert info["name"] == "MusicGen Small"
            assert info["size_mb"] == 300

    @pytest.mark.asyncio
    async def test_unload_model(self, model_manager):
        """Test model unloading."""
        model_id = "test_model"
        mock_model = Mock()

        # Load model first
        model_manager._cache._models[model_id] = mock_model

        # Unload model
        success = await model_manager.unload_model(model_id)

        assert success is True
        assert model_id not in model_manager._cache._models

    def test_cache_size_management(self, model_manager):
        """Test cache size management."""
        # Fill cache beyond capacity
        for i in range(5):  # More than cache size (3)
            model_manager._cache._models[f"model_{i}"] = Mock()

        # Cache should enforce size limit
        model_manager._cache._enforce_size_limit()

        assert len(model_manager._cache._models) <= model_manager._config.model_cache_size


@pytest.mark.unit
class TestModelCache:
    """Test ModelCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create ModelCache instance."""
        return ModelCache(max_size=3)

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache._max_size == 3
        assert len(cache._models) == 0
        assert len(cache._access_times) == 0

    def test_cache_add_model(self, cache):
        """Test adding model to cache."""
        model_id = "test_model"
        model = Mock()

        cache.add_model(model_id, model)

        assert model_id in cache._models
        assert cache._models[model_id] is model
        assert model_id in cache._access_times

    def test_cache_get_model(self, cache):
        """Test getting model from cache."""
        model_id = "test_model"
        model = Mock()

        cache.add_model(model_id, model)
        retrieved_model = cache.get_model(model_id)

        assert retrieved_model is model

    def test_cache_get_nonexistent_model(self, cache):
        """Test getting nonexistent model from cache."""
        result = cache.get_model("nonexistent")
        assert result is None

    def test_cache_size_limit_enforcement(self, cache):
        """Test cache size limit enforcement."""
        # Add more models than cache capacity
        for i in range(5):
            cache.add_model(f"model_{i}", Mock())
            cache._enforce_size_limit()

        # Should not exceed max size
        assert len(cache._models) <= cache._max_size

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction policy."""
        import time

        # Add models
        for i in range(3):
            cache.add_model(f"model_{i}", Mock())
            time.sleep(0.01)  # Ensure different timestamps

        # Access model_0 to make it most recent
        cache.get_model("model_0")

        # Add one more model (should evict model_1, the oldest unaccessed)
        cache.add_model("model_3", Mock())
        cache._enforce_size_limit()

        assert "model_0" in cache._models  # Recently accessed
        assert "model_3" in cache._models  # Just added
        assert len(cache._models) <= cache._max_size

    def test_cache_clear(self, cache):
        """Test clearing the cache."""
        # Add some models
        for i in range(3):
            cache.add_model(f"model_{i}", Mock())

        cache.clear()

        assert len(cache._models) == 0
        assert len(cache._access_times) == 0

    def test_cache_remove_model(self, cache):
        """Test removing specific model from cache."""
        model_id = "test_model"
        cache.add_model(model_id, Mock())

        success = cache.remove_model(model_id)

        assert success is True
        assert model_id not in cache._models
        assert model_id not in cache._access_times

    def test_cache_memory_usage_estimation(self, cache):
        """Test memory usage estimation."""
        # Add mock models with size estimation
        for i in range(3):
            model = Mock()
            model.numel = Mock(return_value=1000000)  # 1M parameters
            cache.add_model(f"model_{i}", model)

        memory_usage = cache.estimate_memory_usage_mb()

        # Should estimate some memory usage
        assert memory_usage > 0


@pytest.mark.unit
class TestModelLoader:
    """Test ModelLoader functionality."""

    @pytest.fixture
    def loader(self):
        """Create ModelLoader instance."""
        return ModelLoader(device="cpu")

    def test_loader_initialization(self, loader):
        """Test loader initialization."""
        assert loader._device == "cpu"
        assert hasattr(loader, "_model_registry")

    @pytest.mark.asyncio
    async def test_load_pretrained_model(self, loader):
        """Test loading pretrained model."""
        model_id = "facebook/musicgen-small"

        with patch("music_gen.models.musicgen.MusicGenModel") as MockModel:
            mock_model = Mock()
            MockModel.from_pretrained.return_value = mock_model

            model = await loader.load_model(model_id)

            assert model is mock_model
            MockModel.from_pretrained.assert_called_once_with(model_id)

    @pytest.mark.asyncio
    async def test_load_custom_model(self, loader):
        """Test loading custom model from file."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            model_path = tmp_file.name

            # Mock model state dict
            mock_state_dict = {
                "layer.weight": torch.randn(100, 100),
                "layer.bias": torch.randn(100),
            }

            with patch("torch.load", return_value=mock_state_dict), patch(
                "music_gen.models.musicgen.MusicGenModel"
            ) as MockModel:
                mock_model = Mock()
                MockModel.return_value = mock_model

                model = await loader.load_model(model_path)

                assert model is mock_model
                mock_model.load_state_dict.assert_called_once_with(mock_state_dict)

    @pytest.mark.asyncio
    async def test_load_model_failure(self, loader):
        """Test model loading failure."""
        model_id = "nonexistent/model"

        with patch("music_gen.models.musicgen.MusicGenModel") as MockModel:
            MockModel.from_pretrained.side_effect = Exception("Model not found")

            with pytest.raises(ModelLoadError):
                await loader.load_model(model_id)

    def test_list_available_models(self, loader):
        """Test listing available models."""
        with patch.object(loader, "_get_pretrained_models") as mock_pretrained, patch.object(
            loader, "_get_custom_models"
        ) as mock_custom:
            mock_pretrained.return_value = [
                {"id": "facebook/musicgen-small", "type": "pretrained"},
                {"id": "facebook/musicgen-medium", "type": "pretrained"},
            ]
            mock_custom.return_value = [
                {"id": "custom_model_1", "type": "custom"},
            ]

            models = loader.list_available_models()

            assert len(models) == 3
            pretrained_count = sum(1 for m in models if m["type"] == "pretrained")
            custom_count = sum(1 for m in models if m["type"] == "custom")
            assert pretrained_count == 2
            assert custom_count == 1

    def test_get_model_info(self, loader):
        """Test getting model information."""
        model_id = "facebook/musicgen-small"

        with patch.object(loader, "_fetch_model_metadata") as mock_metadata:
            mock_metadata.return_value = {
                "name": "MusicGen Small",
                "size_mb": 300,
                "parameters": "300M",
                "description": "Small model for quick generation",
            }

            info = loader.get_model_info(model_id)

            assert info["name"] == "MusicGen Small"
            assert info["size_mb"] == 300


@pytest.mark.unit
class TestModelValidator:
    """Test ModelValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create ModelValidator instance."""
        return ModelValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert hasattr(validator, "_validation_rules")

    def test_validate_model_structure(self, validator):
        """Test model structure validation."""
        # Create mock model with expected attributes
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 2048
        mock_model.config.hidden_size = 1024
        mock_model.generate = Mock()

        # Should pass validation
        result = validator.validate_model(mock_model)
        assert result is True

    def test_validate_model_missing_attributes(self, validator):
        """Test validation with missing model attributes."""
        # Create mock model missing required attributes
        mock_model = Mock()
        mock_model.config = Mock()
        # Missing vocab_size
        mock_model.config.hidden_size = 1024

        with pytest.raises(ModelValidationError):
            validator.validate_model(mock_model)

    def test_validate_model_config(self, validator):
        """Test model configuration validation."""
        valid_config = {"vocab_size": 2048, "hidden_size": 1024, "num_layers": 12, "num_heads": 16}

        result = validator.validate_config(valid_config)
        assert result is True

    def test_validate_invalid_config(self, validator):
        """Test validation with invalid configuration."""
        invalid_configs = [
            {"vocab_size": 0},  # Invalid vocab size
            {"vocab_size": 2048, "hidden_size": -1},  # Negative hidden size
            {
                "vocab_size": 2048,
                "hidden_size": 1024,
                "num_heads": 7,
            },  # Hidden size not divisible by heads
        ]

        for config in invalid_configs:
            with pytest.raises(ModelValidationError):
                validator.validate_config(config)

    def test_validate_model_parameters(self, validator):
        """Test model parameter validation."""
        mock_model = Mock()

        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000000  # 1M parameters
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = [mock_param]

        result = validator._validate_parameters(mock_model)
        assert result is True

    def test_validate_model_device_placement(self, validator):
        """Test model device placement validation."""
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]

        # Should validate device consistency
        result = validator._validate_device_placement(mock_model)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
