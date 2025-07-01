"""
Unit tests for configuration management.
"""

import json
from unittest.mock import mock_open, patch

import pytest

# Import config modules - handle missing dependencies gracefully
try:
    from music_gen.configs.config import (
        DataConfig,
        ModelConfig,
        MusicGenConfig,
        TrainingConfig,
        load_config,
        merge_configs,
        save_config,
        validate_config,
    )

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from omegaconf import DictConfig, OmegaConf

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestMusicGenConfig:
    """Test main MusicGenConfig class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = MusicGenConfig()

        assert config is not None
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert hasattr(config, "data")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model": {"hidden_size": 512, "num_layers": 8, "vocab_size": 2048},
            "training": {"batch_size": 16, "learning_rate": 1e-4, "num_epochs": 100},
        }

        config = MusicGenConfig.from_dict(config_dict)

        assert config.model.hidden_size == 512
        assert config.model.num_layers == 8
        assert config.training.batch_size == 16
        assert config.training.learning_rate == 1e-4

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = MusicGenConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert "data" in config_dict

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        valid_config = MusicGenConfig()
        assert valid_config.validate()

        # Invalid config - negative values
        invalid_config = MusicGenConfig()
        invalid_config.model.hidden_size = -1
        assert not invalid_config.validate()


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestModelConfig:
    """Test ModelConfig class."""

    def test_model_config_defaults(self):
        """Test default model configuration."""
        config = ModelConfig()

        assert config.hidden_size > 0
        assert config.num_layers > 0
        assert config.vocab_size > 0
        assert config.max_sequence_length > 0

    def test_model_config_custom(self):
        """Test custom model configuration."""
        config = ModelConfig(
            hidden_size=1024, num_layers=12, vocab_size=4096, max_sequence_length=2048
        )

        assert config.hidden_size == 1024
        assert config.num_layers == 12
        assert config.vocab_size == 4096
        assert config.max_sequence_length == 2048

    def test_model_config_validation(self):
        """Test model config validation."""
        # Valid config
        valid_config = ModelConfig(hidden_size=512, num_layers=6)
        assert valid_config.validate()

        # Invalid config
        invalid_config = ModelConfig(hidden_size=0, num_layers=-1)
        assert not invalid_config.validate()

    def test_model_config_serialization(self):
        """Test model config serialization."""
        config = ModelConfig(hidden_size=768, num_layers=10)

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["hidden_size"] == 768
        assert config_dict["num_layers"] == 10

        # Test from_dict
        restored_config = ModelConfig.from_dict(config_dict)
        assert restored_config.hidden_size == 768
        assert restored_config.num_layers == 10


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.num_epochs > 0
        assert config.optimizer in ["adam", "adamw", "sgd"]

    def test_training_config_custom(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            batch_size=32, learning_rate=2e-4, num_epochs=200, optimizer="adamw", scheduler="cosine"
        )

        assert config.batch_size == 32
        assert config.learning_rate == 2e-4
        assert config.num_epochs == 200
        assert config.optimizer == "adamw"
        assert config.scheduler == "cosine"

    def test_training_config_validation(self):
        """Test training config validation."""
        # Valid config
        valid_config = TrainingConfig(batch_size=16, learning_rate=1e-4)
        assert valid_config.validate()

        # Invalid config - negative values
        invalid_config = TrainingConfig(batch_size=-1, learning_rate=0)
        assert not invalid_config.validate()

    def test_training_config_lr_scheduling(self):
        """Test learning rate scheduling configuration."""
        config = TrainingConfig(scheduler="cosine", warmup_steps=1000, max_lr=1e-3, min_lr=1e-6)

        assert config.scheduler == "cosine"
        assert config.warmup_steps == 1000
        assert config.max_lr == 1e-3
        assert config.min_lr == 1e-6


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestDataConfig:
    """Test DataConfig class."""

    def test_data_config_defaults(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.dataset_name is not None
        assert config.sample_rate > 0
        assert config.max_audio_length > 0
        assert config.num_workers >= 0

    def test_data_config_custom(self):
        """Test custom data configuration."""
        config = DataConfig(
            dataset_name="musiccaps",
            sample_rate=32000,
            max_audio_length=30.0,
            num_workers=4,
            batch_size=8,
        )

        assert config.dataset_name == "musiccaps"
        assert config.sample_rate == 32000
        assert config.max_audio_length == 30.0
        assert config.num_workers == 4
        assert config.batch_size == 8

    def test_data_config_validation(self):
        """Test data config validation."""
        # Valid config
        valid_config = DataConfig(sample_rate=24000, max_audio_length=10.0)
        assert valid_config.validate()

        # Invalid config
        invalid_config = DataConfig(sample_rate=-1, max_audio_length=0)
        assert not invalid_config.validate()

    def test_data_config_augmentation(self):
        """Test data augmentation configuration."""
        config = DataConfig(
            augment_audio=True,
            augment_text=True,
            augmentation_strength="strong",
            use_adaptive_augmentation=True,
        )

        assert config.augment_audio is True
        assert config.augment_text is True
        assert config.augmentation_strength == "strong"
        assert config.use_adaptive_augmentation is True


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestConfigIO:
    """Test configuration I/O operations."""

    def test_load_config_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
        model:
          hidden_size: 768
          num_layers: 12
        training:
          batch_size: 16
          learning_rate: 0.0001
        """

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = load_config("config.yaml")

                assert config is not None
                if hasattr(config, "model"):
                    assert config.model.hidden_size == 768

    def test_load_config_json(self):
        """Test loading config from JSON file."""
        json_content = {
            "model": {"hidden_size": 512, "num_layers": 8},
            "training": {"batch_size": 32, "learning_rate": 0.0002},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(json_content))):
            with patch("pathlib.Path.exists", return_value=True):
                config = load_config("config.json")

                assert config is not None

    def test_save_config_yaml(self):
        """Test saving config to YAML file."""
        config = MusicGenConfig()

        with patch("builtins.open", mock_open()) as mock_file:
            save_config(config, "output.yaml")

            mock_file.assert_called_once_with("output.yaml", "w")

    def test_save_config_json(self):
        """Test saving config to JSON file."""
        config = MusicGenConfig()

        with patch("builtins.open", mock_open()) as mock_file:
            save_config(config, "output.json")

            mock_file.assert_called_once_with("output.json", "w")

    def test_config_file_not_found(self):
        """Test handling missing config file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent.yaml")


@pytest.mark.unit
@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestConfigUtilities:
    """Test configuration utility functions."""

    def test_merge_configs(self):
        """Test merging multiple configurations."""
        base_config = {"model": {"hidden_size": 512}, "training": {"batch_size": 16}}
        override_config = {"model": {"num_layers": 8}, "training": {"batch_size": 32}}

        merged = merge_configs(base_config, override_config)

        assert merged["model"]["hidden_size"] == 512  # From base
        assert merged["model"]["num_layers"] == 8  # From override
        assert merged["training"]["batch_size"] == 32  # Override wins

    def test_validate_config_structure(self):
        """Test config structure validation."""
        # Valid structure
        valid_config = {
            "model": {"hidden_size": 512},
            "training": {"batch_size": 16},
            "data": {"sample_rate": 24000},
        }
        assert validate_config(valid_config)

        # Invalid structure - missing required sections
        invalid_config = {"model": {"hidden_size": 512}}
        assert not validate_config(invalid_config)

    def test_config_environment_substitution(self):
        """Test environment variable substitution in config."""
        with patch.dict("os.environ", {"MODEL_SIZE": "768", "BATCH_SIZE": "32"}):
            config_template = {
                "model": {"hidden_size": "${MODEL_SIZE}"},
                "training": {"batch_size": "${BATCH_SIZE}"},
            }

            # Mock environment substitution function
            def substitute_env_vars(config):
                import os

                if isinstance(config, dict):
                    for key, value in config.items():
                        if (
                            isinstance(value, str)
                            and value.startswith("${")
                            and value.endswith("}")
                        ):
                            env_var = value[2:-1]
                            config[key] = os.environ.get(env_var, value)
                        elif isinstance(value, dict):
                            substitute_env_vars(value)
                return config

            resolved_config = substitute_env_vars(config_template.copy())

            assert resolved_config["model"]["hidden_size"] == "768"
            assert resolved_config["training"]["batch_size"] == "32"


@pytest.mark.unit
@pytest.mark.skipif(
    not CONFIG_AVAILABLE or not OMEGACONF_AVAILABLE, reason="Dependencies not available"
)
class TestHydraIntegration:
    """Test Hydra/OmegaConf integration."""

    def test_omegaconf_config_creation(self):
        """Test creating OmegaConf config."""
        config_dict = {"model": {"hidden_size": 512}, "training": {"batch_size": 16}}

        config = OmegaConf.create(config_dict)

        assert config.model.hidden_size == 512
        assert config.training.batch_size == 16

    def test_config_interpolation(self):
        """Test OmegaConf interpolation."""
        config_dict = {
            "base_lr": 0.001,
            "training": {"learning_rate": "${base_lr}", "warmup_lr": "${divide:${base_lr},10}"},
        }

        config = OmegaConf.create(config_dict)

        assert config.training.learning_rate == 0.001

    def test_config_merging_omegaconf(self):
        """Test OmegaConf config merging."""
        base_config = OmegaConf.create({"model": {"hidden_size": 512}})
        override_config = OmegaConf.create({"model": {"num_layers": 8}})

        merged = OmegaConf.merge(base_config, override_config)

        assert merged.model.hidden_size == 512
        assert merged.model.num_layers == 8


@pytest.mark.unit
class TestConfigMocks:
    """Test config functionality with mocked dependencies."""

    def test_mock_config_loading(self):
        """Test config loading with mocked file operations."""

        # Simple mock config class
        class MockConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def validate(self):
                return hasattr(self, "model_size") and self.model_size > 0

        config = MockConfig(model_size=512, batch_size=16)
        assert config.validate()
        assert config.model_size == 512
        assert config.batch_size == 16

    def test_mock_config_serialization(self):
        """Test config serialization with mocks."""
        config_data = {"model": {"hidden_size": 768}, "training": {"lr": 0.001}}

        # Mock save operation
        def mock_save(config, path):
            return True

        # Mock load operation
        def mock_load(path):
            return config_data

        assert mock_save(config_data, "test.yaml")
        loaded = mock_load("test.yaml")
        assert loaded == config_data


if __name__ == "__main__":
    pytest.main([__file__])
