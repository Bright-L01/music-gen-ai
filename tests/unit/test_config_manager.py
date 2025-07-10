"""
Unit tests for configuration manager.

Tests configuration loading, validation, and management functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from music_gen.core.config_manager import (
    ConfigManager,
    ConfigValidationResult,
    ConfigurationError,
    load_config,
    get_config,
    validate_config_file,
)
from music_gen.core.config_models import MusicGenConfig


@pytest.mark.unit
class TestConfigManager:
    """Test ConfigManager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()

            # Create app subdirectory
            app_dir = config_dir / "app"
            app_dir.mkdir()

            # Create basic config files
            base_config = {
                "app": {"name": "Test MusicGen", "version": "1.0.0"},
                "api": {"port": 8000},
                "model": {"default_model": "facebook/musicgen-small"},
            }

            dev_config = {
                "defaults": ["base"],
                "app": {"environment": "development", "debug": True},
                "auth": {"enabled": False},
            }

            prod_config = {
                "defaults": ["base"],
                "app": {"environment": "production", "debug": False},
                "auth": {"enabled": True, "jwt_secret": "${JWT_SECRET:prod_secret_32_chars_long}"},
                "limits": {"max_requests_per_hour": 1000},
            }

            # Write config files
            import yaml

            with open(config_dir / "base.yaml", "w") as f:
                yaml.dump(base_config, f)

            with open(app_dir / "development.yaml", "w") as f:
                yaml.dump(dev_config, f)

            with open(app_dir / "production.yaml", "w") as f:
                yaml.dump(prod_config, f)

            yield str(config_dir)

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(config_dir=temp_config_dir)

    def test_config_manager_initialization(self, config_manager):
        """Test ConfigManager initialization."""
        assert config_manager is not None
        assert config_manager.config_dir is not None
        assert not config_manager._is_initialized

    def test_config_manager_default_directory(self):
        """Test default configuration directory detection."""
        manager = ConfigManager()
        assert manager.config_dir is not None
        assert "configs" in manager.config_dir

    @patch("music_gen.core.config_manager.initialize_config_dir")
    @patch("music_gen.core.config_manager.compose")
    def test_initialize_success(self, mock_compose, mock_initialize, config_manager):
        """Test successful configuration initialization."""
        mock_config = Mock()
        mock_compose.return_value = mock_config

        config_manager.initialize("base")

        assert config_manager._is_initialized
        assert config_manager._hydra_config is mock_config
        mock_initialize.assert_called_once()
        mock_compose.assert_called_once_with(config_name="base", overrides=[])

    @patch("music_gen.core.config_manager.initialize_config_dir")
    def test_initialize_failure(self, mock_initialize, config_manager):
        """Test configuration initialization failure."""
        mock_initialize.side_effect = Exception("Init failed")

        with pytest.raises(Exception, match="Init failed"):
            config_manager.initialize("base")

        assert not config_manager._is_initialized

    @patch("music_gen.core.config_manager.GlobalHydra")
    @patch("music_gen.core.config_manager.initialize_config_dir")
    @patch("music_gen.core.config_manager.compose")
    def test_load_config_success(
        self, mock_compose, mock_initialize, mock_global_hydra, config_manager
    ):
        """Test successful configuration loading."""
        # Mock Hydra components
        mock_global_hydra.return_value.is_initialized.return_value = False

        mock_hydra_config = {
            "app": {"name": "Test", "environment": "development"},
            "api": {"port": 8000},
            "auth": {"enabled": False},
            "model": {"default_model": "facebook/musicgen-small"},
        }
        mock_compose.return_value = mock_hydra_config

        # Load configuration
        config = config_manager.load_config("base", validate=True)

        assert isinstance(config, MusicGenConfig)
        assert config.app.name == "Test"
        assert config.api.port == 8000
        assert not config.auth.enabled

    def test_environment_variable_substitution(self, config_manager):
        """Test environment variable substitution."""
        test_config = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}",
                "password": "${DB_PASSWORD}",
            },
            "features": {"enabled": "${FEATURES_ENABLED:true}"},
        }

        # Set environment variables
        os.environ["DB_HOST"] = "prod.example.com"
        os.environ["DB_PORT"] = "5433"
        os.environ["FEATURES_ENABLED"] = "false"

        try:
            result = config_manager._substitute_env_vars(test_config)

            assert result["database"]["host"] == "prod.example.com"
            assert result["database"]["port"] == 5433  # Should be converted to int
            assert result["database"]["password"] is None  # No default, not set
            assert result["features"]["enabled"] is False  # Should be converted to bool

        finally:
            # Clean up environment variables
            for var in ["DB_HOST", "DB_PORT", "FEATURES_ENABLED"]:
                os.environ.pop(var, None)

    def test_validate_config_success(self, config_manager):
        """Test successful configuration validation."""
        valid_config = {
            "app": {"name": "Test", "environment": "development"},
            "api": {"port": 8000},
            "auth": {"enabled": False},
            "model": {"default_model": "facebook/musicgen-small"},
        }

        result = config_manager.validate_config(valid_config)

        assert result.is_valid
        assert result.config is not None
        assert isinstance(result.config, MusicGenConfig)
        assert result.errors is None

    def test_validate_config_failure(self, config_manager):
        """Test configuration validation failure."""
        invalid_config = {
            "app": {"name": ""},  # Empty name should fail
            "api": {"port": 70000},  # Invalid port
            "auth": {"enabled": True, "jwt_secret": "short"},  # Weak secret
        }

        result = config_manager.validate_config(invalid_config)

        assert not result.is_valid
        assert result.config is None
        assert result.errors is not None
        assert len(result.errors) > 0

    def test_validate_config_with_warnings(self, config_manager):
        """Test configuration validation with warnings."""
        config_with_warnings = {
            "app": {"name": "Test", "environment": "development"},
            "model": {"cache_dir": "/nonexistent/path"},  # Should generate warning
            "audio": {"output_dir": "/another/nonexistent/path"},
        }

        with patch.object(MusicGenConfig, "validate_paths", return_value=["Path warning"]):
            result = config_manager.validate_config(config_with_warnings)

        assert result.is_valid
        assert result.warnings is not None
        assert len(result.warnings) > 0

    @patch("music_gen.core.config_manager.initialize_config_dir")
    @patch("music_gen.core.config_manager.compose")
    def test_load_config_with_environment(self, mock_compose, mock_initialize, config_manager):
        """Test loading configuration with environment override."""
        mock_config = {
            "app": {"environment": "production", "debug": False},
            "auth": {"enabled": True, "jwt_secret": "prod_secret_32_chars_long"},
            "limits": {"max_requests_per_hour": 1000},
        }
        mock_compose.return_value = mock_config

        config = config_manager.load_config(environment="production")

        assert config.app.environment == "production"
        assert not config.app.debug
        assert config.auth.enabled

    @patch("music_gen.core.config_manager.initialize_config_dir")
    @patch("music_gen.core.config_manager.compose")
    def test_load_config_with_overrides(self, mock_compose, mock_initialize, config_manager):
        """Test loading configuration with overrides."""
        mock_config = {
            "app": {"name": "Test"},
            "api": {"port": 9000},  # Override value
            "model": {"cache_size": 10},  # Override value
        }
        mock_compose.return_value = mock_config

        overrides = ["api.port=9000", "model.cache_size=10"]
        config = config_manager.load_config(overrides=overrides)

        assert config.api.port == 9000
        assert config.model.cache_size == 10

    def test_get_config_when_none_loaded(self, config_manager):
        """Test getting configuration when none is loaded."""
        config = config_manager.get_config()
        assert config is None

    def test_save_config_schema(self, config_manager):
        """Test saving configuration schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_path = f.name

        try:
            config_manager.save_config_schema(schema_path)

            # Verify file was created and contains valid JSON
            assert Path(schema_path).exists()

            import json

            with open(schema_path, "r") as f:
                schema = json.load(f)

            assert "title" in schema
            assert "properties" in schema

        finally:
            Path(schema_path).unlink(missing_ok=True)

    def test_generate_example_config(self, config_manager):
        """Test generating example configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            example_path = f.name

        try:
            config_manager.generate_example_config(example_path, "production")

            # Verify file was created
            assert Path(example_path).exists()

            # Load and verify content
            import yaml

            with open(example_path, "r") as f:
                example_config = yaml.safe_load(f)

            assert example_config["app"]["environment"] == "production"
            assert not example_config["app"]["debug"]
            assert example_config["auth"]["enabled"]

        finally:
            Path(example_path).unlink(missing_ok=True)

    def test_update_config(self, config_manager):
        """Test updating configuration."""
        # First load a base config
        base_config = MusicGenConfig(app={"name": "Original"}, api={"port": 8000})
        config_manager._config = base_config

        # Update configuration
        updates = {"app": {"name": "Updated"}, "api": {"workers": 4}}

        updated_config = config_manager.update_config(updates)

        assert updated_config.app.name == "Updated"
        assert updated_config.api.port == 8000  # Should preserve original
        assert updated_config.api.workers == 4  # Should add new value

    def test_update_config_validation_failure(self, config_manager):
        """Test updating configuration with validation failure."""
        base_config = MusicGenConfig()
        config_manager._config = base_config

        # Invalid update
        invalid_updates = {"api": {"port": 70000}}  # Invalid port

        with pytest.raises(ConfigurationError):
            config_manager.update_config(invalid_updates, validate=True)

    def test_export_config_yaml(self, config_manager):
        """Test exporting configuration to YAML."""
        config = MusicGenConfig(app={"name": "Test Export"})
        config_manager._config = config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            export_path = f.name

        try:
            config_manager.export_config(export_path, "yaml")

            # Verify file exists and contains expected content
            assert Path(export_path).exists()

            import yaml

            with open(export_path, "r") as f:
                exported_config = yaml.safe_load(f)

            assert exported_config["app"]["name"] == "Test Export"

        finally:
            Path(export_path).unlink(missing_ok=True)

    def test_export_config_json(self, config_manager):
        """Test exporting configuration to JSON."""
        config = MusicGenConfig(app={"name": "Test Export"})
        config_manager._config = config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            config_manager.export_config(export_path, "json")

            # Verify file exists and contains expected content
            assert Path(export_path).exists()

            import json

            with open(export_path, "r") as f:
                exported_config = json.load(f)

            assert exported_config["app"]["name"] == "Test Export"

        finally:
            Path(export_path).unlink(missing_ok=True)

    def test_export_config_invalid_format(self, config_manager):
        """Test exporting configuration with invalid format."""
        config = MusicGenConfig()
        config_manager._config = config

        with pytest.raises(ValueError, match="Unsupported format"):
            config_manager.export_config("test.xml", "xml")

    def test_context_manager(self, config_manager):
        """Test ConfigManager as context manager."""
        with patch("music_gen.core.config_manager.GlobalHydra") as mock_global_hydra:
            mock_hydra = Mock()
            mock_global_hydra.return_value = mock_hydra
            mock_hydra.is_initialized.return_value = True

            with config_manager as cm:
                assert cm is config_manager

            # Should clear Hydra on exit
            mock_hydra.clear.assert_called_once()

    def test_deep_merge(self, config_manager):
        """Test deep merging of configuration dictionaries."""
        base = {
            "app": {"name": "Base", "version": "1.0"},
            "api": {"port": 8000, "host": "localhost"},
            "features": {"a": True, "b": False},
        }

        updates = {
            "app": {"name": "Updated"},  # Should update name, keep version
            "api": {"port": 9000},  # Should update port, keep host
            "features": {"b": True, "c": True},  # Should update b, add c, keep a
            "new_section": {"enabled": True},  # Should add new section
        }

        result = config_manager._deep_merge(base, updates)

        assert result["app"]["name"] == "Updated"
        assert result["app"]["version"] == "1.0"  # Preserved
        assert result["api"]["port"] == 9000
        assert result["api"]["host"] == "localhost"  # Preserved
        assert result["features"]["a"] is True  # Preserved
        assert result["features"]["b"] is True  # Updated
        assert result["features"]["c"] is True  # Added
        assert result["new_section"]["enabled"] is True  # Added


@pytest.mark.unit
class TestConfigManagerUtilities:
    """Test configuration manager utility functions."""

    @patch("music_gen.core.config_manager.get_config_manager")
    def test_load_config_function(self, mock_get_manager):
        """Test load_config convenience function."""
        mock_manager = Mock()
        mock_config = Mock()
        mock_manager.load_config.return_value = mock_config
        mock_get_manager.return_value = mock_manager

        # Test with environment variable
        with patch.dict(os.environ, {"MUSICGEN_ENV": "staging"}):
            result = load_config("base", overrides=["test=value"])

        assert result is mock_config
        mock_manager.load_config.assert_called_once_with("base", "staging", ["test=value"])

    @patch("music_gen.core.config_manager.get_config_manager")
    def test_get_config_function(self, mock_get_manager):
        """Test get_config convenience function."""
        mock_manager = Mock()
        mock_config = Mock()
        mock_manager.get_config.return_value = mock_config
        mock_get_manager.return_value = mock_manager

        result = get_config()

        assert result is mock_config
        mock_manager.get_config.assert_called_once()

    def test_validate_config_file_success(self):
        """Test validating a configuration file."""
        valid_config = {"app": {"name": "Test"}, "api": {"port": 8000}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            import yaml

            yaml.dump(valid_config, f)

        try:
            result = validate_config_file(config_path)

            assert result.is_valid
            assert result.config is not None
            assert result.errors is None

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_validate_config_file_failure(self):
        """Test validating an invalid configuration file."""
        invalid_config = {"api": {"port": "invalid_port"}}  # Should be integer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name
            import yaml

            yaml.dump(invalid_config, f)

        try:
            result = validate_config_file(config_path)

            assert not result.is_valid
            assert result.config is None
            assert result.errors is not None

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_validate_config_file_not_found(self):
        """Test validating a non-existent configuration file."""
        result = validate_config_file("/nonexistent/config.yaml")

        assert not result.is_valid
        assert result.errors is not None
        assert "Failed to load configuration file" in result.errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
