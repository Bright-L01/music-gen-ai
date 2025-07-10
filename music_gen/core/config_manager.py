"""
Configuration management system for MusicGen AI.

Provides unified configuration loading, validation, and management using
Hydra for YAML configuration and Pydantic for validation.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from .config_models import MusicGenConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    config: Optional[MusicGenConfig] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class ConfigManager:
    """
    Unified configuration manager for MusicGen AI.

    Handles configuration loading from Hydra YAML files with Pydantic validation,
    environment variable substitution, and schema generation.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to project config directory.
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self._config: Optional[MusicGenConfig] = None
        self._hydra_config: Optional[DictConfig] = None
        self._is_initialized = False

    def _get_default_config_dir(self) -> str:
        """Get default configuration directory."""
        # Find the project root by looking for setup.py or pyproject.toml
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            if any(
                (current_dir / name).exists() for name in ["setup.py", "pyproject.toml", ".git"]
            ):
                return str(current_dir / "configs")
            current_dir = current_dir.parent

        # Fallback to relative path
        return str(Path(__file__).parent.parent.parent / "configs")

    def initialize(self, config_name: str = "base", overrides: Optional[List[str]] = None) -> None:
        """
        Initialize Hydra configuration system.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
            overrides: List of configuration overrides (e.g., ["model.device=cuda"])
        """
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra().clear()

            # Initialize Hydra with our config directory
            config_dir = Path(self.config_dir).absolute()
            if not config_dir.exists():
                raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

            initialize_config_dir(config_dir=str(config_dir), version_base=None)

            # Load configuration
            overrides = overrides or []
            self._hydra_config = compose(config_name=config_name, overrides=overrides)

            self._is_initialized = True
            logger.info(f"Configuration initialized from {config_dir}/{config_name}.yaml")

        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise

    def load_config(
        self,
        config_name: str = "base",
        environment: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        validate: bool = True,
    ) -> MusicGenConfig:
        """
        Load and validate configuration.

        Args:
            config_name: Base configuration name
            environment: Environment override (development, staging, production)
            overrides: Additional configuration overrides
            validate: Whether to validate configuration with Pydantic

        Returns:
            Validated MusicGenConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Determine effective config name
            if environment:
                # Use environment-specific config if available
                env_config_path = Path(self.config_dir) / "app" / f"{environment}.yaml"
                if env_config_path.exists():
                    config_name = f"app/{environment}"
                else:
                    logger.warning(
                        f"Environment config {environment}.yaml not found, using {config_name}"
                    )

            # Initialize Hydra if not already done
            if not self._is_initialized:
                self.initialize(config_name, overrides)
            else:
                # Re-compose with new parameters
                overrides = overrides or []
                self._hydra_config = compose(config_name=config_name, overrides=overrides)

            # Convert OmegaConf to dict for Pydantic
            config_dict = OmegaConf.to_container(self._hydra_config, resolve=True)

            # Perform environment variable substitution
            config_dict = self._substitute_env_vars(config_dict)

            if validate:
                # Validate with Pydantic
                validation_result = self.validate_config(config_dict)
                if not validation_result.is_valid:
                    raise ConfigurationError(
                        f"Configuration validation failed: {validation_result.errors}"
                    )
                self._config = validation_result.config
            else:
                # Create config without validation
                self._config = MusicGenConfig.parse_obj(config_dict)

            logger.info(f"Configuration loaded successfully: {self._config.app.environment}")
            return self._config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def validate_config(self, config_dict: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate configuration dictionary with Pydantic.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            ConfigValidationResult with validation details
        """
        errors = []
        warnings = []

        try:
            # Validate with Pydantic
            config = MusicGenConfig.parse_obj(config_dict)

            # Additional custom validations
            path_issues = config.validate_paths()
            if path_issues:
                warnings.extend(path_issues)

            return ConfigValidationResult(
                is_valid=True, config=config, errors=None, warnings=warnings if warnings else None
            )

        except ValidationError as e:
            # Extract validation errors
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_msg = f"{field_path}: {error['msg']}"
                errors.append(error_msg)

            return ConfigValidationResult(
                is_valid=False, config=None, errors=errors, warnings=warnings if warnings else None
            )

        except Exception as e:
            return ConfigValidationResult(
                is_valid=False,
                config=None,
                errors=[f"Unexpected validation error: {str(e)}"],
                warnings=None,
            )

    def _substitute_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in configuration.

        Handles patterns like:
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration with environment variables substituted
        """

        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable name and default value
                var_expr = value[2:-1]  # Remove ${ and }

                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                else:
                    var_name, default_value = var_expr, None

                # Get environment variable
                env_value = os.getenv(var_name, default_value)

                if env_value is None:
                    logger.warning(
                        f"Environment variable {var_name} not set and no default provided"
                    )
                    return value  # Return original value

                # Try to convert to appropriate type
                if env_value.lower() in ("true", "false"):
                    return env_value.lower() == "true"
                elif env_value.isdigit():
                    return int(env_value)
                elif env_value.replace(".", "").isdigit():
                    return float(env_value)
                else:
                    return env_value

            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config_dict)

    def get_config(self) -> Optional[MusicGenConfig]:
        """Get the currently loaded configuration."""
        return self._config

    def reload_config(self) -> MusicGenConfig:
        """Reload configuration from disk."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")

        # Re-load with same parameters
        return self.load_config()

    def save_config_schema(self, output_path: str) -> None:
        """
        Save configuration schema to JSON file.

        Args:
            output_path: Path to save schema file
        """
        schema = MusicGenConfig.schema()

        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2, sort_keys=True)

        logger.info(f"Configuration schema saved to {output_path}")

    def generate_example_config(self, output_path: str, environment: str = "development") -> None:
        """
        Generate example configuration file.

        Args:
            output_path: Path to save example configuration
            environment: Environment type for the example
        """
        # Create default config
        config = MusicGenConfig()
        config.app.environment = environment

        # Adjust settings for environment
        if environment == "production":
            config.app.debug = False
            config.auth.enabled = True
            config.api.workers = 4
            config.monitoring.enabled = True
        elif environment == "staging":
            config.app.debug = False
            config.auth.enabled = True
            config.api.workers = 2

        # Convert to dict and save
        config_dict = config.dict()

        with open(output_path, "w") as f:
            # Use OmegaConf to create nice YAML output
            yaml_config = OmegaConf.create(config_dict)
            OmegaConf.save(yaml_config, f)

        logger.info(f"Example {environment} configuration saved to {output_path}")

    def validate_current_config(self) -> ConfigValidationResult:
        """Validate the currently loaded configuration."""
        if not self._config:
            return ConfigValidationResult(is_valid=False, errors=["No configuration loaded"])

        return self.validate_config(self._config.dict())

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        if not self._config:
            return {"status": "No configuration loaded"}

        return self._config.get_environment_summary()

    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> MusicGenConfig:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate after update

        Returns:
            Updated configuration
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")

        # Deep merge updates
        current_dict = self._config.dict()
        updated_dict = self._deep_merge(current_dict, updates)

        if validate:
            validation_result = self.validate_config(updated_dict)
            if not validation_result.is_valid:
                raise ConfigurationError(
                    f"Configuration update validation failed: {validation_result.errors}"
                )
            self._config = validation_result.config
        else:
            self._config = MusicGenConfig.parse_obj(updated_dict)

        return self._config

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def export_config(self, output_path: str, format: str = "yaml") -> None:
        """
        Export current configuration to file.

        Args:
            output_path: Output file path
            format: Export format (yaml, json)
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")

        config_dict = self._config.dict()

        if format.lower() == "yaml":
            yaml_config = OmegaConf.create(config_dict)
            OmegaConf.save(yaml_config, output_path)
        elif format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Configuration exported to {output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()


class ConfigurationError(Exception):
    """Configuration-related error."""


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(
    config_name: str = "base",
    environment: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> MusicGenConfig:
    """
    Convenience function to load configuration.

    Args:
        config_name: Configuration name
        environment: Environment override
        overrides: Configuration overrides

    Returns:
        Loaded and validated configuration
    """
    # Try to get environment from environment variable if not specified
    if environment is None:
        environment = os.getenv("MUSICGEN_ENV", "development")

    manager = get_config_manager()
    return manager.load_config(config_name, environment, overrides)


def get_config() -> Optional[MusicGenConfig]:
    """Get currently loaded configuration."""
    manager = get_config_manager()
    return manager.get_config()


def validate_config_file(config_path: str) -> ConfigValidationResult:
    """
    Validate a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validation result
    """
    try:
        # Load YAML file
        config_dict = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

        # Validate with Pydantic
        manager = ConfigManager()
        return manager.validate_config(config_dict)

    except Exception as e:
        return ConfigValidationResult(
            is_valid=False, errors=[f"Failed to load configuration file: {str(e)}"]
        )
