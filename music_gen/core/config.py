"""
Application configuration management.

Provides centralized configuration with environment variable support,
validation, and type safety.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AppConfig:
    """Main application configuration."""

    # General settings
    app_name: str = "Music Gen AI"
    version: str = "1.0.0"
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Paths
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_cache_dir: Path = field(default_factory=lambda: Path("models/cache"))
    audio_cache_dir: Path = field(default_factory=lambda: Path("audio/cache"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Model settings
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "facebook/musicgen-small")
    )
    model_device: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    model_cache_size: int = field(default_factory=lambda: int(os.getenv("MODEL_CACHE_SIZE", "3")))

    # Audio settings
    default_sample_rate: int = 32000
    default_duration: float = 30.0
    max_duration: float = 300.0  # 5 minutes
    audio_format: str = "wav"

    # API settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    api_cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",")
    )

    # Database settings (for future use)
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))

    # Security settings
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY"))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expiry_hours: int = field(default_factory=lambda: int(os.getenv("JWT_EXPIRY_HOURS", "24")))
    session_secret_key: str = field(default_factory=lambda: os.getenv("SESSION_SECRET_KEY"))
    csrf_secret_key: str = field(default_factory=lambda: os.getenv("CSRF_SECRET_KEY"))

    # Training settings
    training_batch_size: int = 32
    training_learning_rate: float = 1e-4
    training_num_epochs: int = 10
    training_mixed_precision: bool = True

    # Performance settings
    max_concurrent_generations: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_GENERATIONS", "5"))
    )
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "300")))

    # Logging settings
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Validate and create directories after initialization."""
        # Convert string paths to Path objects
        self.base_dir = Path(self.base_dir)
        self.data_dir = self.base_dir / self.data_dir
        self.model_cache_dir = self.base_dir / self.model_cache_dir
        self.audio_cache_dir = self.base_dir / self.audio_cache_dir
        self.output_dir = self.base_dir / self.output_dir
        self.log_dir = self.base_dir / self.log_dir

        # Create directories if they don't exist
        for dir_path in [
            self.data_dir,
            self.model_cache_dir,
            self.audio_cache_dir,
            self.output_dir,
            self.log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Validate security settings in production
        if self.environment == "production":
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY environment variable must be set in production")
            if not self.session_secret_key:
                raise ValueError(
                    "SESSION_SECRET_KEY environment variable must be set in production"
                )
            if not self.csrf_secret_key:
                raise ValueError("CSRF_SECRET_KEY environment variable must be set in production")

            # Validate secret strength
            for secret_name, secret_value in [
                ("JWT_SECRET_KEY", self.jwt_secret_key),
                ("SESSION_SECRET_KEY", self.session_secret_key),
                ("CSRF_SECRET_KEY", self.csrf_secret_key),
            ]:
                if secret_value and len(secret_value) < 32:
                    raise ValueError(
                        f"{secret_name} must be at least 32 characters long for security"
                    )

    @classmethod
    def from_file(cls, config_path: str) -> "AppConfig":
        """Load configuration from file (JSON or YAML)."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                config_data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        path = Path(config_path)
        config_dict = self.to_dict()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")


# Import torch here to avoid circular imports
import torch

# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from environment variable or use defaults
        config_path = os.getenv("MUSIC_GEN_CONFIG")
        if config_path and Path(config_path).exists():
            _config = AppConfig.from_file(config_path)
        else:
            _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
