"""
Unit tests for configuration models.

Tests Pydantic validation models for comprehensive configuration validation.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from music_gen.core.config_models import (
    MusicGenConfig,
    ApiConfig,
    AuthConfig,
    ModelConfig,
    GenerationConfig,
    AudioConfig,
    ResourceConfig,
    MonitoringConfig,
    ExternalServicesConfig,
    FeaturesConfig,
    ValidationConfig,
    CorsConfig,
    RateLimitingConfig,
    ProductionConfig,
    StagingConfig,
    LimitsConfig,
)


@pytest.mark.unit
class TestConfigModels:
    """Test configuration model validation."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = MusicGenConfig()

        assert config.app.name == "MusicGen AI"
        assert config.app.version == "1.0.0"
        assert config.app.environment == "development"
        assert config.model.default_model == "facebook/musicgen-small"
        assert config.generation.default_duration == 10.0

    def test_api_config_validation(self):
        """Test API configuration validation."""
        # Valid configuration
        api_config = ApiConfig(host="0.0.0.0", port=8000, workers=4, debug=False)
        assert api_config.port == 8000
        assert api_config.workers == 4

        # Invalid port
        with pytest.raises(ValidationError):
            ApiConfig(port=70000)  # Port too high

        # Invalid workers
        with pytest.raises(ValidationError):
            ApiConfig(workers=0)  # Must be >= 1

    def test_auth_config_validation(self):
        """Test authentication configuration validation."""
        # Valid disabled auth
        auth_config = AuthConfig(enabled=False)
        assert not auth_config.enabled

        # Valid enabled auth with strong secret
        auth_config = AuthConfig(
            enabled=True, jwt_secret="a" * 32, jwt_algorithm="RS256"  # 32 character secret
        )
        assert auth_config.enabled
        assert auth_config.jwt_algorithm == "RS256"

        # Invalid: enabled auth with weak secret
        with pytest.raises(ValidationError):
            AuthConfig(enabled=True, jwt_secret="weak")  # Too short for production

    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid configuration
        model_config = ModelConfig(
            default_model="facebook/musicgen-medium", cache_size=5, device="cpu"
        )
        assert model_config.default_model == "facebook/musicgen-medium"
        assert model_config.cache_size == 5

        # Invalid cache size
        with pytest.raises(ValidationError):
            ModelConfig(cache_size=0)  # Must be >= 1

        # Invalid cache size too large
        with pytest.raises(ValidationError):
            ModelConfig(cache_size=25)  # Must be <= 20

        # Empty model name
        with pytest.raises(ValidationError):
            ModelConfig(default_model="")

    def test_generation_config_validation(self):
        """Test generation configuration validation."""
        # Valid configuration
        gen_config = GenerationConfig(
            default_duration=15.0, max_duration=60.0, default_temperature=0.8
        )
        assert gen_config.default_duration == 15.0
        assert gen_config.max_duration == 60.0

        # Invalid: max duration less than default
        with pytest.raises(ValidationError):
            GenerationConfig(default_duration=30.0, max_duration=20.0)  # Less than default

        # Invalid temperature
        with pytest.raises(ValidationError):
            GenerationConfig(default_temperature=3.0)  # Too high

        # Invalid batch size
        with pytest.raises(ValidationError):
            GenerationConfig(max_batch_size=0)  # Must be >= 1

    def test_audio_config_validation(self):
        """Test audio configuration validation."""
        # Valid configuration
        audio_config = AudioConfig(sample_rate=44100, format="wav", bit_depth=16, channels=2)
        assert audio_config.sample_rate == 44100
        assert audio_config.format == "wav"
        assert audio_config.channels == 2

        # Invalid sample rate
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=12000)  # Not in supported rates

        # Invalid format
        with pytest.raises(ValidationError):
            AudioConfig(format="xyz")  # Not supported

        # Invalid bit depth
        with pytest.raises(ValidationError):
            AudioConfig(bit_depth=12)  # Not supported

    def test_resource_config_validation(self):
        """Test resource configuration validation."""
        # Valid configuration
        resource_config = ResourceConfig(
            max_memory_usage_percent=80.0, cleanup_threshold_percent=70.0, gpu_memory_fraction=0.8
        )
        assert resource_config.max_memory_usage_percent == 80.0
        assert resource_config.cleanup_threshold_percent == 70.0

        # Invalid: cleanup threshold >= max usage
        with pytest.raises(ValidationError):
            ResourceConfig(
                max_memory_usage_percent=80.0, cleanup_threshold_percent=85.0  # Higher than max
            )

        # Invalid GPU memory fraction
        with pytest.raises(ValidationError):
            ResourceConfig(gpu_memory_fraction=1.5)  # > 1.0

    def test_cors_config_validation(self):
        """Test CORS configuration validation."""
        cors_config = CorsConfig(
            enabled=True,
            origins=["http://localhost:3000", "https://example.com"],
            methods=["GET", "POST"],
            credentials=True,
        )

        assert cors_config.enabled
        assert len(cors_config.origins) == 2
        assert "GET" in cors_config.methods

    def test_rate_limiting_config_validation(self):
        """Test rate limiting configuration validation."""
        # Valid configuration
        rate_config = RateLimitingConfig(enabled=True, requests_per_minute=100, burst_limit=20)
        assert rate_config.enabled
        assert rate_config.requests_per_minute == 100

        # Invalid requests per minute
        with pytest.raises(ValidationError):
            RateLimitingConfig(requests_per_minute=0)  # Must be >= 1

        # Invalid burst limit
        with pytest.raises(ValidationError):
            RateLimitingConfig(burst_limit=0)  # Must be >= 1

    def test_external_services_config(self):
        """Test external services configuration."""
        services_config = ExternalServicesConfig()

        # Default should have all services disabled
        assert not services_config.wandb.enabled
        assert not services_config.redis.enabled
        assert not services_config.postgresql.enabled

        # Test with enabled services
        services_config = ExternalServicesConfig(
            wandb={"enabled": True, "project": "test-project"},
            redis={"enabled": True, "host": "redis.example.com"},
            postgresql={"enabled": True, "host": "postgres.example.com"},
        )

        assert services_config.wandb.enabled
        assert services_config.wandb.project == "test-project"
        assert services_config.redis.enabled
        assert services_config.redis.host == "redis.example.com"

    def test_features_config(self):
        """Test features configuration."""
        features_config = FeaturesConfig(
            streaming_generation=True,
            multi_instrument=True,
            batch_processing=True,
            audio_effects=False,
        )

        assert features_config.streaming_generation
        assert features_config.multi_instrument
        assert features_config.batch_processing
        assert not features_config.audio_effects

    def test_validation_config(self):
        """Test validation configuration."""
        validation_config = ValidationConfig(
            strict_mode=True, max_prompt_length=500, min_duration=1.0, max_file_size_mb=50
        )

        assert validation_config.strict_mode
        assert validation_config.max_prompt_length == 500

        # Invalid prompt length
        with pytest.raises(ValidationError):
            ValidationConfig(max_prompt_length=0)  # Must be >= 1

        # Invalid duration
        with pytest.raises(ValidationError):
            ValidationConfig(min_duration=0.0)  # Must be > 0.01

    def test_production_config_validation(self):
        """Test production-specific configuration validation."""
        production_config = ProductionConfig()

        # Should have secure defaults
        assert not production_config.high_availability.enabled  # Disabled by default
        assert production_config.security.response_sanitization
        assert production_config.backup.encryption

    def test_staging_config_validation(self):
        """Test staging-specific configuration validation."""
        staging_config = StagingConfig(
            load_testing={"enabled": True, "max_concurrent_users": 20},
            data_collection={"enabled": True, "retention_days": 14},
        )

        assert staging_config.load_testing.enabled
        assert staging_config.load_testing.max_concurrent_users == 20
        assert staging_config.data_collection.retention_days == 14

    def test_limits_config_validation(self):
        """Test resource limits configuration validation."""
        limits_config = LimitsConfig(
            max_requests_per_hour=5000,
            max_generation_time_minutes=10,
            max_concurrent_generations=20,
        )

        assert limits_config.max_requests_per_hour == 5000
        assert limits_config.max_generation_time_minutes == 10

        # Invalid limits
        with pytest.raises(ValidationError):
            LimitsConfig(max_requests_per_hour=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            LimitsConfig(max_generation_time_minutes=0)  # Must be >= 1

    def test_complete_config_validation(self):
        """Test complete configuration validation."""
        config_dict = {
            "app": {"name": "Test MusicGen", "version": "1.0.0", "environment": "production"},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4, "debug": False},
            "auth": {
                "enabled": True,
                "jwt_secret": "production_secret_key_32_chars_long",
                "jwt_algorithm": "RS256",
            },
            "model": {
                "default_model": "facebook/musicgen-large",
                "cache_size": 5,
                "device": "cuda",
            },
            "generation": {"default_duration": 15.0, "max_duration": 120.0, "quality_mode": "high"},
            "limits": {"max_requests_per_hour": 1000, "max_generation_time_minutes": 10},
        }

        config = MusicGenConfig.parse_obj(config_dict)

        assert config.app.environment == "production"
        assert config.auth.enabled
        assert config.model.default_model == "facebook/musicgen-large"
        assert config.generation.quality_mode == "high"

    def test_environment_consistency_validation(self):
        """Test environment consistency validation."""
        # Production environment without required settings should fail
        with pytest.raises(ValidationError, match="Authentication must be enabled in production"):
            MusicGenConfig(app={"environment": "production"}, auth={"enabled": False})

        # Production with debug enabled should fail
        with pytest.raises(ValidationError, match="Debug mode must be disabled in production"):
            MusicGenConfig(
                app={"environment": "production", "debug": True},
                auth={"enabled": True, "jwt_secret": "a" * 32},
            )

        # Production without limits should fail
        with pytest.raises(ValidationError, match="Resource limits must be defined in production"):
            MusicGenConfig(
                app={"environment": "production", "debug": False},
                auth={"enabled": True, "jwt_secret": "a" * 32}
                # Missing limits
            )

    def test_config_schema_generation(self):
        """Test configuration schema generation."""
        schema = MusicGenConfig.schema()

        assert "title" in schema
        assert "properties" in schema
        assert "required" in schema

        # Check that key sections are present
        properties = schema["properties"]
        assert "app" in properties
        assert "api" in properties
        assert "model" in properties
        assert "generation" in properties
        assert "audio" in properties

    def test_config_environment_summary(self):
        """Test configuration environment summary."""
        config = MusicGenConfig(
            app={"environment": "staging"},
            auth={"enabled": True},
            model={"default_model": "facebook/musicgen-medium"},
            features={"streaming_generation": True, "batch_processing": True},
        )

        summary = config.get_environment_summary()

        assert summary["environment"] == "staging"
        assert summary["auth_enabled"] is True
        assert summary["default_model"] == "facebook/musicgen-medium"
        assert summary["feature_flags"]["streaming"] is True
        assert summary["feature_flags"]["batch_processing"] is True

    def test_config_path_validation(self):
        """Test configuration path validation."""
        config = MusicGenConfig(
            model={"cache_dir": "/tmp/test_models"},
            audio={"output_dir": "/tmp/test_audio"},
            data={"datasets_dir": "/tmp/test_datasets"},
        )

        # This should not raise any validation errors during model creation
        assert config.model.cache_dir == "/tmp/test_models"
        assert config.audio.output_dir == "/tmp/test_audio"
        assert config.data.datasets_dir == "/tmp/test_datasets"

    def test_invalid_config_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            MusicGenConfig(
                app={"name": "Test"}, unknown_field="should_fail"  # Extra field should be rejected
            )

    def test_config_field_constraints(self):
        """Test various field constraints."""
        # Test string length constraints
        with pytest.raises(ValidationError):
            MusicGenConfig(app={"name": ""})  # Empty name should fail

        # Test numeric constraints
        with pytest.raises(ValidationError):
            MusicGenConfig(generation={"default_temperature": 0.0})  # Below minimum

        with pytest.raises(ValidationError):
            MusicGenConfig(generation={"default_temperature": 3.0})  # Above maximum

        # Test choice constraints
        with pytest.raises(ValidationError):
            MusicGenConfig(audio={"format": "invalid_format"})  # Invalid choice

    def test_nested_config_validation(self):
        """Test nested configuration validation."""
        # Test valid nested config
        config = MusicGenConfig(
            external_services={
                "redis": {"enabled": True, "host": "redis.example.com", "port": 6379, "ssl": True}
            }
        )

        assert config.external_services.redis.enabled
        assert config.external_services.redis.ssl

        # Test invalid nested config
        with pytest.raises(ValidationError):
            MusicGenConfig(
                external_services={"redis": {"enabled": True, "port": 70000}}  # Invalid port
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
