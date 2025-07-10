"""
Pydantic models for configuration validation.

Provides comprehensive validation for all application configuration settings
with automatic schema generation and type safety.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator, validator

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CorsConfig(BaseModel):
    """CORS configuration model."""

    enabled: bool = True
    origins: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    headers: List[str] = Field(default=["*"])
    credentials: bool = True
    max_age: Optional[int] = None


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration model."""

    enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    burst_limit: int = Field(default=10, ge=1, le=1000)
    per_user_limit: Optional[int] = Field(default=None, ge=1)


class ApiConfig(BaseModel):
    """API configuration model."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    reload: bool = True
    debug: bool = True
    cors: CorsConfig = Field(default_factory=CorsConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)

    @validator("host")
    def validate_host(cls, v):
        """Validate host format."""
        if v not in ["0.0.0.0", "127.0.0.1", "localhost"] and not v.startswith("0.0.0.0"):
            # Allow custom hosts but validate basic format
            if not v.replace(".", "").replace(":", "").replace("-", "").replace("_", "").isalnum():
                raise ValueError("Invalid host format")
        return v


class AuthConfig(BaseModel):
    """Authentication configuration model."""

    enabled: bool = False
    jwt_secret: str = Field(default="default_dev_secret")
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"] = "HS256"
    jwt_expiration_hours: int = Field(default=24, ge=1, le=8760)  # Max 1 year
    api_key_required: bool = False
    allowed_api_keys: List[str] = Field(default_factory=list)
    rate_limiting: Optional[RateLimitingConfig] = None

    @validator("jwt_secret")
    def validate_jwt_secret(cls, v, values):
        """Validate JWT secret security."""
        if values.get("enabled", False) and len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters when auth is enabled")
        return v


class ModelConfig(BaseModel):
    """Model configuration model."""

    default_model: str = Field(default="facebook/musicgen-small")
    cache_size: int = Field(default=3, ge=1, le=20)
    cache_dir: str = Field(default="./cache/models")
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    download_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    preload_models: Optional[List[str]] = None

    @validator("default_model")
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @validator("device")
    def validate_device(cls, v):
        """Validate device availability."""
        if not TORCH_AVAILABLE:
            # If torch is not available, only allow cpu
            if v not in ["auto", "cpu"]:
                raise ValueError(
                    f"Device '{v}' requested but torch is not available. Use 'cpu' or 'auto'."
                )
            return "cpu"  # Force cpu when torch unavailable

        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        elif v == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError("MPS device requested but not available")
        return v


class GenerationConfig(BaseModel):
    """Generation configuration model."""

    default_duration: float = Field(default=10.0, ge=0.1, le=300.0)
    max_duration: float = Field(default=60.0, ge=1.0, le=600.0)
    default_temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    default_top_k: int = Field(default=250, ge=1, le=2048)
    default_top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    max_batch_size: int = Field(default=4, ge=1, le=32)
    timeout_seconds: int = Field(default=120, ge=10, le=3600)
    quality_mode: Literal["fast", "standard", "high"] = "standard"
    priority_queue_enabled: bool = False

    @validator("max_duration")
    def validate_max_duration(cls, v, values):
        """Ensure max duration is greater than default duration."""
        default_duration = values.get("default_duration", 10.0)
        if v <= default_duration:
            raise ValueError("max_duration must be greater than default_duration")
        return v


class AudioConfig(BaseModel):
    """Audio configuration model."""

    sample_rate: int = Field(default=24000, ge=8000, le=96000)
    format: Literal["wav", "mp3", "flac", "ogg"] = "wav"
    bit_depth: Literal[8, 16, 24, 32] = 16
    channels: Literal[1, 2] = 1
    normalization: Literal["peak", "rms", "lufs", "none"] = "peak"
    output_dir: str = Field(default="./outputs/audio")
    temp_dir: str = Field(default="./tmp/audio")
    cleanup_temp_files: bool = True
    backup_outputs: bool = False
    backup_retention_days: Optional[int] = Field(default=None, ge=1, le=365)

    @validator("sample_rate")
    def validate_sample_rate(cls, v):
        """Validate sample rate is supported."""
        supported_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
        if v not in supported_rates:
            raise ValueError(f"Sample rate {v} not in supported rates: {supported_rates}")
        return v


class ResourceConfig(BaseModel):
    """Resource management configuration model."""

    max_memory_usage_percent: float = Field(default=85.0, ge=10.0, le=95.0)
    max_cpu_usage_percent: float = Field(default=90.0, ge=10.0, le=95.0)
    cleanup_threshold_percent: float = Field(default=80.0, ge=10.0, le=90.0)
    monitoring_interval_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    auto_cleanup_enabled: bool = True
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    memory_pressure_handling: Literal["conservative", "moderate", "aggressive"] = "moderate"

    @validator("cleanup_threshold_percent")
    def validate_cleanup_threshold(cls, v, values):
        """Ensure cleanup threshold is less than max usage."""
        max_memory = values.get("max_memory_usage_percent", 85.0)
        if v >= max_memory:
            raise ValueError("cleanup_threshold_percent must be less than max_memory_usage_percent")
        return v


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration model."""

    num_workers: int = Field(default=4, ge=1, le=32)
    batch_size: int = Field(default=32, ge=1, le=512)
    max_sequence_length: int = Field(default=1024, ge=64, le=4096)
    persistent_workers: bool = False


class DataConfig(BaseModel):
    """Data configuration model."""

    datasets_dir: str = Field(default="./data/datasets")
    cache_dir: str = Field(default="./cache/data")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class TrainingConfig(BaseModel):
    """Training configuration model."""

    enabled: bool = False
    checkpoints_dir: str = Field(default="./checkpoints")
    logs_dir: str = Field(default="./logs")
    tensorboard_dir: str = Field(default="./runs")
    save_every_n_steps: int = Field(default=1000, ge=1)
    validate_every_n_steps: int = Field(default=500, ge=1)
    max_checkpoints: int = Field(default=5, ge=1, le=50)


class MetricsConfig(BaseModel):
    """Metrics configuration model."""

    enabled: bool = True
    port: int = Field(default=9090, ge=1024, le=65535)
    endpoint: str = Field(default="/metrics")
    detailed_metrics: bool = False
    custom_metrics: bool = False


class HealthCheckConfig(BaseModel):
    """Health check configuration model."""

    enabled: bool = True
    endpoint: str = Field(default="/health")
    detailed_endpoint: str = Field(default="/health/detailed")
    liveness_probe: Optional[str] = None
    readiness_probe: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text", "colored"] = "json"
    file_rotation: bool = True
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=100)
    console_output: bool = True
    structured_logging: bool = False
    correlation_id: bool = False


class MonitoringConfig(BaseModel):
    """Monitoring configuration model."""

    enabled: bool = True
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class WandbConfig(BaseModel):
    """Weights & Biases configuration model."""

    enabled: bool = False
    project: str = Field(default="musicgen-ai")
    entity: Optional[str] = None
    api_key: Optional[str] = None
    log_frequency: Literal["minimal", "normal", "verbose"] = "normal"


class RedisConfig(BaseModel):
    """Redis configuration model."""

    enabled: bool = False
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[str] = None
    ssl: bool = False
    ssl_verify: bool = True
    connection_pool_size: int = Field(default=10, ge=1, le=100)
    max_connections: int = Field(default=50, ge=1, le=500)
    health_check_interval: int = Field(default=30, ge=5, le=300)


class PostgreSQLConfig(BaseModel):
    """PostgreSQL configuration model."""

    enabled: bool = False
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="musicgen")
    username: str = Field(default="musicgen")
    password: Optional[str] = None
    ssl_mode: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ] = "prefer"
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=200)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    health_check_interval: int = Field(default=60, ge=10, le=600)


class ExternalServicesConfig(BaseModel):
    """External services configuration model."""

    wandb: WandbConfig = Field(default_factory=WandbConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)


class FeaturesConfig(BaseModel):
    """Feature flags configuration model."""

    streaming_generation: bool = False
    multi_instrument: bool = False
    batch_processing: bool = True
    conditioning: bool = True
    model_switching: bool = True
    audio_effects: bool = False
    midi_export: bool = False


class DevToolsConfig(BaseModel):
    """Development tools configuration model."""

    profiling_enabled: bool = False
    debug_sql: bool = False
    mock_slow_operations: bool = False
    test_mode: bool = False
    hot_reload: bool = True
    auto_open_browser: bool = False
    development_server: bool = False


class ValidationConfig(BaseModel):
    """Validation configuration model."""

    strict_mode: bool = False
    validate_audio_output: bool = True
    validate_model_outputs: bool = True
    max_prompt_length: int = Field(default=1000, ge=1, le=10000)
    min_duration: float = Field(default=0.1, ge=0.01, le=10.0)
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)


class RetryPolicyConfig(BaseModel):
    """Retry policy configuration model."""

    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    max_delay: int = Field(default=60, ge=1, le=3600)


class HighAvailabilityConfig(BaseModel):
    """High availability configuration model."""

    enabled: bool = False
    health_checks: bool = True
    graceful_shutdown_timeout: int = Field(default=30, ge=5, le=300)
    circuit_breaker: bool = False
    retry_policy: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)


class PerformanceConfig(BaseModel):
    """Performance optimization configuration model."""

    connection_pooling: bool = True
    caching_strategy: Literal["none", "conservative", "moderate", "aggressive"] = "moderate"
    preload_critical_data: bool = False
    optimize_for_latency: bool = False


class SecurityConfig(BaseModel):
    """Security configuration model."""

    request_validation: Literal["none", "basic", "strict"] = "basic"
    response_sanitization: bool = True
    security_headers: bool = True
    audit_logging: bool = False
    ip_whitelisting: bool = False
    ddos_protection: bool = False


class AlertingConfig(BaseModel):
    """Alerting configuration model."""

    enabled: bool = False
    channels: List[Literal["email", "slack", "pagerduty", "webhook"]] = Field(default_factory=list)
    escalation_policy: bool = False


class ProductionMonitoringConfig(BaseModel):
    """Production monitoring configuration model."""

    apm_enabled: bool = False
    error_tracking: bool = False
    performance_monitoring: bool = False
    uptime_monitoring: bool = False
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)


class BackupConfig(BaseModel):
    """Backup configuration model."""

    enabled: bool = False
    frequency: Literal["hourly", "daily", "weekly"] = "daily"
    retention_days: int = Field(default=30, ge=1, le=365)
    encryption: bool = True
    offsite_backup: bool = False


class ProductionConfig(BaseModel):
    """Production-specific configuration model."""

    high_availability: HighAvailabilityConfig = Field(default_factory=HighAvailabilityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: ProductionMonitoringConfig = Field(default_factory=ProductionMonitoringConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)


class LoadTestingConfig(BaseModel):
    """Load testing configuration model."""

    enabled: bool = False
    max_concurrent_users: int = Field(default=10, ge=1, le=1000)
    test_data_generation: bool = False


class DataCollectionConfig(BaseModel):
    """Data collection configuration model."""

    enabled: bool = False
    anonymize_requests: bool = True
    store_generated_audio: bool = False
    retention_days: int = Field(default=30, ge=1, le=365)


class FeatureTogglesConfig(BaseModel):
    """Feature toggles configuration model."""

    new_model_testing: bool = False
    experimental_features: bool = False
    beta_endpoints: bool = False


class StagingConfig(BaseModel):
    """Staging-specific configuration model."""

    load_testing: LoadTestingConfig = Field(default_factory=LoadTestingConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    feature_toggles: FeatureTogglesConfig = Field(default_factory=FeatureTogglesConfig)


class LimitsConfig(BaseModel):
    """Resource limits and quotas configuration model."""

    max_requests_per_hour: int = Field(default=1000, ge=1, le=100000)
    max_generation_time_minutes: int = Field(default=5, ge=1, le=60)
    max_concurrent_generations: int = Field(default=10, ge=1, le=100)
    max_queue_size: int = Field(default=100, ge=1, le=10000)
    max_user_storage_mb: int = Field(default=1000, ge=10, le=100000)


class AppConfig(BaseModel):
    """Main application configuration model."""

    name: str = Field(default="MusicGen AI")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Production-ready text-to-music generation system")
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True


class MusicGenConfig(BaseModel):
    """
    Complete MusicGen AI configuration model.

    This is the root configuration model that validates the entire
    application configuration with comprehensive type safety and validation.
    """

    # Core application settings
    app: AppConfig = Field(default_factory=AppConfig)

    # Service configurations
    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # Model and generation
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

    # System resources
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Monitoring and observability
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # External integrations
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)

    # Feature management
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    dev_tools: DevToolsConfig = Field(default_factory=DevToolsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Environment-specific configurations
    production: Optional[ProductionConfig] = None
    staging: Optional[StagingConfig] = None
    limits: Optional[LimitsConfig] = None

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Reject unknown fields
        use_enum_values = True
        validate_all = True
        anystr_strip_whitespace = True

    @model_validator(mode="before")
    def validate_environment_consistency(cls, values):
        """Validate environment-specific configuration consistency."""
        if not isinstance(values, dict):
            return values

        app_values = values.get("app", {})
        environment = app_values.get("environment", "development")

        if environment == "production":
            # Validate production requirements
            auth = values.get("auth", {})
            if not auth.get("enabled", False):
                raise ValueError("Authentication must be enabled in production")

            # Check if debug is explicitly set to True in production
            debug_value = app_values.get("debug")
            if debug_value is True:  # Only fail if explicitly set to True
                raise ValueError("Debug mode must be disabled in production")

            limits = values.get("limits")
            if not limits:
                raise ValueError("Resource limits must be defined in production")

        return values

    @validator("app")
    def validate_app_config(cls, v, values):
        """Validate application configuration."""
        if not v.name or len(v.name.strip()) == 0:
            raise ValueError("Application name cannot be empty")
        return v

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for the configuration."""
        return self.schema()

    def validate_paths(self) -> List[str]:
        """Validate all configured paths exist or can be created."""
        issues = []

        # Check critical paths
        paths_to_check = [
            self.model.cache_dir,
            self.audio.output_dir,
            self.audio.temp_dir,
            self.data.datasets_dir,
            self.data.cache_dir,
        ]

        if self.training.enabled:
            paths_to_check.extend(
                [
                    self.training.checkpoints_dir,
                    self.training.logs_dir,
                    self.training.tensorboard_dir,
                ]
            )

        for path_str in paths_to_check:
            try:
                path = Path(path_str)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create path {path_str}: {e}")

        return issues

    def get_environment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "environment": self.app.environment,
            "debug_mode": self.app.debug,
            "api_workers": self.api.workers,
            "auth_enabled": self.auth.enabled,
            "default_model": self.model.default_model,
            "cache_size": self.model.cache_size,
            "max_duration": self.generation.max_duration,
            "monitoring_enabled": self.monitoring.enabled,
            "external_services": {
                "wandb": self.external_services.wandb.enabled,
                "redis": self.external_services.redis.enabled,
                "postgresql": self.external_services.postgresql.enabled,
            },
            "feature_flags": {
                "streaming": self.features.streaming_generation,
                "multi_instrument": self.features.multi_instrument,
                "batch_processing": self.features.batch_processing,
            },
        }
