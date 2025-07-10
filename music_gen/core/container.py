"""
Dependency Injection Container for Music Gen AI.

This module provides a centralized DI container using the injector library,
managing all dependencies and their lifecycles.
"""

from typing import Optional

from injector import Injector, Module, provider, singleton

from music_gen.application.services import (
    AudioProcessingServiceImpl,
    GenerationServiceImpl,
    ModelServiceImpl,
    TrainingServiceImpl,
)
from music_gen.core.config import AppConfig
from music_gen.core.interfaces.repositories import (
    AudioRepository,
    MetadataRepository,
    ModelRepository,
    TaskRepository,
)
from music_gen.core.interfaces.services import (
    AudioProcessingService,
    GenerationService,
    ModelService,
    TrainingService,
)
from music_gen.infrastructure.repositories import (
    FileSystemAudioRepository,
    FileSystemMetadataRepository,
    FileSystemModelRepository,
    InMemoryTaskRepository,
    PostgreSQLTaskRepository,
    RedisTaskRepository,
)
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    RedisTaskRepositoryAdvanced,
)


class CoreModule(Module):
    """Core dependency injection module."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

    @singleton
    @provider
    def provide_app_config(self) -> AppConfig:
        """Provide application configuration."""
        return self.config

    # Repository providers
    @singleton
    @provider
    def provide_model_repository(self, config: AppConfig) -> ModelRepository:
        """Provide model repository implementation."""
        return FileSystemModelRepository(config.model_cache_dir)

    @singleton
    @provider
    def provide_task_repository(self, config: AppConfig) -> TaskRepository:
        """Provide task repository implementation based on configuration."""
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        # Use database URL if available (PostgreSQL preferred)
        if config.database_url:
            logger.info("Using PostgreSQL task repository")
            return PostgreSQLTaskRepository(config.database_url)

        # Use Redis if available
        elif config.redis_url:
            # Check if advanced Redis features are enabled
            use_advanced_redis = getattr(config, "use_advanced_redis", True)

            if use_advanced_redis:
                logger.info("Using advanced Redis task repository with Streams support")
                repo = RedisTaskRepositoryAdvanced(config.redis_url)
                # Initialize repository in background
                asyncio.create_task(repo.initialize())
                return repo
            else:
                logger.info("Using basic Redis task repository")
                return RedisTaskRepository(config.redis_url)

        # Fall back to in-memory for development
        else:
            logger.warning(
                "Using in-memory task repository. Configure DATABASE_URL or REDIS_URL for production."
            )
            return InMemoryTaskRepository()

    @singleton
    @provider
    def provide_metadata_repository(self, config: AppConfig) -> MetadataRepository:
        """Provide metadata repository implementation."""
        return FileSystemMetadataRepository(config.data_dir)

    @singleton
    @provider
    def provide_audio_repository(self, config: AppConfig) -> AudioRepository:
        """Provide audio repository implementation."""
        return FileSystemAudioRepository(config.audio_cache_dir)

    # Service providers
    @singleton
    @provider
    def provide_model_service(self, model_repo: ModelRepository, config: AppConfig) -> ModelService:
        """Provide model service implementation."""
        return ModelServiceImpl(model_repo, config)

    @singleton
    @provider
    def provide_audio_processing_service(
        self, audio_repo: AudioRepository, config: AppConfig
    ) -> AudioProcessingService:
        """Provide audio processing service."""
        return AudioProcessingServiceImpl(audio_repo, config)

    @singleton
    @provider
    def provide_generation_service(
        self,
        model_service: ModelService,
        audio_service: AudioProcessingService,
        task_repo: TaskRepository,
    ) -> GenerationService:
        """Provide generation service implementation."""
        return GenerationServiceImpl(model_service, audio_service, task_repo)

    @singleton
    @provider
    def provide_training_service(
        self, model_service: ModelService, metadata_repo: MetadataRepository, config: AppConfig
    ) -> TrainingService:
        """Provide training service implementation."""
        return TrainingServiceImpl(model_service, metadata_repo, config)


class Container:
    """Main dependency injection container."""

    _instance: Optional[Injector] = None

    @classmethod
    def get_instance(cls, config: Optional[AppConfig] = None) -> Injector:
        """Get or create the DI container instance."""
        if cls._instance is None:
            cls._instance = Injector([CoreModule(config)])
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the container (useful for testing)."""
        cls._instance = None


def get_container(config: Optional[AppConfig] = None) -> Injector:
    """Get the dependency injection container."""
    return Container.get_instance(config)
