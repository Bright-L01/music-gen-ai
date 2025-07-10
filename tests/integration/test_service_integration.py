"""
Integration tests for service-to-service communication.

Tests the interaction between different services in the MusicGen AI system,
including model service, generation service, audio processing, and repository layers.
"""

import asyncio
import pytest
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from typing import Dict, Any

from music_gen.application.services import (
    ModelServiceImpl,
    GenerationServiceImpl,
    AudioProcessingServiceImpl,
)
from music_gen.infrastructure.repositories import (
    InMemoryTaskRepository,
    InMemoryModelRepository,
    InMemoryAudioRepository,
)
from music_gen.core.interfaces.services import (
    GenerationRequest,
    GenerationResult,
)
from music_gen.core.config import AppConfig
from music_gen.core.exceptions import (
    GenerationError,
    ModelLoadError,
    AudioProcessingError,
)


@pytest.mark.integration
class TestModelGenerationIntegration:
    """Test integration between model service and generation service."""

    @pytest.fixture
    async def model_repository(self):
        """Create in-memory model repository."""
        repo = InMemoryModelRepository()

        # Pre-populate with test model
        test_model_data = {
            "state_dict": {"weight": torch.randn(100, 100)},
            "config": {
                "vocab_size": 2048,
                "hidden_size": 512,
                "sample_rate": 24000,
                "hop_length": 512,
            },
            "model_type": "MusicGenModel",
        }
        await repo.save_model("test_model", test_model_data)

        return repo

    @pytest.fixture
    def app_config(self):
        """Create test application configuration."""
        return AppConfig(
            model_cache_size=2,
            model_device="cpu",
            generation_timeout=30.0,
        )

    @pytest.fixture
    async def model_service(self, model_repository, app_config):
        """Create model service with repository."""
        service = ModelServiceImpl(model_repository, app_config)
        return service

    @pytest.fixture
    async def task_repository(self):
        """Create task repository for tracking generation tasks."""
        return InMemoryTaskRepository(max_tasks=100)

    @pytest.fixture
    async def audio_service(self):
        """Create audio processing service."""
        audio_repo = InMemoryAudioRepository()
        config = AppConfig()
        return AudioProcessingServiceImpl(audio_repo, config)

    @pytest.fixture
    async def generation_service(self, model_service, audio_service, task_repository):
        """Create generation service with dependencies."""
        return GenerationServiceImpl(model_service, audio_service, task_repository)

    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self, generation_service, model_service):
        """Test complete generation pipeline integration."""
        # Mock the actual model components that require external dependencies
        with patch.object(model_service, "load_model") as mock_load:
            mock_model = Mock()
            mock_model.config.sample_rate = 24000
            mock_model.config.hop_length = 512
            mock_model.generate.return_value = torch.randint(0, 2048, (1, 100))
            mock_model.encode_text.return_value = torch.randn(1, 10, 768)
            mock_model.decode_tokens.return_value = torch.randn(1, 24000)
            mock_load.return_value = mock_model

            # Create generation request
            request = GenerationRequest(
                prompt="Upbeat electronic dance music",
                duration=10.0,
                model_id="test_model",
                temperature=0.8,
            )

            # Execute generation
            result = await generation_service.generate(request)

            # Verify result
            assert isinstance(result, GenerationResult)
            assert result.duration == 10.0
            assert result.sample_rate == 24000
            assert isinstance(result.audio, torch.Tensor)

            # Verify service interactions
            mock_load.assert_called_once_with("test_model")
            mock_model.encode_text.assert_called_once()
            mock_model.generate.assert_called_once()
            mock_model.decode_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_caching(self, model_service):
        """Test model loading and caching integration."""
        with patch.object(model_service, "_load_from_repository") as mock_load_repo:
            # Mock model loading
            mock_model = Mock()
            mock_load_repo.return_value = mock_model

            # Load model twice
            model1 = await model_service.load_model("test_model")
            model2 = await model_service.load_model("test_model")

            # Should return same cached instance
            assert model1 is model2

            # Repository should only be called once
            mock_load_repo.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_tracking_integration(self, generation_service, task_repository):
        """Test task tracking throughout generation process."""
        with patch.object(generation_service._model_service, "load_model") as mock_load:
            mock_model = Mock()
            mock_model.config.sample_rate = 24000
            mock_model.generate.return_value = torch.randint(0, 2048, (1, 100))
            mock_model.encode_text.return_value = torch.randn(1, 10, 768)
            mock_model.decode_tokens.return_value = torch.randn(1, 24000)
            mock_load.return_value = mock_model

            request = GenerationRequest(
                prompt="Test music",
                duration=5.0,
            )

            # Execute generation
            result = await generation_service.generate(request)

            # Verify task was tracked
            tasks = await task_repository.list_tasks()
            assert len(tasks) == 1

            task = tasks[0]
            assert task["status"] == "completed"
            assert task["request"]["prompt"] == "Test music"
            assert "task_id" in result.metadata

    @pytest.mark.asyncio
    async def test_error_propagation(self, generation_service, model_service):
        """Test error propagation through service layers."""
        # Simulate model loading error
        with patch.object(model_service, "load_model") as mock_load:
            mock_load.side_effect = ModelLoadError("Model not found")

            request = GenerationRequest(prompt="Test music")

            with pytest.raises(GenerationError):
                await generation_service.generate(request)

    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(self, generation_service, model_service):
        """Test handling of concurrent generation requests."""
        with patch.object(model_service, "load_model") as mock_load:
            mock_model = Mock()
            mock_model.config.sample_rate = 24000
            mock_model.generate.return_value = torch.randint(0, 2048, (1, 50))
            mock_model.encode_text.return_value = torch.randn(1, 10, 768)
            mock_model.decode_tokens.return_value = torch.randn(1, 12000)
            mock_load.return_value = mock_model

            # Create multiple requests
            requests = [GenerationRequest(prompt=f"Music {i}", duration=5.0) for i in range(5)]

            # Execute concurrently
            tasks = [generation_service.generate(req) for req in requests]
            results = await asyncio.gather(*tasks)

            # Verify all succeeded
            assert len(results) == 5
            for result in results:
                assert isinstance(result, GenerationResult)
                assert result.duration == 5.0


@pytest.mark.integration
class TestAudioProcessingIntegration:
    """Test integration of audio processing with other services."""

    @pytest.fixture
    async def audio_repository(self):
        """Create audio repository."""
        return InMemoryAudioRepository()

    @pytest.fixture
    async def audio_service(self, audio_repository):
        """Create audio processing service."""
        config = AppConfig()
        return AudioProcessingServiceImpl(audio_repository, config)

    @pytest.mark.asyncio
    async def test_audio_generation_post_processing(self, audio_service):
        """Test audio post-processing integration."""
        # Create raw generated audio
        raw_audio = torch.randn(1, 24000) * 2.0  # Loud, unnormalized

        # Apply processing chain
        processing_operations = [
            {"type": "normalize", "params": {"method": "peak"}},
            {"type": "fade", "params": {"fade_in": 0.1, "fade_out": 0.1}},
        ]

        processed_audio = await audio_service.process_audio(
            raw_audio, sample_rate=24000, operations=processing_operations
        )

        # Verify processing was applied
        assert processed_audio.shape == raw_audio.shape
        assert processed_audio.abs().max() <= 1.0  # Normalized
        assert processed_audio[0, 0] == 0.0  # Fade in
        assert processed_audio[0, -1] == 0.0  # Fade out

    @pytest.mark.asyncio
    async def test_audio_storage_retrieval(self, audio_service, audio_repository):
        """Test audio storage and retrieval integration."""
        # Create test audio
        test_audio = torch.randn(1, 24000)

        # Save audio
        audio_id = "test_audio_001"
        file_path = await audio_repository.save_audio(
            audio_id, test_audio, sample_rate=24000, metadata={"source": "test"}
        )

        assert file_path is not None

        # Retrieve audio
        retrieved_audio, metadata = await audio_repository.load_audio(audio_id)

        assert torch.allclose(retrieved_audio, test_audio, atol=1e-6)
        assert metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_batch_audio_processing(self, audio_service):
        """Test batch audio processing integration."""
        # Create batch of audio
        batch_audio = [torch.randn(1, 24000) for _ in range(4)]

        # Process batch
        processing_operations = [
            {"type": "normalize", "params": {"method": "rms"}},
        ]

        processed_batch = []
        for audio in batch_audio:
            processed = await audio_service.process_audio(
                audio, sample_rate=24000, operations=processing_operations
            )
            processed_batch.append(processed)

        # Verify batch processing
        assert len(processed_batch) == 4
        for processed in processed_batch:
            rms = torch.sqrt(torch.mean(processed**2))
            assert torch.abs(rms - 1.0) < 0.1  # RMS normalized


@pytest.mark.integration
class TestRepositoryIntegration:
    """Test integration between different repository layers."""

    @pytest.fixture
    async def repositories(self):
        """Create all repository instances."""
        return {
            "task": InMemoryTaskRepository(max_tasks=100),
            "model": InMemoryModelRepository(),
            "audio": InMemoryAudioRepository(),
        }

    @pytest.mark.asyncio
    async def test_cross_repository_data_flow(self, repositories):
        """Test data flow across different repositories."""
        task_repo = repositories["task"]
        model_repo = repositories["model"]
        audio_repo = repositories["audio"]

        # 1. Save model
        model_data = {
            "state_dict": {"weight": torch.randn(10, 10)},
            "config": {"hidden_size": 512},
        }
        await model_repo.save_model("test_model", model_data)

        # 2. Create generation task
        task_id = "task_001"
        task_data = {
            "model_id": "test_model",
            "prompt": "Test music",
            "status": "pending",
        }
        await task_repo.create_task(task_id, task_data)

        # 3. Save generated audio
        generated_audio = torch.randn(1, 24000)
        audio_path = await audio_repo.save_audio(
            "audio_001", generated_audio, sample_rate=24000, metadata={"task_id": task_id}
        )

        # 4. Update task with audio reference
        await task_repo.update_task(
            task_id,
            {
                "status": "completed",
                "audio_path": audio_path,
            },
        )

        # Verify data consistency
        final_task = await task_repo.get_task(task_id)
        assert final_task["status"] == "completed"
        assert final_task["audio_path"] == audio_path

        retrieved_model = await model_repo.load_model("test_model")
        assert retrieved_model is not None

        retrieved_audio, metadata = await audio_repo.load_audio("audio_001")
        assert torch.equal(retrieved_audio, generated_audio)
        assert metadata["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_repository_cleanup_integration(self, repositories):
        """Test cleanup operations across repositories."""
        task_repo = repositories["task"]
        audio_repo = repositories["audio"]

        # Create multiple tasks and audio files
        task_ids = []
        audio_ids = []

        for i in range(10):
            task_id = f"task_{i:03d}"
            audio_id = f"audio_{i:03d}"

            # Create task
            await task_repo.create_task(
                task_id,
                {
                    "prompt": f"Music {i}",
                    "status": "completed" if i < 5 else "failed",
                },
            )
            task_ids.append(task_id)

            # Create audio
            audio = torch.randn(1, 12000)
            await audio_repo.save_audio(
                audio_id, audio, sample_rate=24000, metadata={"task_id": task_id}
            )
            audio_ids.append(audio_id)

        # Test cleanup of failed tasks
        await task_repo.cleanup_failed_tasks()

        # Verify only completed tasks remain
        remaining_tasks = await task_repo.list_tasks()
        assert len(remaining_tasks) == 5
        assert all(task["status"] == "completed" for task in remaining_tasks)

    @pytest.mark.asyncio
    async def test_repository_transaction_simulation(self, repositories):
        """Test transaction-like behavior across repositories."""
        task_repo = repositories["task"]
        audio_repo = repositories["audio"]

        task_id = "atomic_task_001"
        audio_id = "atomic_audio_001"

        try:
            # Simulate atomic operation
            await task_repo.create_task(
                task_id,
                {
                    "status": "processing",
                    "prompt": "Atomic test",
                },
            )

            # Simulate processing...
            audio = torch.randn(1, 24000)
            audio_path = await audio_repo.save_audio(audio_id, audio, sample_rate=24000)

            # Simulate error during final update
            if False:  # Simulate condition
                raise Exception("Simulated processing error")

            # Complete transaction
            await task_repo.update_task(
                task_id,
                {
                    "status": "completed",
                    "audio_path": audio_path,
                },
            )

        except Exception as e:
            # Cleanup on error
            await task_repo.update_task(
                task_id,
                {
                    "status": "failed",
                    "error": str(e),
                },
            )
            # In real implementation, would also cleanup audio

        # Verify final state
        final_task = await task_repo.get_task(task_id)
        assert final_task["status"] == "completed"


@pytest.mark.integration
class TestAPIServiceIntegration:
    """Test integration between API layer and services."""

    @pytest.fixture
    async def service_container(self):
        """Create container with all services."""
        # Create repositories
        task_repo = InMemoryTaskRepository()
        model_repo = InMemoryModelRepository()
        audio_repo = InMemoryAudioRepository()

        # Create services
        config = AppConfig()
        model_service = ModelServiceImpl(model_repo, config)
        audio_service = AudioProcessingServiceImpl(audio_repo, config)
        generation_service = GenerationServiceImpl(model_service, audio_service, task_repo)

        return {
            "generation": generation_service,
            "model": model_service,
            "audio": audio_service,
            "repositories": {
                "task": task_repo,
                "model": model_repo,
                "audio": audio_repo,
            },
        }

    @pytest.mark.asyncio
    async def test_api_generation_workflow(self, service_container):
        """Test complete API generation workflow."""
        generation_service = service_container["generation"]

        with patch.object(generation_service._model_service, "load_model") as mock_load:
            # Mock model
            mock_model = Mock()
            mock_model.config.sample_rate = 24000
            mock_model.generate.return_value = torch.randint(0, 2048, (1, 100))
            mock_model.encode_text.return_value = torch.randn(1, 10, 768)
            mock_model.decode_tokens.return_value = torch.randn(1, 24000)
            mock_load.return_value = mock_model

            # Simulate API request
            request = GenerationRequest(
                prompt="API test music",
                duration=15.0,
                temperature=0.7,
            )

            # Execute generation
            result = await generation_service.generate(request)

            # Verify API-compatible result
            assert isinstance(result, GenerationResult)
            assert "task_id" in result.metadata
            assert "generation_time" in result.metadata
            assert result.audio.shape[-1] > 0

    @pytest.mark.asyncio
    async def test_service_health_check(self, service_container):
        """Test health check across all services."""
        # Check model service
        models = await service_container["model"].list_available_models()
        assert isinstance(models, list)

        # Check repositories
        task_repo = service_container["repositories"]["task"]
        tasks = await task_repo.list_tasks()
        assert isinstance(tasks, list)

        # All services should be responsive
        assert True  # If we get here, services are healthy

    @pytest.mark.asyncio
    async def test_service_error_handling(self, service_container):
        """Test error handling across service layers."""
        generation_service = service_container["generation"]

        # Test invalid request
        with pytest.raises(Exception):  # Should catch validation error
            invalid_request = GenerationRequest(
                prompt="",  # Empty prompt
                duration=-1.0,  # Invalid duration
            )
            await generation_service.generate(invalid_request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
