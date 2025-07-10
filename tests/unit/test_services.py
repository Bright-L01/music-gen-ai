"""
Unit tests for service implementations.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import torch
from datetime import datetime

from music_gen.application.services import (
    ModelServiceImpl,
    GenerationServiceImpl,
    AudioProcessingServiceImpl,
    TrainingServiceImpl,
)
from music_gen.core.interfaces.services import (
    GenerationRequest,
    GenerationResult,
    TrainingConfig,
)
from music_gen.core.config import AppConfig
from music_gen.core.exceptions import (
    ModelLoadError,
    GenerationError,
    AudioProcessingError,
    TrainingError,
)


@pytest.mark.unit
class TestModelService:
    """Test model service implementation."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.exists = AsyncMock(return_value=False)
        repo.save_model = AsyncMock()
        repo.load_model = AsyncMock()
        repo.list_models = AsyncMock(return_value=[])
        repo.get_metadata = AsyncMock(return_value=None)
        repo.delete_model = AsyncMock()
        return repo

    @pytest.fixture
    def config(self):
        """Create test config."""
        return AppConfig(model_cache_size=2, model_device="cpu")

    @pytest.fixture
    def service(self, mock_repository, config):
        """Create service instance."""
        return ModelServiceImpl(mock_repository, config)

    @pytest.mark.asyncio
    async def test_load_model_from_cache(self, service):
        """Test loading model from cache."""
        # Manually add model to cache
        mock_model = Mock()
        service._model_cache["test_model"] = mock_model

        # Load should return cached model
        model = await service.load_model("test_model")
        assert model is mock_model

        # Repository should not be called
        service._repository.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_custom_model(self, service, mock_repository):
        """Test loading custom model from repository."""
        model_id = "custom_model"
        mock_repository.exists.return_value = True
        mock_repository.load_model.return_value = {
            "state_dict": {"weight": torch.randn(10, 10)},
            "config": {},
            "model_type": "MusicGenModel",
        }

        with patch("music_gen.application.services.model_service.MusicGenModel") as MockModel:
            mock_instance = Mock()
            MockModel.return_value = mock_instance

            model = await service.load_model(model_id)

            assert model is mock_instance
            mock_repository.load_model.assert_called_once_with(model_id)
            assert model_id in service._model_cache

    @pytest.mark.asyncio
    async def test_load_pretrained_model(self, service, mock_repository):
        """Test loading pre-trained model."""
        model_id = "facebook/musicgen-small"
        mock_repository.exists.return_value = False

        with patch("music_gen.application.services.model_service.MusicGenModel") as MockModel:
            mock_instance = Mock()
            MockModel.return_value = mock_instance

            model = await service.load_model(model_id)

            assert model is mock_instance
            mock_repository.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_model(self, service, mock_repository):
        """Test saving a model."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        mock_model.config = Mock(__dict__={"hidden_size": 768})
        mock_model.parameters.return_value = [torch.randn(1)]

        model_id = "test_model"
        metadata = {"version": "1.0"}

        await service.save_model(mock_model, model_id, metadata)

        mock_repository.save_model.assert_called_once()
        call_args = mock_repository.save_model.call_args
        assert call_args[0][0] == model_id
        assert "state_dict" in call_args[0][1]
        assert call_args[0][2]["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_list_available_models(self, service, mock_repository):
        """Test listing available models."""
        mock_repository.list_models.return_value = ["custom_model_1", "custom_model_2"]
        mock_repository.get_metadata.side_effect = [
            {"version": "1.0"},
            {"version": "2.0"},
        ]

        models = await service.list_available_models()

        # Should include custom and pre-trained models
        assert len(models) >= 5  # 2 custom + 3 pre-trained

        custom_models = [m for m in models if m["type"] == "custom"]
        assert len(custom_models) == 2

        pretrained_models = [m for m in models if m["type"] == "pretrained"]
        assert len(pretrained_models) >= 3

    @pytest.mark.asyncio
    async def test_cache_management(self, service, mock_repository):
        """Test cache size management."""
        # Fill cache to limit
        for i in range(3):  # One more than cache size
            service._model_cache[f"model_{i}"] = Mock()

        # Load new model should trigger cache cleanup
        await service._manage_cache_size()

        # Cache should be at limit
        assert len(service._model_cache) == service._config.model_cache_size


@pytest.mark.unit
class TestGenerationService:
    """Test generation service implementation."""

    @pytest.fixture
    def mock_model_service(self):
        """Create mock model service."""
        service = AsyncMock()
        mock_model = Mock()
        mock_model.config.sample_rate = 24000
        mock_model.config.hop_length = 512
        mock_model.generate.return_value = torch.randn(1, 48000)  # 2 seconds
        service.load_model.return_value = mock_model
        return service

    @pytest.fixture
    def mock_audio_service(self):
        """Create mock audio service."""
        service = AsyncMock()
        service.normalize_audio.side_effect = lambda x, **kwargs: x
        return service

    @pytest.fixture
    def mock_task_repository(self):
        """Create mock task repository."""
        repo = AsyncMock()
        repo.create_task = AsyncMock()
        repo.update_task = AsyncMock()
        return repo

    @pytest.fixture
    def service(self, mock_model_service, mock_audio_service, mock_task_repository):
        """Create service instance."""
        return GenerationServiceImpl(mock_model_service, mock_audio_service, mock_task_repository)

    @pytest.mark.asyncio
    async def test_generate_success(self, service, mock_task_repository):
        """Test successful generation."""
        request = GenerationRequest(
            prompt="Test music",
            duration=2.0,
            temperature=1.0,
        )

        result = await service.generate(request)

        assert isinstance(result, GenerationResult)
        assert result.duration == 2.0
        assert result.sample_rate == 24000
        assert isinstance(result.audio, torch.Tensor)

        # Task should be created and updated
        assert mock_task_repository.create_task.called
        assert mock_task_repository.update_task.called

        # Check status updates
        update_calls = mock_task_repository.update_task.call_args_list
        statuses = [call[0][1]["status"] for call in update_calls]
        assert "loading_model" in statuses
        assert "generating" in statuses
        assert "post_processing" in statuses
        assert "completed" in statuses

    @pytest.mark.asyncio
    async def test_generate_with_conditioning(self, service):
        """Test generation with conditioning."""
        request = GenerationRequest(
            prompt="Jazz music",
            duration=2.0,
        )
        conditioning = {
            "genre": "jazz",
            "mood": "upbeat",
            "tempo": 120,
        }

        result = await service.generate_with_conditioning(request, conditioning)

        assert isinstance(result, GenerationResult)
        assert result.metadata["parameters"]["conditioning"] == conditioning

    @pytest.mark.asyncio
    async def test_generate_failure(self, service, mock_model_service, mock_task_repository):
        """Test generation failure handling."""
        mock_model_service.load_model.side_effect = Exception("Model load failed")

        request = GenerationRequest(prompt="Test music")

        with pytest.raises(GenerationError):
            await service.generate(request)

        # Task should be marked as failed
        update_calls = mock_task_repository.update_task.call_args_list
        last_update = update_calls[-1][0][1]
        assert last_update["status"] == "failed"
        assert "error" in last_update

    @pytest.mark.asyncio
    async def test_generate_continuation(self, service, mock_model_service):
        """Test generating continuation of existing audio."""
        context_audio = torch.randn(1, 24000)  # 1 second context
        mock_model = mock_model_service.load_model.return_value
        mock_model.encode_audio = Mock(return_value=torch.randn(1, 10, 256))

        result = await service.generate_continuation(
            context_audio,
            duration=2.0,
            temperature=0.8,
        )

        assert isinstance(result, GenerationResult)
        assert result.audio.shape[-1] > context_audio.shape[-1]
        assert result.metadata["type"] == "continuation"

    @pytest.mark.asyncio
    async def test_get_supported_models(self, service, mock_model_service):
        """Test getting supported models."""
        mock_model_service.list_available_models.return_value = [
            {"id": "model_1"},
            {"id": "model_2"},
        ]

        models = await service.get_supported_models()

        assert models == ["model_1", "model_2"]


@pytest.mark.unit
class TestAudioProcessingService:
    """Test audio processing service implementation."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock audio repository."""
        return AsyncMock()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return AppConfig()

    @pytest.fixture
    def service(self, mock_repository, config):
        """Create service instance."""
        return AudioProcessingServiceImpl(mock_repository, config)

    @pytest.mark.asyncio
    async def test_process_audio_chain(self, service):
        """Test processing audio with operation chain."""
        audio = torch.randn(1, 24000)
        sample_rate = 24000

        operations = [
            {"type": "normalize", "params": {"method": "peak"}},
            {"type": "fade", "params": {"fade_in": 0.1, "fade_out": 0.1}},
        ]

        result = await service.process_audio(audio, sample_rate, operations)

        assert isinstance(result, torch.Tensor)
        assert result.shape == audio.shape

        # Check normalization (peak should be 1.0)
        assert result.abs().max() <= 1.0

        # Check fade (start and end should be 0)
        assert result[0, 0] == 0.0
        assert result[0, -1] == 0.0

    @pytest.mark.asyncio
    async def test_normalize_audio_methods(self, service):
        """Test different normalization methods."""
        audio = torch.randn(1, 24000) * 0.5

        # Peak normalization
        peak_norm = await service.normalize_audio(audio, "peak")
        assert peak_norm.abs().max() == pytest.approx(1.0, rel=1e-6)

        # RMS normalization
        rms_norm = await service.normalize_audio(audio, "rms")
        rms = torch.sqrt(torch.mean(rms_norm**2))
        assert rms == pytest.approx(1.0, rel=1e-6)

        # LUFS normalization (simplified)
        lufs_norm = await service.normalize_audio(audio, "lufs")
        assert isinstance(lufs_norm, torch.Tensor)

    @pytest.mark.asyncio
    async def test_resample_audio(self, service):
        """Test audio resampling."""
        audio = torch.randn(1, 24000)

        # Resample from 24kHz to 48kHz
        resampled = await service.resample_audio(audio, 24000, 48000)
        assert resampled.shape[-1] == 48000

        # No resampling if same rate
        same = await service.resample_audio(audio, 24000, 24000)
        assert torch.equal(same, audio)

    @pytest.mark.asyncio
    async def test_mix_tracks(self, service):
        """Test mixing multiple tracks."""
        track1 = torch.ones(1, 24000) * 0.5
        track2 = torch.ones(1, 24000) * 0.3
        track3 = torch.ones(1, 20000) * 0.4  # Shorter track

        # Mix with equal weights
        mixed = await service.mix_tracks([track1, track2, track3])

        assert mixed.shape[-1] == 24000  # Length of longest track

        # Mix with custom weights
        mixed_weighted = await service.mix_tracks([track1, track2], weights=[0.7, 0.3])
        assert isinstance(mixed_weighted, torch.Tensor)

    @pytest.mark.asyncio
    async def test_apply_effects(self, service):
        """Test applying audio effects."""
        audio = torch.randn(1, 24000)

        effects = [
            {"type": "reverb", "params": {"room_size": 0.8}},
            {"type": "compression", "params": {"threshold": -10}},
        ]

        result = await service.apply_effects(audio, effects)

        assert isinstance(result, torch.Tensor)
        assert result.shape == audio.shape

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling in audio processing."""
        audio = torch.randn(1, 24000)

        # Invalid normalization method
        with pytest.raises(AudioProcessingError):
            await service.normalize_audio(audio, "invalid_method")

        # Empty track list for mixing
        with pytest.raises(ValueError):
            await service.mix_tracks([])


@pytest.mark.unit
class TestTrainingService:
    """Test training service implementation."""

    @pytest.fixture
    def mock_model_service(self):
        """Create mock model service."""
        service = AsyncMock()
        mock_model = Mock()
        service.load_model.return_value = mock_model
        service.save_model = AsyncMock()
        return service

    @pytest.fixture
    def mock_metadata_repository(self):
        """Create mock metadata repository."""
        repo = AsyncMock()
        repo.load_metadata.return_value = {
            "path": "/data/musiccaps",
            "samples": 1000,
        }
        return repo

    @pytest.fixture
    def config(self):
        """Create test config."""
        return AppConfig()

    @pytest.fixture
    def service(self, mock_model_service, mock_metadata_repository, config):
        """Create service instance."""
        return TrainingServiceImpl(mock_model_service, mock_metadata_repository, config)

    @pytest.mark.asyncio
    async def test_train_model(self, service, mock_model_service):
        """Test model training."""
        model_id = "test_model"
        dataset_id = "test_dataset"
        config = TrainingConfig(
            batch_size=16,
            num_epochs=2,
        )

        with patch("music_gen.application.services.training_service.MusicDataModule"):
            with patch("music_gen.application.services.training_service.MusicGenLightningModule"):
                with patch(
                    "music_gen.application.services.training_service.pl.Trainer"
                ) as MockTrainer:
                    mock_trainer = Mock()
                    mock_trainer.current_epoch = 2
                    mock_trainer.callback_metrics = {"train_loss": 0.5}
                    MockTrainer.return_value = mock_trainer

                    result = await service.train_model(model_id, dataset_id, config)

                    assert "trained_model_id" in result
                    assert result["epochs_trained"] == 2
                    assert result["final_loss"] == 0.5

                    # Model should be saved
                    mock_model_service.save_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_training_status(self, service):
        """Test getting training job status."""
        # Create a job
        job_id = "test_job"
        service._training_jobs[job_id] = {
            "id": job_id,
            "status": "training",
            "progress": 0.5,
        }

        status = await service.get_training_status(job_id)

        assert status["id"] == job_id
        assert status["status"] == "training"
        assert status["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_stop_training(self, service):
        """Test stopping a training job."""
        job_id = "test_job"
        service._training_jobs[job_id] = {
            "id": job_id,
            "status": "training",
        }

        await service.stop_training(job_id)

        assert service._training_jobs[job_id]["status"] == "stopped"
        assert "stopped_at" in service._training_jobs[job_id]

    @pytest.mark.asyncio
    async def test_list_training_jobs(self, service):
        """Test listing training jobs."""
        # Create multiple jobs
        for i in range(3):
            service._training_jobs[f"job_{i}"] = {
                "id": f"job_{i}",
                "status": "completed" if i < 2 else "failed",
                "started_at": datetime.utcnow().isoformat(),
            }

        # List all jobs
        all_jobs = await service.list_training_jobs()
        assert len(all_jobs) == 3

        # List by status
        completed_jobs = await service.list_training_jobs(status="completed")
        assert len(completed_jobs) == 2

    @pytest.mark.asyncio
    async def test_fine_tune_model(self, service):
        """Test model fine-tuning."""
        with patch.object(service, "train_model") as mock_train:
            mock_train.return_value = {"trained_model_id": "fine_tuned_model"}

            model_id = await service.fine_tune_model("base_model", "dataset", TrainingConfig())

            assert model_id == "fine_tuned_model"

            # Should use lower learning rate
            call_args = mock_train.call_args[0]
            assert call_args[2].learning_rate < TrainingConfig().learning_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
