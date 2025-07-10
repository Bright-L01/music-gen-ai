"""
Unit tests for repository implementations.
"""

import asyncio
import json
from pathlib import Path
import tempfile
import shutil
import pytest
import torch
from datetime import datetime

from music_gen.infrastructure.repositories import (
    FileSystemModelRepository,
    InMemoryTaskRepository,
    FileSystemMetadataRepository,
    FileSystemAudioRepository,
)
from music_gen.core.exceptions import (
    ModelNotFoundError,
    TaskNotFoundError,
    MetadataNotFoundError,
    AudioNotFoundError,
)


@pytest.mark.unit
class TestFileSystemModelRepository:
    """Test file system model repository."""

    @pytest.fixture
    async def repository(self):
        """Create repository with temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = FileSystemModelRepository(Path(tmp_dir))
            yield repo

    @pytest.mark.asyncio
    async def test_save_and_load_model(self, repository):
        """Test saving and loading a model."""
        model_id = "test_model"
        model_state = {
            "state_dict": {"weight": torch.randn(10, 10)},
            "config": {"hidden_size": 768},
        }
        metadata = {"version": "1.0", "trained_on": "test_data"}

        # Save model
        await repository.save_model(model_id, model_state, metadata)

        # Check if exists
        assert await repository.exists(model_id)

        # Load model
        loaded_state = await repository.load_model(model_id)
        assert "state_dict" in loaded_state
        assert torch.allclose(
            loaded_state["state_dict"]["weight"], model_state["state_dict"]["weight"]
        )

        # Load metadata
        loaded_metadata = await repository.get_metadata(model_id)
        assert loaded_metadata == metadata

    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, repository):
        """Test loading a model that doesn't exist."""
        with pytest.raises(ModelNotFoundError):
            await repository.load_model("nonexistent_model")

    @pytest.mark.asyncio
    async def test_delete_model(self, repository):
        """Test deleting a model."""
        model_id = "test_model"
        model_state = {"state_dict": {}}

        await repository.save_model(model_id, model_state)
        assert await repository.exists(model_id)

        await repository.delete_model(model_id)
        assert not await repository.exists(model_id)

    @pytest.mark.asyncio
    async def test_list_models(self, repository):
        """Test listing models."""
        # Save multiple models
        for i in range(3):
            await repository.save_model(f"model_{i}", {"state_dict": {}}, {"index": i})

        models = await repository.list_models()
        assert len(models) == 3
        assert all(f"model_{i}" in models for i in range(3))

    @pytest.mark.asyncio
    async def test_special_characters_in_model_id(self, repository):
        """Test handling special characters in model ID."""
        model_id = "facebook/musicgen-small"
        model_state = {"state_dict": {}}

        await repository.save_model(model_id, model_state)
        assert await repository.exists(model_id)

        loaded = await repository.load_model(model_id)
        assert loaded["state_dict"] == model_state["state_dict"]


@pytest.mark.unit
class TestInMemoryTaskRepository:
    """Test in-memory task repository."""

    @pytest.fixture
    async def repository(self):
        """Create repository instance."""
        return InMemoryTaskRepository(max_tasks=10)

    @pytest.mark.asyncio
    async def test_create_and_get_task(self, repository):
        """Test creating and retrieving a task."""
        task_id = "task_1"
        task_data = {
            "status": "pending",
            "prompt": "Test music",
        }

        await repository.create_task(task_id, task_data)

        retrieved = await repository.get_task(task_id)
        assert retrieved is not None
        assert retrieved["status"] == "pending"
        assert retrieved["prompt"] == "Test music"
        assert "created_at" in retrieved

    @pytest.mark.asyncio
    async def test_update_task(self, repository):
        """Test updating a task."""
        task_id = "task_1"
        await repository.create_task(task_id, {"status": "pending"})

        await repository.update_task(task_id, {"status": "completed", "result": "success"})

        task = await repository.get_task(task_id)
        assert task["status"] == "completed"
        assert task["result"] == "success"
        assert "updated_at" in task

    @pytest.mark.asyncio
    async def test_update_nonexistent_task(self, repository):
        """Test updating a task that doesn't exist."""
        with pytest.raises(TaskNotFoundError):
            await repository.update_task("nonexistent", {"status": "completed"})

    @pytest.mark.asyncio
    async def test_delete_task(self, repository):
        """Test deleting a task."""
        task_id = "task_1"
        await repository.create_task(task_id, {"status": "pending"})

        await repository.delete_task(task_id)

        task = await repository.get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_list_tasks_with_filter(self, repository):
        """Test listing tasks with status filter."""
        # Create tasks with different statuses
        await repository.create_task("task_1", {"status": "pending"})
        await repository.create_task("task_2", {"status": "completed"})
        await repository.create_task("task_3", {"status": "pending"})
        await repository.create_task("task_4", {"status": "failed"})

        # List all tasks
        all_tasks = await repository.list_tasks()
        assert len(all_tasks) == 4

        # List pending tasks
        pending_tasks = await repository.list_tasks(status="pending")
        assert len(pending_tasks) == 2

        # List with pagination
        paginated = await repository.list_tasks(limit=2, offset=1)
        assert len(paginated) == 2

    @pytest.mark.asyncio
    async def test_max_tasks_limit(self, repository):
        """Test that max tasks limit is enforced."""
        # Create more tasks than the limit
        for i in range(15):
            await repository.create_task(f"task_{i}", {"index": i})

        # Should only keep the last 10 tasks
        all_tasks = await repository.list_tasks()
        assert len(all_tasks) == 10

        # Oldest tasks should be removed
        assert await repository.get_task("task_0") is None
        assert await repository.get_task("task_14") is not None


@pytest.mark.unit
class TestFileSystemMetadataRepository:
    """Test file system metadata repository."""

    @pytest.fixture
    async def repository(self):
        """Create repository with temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = FileSystemMetadataRepository(Path(tmp_dir))
            yield repo

    @pytest.mark.asyncio
    async def test_save_and_load_metadata(self, repository):
        """Test saving and loading metadata."""
        dataset_id = "test_dataset"
        metadata = {
            "name": "Test Dataset",
            "samples": 1000,
            "duration": 3600.0,
            "tags": ["music", "test"],
        }

        await repository.save_metadata(dataset_id, metadata)

        loaded = await repository.load_metadata(dataset_id)
        assert loaded["name"] == "Test Dataset"
        assert loaded["samples"] == 1000
        assert loaded["dataset_id"] == dataset_id
        assert "saved_at" in loaded

    @pytest.mark.asyncio
    async def test_update_metadata(self, repository):
        """Test updating metadata."""
        dataset_id = "test_dataset"
        await repository.save_metadata(dataset_id, {"samples": 1000})

        await repository.update_metadata(dataset_id, {"samples": 2000, "verified": True})

        loaded = await repository.load_metadata(dataset_id)
        assert loaded["samples"] == 2000
        assert loaded["verified"] is True
        assert "updated_at" in loaded

    @pytest.mark.asyncio
    async def test_search_metadata(self, repository):
        """Test searching metadata."""
        # Save multiple datasets
        await repository.save_metadata(
            "dataset_1",
            {
                "samples": 1000,
                "genre": "rock",
                "quality": "high",
            },
        )
        await repository.save_metadata(
            "dataset_2",
            {
                "samples": 2000,
                "genre": "jazz",
                "quality": "high",
            },
        )
        await repository.save_metadata(
            "dataset_3",
            {
                "samples": 1500,
                "genre": "rock",
                "quality": "medium",
            },
        )

        # Search by exact match
        results = await repository.search_metadata({"genre": "rock"})
        assert len(results) == 2

        # Search by multiple criteria
        results = await repository.search_metadata(
            {
                "genre": "rock",
                "quality": "high",
            }
        )
        assert len(results) == 1

        # Search with range query
        results = await repository.search_metadata({"samples": {"$gte": 1500}})
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_metadata(self, repository):
        """Test deleting metadata."""
        dataset_id = "test_dataset"
        await repository.save_metadata(dataset_id, {"name": "Test"})

        await repository.delete_metadata(dataset_id)

        with pytest.raises(MetadataNotFoundError):
            await repository.load_metadata(dataset_id)


@pytest.mark.unit
class TestFileSystemAudioRepository:
    """Test file system audio repository."""

    @pytest.fixture
    async def repository(self):
        """Create repository with temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = FileSystemAudioRepository(Path(tmp_dir))
            yield repo

    @pytest.mark.asyncio
    async def test_save_and_load_audio(self, repository):
        """Test saving and loading audio."""
        audio_id = "test_audio"
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        # Create test audio
        audio_data = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, samples))
        audio_data = audio_data.unsqueeze(0)  # Add channel dimension

        # Save audio
        path = await repository.save_audio(audio_id, audio_data, sample_rate)
        assert path.endswith(".wav")

        # Check existence
        assert await repository.exists(audio_id)

        # Load audio
        loaded_audio, loaded_sr = await repository.load_audio(audio_id)
        assert loaded_sr == sample_rate
        assert loaded_audio.shape == audio_data.shape

        # Audio should be approximately equal (allowing for encoding/decoding)
        assert torch.allclose(loaded_audio, audio_data, atol=1e-3)

    @pytest.mark.asyncio
    async def test_load_nonexistent_audio(self, repository):
        """Test loading audio that doesn't exist."""
        with pytest.raises(AudioNotFoundError):
            await repository.load_audio("nonexistent_audio")

    @pytest.mark.asyncio
    async def test_delete_audio(self, repository):
        """Test deleting audio."""
        audio_id = "test_audio"
        audio_data = torch.randn(1, 24000)

        await repository.save_audio(audio_id, audio_data, 24000)
        assert await repository.exists(audio_id)

        await repository.delete_audio(audio_id)
        assert not await repository.exists(audio_id)

    @pytest.mark.asyncio
    async def test_get_audio_url(self, repository):
        """Test getting audio URL."""
        audio_id = "test_audio"
        audio_data = torch.randn(1, 24000)

        await repository.save_audio(audio_id, audio_data, 24000)

        url = await repository.get_audio_url(audio_id)
        assert url.startswith("file://")
        assert audio_id.replace("/", "_") in url

    @pytest.mark.asyncio
    async def test_list_audio(self, repository):
        """Test listing audio files."""
        # Save multiple audio files
        for i in range(5):
            await repository.save_audio(f"audio_{i}", torch.randn(1, 24000), 24000)

        # List all
        audio_list = await repository.list_audio()
        assert len(audio_list) == 5

        # List with prefix
        audio_list = await repository.list_audio(prefix="audio_")
        assert len(audio_list) == 5

        # List with limit
        audio_list = await repository.list_audio(limit=3)
        assert len(audio_list) == 3

    @pytest.mark.asyncio
    async def test_handle_different_audio_shapes(self, repository):
        """Test handling different audio tensor shapes."""
        audio_id = "test_audio"
        sample_rate = 24000

        # Test 1D audio
        audio_1d = torch.randn(24000)
        await repository.save_audio(audio_id, audio_1d, sample_rate)
        loaded, _ = await repository.load_audio(audio_id)
        assert loaded.dim() == 2  # Should have channel dimension

        # Test 3D audio (batch dimension)
        await repository.delete_audio(audio_id)
        audio_3d = torch.randn(1, 2, 24000)  # batch, channels, samples
        await repository.save_audio(audio_id, audio_3d, sample_rate)
        loaded, _ = await repository.load_audio(audio_id)
        assert loaded.shape == (2, 24000)  # Batch removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
