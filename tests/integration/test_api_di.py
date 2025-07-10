"""
Integration tests for API endpoints with dependency injection.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
import pytest
import torch
from fastapi.testclient import TestClient
from injector import Injector, Module, provider, singleton

from music_gen.api.endpoints.generation_refactored import router as generation_router
from music_gen.core.interfaces.services import (
    GenerationService,
    GenerationRequest,
    GenerationResult,
)
from music_gen.core.interfaces.repositories import (
    TaskRepository,
    AudioRepository,
)
from music_gen.core.config import AppConfig
from music_gen.core.container import get_container, Container


class MockModule(Module):
    """Mock dependency injection module for testing."""

    @singleton
    @provider
    def provide_generation_service(self) -> GenerationService:
        """Provide mock generation service."""
        service = AsyncMock(spec=GenerationService)

        # Mock successful generation
        async def mock_generate(request):
            return GenerationResult(
                audio=torch.randn(1, 48000),
                sample_rate=24000,
                duration=2.0,
                metadata={
                    "prompt": request.prompt,
                    "model": "test_model",
                },
            )

        service.generate.side_effect = mock_generate
        service.get_supported_models.return_value = ["test_model"]
        return service

    @singleton
    @provider
    def provide_task_repository(self) -> TaskRepository:
        """Provide mock task repository."""
        repo = AsyncMock(spec=TaskRepository)
        repo._tasks = {}  # In-memory storage for testing

        async def create_task(task_id, data):
            repo._tasks[task_id] = data

        async def get_task(task_id):
            return repo._tasks.get(task_id)

        async def update_task(task_id, updates):
            if task_id in repo._tasks:
                repo._tasks[task_id].update(updates)

        async def list_tasks(**kwargs):
            return list(repo._tasks.values())

        repo.create_task.side_effect = create_task
        repo.get_task.side_effect = get_task
        repo.update_task.side_effect = update_task
        repo.list_tasks.side_effect = list_tasks

        return repo

    @singleton
    @provider
    def provide_audio_repository(self) -> AudioRepository:
        """Provide mock audio repository."""
        repo = AsyncMock(spec=AudioRepository)
        repo.save_audio.return_value = "/tmp/test_audio.wav"
        return repo


@pytest.fixture
def test_app():
    """Create test FastAPI app with mocked dependencies."""
    from fastapi import FastAPI

    # Reset container and inject mocks
    Container.reset()
    container = Injector([MockModule()])
    Container._instance = container

    # Create app
    app = FastAPI()
    app.include_router(generation_router, prefix="/api/v1/generate")

    yield app

    # Reset container after test
    Container.reset()


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.mark.integration
class TestGenerationAPIWithDI:
    """Test generation API with dependency injection."""

    def test_generate_music(self, client):
        """Test music generation endpoint."""
        request_data = {
            "prompt": "Upbeat jazz music",
            "duration": 10.0,
            "temperature": 0.8,
        }

        response = client.post("/api/v1/generate/", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_get_generation_status(self, client):
        """Test getting generation status."""
        # First create a generation
        response = client.post("/api/v1/generate/", json={"prompt": "Test music"})
        task_id = response.json()["task_id"]

        # Get status
        response = client.get(f"/api/v1/generate/{task_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] in ["pending", "processing", "completed", "failed"]

    def test_get_nonexistent_task(self, client):
        """Test getting status of non-existent task."""
        response = client.get("/api/v1/generate/nonexistent_task_id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"

    def test_batch_generation(self, client):
        """Test batch generation endpoint."""
        batch_request = {
            "requests": [
                {"prompt": "Jazz music", "duration": 10.0},
                {"prompt": "Rock music", "duration": 15.0},
                {"prompt": "Classical music", "duration": 20.0},
            ]
        }

        response = client.post("/api/v1/generate/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert "task_ids" in data
        assert len(data["task_ids"]) == 3
        assert data["status"] == "pending"

    def test_get_batch_status(self, client):
        """Test getting batch generation status."""
        # Create batch
        batch_request = {
            "requests": [
                {"prompt": "Music 1"},
                {"prompt": "Music 2"},
            ]
        }
        response = client.post("/api/v1/generate/batch", json=batch_request)
        batch_id = response.json()["batch_id"]

        # Get batch status
        response = client.get(f"/api/v1/generate/batch/{batch_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["total"] == 2
        assert "completed" in data
        assert "failed" in data
        assert "pending" in data

    def test_generation_with_all_parameters(self, client):
        """Test generation with all optional parameters."""
        request_data = {
            "prompt": "Epic orchestral music",
            "duration": 30.0,
            "temperature": 1.2,
            "top_k": 100,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "num_beams": 4,
            "guidance_scale": 4.0,
            "genre": "orchestral",
            "mood": "epic",
            "tempo": 140,
            "instruments": ["violin", "cello", "timpani"],
            "seed": 42,
        }

        response = client.post("/api/v1/generate/", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_generation_validation(self, client):
        """Test request validation."""
        # Missing required field
        response = client.post("/api/v1/generate/", json={})
        assert response.status_code == 422

        # Invalid duration
        response = client.post(
            "/api/v1/generate/",
            json={
                "prompt": "Test",
                "duration": 500.0,  # Too long
            },
        )
        assert response.status_code == 422

        # Invalid temperature
        response = client.post(
            "/api/v1/generate/",
            json={
                "prompt": "Test",
                "temperature": 3.0,  # Too high
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_background_task_execution(self, test_app):
        """Test that background tasks execute properly."""
        from music_gen.api.endpoints.generation_refactored import generate_music_task

        # Get dependencies
        container = get_container()
        gen_service = container.get(GenerationService)
        task_repo = container.get(TaskRepository)

        # Create task
        task_id = "test_task"
        await task_repo.create_task(task_id, {"status": "pending"})

        # Run generation task
        from music_gen.api.endpoints.generation_refactored import GenerationRequest as APIRequest

        request = APIRequest(prompt="Test music", duration=10.0)

        await generate_music_task(task_id, request, gen_service, task_repo)

        # Check task was updated
        task = await task_repo.get_task(task_id)
        assert task["status"] == "completed"
        assert "audio_path" in task
        assert "metadata" in task

    def test_dependency_injection_isolation(self, client):
        """Test that DI properly isolates dependencies."""
        # Get the injected service
        container = get_container()
        gen_service = container.get(GenerationService)

        # Verify it's our mock
        assert isinstance(gen_service, AsyncMock)

        # Make request
        response = client.post("/api/v1/generate/", json={"prompt": "Test"})
        assert response.status_code == 200

        # Service should have been called
        assert gen_service.generate.called


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling with DI."""

    @pytest.fixture
    def failing_app(self):
        """Create app with failing services."""
        from fastapi import FastAPI

        # Create module with failing services
        class FailingModule(Module):
            @singleton
            @provider
            def provide_generation_service(self) -> GenerationService:
                service = AsyncMock(spec=GenerationService)
                service.generate.side_effect = Exception("Generation failed")
                return service

            @singleton
            @provider
            def provide_task_repository(self) -> TaskRepository:
                return AsyncMock(spec=TaskRepository)

            @singleton
            @provider
            def provide_audio_repository(self) -> AudioRepository:
                return AsyncMock(spec=AudioRepository)

        # Reset and configure container
        Container.reset()
        container = Injector([FailingModule()])
        Container._instance = container

        # Create app
        app = FastAPI()
        app.include_router(generation_router, prefix="/api/v1/generate")

        yield app

        Container.reset()

    def test_service_failure_handling(self, failing_app):
        """Test handling of service failures."""
        client = TestClient(failing_app)

        # This should trigger the background task but not fail the request
        response = client.post("/api/v1/generate/", json={"prompt": "Test"})
        assert response.status_code == 200  # Request succeeds

        # The task will fail in the background
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"


@pytest.mark.integration
class TestAPIPerformance:
    """Test API performance characteristics."""

    def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import concurrent.futures

        def make_request(i):
            return client.post(
                "/api/v1/generate/",
                json={
                    "prompt": f"Test music {i}",
                    "duration": 10.0,
                },
            )

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique task IDs
        task_ids = [r.json()["task_id"] for r in responses]
        assert len(set(task_ids)) == 10

    def test_large_batch_request(self, client):
        """Test handling large batch requests."""
        # Maximum allowed batch size
        batch_request = {"requests": [{"prompt": f"Music {i}", "duration": 10.0} for i in range(5)]}

        response = client.post("/api/v1/generate/batch", json=batch_request)
        assert response.status_code == 200

        # Too many requests
        large_batch = {
            "requests": [
                {"prompt": f"Music {i}", "duration": 10.0} for i in range(10)  # More than max
            ]
        }

        response = client.post("/api/v1/generate/batch", json=large_batch)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
