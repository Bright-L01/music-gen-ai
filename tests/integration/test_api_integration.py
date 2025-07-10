"""
Integration tests for API endpoints and workflows.

Tests the complete API integration including request handling,
service orchestration, response formatting, and error handling.
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import httpx

from music_gen.api.main import create_app
from music_gen.api.endpoints import generation, monitoring
from music_gen.core.container import Container
from music_gen.infrastructure.repositories import (
    InMemoryTaskRepository,
    InMemoryModelRepository,
    InMemoryAudioRepository,
)
from music_gen.application.services import (
    GenerationServiceImpl,
    ModelServiceImpl,
    AudioProcessingServiceImpl,
)
from music_gen.core.config import AppConfig


@pytest.mark.integration
class TestGenerationAPIIntegration:
    """Test generation API integration."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application."""
        # Reset container
        Container.reset()

        # Create test repositories
        task_repo = InMemoryTaskRepository()
        model_repo = InMemoryModelRepository()
        audio_repo = InMemoryAudioRepository()

        # Create test services
        config = AppConfig()
        model_service = ModelServiceImpl(model_repo, config)
        audio_service = AudioProcessingServiceImpl(audio_repo, config)
        generation_service = GenerationServiceImpl(model_service, audio_service, task_repo)

        # Mock container to return test services
        container_mock = Mock()
        container_mock.get.side_effect = lambda service_type: {
            "GenerationService": generation_service,
            "ModelService": model_service,
            "AudioService": audio_service,
        }.get(service_type.__name__, Mock())

        Container._instance = container_mock

        # Create app
        app = create_app()

        yield app

        # Cleanup
        Container.reset()

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_generate_music_endpoint(self, client):
        """Test music generation endpoint."""
        # Mock the generation service
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()

            # Mock generation result
            from music_gen.core.interfaces.services import GenerationResult
            import torch

            mock_result = GenerationResult(
                audio=torch.randn(1, 24000),
                sample_rate=24000,
                duration=10.0,
                metadata={
                    "task_id": "test_task_123",
                    "prompt": "Test music",
                    "generation_time": {"total_seconds": 5.2},
                },
            )
            mock_service.generate.return_value = mock_result
            mock_get_service.return_value = mock_service

            # Make request
            request_data = {
                "prompt": "Upbeat jazz music",
                "duration": 10.0,
                "temperature": 0.8,
            }

            response = client.post("/api/v1/generate/", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert "task_id" in data
            assert data["status"] == "completed"
            assert data["duration"] == 10.0
            assert "audio_url" in data

    def test_generate_music_validation_error(self, client):
        """Test generation with validation errors."""
        # Invalid request data
        invalid_requests = [
            {},  # Missing prompt
            {"prompt": ""},  # Empty prompt
            {"prompt": "Test", "duration": -1},  # Invalid duration
            {"prompt": "Test", "temperature": 2.0},  # Invalid temperature
        ]

        for invalid_request in invalid_requests:
            response = client.post("/api/v1/generate/", json=invalid_request)
            assert response.status_code == 422  # Validation error

    def test_generate_music_with_conditioning(self, client):
        """Test generation with conditioning parameters."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_with_conditioning = AsyncMock()

            from music_gen.core.interfaces.services import GenerationResult
            import torch

            mock_result = GenerationResult(
                audio=torch.randn(1, 24000),
                sample_rate=24000,
                duration=15.0,
                metadata={
                    "task_id": "test_task_456",
                    "conditioning": {"genre": "jazz", "mood": "upbeat"},
                },
            )
            mock_service.generate_with_conditioning.return_value = mock_result
            mock_get_service.return_value = mock_service

            request_data = {
                "prompt": "Jazz music",
                "duration": 15.0,
                "conditioning": {"genre": "jazz", "mood": "upbeat", "tempo": 120},
            }

            response = client.post("/api/v1/generate/conditional", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["conditioning"]["genre"] == "jazz"

    def test_batch_generation_endpoint(self, client):
        """Test batch generation endpoint."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()

            # Mock batch results
            from music_gen.core.interfaces.services import GenerationResult
            import torch

            batch_results = [
                GenerationResult(
                    audio=torch.randn(1, 12000),
                    sample_rate=24000,
                    duration=5.0,
                    metadata={"task_id": f"batch_task_{i}"},
                )
                for i in range(3)
            ]
            mock_service.generate_batch = AsyncMock(return_value=batch_results)
            mock_get_service.return_value = mock_service

            request_data = {
                "requests": [
                    {"prompt": "Jazz music", "duration": 5.0},
                    {"prompt": "Rock music", "duration": 5.0},
                    {"prompt": "Classical music", "duration": 5.0},
                ]
            }

            response = client.post("/api/v1/generate/batch", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 3

    def test_get_generation_status(self, client):
        """Test getting generation status."""
        with patch("music_gen.api.endpoints.generation.get_task_repository") as mock_get_repo:
            mock_repo = AsyncMock()
            task_data = {
                "id": "test_task_789",
                "status": "completed",
                "progress": 1.0,
                "request": {"prompt": "Test music"},
                "result": {"duration": 10.0, "audio_url": "/audio/test.wav"},
            }
            mock_repo.get_task.return_value = task_data
            mock_get_repo.return_value = mock_repo

            response = client.get("/api/v1/generate/test_task_789")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test_task_789"
            assert data["status"] == "completed"

    def test_list_generation_tasks(self, client):
        """Test listing generation tasks."""
        with patch("music_gen.api.endpoints.generation.get_task_repository") as mock_get_repo:
            mock_repo = AsyncMock()
            tasks = [
                {
                    "id": f"task_{i}",
                    "status": "completed" if i % 2 == 0 else "processing",
                    "created_at": "2024-01-01T00:00:00Z",
                }
                for i in range(5)
            ]
            mock_repo.list_tasks.return_value = tasks
            mock_get_repo.return_value = mock_repo

            response = client.get("/api/v1/generate/")

            assert response.status_code == 200
            data = response.json()
            assert len(data["tasks"]) == 5

    def test_cancel_generation_task(self, client):
        """Test canceling a generation task."""
        with patch("music_gen.api.endpoints.generation.get_task_repository") as mock_get_repo:
            mock_repo = AsyncMock()
            mock_repo.update_task = AsyncMock()
            mock_get_repo.return_value = mock_repo

            response = client.delete("/api/v1/generate/test_task_cancel")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"


@pytest.mark.integration
class TestMonitoringAPIIntegration:
    """Test monitoring and resource management API integration."""

    @pytest.fixture
    def monitoring_client(self, test_app):
        """Create client for monitoring endpoints."""
        return TestClient(test_app)

    def test_system_status_endpoint(self, monitoring_client):
        """Test system status monitoring."""
        with patch("music_gen.api.endpoints.monitoring.get_resource_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_system_status.return_value = {
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 67.8,
                "gpu_usage_percent": 23.1,
                "disk_usage_percent": 34.5,
                "active_tasks": 3,
                "queue_size": 2,
            }
            mock_get_manager.return_value = mock_manager

            response = monitoring_client.get("/api/v1/monitoring/status")

            assert response.status_code == 200
            data = response.json()
            assert "cpu_usage_percent" in data
            assert "memory_usage_percent" in data
            assert "active_tasks" in data

    def test_resource_usage_endpoint(self, monitoring_client):
        """Test resource usage monitoring."""
        with patch("music_gen.api.endpoints.monitoring.get_resource_monitor") as mock_get_monitor:
            mock_monitor = Mock()
            mock_monitor.get_usage_history.return_value = {
                "timestamps": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
                "cpu_usage": [45.2, 47.1],
                "memory_usage": [67.8, 69.2],
                "gpu_usage": [23.1, 25.4],
            }
            mock_get_monitor.return_value = mock_monitor

            response = monitoring_client.get("/api/v1/monitoring/resources")

            assert response.status_code == 200
            data = response.json()
            assert "timestamps" in data
            assert "cpu_usage" in data

    def test_performance_metrics_endpoint(self, monitoring_client):
        """Test performance metrics endpoint."""
        with patch(
            "music_gen.api.endpoints.monitoring.get_metrics_collector"
        ) as mock_get_collector:
            mock_collector = Mock()
            mock_collector.get_performance_metrics.return_value = {
                "average_generation_time": 8.5,
                "requests_per_minute": 12.3,
                "error_rate_percent": 0.8,
                "queue_wait_time": 2.1,
                "model_load_time": 3.2,
            }
            mock_get_collector.return_value = mock_collector

            response = monitoring_client.get("/api/v1/monitoring/metrics")

            assert response.status_code == 200
            data = response.json()
            assert "average_generation_time" in data
            assert "requests_per_minute" in data

    def test_alerts_endpoint(self, monitoring_client):
        """Test alerts monitoring endpoint."""
        with patch("music_gen.api.endpoints.monitoring.get_alert_manager") as mock_get_alerts:
            mock_alerts = Mock()
            mock_alerts.get_active_alerts.return_value = [
                {
                    "id": "alert_1",
                    "type": "high_memory_usage",
                    "severity": "warning",
                    "message": "Memory usage above 80%",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ]
            mock_get_alerts.return_value = mock_alerts

            response = monitoring_client.get("/api/v1/monitoring/alerts")

            assert response.status_code == 200
            data = response.json()
            assert len(data["alerts"]) == 1
            assert data["alerts"][0]["type"] == "high_memory_usage"


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.fixture
    def error_client(self, test_app):
        """Create client for error testing."""
        return TestClient(test_app)

    def test_service_unavailable_error(self, error_client):
        """Test handling when services are unavailable."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_get_service.side_effect = Exception("Service unavailable")

            request_data = {"prompt": "Test music", "duration": 10.0}
            response = error_client.post("/api/v1/generate/", json=request_data)

            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    def test_generation_timeout_error(self, error_client):
        """Test handling of generation timeouts."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate.side_effect = asyncio.TimeoutError("Generation timed out")
            mock_get_service.return_value = mock_service

            request_data = {"prompt": "Test music", "duration": 30.0}
            response = error_client.post("/api/v1/generate/", json=request_data)

            assert response.status_code == 408  # Request timeout

    def test_model_load_error(self, error_client):
        """Test handling of model loading errors."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()
            from music_gen.core.exceptions import ModelLoadError

            mock_service.generate.side_effect = ModelLoadError("Model not found")
            mock_get_service.return_value = mock_service

            request_data = {"prompt": "Test music", "duration": 10.0}
            response = error_client.post("/api/v1/generate/", json=request_data)

            assert response.status_code == 404  # Not found

    def test_resource_exhaustion_error(self, error_client):
        """Test handling of resource exhaustion."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()
            from music_gen.core.exceptions import InsufficientResourcesError

            mock_service.generate.side_effect = InsufficientResourcesError("Out of memory")
            mock_get_service.return_value = mock_service

            request_data = {"prompt": "Test music", "duration": 10.0}
            response = error_client.post("/api/v1/generate/", json=request_data)

            assert response.status_code == 503  # Service unavailable

    def test_invalid_json_request(self, error_client):
        """Test handling of invalid JSON requests."""
        response = error_client.post(
            "/api/v1/generate/",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Unprocessable entity

    def test_missing_content_type(self, error_client):
        """Test handling of missing content type."""
        response = error_client.post(
            "/api/v1/generate/", content=json.dumps({"prompt": "Test", "duration": 10.0})
        )

        # Should still work or give appropriate error
        assert response.status_code in [200, 400, 415, 422]

    def test_rate_limiting(self, error_client):
        """Test rate limiting behavior."""
        # Make multiple rapid requests
        request_data = {"prompt": "Test music", "duration": 5.0}

        responses = []
        for _ in range(10):
            response = error_client.post("/api/v1/generate/", json=request_data)
            responses.append(response)

        # At least some requests should succeed
        success_codes = [r.status_code for r in responses if r.status_code == 200]
        assert len(success_codes) > 0

        # May have rate limiting responses
        rate_limited = [r.status_code for r in responses if r.status_code == 429]
        # Rate limiting is optional for this test


@pytest.mark.integration
class TestAPIWorkflows:
    """Test complete API workflows and user journeys."""

    @pytest.fixture
    def workflow_client(self, test_app):
        """Create client for workflow testing."""
        return TestClient(test_app)

    def test_complete_generation_workflow(self, workflow_client):
        """Test complete generation workflow from request to completion."""
        with patch(
            "music_gen.api.endpoints.generation.get_generation_service"
        ) as mock_get_service, patch(
            "music_gen.api.endpoints.generation.get_task_repository"
        ) as mock_get_repo:
            # Setup mocks
            mock_service = AsyncMock()
            mock_repo = AsyncMock()

            from music_gen.core.interfaces.services import GenerationResult
            import torch

            task_id = "workflow_test_123"

            # Mock generation result
            mock_result = GenerationResult(
                audio=torch.randn(1, 48000),
                sample_rate=24000,
                duration=20.0,
                metadata={"task_id": task_id},
            )
            mock_service.generate.return_value = mock_result

            # Mock task tracking
            task_states = {
                "pending": {"id": task_id, "status": "pending", "progress": 0.0},
                "processing": {"id": task_id, "status": "processing", "progress": 0.5},
                "completed": {
                    "id": task_id,
                    "status": "completed",
                    "progress": 1.0,
                    "result": {"audio_url": f"/audio/{task_id}.wav"},
                },
            }

            call_count = 0

            def mock_get_task(tid):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return task_states["pending"]
                elif call_count == 2:
                    return task_states["processing"]
                else:
                    return task_states["completed"]

            mock_repo.get_task.side_effect = mock_get_task

            mock_get_service.return_value = mock_service
            mock_get_repo.return_value = mock_repo

            # 1. Submit generation request
            request_data = {
                "prompt": "Epic orchestral soundtrack",
                "duration": 20.0,
                "temperature": 0.7,
            }

            response = workflow_client.post("/api/v1/generate/", json=request_data)
            assert response.status_code == 200

            submission_data = response.json()
            returned_task_id = submission_data["task_id"]

            # 2. Check status (pending)
            response = workflow_client.get(f"/api/v1/generate/{returned_task_id}")
            assert response.status_code == 200

            status_data = response.json()
            assert status_data["status"] == "pending"

            # 3. Check status (processing)
            response = workflow_client.get(f"/api/v1/generate/{returned_task_id}")
            assert response.status_code == 200

            status_data = response.json()
            assert status_data["status"] == "processing"
            assert status_data["progress"] == 0.5

            # 4. Check status (completed)
            response = workflow_client.get(f"/api/v1/generate/{returned_task_id}")
            assert response.status_code == 200

            final_data = response.json()
            assert final_data["status"] == "completed"
            assert final_data["progress"] == 1.0
            assert "audio_url" in final_data["result"]

    def test_batch_workflow(self, workflow_client):
        """Test batch generation workflow."""
        with patch("music_gen.api.endpoints.generation.get_generation_service") as mock_get_service:
            mock_service = AsyncMock()

            # Mock batch results
            from music_gen.core.interfaces.services import GenerationResult
            import torch

            batch_results = [
                GenerationResult(
                    audio=torch.randn(1, 24000),
                    sample_rate=24000,
                    duration=10.0,
                    metadata={"task_id": f"batch_{i}"},
                )
                for i in range(3)
            ]
            mock_service.generate_batch = AsyncMock(return_value=batch_results)
            mock_get_service.return_value = mock_service

            # Submit batch request
            batch_request = {
                "requests": [
                    {"prompt": "Jazz piano", "duration": 10.0},
                    {"prompt": "Rock guitar", "duration": 10.0},
                    {"prompt": "Classical violin", "duration": 10.0},
                ]
            }

            response = workflow_client.post("/api/v1/generate/batch", json=batch_request)
            assert response.status_code == 200

            batch_data = response.json()
            assert len(batch_data["results"]) == 3

            # Verify each result
            for i, result in enumerate(batch_data["results"]):
                assert result["duration"] == 10.0
                assert "task_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
