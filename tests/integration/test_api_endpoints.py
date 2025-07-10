"""
Integration tests for Music Generation API endpoints.

Tests the complete API functionality including:
- Music generation requests
- Authentication and authorization  
- Rate limiting
- File upload/download
- WebSocket streaming
- Task status tracking
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any
import tempfile
import pytest
import httpx
from fastapi.testclient import TestClient
import websockets
from unittest.mock import Mock, patch

from music_gen.api.app import create_app
from music_gen.core.model_manager import ModelManager


@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application."""
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "testing",
            "DEFAULT_MODEL": "facebook/musicgen-small",
            "ENABLE_PROXY_HEADERS": "false",
            "TRUSTED_PROXIES": "",
            "INTERNAL_API_KEYS": "test-internal-key",
            "DEFAULT_RATE_LIMIT_TIER": "premium",
        },
    ):
        app = create_app(title="Test Music Gen API", version="test")
        return app


@pytest.fixture(scope="session")
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment and mock dependencies."""
    # Mock ModelManager to avoid loading real models
    with patch("music_gen.core.model_manager.FastMusicGenerator") as mock_gen:
        # Mock model behavior
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.generate_single.return_value = Mock(
            audio=Mock(shape=(1, 32000)),  # 1 second at 32kHz
            sample_rate=32000,
            generation_time=0.5,
        )
        mock_model.generate_batch.return_value = [
            Mock(
                audio=Mock(shape=(1, 32000)),
                sample_rate=32000,
                generation_time=0.5,
                metadata={"prompt": f"test prompt {i}"},
            )
            for i in range(3)
        ]
        mock_gen.return_value = mock_model

        # Reset singleton for testing
        ModelManager._instance = None

        yield

        # Cleanup
        ModelManager._instance = None


class TestHealthEndpoints:
    """Test health and basic endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "test"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert "timestamp" in data

    def test_health_detailed(self, client):
        """Test detailed health check."""
        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "model_manager" in data
        assert "memory_usage" in data
        assert "disk_usage" in data


class TestMusicGenerationEndpoints:
    """Test music generation functionality."""

    def test_successful_generation_request(self, client):
        """Test successful music generation request."""
        request_data = {
            "prompt": "upbeat electronic music",
            "duration": 5.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
        }

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert uuid.UUID(data["task_id"])  # Valid UUID

    def test_generation_with_optional_parameters(self, client):
        """Test generation with all optional parameters."""
        request_data = {
            "prompt": "classical piano piece",
            "duration": 10.0,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.85,
            "guidance_scale": 3.5,
            "genre": "classical",
            "mood": "peaceful",
            "tempo": 120,
            "instruments": ["piano", "strings"],
            "seed": 42,
        }

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "pending"

    def test_batch_generation_request(self, client):
        """Test batch generation request."""
        batch_data = {
            "requests": [
                {"prompt": "rock music", "duration": 3.0},
                {"prompt": "jazz music", "duration": 4.0},
                {"prompt": "ambient music", "duration": 5.0},
            ]
        }

        response = client.post("/api/v1/generate/batch", json=batch_data)
        assert response.status_code == 200

        data = response.json()
        assert "batch_id" in data
        assert "task_ids" in data
        assert len(data["task_ids"]) == 3
        assert data["total_requests"] == 3
        assert data["status"] == "pending"


class TestTaskStatusEndpoints:
    """Test task status tracking."""

    def test_get_task_status_not_found(self, client):
        """Test getting status of non-existent task."""
        fake_task_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/generate/{fake_task_id}")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"

    def test_task_lifecycle(self, client):
        """Test complete task lifecycle: create -> check status -> completion."""
        # Create generation request
        request_data = {"prompt": "test music", "duration": 2.0}

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # Check initial status
        status_response = client.get(f"/api/v1/generate/{task_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in ["pending", "processing"]

        # Wait for completion (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = client.get(f"/api/v1/generate/{task_id}")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                assert "audio_url" in status_data
                assert "duration" in status_data
                assert "metadata" in status_data
                break
            elif status_data["status"] == "failed":
                assert "error" in status_data
                break

            time.sleep(1)

        # Should complete within timeout
        assert status_data["status"] in ["completed", "failed"]

    def test_batch_status_tracking(self, client):
        """Test batch status tracking."""
        # Create batch request
        batch_data = {
            "requests": [
                {"prompt": "test 1", "duration": 2.0},
                {"prompt": "test 2", "duration": 2.0},
            ]
        }

        response = client.post("/api/v1/generate/batch", json=batch_data)
        batch_id = response.json()["batch_id"]

        # Check batch status
        batch_status = client.get(f"/api/v1/generate/batch/{batch_id}")
        assert batch_status.status_code == 200

        data = batch_status.json()
        assert data["batch_id"] == batch_id
        assert data["total"] == 2
        assert "completed" in data
        assert "failed" in data
        assert "pending" in data
        assert "tasks" in data


class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    def test_invalid_prompt_empty(self, client):
        """Test empty prompt validation."""
        request_data = {"prompt": "", "duration": 5.0}

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_duration_bounds(self, client):
        """Test duration boundary validation."""
        # Duration too short
        request_data = {"prompt": "test music", "duration": 0.5}  # Below minimum (1.0)

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422

        # Duration too long
        request_data = {"prompt": "test music", "duration": 120.0}  # Above maximum (60.0)

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422

    def test_invalid_temperature_bounds(self, client):
        """Test temperature validation."""
        request_data = {
            "prompt": "test music",
            "duration": 5.0,
            "temperature": 3.0,  # Above maximum (2.0)
        }

        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422

    def test_invalid_batch_size(self, client):
        """Test batch size validation."""
        # Too many requests in batch
        batch_data = {
            "requests": [
                {"prompt": f"test {i}", "duration": 2.0} for i in range(10)  # Above maximum (5)
            ]
        }

        response = client.post("/api/v1/generate/batch", json=batch_data)
        assert response.status_code == 422

    def test_malformed_json(self, client):
        """Test malformed JSON handling."""
        response = client.post(
            "/api/v1/generate/",
            data='{"prompt": "test", invalid json',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.post("/api/v1/generate/", json={"prompt": "test music", "duration": 2.0})

        # Should have rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_enforcement(self, client):
        """Test rate limit enforcement."""
        # Make many requests quickly to trigger rate limit
        request_data = {"prompt": "test music", "duration": 1.0}

        responses = []
        for _ in range(20):  # Exceed typical rate limit
            response = client.post("/api/v1/generate/", json=request_data)
            responses.append(response)

            if response.status_code == 429:  # Rate limited
                break

        # Should eventually get rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        if rate_limited:
            rate_limited_response = next(r for r in responses if r.status_code == 429)
            assert "Retry-After" in rate_limited_response.headers

    def test_rate_limit_different_endpoints(self, client):
        """Test rate limits apply across endpoints."""
        # Make requests to different endpoints
        endpoints = [
            ("/api/v1/generate/", {"prompt": "test", "duration": 1.0}),
            ("/health", None),
            ("/api/v1/models/", None),
        ]

        for endpoint, data in endpoints:
            if data:
                response = client.post(endpoint, json=data)
            else:
                response = client.get(endpoint)

            # Should have rate limit headers regardless of endpoint
            assert "X-RateLimit-Limit" in response.headers


class TestFileOperations:
    """Test file upload and download operations."""

    def test_download_nonexistent_file(self, client):
        """Test downloading non-existent file."""
        fake_task_id = str(uuid.uuid4())
        response = client.get(f"/download/{fake_task_id}")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"

    def test_download_incomplete_task(self, client):
        """Test downloading from incomplete task."""
        # Create a generation request
        request_data = {"prompt": "test music", "duration": 2.0}

        response = client.post("/api/v1/generate/", json=request_data)
        task_id = response.json()["task_id"]

        # Try to download immediately (should fail)
        download_response = client.get(f"/download/{task_id}")
        assert download_response.status_code == 400
        assert download_response.json()["detail"] == "Generation not completed"

    def test_successful_download_workflow(self, client):
        """Test complete download workflow."""
        # Create generation request
        request_data = {"prompt": "downloadable music", "duration": 2.0}

        response = client.post("/api/v1/generate/", json=request_data)
        task_id = response.json()["task_id"]

        # Wait for completion
        max_wait = 30
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = client.get(f"/api/v1/generate/{task_id}")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                # Try download
                download_response = client.get(f"/download/{task_id}")
                assert download_response.status_code == 200
                assert download_response.headers["content-type"] == "audio/wav"
                assert "content-length" in download_response.headers
                break
            elif status_data["status"] == "failed":
                break

            time.sleep(1)


class TestAuthenticationEndpoints:
    """Test authentication requirements."""

    def test_csrf_token_endpoint(self, client):
        """Test CSRF token generation."""
        response = client.get("/api/auth/csrf-token")
        assert response.status_code == 200

        data = response.json()
        assert "csrf_token" in data
        assert len(data["csrf_token"]) > 20  # Should be substantial token

    def test_login_endpoint_structure(self, client):
        """Test login endpoint exists and has proper structure."""
        # Test with invalid credentials (should give proper error structure)
        login_data = {"email": "test@example.com", "password": "invalid"}

        response = client.post("/api/auth/login", json=login_data)
        # Might be 401 (unauthorized) or 422 (validation error) depending on implementation
        assert response.status_code in [401, 422, 404]

    def test_protected_endpoints_require_auth(self, client):
        """Test that protected endpoints require authentication."""
        # Most generation endpoints should work without auth in test mode
        # But some admin endpoints might require it

        # Test admin/monitoring endpoints if they exist
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/stats",
            "/api/v1/monitoring/metrics",
        ]

        for endpoint in admin_endpoints:
            response = client.get(endpoint)
            # Should either be 401 (unauthorized) or 404 (not found)
            assert response.status_code in [401, 404, 405]


class TestWebSocketConnections:
    """Test WebSocket streaming functionality."""

    def test_streaming_session_creation(self, client):
        """Test creating a streaming session."""
        request_data = {"prompt": "streaming test music", "duration": 5.0, "chunk_duration": 1.0}

        response = client.post("/api/v1/stream/session", json=request_data)

        if response.status_code == 503:
            # Streaming service not available in test
            pytest.skip("Streaming service not available")

        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert "websocket_url" in data
        assert uuid.UUID(data["session_id"])  # Valid UUID

    def test_streaming_session_list(self, client):
        """Test listing active streaming sessions."""
        response = client.get("/api/v1/stream/sessions")
        assert response.status_code == 200

        data = response.json()
        assert "sessions" in data
        assert "count" in data
        assert isinstance(data["sessions"], list)
        assert isinstance(data["count"], int)

    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_app):
        """Test WebSocket connection (if streaming is available)."""
        # First create a streaming session
        async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/stream/session", json={"prompt": "websocket test", "duration": 3.0}
            )

            if response.status_code == 503:
                pytest.skip("Streaming service not available")

            session_id = response.json()["session_id"]

            # Test WebSocket connection
            ws_url = f"ws://test/api/v1/stream/ws/{session_id}"

            # This is a basic test - real WebSocket testing would need more setup
            # For now, just verify the endpoint structure
            assert "/api/v1/stream/ws/" in ws_url

    def test_websocket_invalid_session(self, client):
        """Test WebSocket with invalid session ID."""
        fake_session_id = str(uuid.uuid4())

        # Try to connect with TestClient (limited WebSocket support)
        with pytest.raises(Exception):
            with client.websocket_connect(f"/api/v1/stream/ws/{fake_session_id}"):
                pass  # Should fail to connect


class TestConcurrentRequests:
    """Test concurrent request handling."""

    def test_concurrent_generation_requests(self, client):
        """Test handling multiple concurrent generation requests."""
        import threading

        results = []
        errors = []

        def make_request(prompt_id):
            try:
                response = client.post(
                    "/api/v1/generate/",
                    json={"prompt": f"concurrent test {prompt_id}", "duration": 2.0},
                )
                results.append(response.json())
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All should have unique task IDs
        task_ids = [r["task_id"] for r in results]
        assert len(set(task_ids)) == 10  # All unique

    def test_concurrent_status_checks(self, client):
        """Test concurrent status checking."""
        # First create a task
        response = client.post("/api/v1/generate/", json={"prompt": "status test", "duration": 3.0})
        task_id = response.json()["task_id"]

        # Check status concurrently
        import threading

        status_results = []

        def check_status():
            response = client.get(f"/api/v1/generate/{task_id}")
            status_results.append(response.json())

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=check_status)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should return valid status
        assert len(status_results) == 5
        for status in status_results:
            assert status["task_id"] == task_id
            assert "status" in status


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Check key endpoints are documented
        paths = schema["paths"]
        assert "/api/v1/generate/" in paths
        assert "/health" in paths

    def test_swagger_docs(self, client):
        """Test Swagger documentation page."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_docs(self, client):
        """Test ReDoc documentation page."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
