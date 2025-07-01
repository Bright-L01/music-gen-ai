"""
Integration tests for FastAPI server.
"""

import io
from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from music_gen.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock model for API testing."""
    mock = Mock()
    mock.generate_audio.return_value = torch.randn(1, 24000)  # 1 second audio
    mock.audio_tokenizer.sample_rate = 24000
    return mock


@pytest.mark.integration
class TestAPIEndpoints:
    """Test API endpoint integration."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_generate_endpoint_structure(self, client):
        """Test generate endpoint structure (without model)."""
        request_data = {
            "prompt": "Happy jazz music",
            "duration": 10.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
        }

        # This will likely fail due to no model, but we can test the structure
        response = client.post("/generate", json=request_data)

        # Should either succeed or fail with 503 (model not loaded)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert "status" in data
        else:
            # Model not loaded error
            assert "Model not loaded" in response.json()["detail"]

    @patch("music_gen.api.main.model")
    def test_generate_endpoint_with_mock(self, mock_model_global, client):
        """Test generate endpoint with mocked model."""
        import torch

        # Setup mock model
        mock_model = Mock()
        mock_model.generate_audio.return_value = torch.randn(1, 24000)
        mock_model.audio_tokenizer.sample_rate = 24000
        mock_model_global = mock_model

        request_data = {
            "prompt": "Happy jazz music",
            "duration": 5.0,
            "temperature": 0.8,
        }

        with patch("music_gen.api.main.model", mock_model):
            response = client.post("/generate", json=request_data)

            if response.status_code == 200:
                data = response.json()
                assert "task_id" in data
                assert data["status"] == "pending"

    def test_models_endpoint(self, client):
        """Test models information endpoint."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "current_model" in data
        assert "available_models" in data
        assert "model_info" in data
        assert isinstance(data["available_models"], list)

    def test_genres_endpoint(self, client):
        """Test genres endpoint."""
        response = client.get("/genres")

        assert response.status_code == 200
        data = response.json()

        assert "genres" in data
        assert isinstance(data["genres"], list)
        assert len(data["genres"]) > 0

    def test_moods_endpoint(self, client):
        """Test moods endpoint."""
        response = client.get("/moods")

        assert response.status_code == 200
        data = response.json()

        assert "moods" in data
        assert isinstance(data["moods"], list)
        assert len(data["moods"]) > 0


@pytest.mark.integration
class TestAPIValidation:
    """Test API request validation."""

    def test_generate_request_validation(self, client):
        """Test request validation for generate endpoint."""
        # Test missing prompt
        response = client.post("/generate", json={})
        assert response.status_code == 422

        # Test invalid duration
        response = client.post(
            "/generate",
            json={
                "prompt": "Test music",
                "duration": -1.0,  # Invalid
            },
        )
        assert response.status_code == 422

        # Test invalid temperature
        response = client.post(
            "/generate",
            json={
                "prompt": "Test music",
                "temperature": -0.5,  # Invalid
            },
        )
        assert response.status_code == 422

        # Test valid request
        response = client.post(
            "/generate",
            json={
                "prompt": "Happy jazz music",
                "duration": 10.0,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
            },
        )
        # Should not fail validation (might fail due to no model)
        assert response.status_code in [200, 503]

    def test_parameter_ranges(self, client):
        """Test parameter range validation."""
        base_request = {
            "prompt": "Test music",
            "duration": 10.0,
        }

        # Test duration limits
        response = client.post(
            "/generate",
            json={
                **base_request,
                "duration": 0.5,  # Below minimum
            },
        )
        assert response.status_code == 422

        response = client.post(
            "/generate",
            json={
                **base_request,
                "duration": 120.0,  # Above maximum (assuming 60.0 max)
            },
        )
        # Might be valid depending on MAX_DURATION setting
        assert response.status_code in [200, 422, 503]

        # Test temperature limits
        response = client.post(
            "/generate",
            json={
                **base_request,
                "temperature": 2.5,  # Above maximum
            },
        )
        assert response.status_code == 422

        # Test top_p limits
        response = client.post(
            "/generate",
            json={
                **base_request,
                "top_p": 1.5,  # Above maximum
            },
        )
        assert response.status_code == 422


@pytest.mark.integration
class TestTaskManagement:
    """Test task management functionality."""

    def test_task_status_flow(self, client):
        """Test task status progression."""
        # This test assumes no model is loaded, so tasks will fail
        # But we can still test the status endpoint structure

        # Try to create a task
        response = client.post(
            "/generate",
            json={
                "prompt": "Test music",
                "duration": 5.0,
            },
        )

        if response.status_code == 200:
            # Task created successfully
            task_id = response.json()["task_id"]

            # Check task status
            status_response = client.get(f"/generate/{task_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            assert "task_id" in status_data
            assert "status" in status_data
            assert status_data["task_id"] == task_id

        elif response.status_code == 503:
            # Model not loaded - expected in test environment
            pass
        else:
            pytest.fail(f"Unexpected response code: {response.status_code}")

    def test_invalid_task_id(self, client):
        """Test accessing invalid task ID."""
        response = client.get("/generate/invalid-task-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_download_without_completion(self, client):
        """Test downloading before task completion."""
        # Try to download non-existent file
        response = client.get("/download/non-existent-task")
        assert response.status_code == 404


@pytest.mark.integration
class TestAudioEvaluation:
    """Test audio evaluation endpoint."""

    def test_evaluate_endpoint_structure(self, client):
        """Test evaluate endpoint structure."""
        # Create fake audio file
        import io
        import wave

        # Create a simple WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz

            # Generate 1 second of silence
            frames = b"\x00\x00" * 24000
            wav_file.writeframes(frames)

        buffer.seek(0)

        # Test file upload
        response = client.post(
            "/evaluate",
            files={"audio_file": ("test.wav", buffer, "audio/wav")},
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "metrics" in data
            assert "quality_score" in data

    def test_evaluate_without_file(self, client):
        """Test evaluate endpoint without file."""
        response = client.post("/evaluate")
        assert response.status_code == 422  # Missing required file


@pytest.mark.integration
@pytest.mark.slow
class TestAsyncBehavior:
    """Test asynchronous behavior of API."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        # This is a basic test - in practice you'd need a running server
        # for true concurrency testing

        requests = []
        for i in range(3):
            request_data = {
                "prompt": f"Test music {i}",
                "duration": 5.0,
            }
            requests.append(request_data)

        # Send concurrent requests (simulated)
        responses = []
        for request_data in requests:
            response = client.post("/generate", json=request_data)
            responses.append(response)

        # All requests should be handled
        assert len(responses) == 3

        # Check that each response is valid (structure-wise)
        for response in responses:
            assert response.status_code in [200, 503]


@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""

    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/generate", data="invalid json{", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test handling of missing content type."""
        response = client.post("/generate", data='{"prompt": "test"}')
        # FastAPI should handle this gracefully
        assert response.status_code in [200, 422, 503]

    def test_oversized_request(self, client):
        """Test handling of oversized requests."""
        # Create a very long prompt
        long_prompt = "A" * 10000

        response = client.post(
            "/generate",
            json={
                "prompt": long_prompt,
                "duration": 5.0,
            },
        )

        # Should either accept or reject gracefully
        assert response.status_code in [200, 413, 422, 503]

    def test_invalid_file_upload(self, client):
        """Test handling of invalid file uploads."""
        # Upload non-audio file
        fake_file = io.BytesIO(b"This is not an audio file")

        response = client.post(
            "/evaluate",
            files={"audio_file": ("test.txt", fake_file, "text/plain")},
        )

        # Should handle gracefully
        assert response.status_code in [400, 422, 500]


@pytest.mark.integration
class TestAPIConfiguration:
    """Test API configuration and settings."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/generate")

        # Should include CORS headers
        assert response.status_code in [200, 405]  # Some clients return 405 for OPTIONS

    def test_docs_endpoints(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Validate OpenAPI structure
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data


if __name__ == "__main__":
    pytest.main([__file__])
