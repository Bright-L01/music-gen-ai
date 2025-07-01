"""
Unit tests for consolidated API functionality.
"""

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Import API modules - handle missing dependencies gracefully
try:
    from music_gen.api.app import app
    from music_gen.api.endpoints import generation, multi_instrument, streaming, health

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestMainAPI:
    """Test main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_app_exists(self):
        """Test that app exists and has correct metadata."""
        assert app is not None
        assert app.title == "Music Gen AI API"
        assert app.version == "1.0.0"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_api_prefix(self, client):
        """Test that API uses correct prefix."""
        # Test that endpoints are under /api/v1
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Test that root redirects to docs
        response = client.get("/")
        assert response.status_code in [200, 307]  # Redirect or docs


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestGenerationEndpoint:
    """Test generation API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_generation_endpoint_validation(self, client):
        """Test input validation on generation endpoint."""
        # Missing required fields
        response = client.post("/api/v1/generate", json={})
        assert response.status_code == 422

        # Invalid duration
        response = client.post("/api/v1/generate", json={"prompt": "Test", "duration": -1})
        assert response.status_code == 422

        # Valid minimal request (will fail with 503 if model not loaded)
        response = client.post("/api/v1/generate", json={"prompt": "Test music"})
        assert response.status_code in [200, 503]

    @patch("music_gen.core.model_manager.ModelManager.get_model")
    @patch("music_gen.utils.audio.save_audio")
    def test_generate_endpoint_mocked(self, mock_save, mock_get_model, client):
        """Test generation endpoint with mocked model."""
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = torch.randn(1, 1, 32000)
        mock_model.sample_rate = 32000
        mock_get_model.return_value = mock_model
        
        # Mock audio save
        mock_save.return_value = "test_output.wav"

        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Happy jazz music", "duration": 10.0}
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "audio_url" in data
        assert data["duration"] == 10.0
        assert data["prompt"] == "Happy jazz music"


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestMultiInstrumentEndpoint:
    """Test multi-instrument API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_multi_instrument_validation(self, client):
        """Test input validation on multi-instrument endpoint."""
        # Missing tracks
        response = client.post("/api/v1/multi-instrument/generate", json={})
        assert response.status_code == 422

        # Empty tracks
        response = client.post("/api/v1/multi-instrument/generate", json={"tracks": []})
        assert response.status_code == 422

        # Valid request structure
        response = client.post(
            "/api/v1/multi-instrument/generate",
            json={
                "tracks": [
                    {"instrument": "piano", "prompt": "Soft piano"},
                    {"instrument": "drums", "prompt": "Jazz drums"}
                ],
                "duration": 30.0
            }
        )
        # Should be 200 or 503 (if model not loaded)
        assert response.status_code in [200, 503]


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestStreamingEndpoint:
    """Test streaming API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_streaming_session_creation(self, client):
        """Test creating streaming session."""
        response = client.post(
            "/api/v1/stream/session",
            json={"prompt": "Streaming test", "duration": 10.0}
        )
        
        # Should be 200 or 503 (if streaming not available)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "websocket_url" in data

    def test_list_sessions(self, client):
        """Test listing active sessions."""
        response = client.get("/api/v1/stream/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data
        assert isinstance(data["sessions"], list)


@pytest.mark.unit
class TestAPIHelpers:
    """Test API helper functions and utilities."""

    def test_endpoint_modules_exist(self):
        """Test that all endpoint modules exist."""
        if API_AVAILABLE:
            assert generation is not None
            assert health is not None
            assert multi_instrument is not None
            assert streaming is not None

    def test_request_models(self):
        """Test that request/response models are properly defined."""
        if API_AVAILABLE:
            from music_gen.api.endpoints.generation import GenerationRequest, GenerationResponse
            from music_gen.api.endpoints.multi_instrument import MultiInstrumentRequest, InstrumentTrack
            from music_gen.api.endpoints.streaming import StreamingRequest, StreamingResponse

            # Test model instantiation
            gen_req = GenerationRequest(prompt="Test")
            assert gen_req.prompt == "Test"
            assert gen_req.duration == 30.0  # default

            track = InstrumentTrack(instrument="piano", prompt="Piano melody")
            assert track.instrument == "piano"
            assert track.volume == 1.0  # default

            stream_req = StreamingRequest(prompt="Stream test")
            assert stream_req.prompt == "Stream test"
            assert stream_req.chunk_duration == 1.0  # default


if __name__ == "__main__":
    pytest.main([__file__])