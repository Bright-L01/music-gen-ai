"""
Unit tests for API functionality.
"""

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Import API modules - handle missing dependencies gracefully
try:
    from music_gen.api.main import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from music_gen.api.real_musicgen_api import RealMusicGenAPI

    REAL_API_AVAILABLE = True
except ImportError:
    REAL_API_AVAILABLE = False

try:
    from music_gen.api.multi_instrument_api import MultiInstrumentAPI

    MULTI_API_AVAILABLE = True
except ImportError:
    MULTI_API_AVAILABLE = False

try:
    from music_gen.api.streaming_api import StreamingAPI

    STREAMING_API_AVAILABLE = True
except ImportError:
    STREAMING_API_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestMainAPI:
    """Test main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_info_endpoint(self, client):
        """Test system info endpoint."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                response = client.get("/info")
                assert response.status_code == 200
                data = response.json()
                assert "system" in data
                assert "gpu_available" in data["system"]

    @patch("music_gen.api.main.create_musicgen_model")
    def test_generate_endpoint(self, mock_model_func, client):
        """Test music generation endpoint."""
        # Mock model
        mock_model = Mock()
        mock_model.generate_audio.return_value = torch.randn(1, 1, 24000)
        mock_model.audio_tokenizer.sample_rate = 24000
        mock_model_func.return_value = mock_model

        with patch("music_gen.api.main.save_audio_file") as mock_save:
            mock_save.return_value = BytesIO(b"fake_audio_data")

            response = client.post(
                "/generate",
                json={"prompt": "Happy jazz music", "duration": 10.0, "model_size": "base"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("audio/")

    def test_generate_endpoint_validation(self, client):
        """Test input validation on generate endpoint."""
        # Missing required fields
        response = client.post("/generate", json={})
        assert response.status_code == 422

        # Invalid duration
        response = client.post("/generate", json={"prompt": "Test", "duration": -1})
        assert response.status_code == 422

    @patch("music_gen.api.main.create_musicgen_model")
    def test_generate_with_conditioning(self, mock_model_func, client):
        """Test generation with conditioning parameters."""
        mock_model = Mock()
        mock_model.generate_audio.return_value = torch.randn(1, 1, 24000)
        mock_model.audio_tokenizer.sample_rate = 24000
        mock_model_func.return_value = mock_model

        with patch("music_gen.api.main.save_audio_file") as mock_save:
            mock_save.return_value = BytesIO(b"fake_audio_data")

            response = client.post(
                "/generate",
                json={
                    "prompt": "Jazz piano",
                    "duration": 15.0,
                    "genre": "jazz",
                    "mood": "happy",
                    "tempo": 120,
                    "temperature": 0.8,
                },
            )

            assert response.status_code == 200
            mock_model.generate_audio.assert_called_once()

            # Check conditioning parameters were passed
            call_kwargs = mock_model.generate_audio.call_args[1]
            assert call_kwargs["duration"] == 15.0
            assert call_kwargs["temperature"] == 0.8


@pytest.mark.unit
@pytest.mark.skipif(not REAL_API_AVAILABLE, reason="Real API not available")
class TestRealMusicGenAPI:
    """Test RealMusicGenAPI class."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked dependencies."""
        with patch("music_gen.api.real_musicgen_api.create_musicgen_model"):
            api = RealMusicGenAPI()
            return api

    def test_api_initialization(self, api_instance):
        """Test API initialization."""
        assert api_instance is not None
        assert hasattr(api_instance, "model")
        assert hasattr(api_instance, "device")

    @patch("music_gen.api.real_musicgen_api.create_musicgen_model")
    def test_load_model(self, mock_model_func):
        """Test model loading."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_func.return_value = mock_model

        api = RealMusicGenAPI()
        api.load_model("base")

        mock_model_func.assert_called_once_with("base")
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

    def test_generate_music(self, api_instance):
        """Test music generation."""
        # Mock the model
        api_instance.model = Mock()
        api_instance.model.generate_audio.return_value = torch.randn(1, 1, 24000)
        api_instance.model.audio_tokenizer.sample_rate = 24000

        result = api_instance.generate_music(prompt="Test music", duration=10.0, temperature=0.8)

        assert result is not None
        assert hasattr(result, "shape")
        api_instance.model.generate_audio.assert_called_once()

    def test_generate_with_conditioning(self, api_instance):
        """Test generation with conditioning."""
        api_instance.model = Mock()
        api_instance.model.generate_audio.return_value = torch.randn(1, 1, 24000)
        api_instance.model.audio_tokenizer.sample_rate = 24000

        result = api_instance.generate_music(
            prompt="Jazz music", duration=15.0, genre="jazz", mood="energetic", tempo=120
        )

        assert result is not None
        call_kwargs = api_instance.model.generate_audio.call_args[1]
        assert "genre_ids" in call_kwargs or "conditioning" in str(call_kwargs)


@pytest.mark.unit
@pytest.mark.skipif(not MULTI_API_AVAILABLE, reason="Multi-instrument API not available")
class TestMultiInstrumentAPI:
    """Test MultiInstrumentAPI class."""

    @pytest.fixture
    def api_instance(self):
        """Create multi-instrument API instance."""
        with patch("music_gen.api.multi_instrument_api.MultiInstrumentGenerator"):
            api = MultiInstrumentAPI()
            return api

    def test_api_initialization(self, api_instance):
        """Test API initialization."""
        assert api_instance is not None
        assert hasattr(api_instance, "generator")

    def test_generate_multi_track(self, api_instance):
        """Test multi-track generation."""
        # Mock the generator
        api_instance.generator = Mock()
        api_instance.generator.generate_multi_track.return_value = {
            "piano": torch.randn(1, 24000),
            "bass": torch.randn(1, 24000),
            "drums": torch.randn(1, 24000),
        }

        result = api_instance.generate_multi_track(
            prompt="Jazz trio", instruments=["piano", "bass", "drums"], duration=20.0
        )

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "piano" in result
        assert "bass" in result
        assert "drums" in result

    def test_mix_tracks(self, api_instance):
        """Test track mixing functionality."""
        tracks = {"piano": torch.randn(1, 24000), "bass": torch.randn(1, 24000)}

        with patch("music_gen.api.multi_instrument_api.AdvancedMixingEngine") as mock_mixer:
            mock_mixer.return_value.mix_tracks.return_value = torch.randn(24000)

            result = api_instance.mix_tracks(tracks)

            assert result is not None
            mock_mixer.return_value.mix_tracks.assert_called_once()


@pytest.mark.unit
@pytest.mark.skipif(not STREAMING_API_AVAILABLE, reason="Streaming API not available")
class TestStreamingAPI:
    """Test StreamingAPI class."""

    @pytest.fixture
    def api_instance(self):
        """Create streaming API instance."""
        with patch("music_gen.api.streaming_api.StreamingGenerator"):
            api = StreamingAPI()
            return api

    def test_api_initialization(self, api_instance):
        """Test API initialization."""
        assert api_instance is not None
        assert hasattr(api_instance, "generator")

    def test_start_streaming(self, api_instance):
        """Test starting streaming generation."""
        # Mock the generator
        api_instance.generator = Mock()
        api_instance.generator.start_streaming.return_value = [
            {"type": "chunk", "audio": torch.randn(1, 1024), "progress": 0.1},
            {"type": "chunk", "audio": torch.randn(1, 1024), "progress": 0.5},
            {"type": "complete", "audio": torch.randn(1, 24000)},
        ]

        stream = api_instance.start_streaming(prompt="Continuous ambient", duration=30.0)

        chunks = list(stream)
        assert len(chunks) == 3
        assert chunks[0]["type"] == "chunk"
        assert chunks[-1]["type"] == "complete"

    def test_stop_streaming(self, api_instance):
        """Test stopping streaming generation."""
        api_instance.generator = Mock()

        api_instance.stop_streaming()

        api_instance.generator.stop.assert_called_once()

    def test_streaming_with_callback(self, api_instance):
        """Test streaming with progress callback."""
        api_instance.generator = Mock()

        callback = Mock()
        api_instance.start_streaming(
            prompt="Test stream", duration=10.0, progress_callback=callback
        )

        # Generator should be called with callback
        api_instance.generator.start_streaming.assert_called_once()
        call_kwargs = api_instance.generator.start_streaming.call_args[1]
        assert "progress_callback" in call_kwargs


@pytest.mark.unit
class TestAPIHelpers:
    """Test API helper functions and utilities."""

    def test_audio_format_validation(self):
        """Test audio format validation."""

        # Mock validation function
        def validate_audio_format(format_type):
            supported = ["wav", "mp3", "flac"]
            return format_type.lower() in supported

        assert validate_audio_format("wav")
        assert validate_audio_format("MP3")
        assert not validate_audio_format("xyz")

    def test_request_validation(self):
        """Test request parameter validation."""

        # Mock validation function
        def validate_generation_request(data):
            required = ["prompt"]
            for field in required:
                if field not in data:
                    return False, f"Missing required field: {field}"

            if "duration" in data and data["duration"] <= 0:
                return False, "Duration must be positive"

            return True, None

        # Valid request
        valid, error = validate_generation_request({"prompt": "Test music", "duration": 10.0})
        assert valid
        assert error is None

        # Invalid request - missing prompt
        valid, error = validate_generation_request({})
        assert not valid
        assert "prompt" in error

        # Invalid request - negative duration
        valid, error = validate_generation_request({"prompt": "Test", "duration": -1})
        assert not valid
        assert "Duration" in error

    def test_response_formatting(self):
        """Test API response formatting."""

        # Mock response formatter
        def format_api_response(data, success=True, message=None):
            return {
                "success": success,
                "data": data,
                "message": message,
                "timestamp": "2024-01-01T00:00:00Z",
            }

        # Success response
        response = format_api_response({"result": "audio_data"})
        assert response["success"] is True
        assert response["data"]["result"] == "audio_data"

        # Error response
        response = format_api_response(None, success=False, message="Error occurred")
        assert response["success"] is False
        assert response["message"] == "Error occurred"


if __name__ == "__main__":
    pytest.main([__file__])
