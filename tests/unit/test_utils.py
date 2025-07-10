"""
Tests for music_gen.streaming.utils
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_gen.streaming.utils import *


class TestUtils:
    """Test utils streaming functionality."""

    def test_audio_to_base64(self):
        """Test audio to base64 conversion."""
        # Create test audio
        sample_rate = 44100
        duration = 0.1  # 100ms
        samples = int(duration * sample_rate)

        # Create mono audio
        audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, samples))
        audio = audio.unsqueeze(0)  # Add channel dimension

        # Convert to base64
        audio_b64 = audio_to_base64(audio, sample_rate)
        assert isinstance(audio_b64, str)
        assert len(audio_b64) > 0

        # Test with batch dimension
        audio_batch = audio.unsqueeze(0)  # Add batch dimension
        audio_b64_batch = audio_to_base64(audio_batch, sample_rate)
        assert isinstance(audio_b64_batch, str)
        assert len(audio_b64_batch) > 0

    def test_base64_to_audio(self):
        """Test base64 to audio conversion."""
        # Create test audio
        sample_rate = 44100
        duration = 0.1
        samples = int(duration * sample_rate)
        original_audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, samples))
        original_audio = original_audio.unsqueeze(0)

        # Convert to base64 and back
        audio_b64 = audio_to_base64(original_audio, sample_rate)
        reconstructed_audio = base64_to_audio(audio_b64, sample_rate)

        assert reconstructed_audio.shape[0] == 1  # Should have channel dimension
        assert reconstructed_audio.shape[1] == samples

        # Should be approximately equal (allowing for quantization)
        original_flat = original_audio[0]
        reconstructed_flat = reconstructed_audio[0]

        # Normalize both for comparison
        original_norm = original_flat / (original_flat.abs().max() + 1e-8)
        reconstructed_norm = reconstructed_flat / (reconstructed_flat.abs().max() + 1e-8)

        # Should be close (allowing for 16-bit quantization)
        assert torch.allclose(original_norm, reconstructed_norm, atol=1e-3)

    def test_streaming_metrics(self):
        """Test StreamingMetrics class."""
        metrics = StreamingMetrics()

        # Test initial values
        assert metrics.total_chunks == 0
        assert metrics.total_duration == 0.0
        assert metrics.start_time > 0

        # Test properties with no data
        assert metrics.average_chunk_time == 0.0
        assert metrics.buffer_underrun_rate == 0.0

        # Add some data
        metrics.total_chunks = 10
        metrics.total_generation_time = 5.0
        metrics.total_duration = 10.0
        metrics.buffer_underruns = 2

        # Test calculated properties
        assert metrics.average_chunk_time == 0.5
        assert metrics.buffer_underrun_rate == 0.2

        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "total_chunks" in metrics_dict
        assert "average_chunk_time" in metrics_dict
        assert "real_time_factor" in metrics_dict

    def test_latency_tracker(self):
        """Test LatencyTracker class."""
        tracker = LatencyTracker(window_size=5)

        # Test initial state
        assert len(tracker.latencies) == 0
        assert len(tracker.timestamps) == 0

        # Add measurements
        for i in range(3):
            tracker.add_measurement(0.1 + i * 0.01)

        assert len(tracker.latencies) == 3
        assert len(tracker.timestamps) == 3
        assert abs(tracker.latencies[0] - 0.1) < 1e-10
        assert abs(tracker.latencies[2] - 0.12) < 1e-10

        # Test window size limit
        for i in range(5):
            tracker.add_measurement(0.2 + i * 0.01)

        assert len(tracker.latencies) == 5  # Should be capped at window_size
        assert len(tracker.timestamps) == 5

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(return_value='{"type": "test"}')
        return ws
