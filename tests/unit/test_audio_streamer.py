"""
Tests for music_gen.streaming.audio_streamer
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from music_gen.streaming.audio_streamer import *


class TestAudioStreamer:
    """Test audio_streamer streaming functionality."""

    def test_audio_chunk_creation(self):
        """Test AudioChunk creation and properties."""
        # Test with 1D audio
        audio_1d = torch.randn(1000)
        chunk = AudioChunk(
            chunk_id=1, audio=audio_1d, sample_rate=24000, duration=1.0, timestamp=0.0
        )

        assert chunk.chunk_id == 1
        assert chunk.sample_rate == 24000
        assert chunk.duration == 1.0
        assert chunk.audio.dim() == 3  # Should be converted to (1, 1, samples)
        assert chunk.num_samples == 1000
        assert chunk.num_channels == 1

        # Test with 2D audio (channels, samples)
        audio_2d = torch.randn(2, 1000)
        chunk_2d = AudioChunk(
            chunk_id=2, audio=audio_2d, sample_rate=24000, duration=1.0, timestamp=1.0
        )

        assert chunk_2d.audio.dim() == 3  # Should be (1, 2, 1000)
        assert chunk_2d.num_channels == 2
        assert chunk_2d.num_samples == 1000

    def test_audio_chunk_to_numpy(self):
        """Test AudioChunk numpy conversion."""
        audio = torch.randn(1, 1000)
        chunk = AudioChunk(chunk_id=1, audio=audio, sample_rate=24000, duration=1.0, timestamp=0.0)

        numpy_audio = chunk.to_numpy()
        assert isinstance(numpy_audio, np.ndarray)
        assert numpy_audio.shape == (1, 1, 1000)

    def test_crossfade_processor_creation(self):
        """Test CrossfadeProcessor creation."""
        processor = CrossfadeProcessor(fade_duration=0.1, sample_rate=24000)

        assert processor.fade_duration == 0.1
        assert processor.sample_rate == 24000
        assert processor.fade_samples == 2400  # 0.1 * 24000
        assert hasattr(processor, "fade_in")
        assert hasattr(processor, "fade_out")
        assert processor.fade_in.shape == (2400,)
        assert processor.fade_out.shape == (2400,)

    def test_crossfade_curves(self):
        """Test crossfade curve generation."""
        processor = CrossfadeProcessor(fade_duration=0.01, sample_rate=1000)

        # Should have 10 samples for 0.01s at 1000Hz
        assert processor.fade_samples == 10

        # Test fade curves
        fade_in = processor.fade_in
        fade_out = processor.fade_out

        # Fade in should start at 0 and end at 1
        assert fade_in[0] == pytest.approx(0.0, abs=1e-6)
        assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)

        # Fade out should start at 1 and end at 0
        assert fade_out[0] == pytest.approx(1.0, abs=1e-6)
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(return_value='{"type": "test"}')
        return ws
