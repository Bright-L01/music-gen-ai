"""
Tests for multi-instrument API endpoints.
"""

import pytest

from music_gen.api.endpoints.multi_instrument import router, MultiInstrumentRequest, MultiInstrumentResponse, InstrumentTrack


class TestMultiInstrumentApi:
    """Test cases for multi-instrument API endpoints."""

    def test_router_exists(self):
        """Test that multi-instrument router exists."""
        assert router is not None
    
    def test_multi_instrument_models(self):
        """Test that request/response models exist."""
        # Test instrument track model
        track = InstrumentTrack(
            instrument="piano",
            prompt="Soft piano melody",
            volume=0.8
        )
        assert track.instrument == "piano"
        assert track.prompt == "Soft piano melody"
        assert track.volume == 0.8
        
        # Test request model
        request = MultiInstrumentRequest(
            tracks=[
                {"instrument": "piano", "prompt": "Piano melody"},
                {"instrument": "drums", "prompt": "Drum beat"}
            ],
            duration=30.0
        )
        assert len(request.tracks) == 2
        assert request.duration == 30.0
