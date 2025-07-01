"""
Tests for generation API endpoints.
"""

import pytest

from music_gen.api.endpoints.generation import router, GenerationRequest, GenerationResponse


class TestGenerationApi:
    """Test cases for generation API endpoints."""

    def test_router_exists(self):
        """Test that generation router exists."""
        assert router is not None
    
    def test_generation_models(self):
        """Test that request/response models exist."""
        # Test request model
        request = GenerationRequest(
            prompt="Generate upbeat music",
            duration=10.0
        )
        assert request.prompt == "Generate upbeat music"
        assert request.duration == 10.0
        
        # Test response model validation
        response = GenerationResponse(
            audio_url="/outputs/test.wav",
            duration=10.0,
            prompt="Generate upbeat music",
            format="wav",
            sample_rate=32000
        )
        assert response.audio_url == "/outputs/test.wav"
        assert response.duration == 10.0
