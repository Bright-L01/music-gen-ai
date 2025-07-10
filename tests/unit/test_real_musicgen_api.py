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
        request = GenerationRequest(prompt="Generate upbeat music", duration=10.0)
        assert request.prompt == "Generate upbeat music"
        assert request.duration == 10.0

        # Test response model validation
        response = GenerationResponse(
            task_id="test-task-123",
            status="completed",
            audio_url="/download/test-task-123",
            duration=10.0,
            metadata={"prompt": "Generate upbeat music"},
        )
        assert response.task_id == "test-task-123"
        assert response.status == "completed"
        assert response.audio_url == "/download/test-task-123"
