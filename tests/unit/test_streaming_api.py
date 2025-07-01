"""
Tests for api/streaming_api.py
"""

import pytest

from music_gen.api.endpoints.streaming import router, StreamingRequest, StreamingResponse


class TestStreamingApi:
    """Test cases for streaming API endpoints."""

    def test_router_exists(self):
        """Test that streaming router exists."""
        assert router is not None
    
    def test_streaming_models(self):
        """Test that request/response models exist."""
        # Test request model
        request = StreamingRequest(
            prompt="Test prompt",
            duration=10.0
        )
        assert request.prompt == "Test prompt"
        assert request.duration == 10.0
        
        # Test response model
        response = StreamingResponse(
            session_id="test-123",
            websocket_url="/api/v1/stream/ws/test-123"
        )
        assert response.session_id == "test-123"
        assert response.websocket_url == "/api/v1/stream/ws/test-123"
