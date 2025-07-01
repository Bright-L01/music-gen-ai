"""
Tests for music_gen.streaming.session
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_gen.streaming.session import *


class TestSession:
    """Test session streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_setup(self):
        """Test streaming setup."""
        # TODO: Implement streaming tests
        pass

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(return_value='{"type": "test"}')
        return ws
