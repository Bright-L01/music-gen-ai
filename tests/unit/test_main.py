"""
Tests for api/main.py
"""

import pytest

from music_gen.api.app import app
from music_gen.api.endpoints import generation, health, multi_instrument, streaming


class TestApp:
    """Test cases for consolidated API app."""

    def test_app_exists(self):
        """Test that FastAPI app exists."""
        assert app is not None
        assert app.title == "Music Gen AI API"
    
    def test_endpoints_imported(self):
        """Test that all endpoint modules are imported."""
        assert generation is not None
        assert health is not None
        assert multi_instrument is not None
        assert streaming is not None
