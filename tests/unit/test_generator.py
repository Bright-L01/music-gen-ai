"""
Tests for music_gen.streaming.generator
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.streaming.generator import *


class TestGeneratorModel:
    """Test generator model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")
