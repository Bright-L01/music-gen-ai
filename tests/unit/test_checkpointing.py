"""
Tests for music_gen.training.checkpointing
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.training.checkpointing import *


class TestCheckpointingModel:
    """Test checkpointing model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")
