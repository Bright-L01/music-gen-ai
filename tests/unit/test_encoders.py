"""
Tests for music_gen.models.encoders
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.models.encoders import *


class TestEncodersModel:
    """Test encoders model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_t5textencoder_creation(self, device):
        """Test T5TextEncoder model creation."""
        model = T5TextEncoder()
        assert isinstance(model, nn.Module)

    def test_t5textencoder_forward(self, device):
        """Test T5TextEncoder forward pass."""
        model = T5TextEncoder().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

    def test_conditioningencoder_creation(self, device):
        """Test ConditioningEncoder model creation."""
        model = ConditioningEncoder()
        assert isinstance(model, nn.Module)

    def test_conditioningencoder_forward(self, device):
        """Test ConditioningEncoder forward pass."""
        model = ConditioningEncoder().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

    def test_multimodalencoder_creation(self, device):
        """Test MultiModalEncoder model creation."""
        model = MultiModalEncoder()
        assert isinstance(model, nn.Module)

    def test_multimodalencoder_forward(self, device):
        """Test MultiModalEncoder forward pass."""
        model = MultiModalEncoder().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass
