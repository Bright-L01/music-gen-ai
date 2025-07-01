"""
Tests for music_gen.models.multi_instrument.conditioning
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.models.multi_instrument.conditioning import *


class TestConditioningModel:
    """Test conditioning model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_instrumentembedding_creation(self, device):
        """Test InstrumentEmbedding model creation."""
        model = InstrumentEmbedding()
        assert isinstance(model, nn.Module)

    def test_instrumentembedding_forward(self, device):
        """Test InstrumentEmbedding forward pass."""
        model = InstrumentEmbedding().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

    def test_instrumentconditioner_creation(self, device):
        """Test InstrumentConditioner model creation."""
        model = InstrumentConditioner()
        assert isinstance(model, nn.Module)

    def test_instrumentconditioner_forward(self, device):
        """Test InstrumentConditioner forward pass."""
        model = InstrumentConditioner().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

    def test_instrumentclassifier_creation(self, device):
        """Test InstrumentClassifier model creation."""
        model = InstrumentClassifier()
        assert isinstance(model, nn.Module)

    def test_instrumentclassifier_forward(self, device):
        """Test InstrumentClassifier forward pass."""
        model = InstrumentClassifier().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass
