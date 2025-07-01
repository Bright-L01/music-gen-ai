"""
Tests for music_gen.models.encodec.audio_tokenizer
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.models.encodec.audio_tokenizer import *


class TestAudioTokenizerModel:
    """Test audio_tokenizer model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_encodectokenizer_creation(self, device):
        """Test EnCodecTokenizer model creation."""
        model = EnCodecTokenizer()
        assert isinstance(model, nn.Module)

    def test_encodectokenizer_forward(self, device):
        """Test EnCodecTokenizer forward pass."""
        model = EnCodecTokenizer().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

    def test_multiresolutiontokenizer_creation(self, device):
        """Test MultiResolutionTokenizer model creation."""
        model = MultiResolutionTokenizer()
        assert isinstance(model, nn.Module)

    def test_multiresolutiontokenizer_forward(self, device):
        """Test MultiResolutionTokenizer forward pass."""
        model = MultiResolutionTokenizer().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass
