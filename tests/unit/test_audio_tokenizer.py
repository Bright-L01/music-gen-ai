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
        model_configs = [
            {"model_name": "facebook/encodec_24khz", "sample_rate": 24000},
            {"model_name": "facebook/encodec_32khz", "sample_rate": 32000},
        ]
        model = MultiResolutionTokenizer(model_configs)
        assert isinstance(model, nn.Module)
        assert model.num_tokenizers == 2

    def test_multiresolutiontokenizer_forward(self, device):
        """Test MultiResolutionTokenizer forward pass."""
        model_configs = [
            {"model_name": "facebook/encodec_24khz", "sample_rate": 24000},
        ]
        model = MultiResolutionTokenizer(model_configs).to(device)

        # Test encoding
        audio = torch.randn(1, 1, 24000).to(device)  # 1 second of audio
        results = model(audio=audio, mode="encode")
        assert isinstance(results, list)
        assert len(results) == 1  # One tokenizer result

        # Test decoding
        codes, scales = results[0]
        audio_decoded = model(multi_codes=results, mode="decode")
        assert isinstance(audio_decoded, list)
        assert len(audio_decoded) == 1
