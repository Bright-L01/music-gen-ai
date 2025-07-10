"""
Tests for music_gen.models.transformer.model
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.models.transformer.model import *


class TestModelModel:
    """Test model model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_rotarypositionalencoding_creation(self, device):
        """Test RotaryPositionalEncoding model creation."""
        model = RotaryPositionalEncoding(dim=64, max_seq_len=512)
        assert isinstance(model, nn.Module)
        assert model.dim == 64
        assert model.max_seq_len == 512

    def test_rotarypositionalencoding_forward(self, device):
        """Test RotaryPositionalEncoding forward pass."""
        model = RotaryPositionalEncoding(dim=64, max_seq_len=512).to(device)
        batch_size, seq_len, hidden_size = 2, 50, 64
        x = torch.randn(batch_size, seq_len, hidden_size).to(device)
        cos, sin = model(x, seq_len)
        assert cos.shape == (seq_len, hidden_size)
        assert sin.shape == (seq_len, hidden_size)

    def test_multiheadattention_creation(self, device, test_config):
        """Test MultiHeadAttention model creation."""
        model = MultiHeadAttention(test_config.transformer)
        assert isinstance(model, nn.Module)
        assert model.num_heads == test_config.transformer.num_heads
        assert model.hidden_size == test_config.transformer.hidden_size

    def test_multiheadattention_forward(self, device, test_config):
        """Test MultiHeadAttention forward pass."""
        model = MultiHeadAttention(test_config.transformer).to(device)
        batch_size, seq_len = 2, 50
        hidden_states = torch.randn(batch_size, seq_len, test_config.transformer.hidden_size).to(
            device
        )

        output, attn_weights = model(hidden_states, use_cache=False)
        assert output.shape == hidden_states.shape
        assert attn_weights is None  # Should be None when use_cache=False

    def test_feedforward_creation(self, device, test_config):
        """Test FeedForward model creation."""
        model = FeedForward(test_config.transformer)
        assert isinstance(model, nn.Module)
        assert model.intermediate_size == test_config.transformer.intermediate_size

    def test_feedforward_forward(self, device, test_config):
        """Test FeedForward forward pass."""
        model = FeedForward(test_config.transformer).to(device)
        batch_size, seq_len = 2, 50
        input_tensor = torch.randn(batch_size, seq_len, test_config.transformer.hidden_size).to(
            device
        )
        output = model(input_tensor)
        assert output.shape == input_tensor.shape

    def test_transformerlayer_creation(self, device, test_config):
        """Test TransformerLayer model creation."""
        model = TransformerLayer(test_config.transformer, layer_idx=0)
        assert isinstance(model, nn.Module)
        assert model.layer_idx == 0
        assert hasattr(model, "self_attn")
        assert hasattr(model, "feed_forward")

    def test_transformerlayer_forward(self, device, test_config):
        """Test TransformerLayer forward pass."""
        model = TransformerLayer(test_config.transformer, layer_idx=0).to(device)
        batch_size, seq_len = 2, 50
        hidden_states = torch.randn(batch_size, seq_len, test_config.transformer.hidden_size).to(
            device
        )

        output, past_kv = model(hidden_states, use_cache=True)
        assert output.shape == hidden_states.shape
        assert past_kv is not None

    def test_musicgentransformer_creation(self, device, test_config):
        """Test MusicGenTransformer model creation."""
        model = MusicGenTransformer(test_config.transformer)
        assert isinstance(model, nn.Module)
        assert model.config == test_config.transformer
        assert hasattr(model, "layers")
        assert len(model.layers) == test_config.transformer.num_layers

    def test_musicgentransformer_forward(self, device, test_config):
        """Test MusicGenTransformer forward pass."""
        model = MusicGenTransformer(test_config.transformer).to(device)
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, test_config.transformer.vocab_size, (batch_size, seq_len)).to(
            device
        )

        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, test_config.transformer.vocab_size)
        assert "hidden_states" in outputs
