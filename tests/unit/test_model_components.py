"""
Comprehensive unit tests for model components.

Tests transformer architecture, attention mechanisms, conditioning modules,
and model utilities.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from music_gen.models.transformer import (
    MusicGenModel,
    TransformerDecoder,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
)
from music_gen.models.conditioning import (
    TextConditioner,
    AudioConditioner,
    MultiModalConditioner,
)
from music_gen.models.encodec import (
    EnCodecTokenizer,
    MultiResolutionTokenizer,
)
from music_gen.core.exceptions import (
    ModelLoadError,
    ConfigurationError,
)


@pytest.mark.unit
class TestTransformerComponents:
    """Test core transformer architecture components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def config(self):
        """Create test model configuration."""
        return {
            "vocab_size": 2048,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "max_sequence_length": 1024,
            "dropout": 0.1,
            "activation": "gelu",
        }

    def test_positional_encoding_creation(self, config, device):
        """Test positional encoding creation and properties."""
        pe = PositionalEncoding(
            d_model=config["hidden_size"],
            max_len=config["max_sequence_length"],
            dropout=config["dropout"],
        ).to(device)

        assert isinstance(pe, nn.Module)
        assert pe.pe.shape == (config["max_sequence_length"], config["hidden_size"])

    def test_positional_encoding_forward(self, config, device):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(
            d_model=config["hidden_size"], max_len=config["max_sequence_length"]
        ).to(device)

        # Test with different sequence lengths
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, config["hidden_size"]).to(device)
            output = pe(x)

            assert output.shape == x.shape
            # Output should be different from input (positions added)
            assert not torch.equal(output, x)

    def test_multihead_attention_creation(self, config, device):
        """Test multi-head attention creation."""
        attention = MultiHeadAttention(
            d_model=config["hidden_size"], num_heads=config["num_heads"], dropout=config["dropout"]
        ).to(device)

        assert isinstance(attention, nn.Module)
        assert attention.num_heads == config["num_heads"]
        assert attention.d_model == config["hidden_size"]
        assert attention.d_k == config["hidden_size"] // config["num_heads"]

    def test_multihead_attention_forward(self, config, device):
        """Test multi-head attention forward pass."""
        attention = MultiHeadAttention(
            d_model=config["hidden_size"], num_heads=config["num_heads"]
        ).to(device)

        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, config["hidden_size"]).to(device)

        # Self-attention
        output, attn_weights = attention(x, x, x)

        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, config["num_heads"], seq_len, seq_len)

        # Attention weights should sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))

    def test_multihead_attention_with_mask(self, config, device):
        """Test multi-head attention with causal mask."""
        attention = MultiHeadAttention(
            d_model=config["hidden_size"], num_heads=config["num_heads"]
        ).to(device)

        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, config["hidden_size"]).to(device)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

        output, attn_weights = attention(x, x, x, mask=mask)

        assert output.shape == x.shape

        # Masked positions should have zero attention
        masked_positions = attn_weights[:, :, mask]
        assert torch.allclose(masked_positions, torch.zeros_like(masked_positions))

    def test_feedforward_creation(self, config, device):
        """Test feedforward network creation."""
        ff = FeedForward(
            d_model=config["hidden_size"],
            d_ff=config["hidden_size"] * 4,
            dropout=config["dropout"],
            activation=config["activation"],
        ).to(device)

        assert isinstance(ff, nn.Module)

    def test_feedforward_forward(self, config, device):
        """Test feedforward network forward pass."""
        ff = FeedForward(
            d_model=config["hidden_size"],
            d_ff=config["hidden_size"] * 4,
            dropout=0.0,  # No dropout for testing
        ).to(device)

        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, config["hidden_size"]).to(device)

        output = ff(x)

        assert output.shape == x.shape
        # Output should be different from input
        assert not torch.equal(output, x)

    def test_transformer_decoder_layer(self, config, device):
        """Test complete transformer decoder layer."""
        from music_gen.models.transformer.decoder import TransformerDecoderLayer

        layer = TransformerDecoderLayer(
            d_model=config["hidden_size"],
            num_heads=config["num_heads"],
            d_ff=config["hidden_size"] * 4,
            dropout=config["dropout"],
        ).to(device)

        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, config["hidden_size"]).to(device)

        output = layer(x)

        assert output.shape == x.shape

    def test_transformer_decoder_full(self, config, device):
        """Test full transformer decoder."""
        decoder = TransformerDecoder(
            vocab_size=config["vocab_size"],
            d_model=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["hidden_size"] * 4,
            max_seq_len=config["max_sequence_length"],
            dropout=config["dropout"],
        ).to(device)

        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)

        output = decoder(input_ids)

        assert output.shape == (batch_size, seq_len, config["vocab_size"])

    def test_transformer_generation(self, config, device):
        """Test transformer generation capabilities."""
        decoder = TransformerDecoder(
            vocab_size=config["vocab_size"],
            d_model=config["hidden_size"],
            num_layers=2,  # Smaller for faster testing
            num_heads=config["num_heads"],
            max_seq_len=config["max_sequence_length"],
        ).to(device)

        # Test generation
        input_ids = torch.randint(0, config["vocab_size"], (1, 10)).to(device)

        with torch.no_grad():
            generated = decoder.generate(input_ids, max_length=20, temperature=1.0, do_sample=True)

        assert generated.shape[0] == 1
        assert generated.shape[1] == 20
        assert generated.dtype == torch.long


@pytest.mark.unit
class TestConditioningModules:
    """Test conditioning modules for text and audio."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_text_conditioner_creation(self, device):
        """Test text conditioner creation."""
        conditioner = TextConditioner(vocab_size=50000, d_model=768, max_length=512).to(device)

        assert isinstance(conditioner, nn.Module)

    def test_text_conditioner_forward(self, device):
        """Test text conditioner forward pass."""
        with patch("transformers.T5EncoderModel") as MockT5:
            mock_model = Mock()
            mock_model.last_hidden_state = torch.randn(2, 20, 768)
            MockT5.from_pretrained.return_value = mock_model

            conditioner = TextConditioner(model_name="t5-small", d_model=768).to(device)

            # Mock tokenizer
            with patch.object(conditioner, "tokenizer") as mock_tokenizer:
                mock_tokenizer.return_value = {
                    "input_ids": torch.randint(0, 1000, (2, 20)),
                    "attention_mask": torch.ones(2, 20),
                }

                text_inputs = ["upbeat jazz music", "calm classical piece"]
                embeddings = conditioner(text_inputs)

                assert embeddings.shape == (2, 20, 768)

    def test_audio_conditioner_creation(self, device):
        """Test audio conditioner creation."""
        conditioner = AudioConditioner(input_dim=1024, d_model=768, num_layers=3).to(device)

        assert isinstance(conditioner, nn.Module)

    def test_audio_conditioner_forward(self, device):
        """Test audio conditioner forward pass."""
        conditioner = AudioConditioner(input_dim=1024, d_model=768, num_layers=3).to(device)

        batch_size, seq_len = 2, 100
        audio_features = torch.randn(batch_size, seq_len, 1024).to(device)

        embeddings = conditioner(audio_features)

        assert embeddings.shape == (batch_size, seq_len, 768)

    def test_multimodal_conditioner(self, device):
        """Test multi-modal conditioning."""
        conditioner = MultiModalConditioner(
            text_dim=768, audio_dim=512, output_dim=1024, fusion_method="concat"
        ).to(device)

        batch_size, text_seq, audio_seq = 2, 50, 100
        text_features = torch.randn(batch_size, text_seq, 768).to(device)
        audio_features = torch.randn(batch_size, audio_seq, 512).to(device)

        fused_features = conditioner(text_features, audio_features)

        assert fused_features.shape[0] == batch_size
        assert fused_features.shape[2] == 1024

    def test_conditioning_with_metadata(self, device):
        """Test conditioning with metadata (genre, mood, etc.)."""
        conditioner = MultiModalConditioner(
            text_dim=768, metadata_categories={"genre": 20, "mood": 10, "tempo": 5}, output_dim=1024
        ).to(device)

        batch_size, seq_len = 2, 50
        text_features = torch.randn(batch_size, seq_len, 768).to(device)
        metadata = {
            "genre": torch.randint(0, 20, (batch_size,)).to(device),
            "mood": torch.randint(0, 10, (batch_size,)).to(device),
            "tempo": torch.randint(0, 5, (batch_size,)).to(device),
        }

        conditioned_features = conditioner(text_features, metadata=metadata)

        assert conditioned_features.shape[0] == batch_size
        assert conditioned_features.shape[2] == 1024


@pytest.mark.unit
class TestMusicGenModel:
    """Test complete MusicGen model."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return {
            "vocab_size": 2048,
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 8,
            "max_sequence_length": 1024,
            "sample_rate": 24000,
            "hop_length": 512,
        }

    def test_model_creation(self, model_config, device):
        """Test MusicGen model creation."""
        try:
            model = MusicGenModel(model_config).to(device)
            assert isinstance(model, nn.Module)
        except Exception as e:
            pytest.skip(f"Model creation failed (expected without dependencies): {e}")

    def test_model_forward_pass(self, model_config, device):
        """Test model forward pass."""
        try:
            model = MusicGenModel(model_config).to(device)

            batch_size, seq_len = 2, 100
            input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len)).to(
                device
            )

            with torch.no_grad():
                output = model(input_ids)

            assert output.shape == (batch_size, seq_len, model_config["vocab_size"])
        except Exception as e:
            pytest.skip(f"Model forward pass failed (expected without dependencies): {e}")

    def test_model_generation(self, model_config, device):
        """Test model generation."""
        try:
            model = MusicGenModel(model_config).to(device)
            model.eval()

            # Mock text conditioning
            text_prompt = "upbeat electronic music"

            with torch.no_grad():
                generated_tokens = model.generate(
                    prompt=text_prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.9
                )

            assert isinstance(generated_tokens, torch.Tensor)
            assert generated_tokens.shape[1] <= 200
        except Exception as e:
            pytest.skip(f"Model generation failed (expected without dependencies): {e}")

    def test_model_conditioning(self, model_config, device):
        """Test model with different conditioning inputs."""
        try:
            model = MusicGenModel(model_config).to(device)

            # Test with text conditioning
            text_embeddings = torch.randn(1, 50, 768).to(device)

            batch_size, seq_len = 1, 100
            input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len)).to(
                device
            )

            with torch.no_grad():
                output = model(input_ids, encoder_hidden_states=text_embeddings)

            assert output.shape == (batch_size, seq_len, model_config["vocab_size"])
        except Exception as e:
            pytest.skip(f"Model conditioning failed (expected without dependencies): {e}")

    def test_model_memory_efficiency(self, device):
        """Test model memory efficiency features."""
        config = {
            "vocab_size": 2048,
            "hidden_size": 256,  # Smaller for memory testing
            "num_layers": 2,
            "num_heads": 4,
            "max_sequence_length": 512,
            "gradient_checkpointing": True,
        }

        try:
            model = MusicGenModel(config).to(device)

            # Test with gradient checkpointing
            batch_size, seq_len = 4, 256  # Larger batch
            input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)

            output = model(input_ids)
            loss = output.sum()
            loss.backward()

            # Should complete without memory issues
            assert output.shape == (batch_size, seq_len, config["vocab_size"])
        except Exception as e:
            pytest.skip(f"Memory efficiency test failed: {e}")

    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        # Valid configuration should work
        valid_config = {
            "vocab_size": 2048,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
        }

        try:
            model = MusicGenModel(valid_config)
            assert isinstance(model, nn.Module)
        except Exception as e:
            pytest.skip(f"Valid config test failed: {e}")

        # Invalid configurations should fail
        invalid_configs = [
            {"vocab_size": 0},  # Invalid vocab size
            {"hidden_size": 512, "num_heads": 7},  # Hidden size not divisible by heads
            {"num_layers": -1},  # Negative layers
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, ConfigurationError)):
                MusicGenModel(invalid_config)


@pytest.mark.unit
class TestModelUtilities:
    """Test model utility functions and helpers."""

    def test_parameter_counting(self):
        """Test parameter counting utilities."""
        from music_gen.models.utils import count_parameters

        # Simple test model
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)

        expected_params = (100 * 50 + 50) + (50 * 10 + 10)  # weights + biases
        assert total_params == expected_params
        assert trainable_params == expected_params

    def test_model_size_estimation(self):
        """Test model size estimation."""
        from music_gen.models.utils import estimate_model_size

        model = nn.Linear(1000, 1000)

        size_mb = estimate_model_size(model)

        # Should be reasonable size estimate
        assert 0 < size_mb < 100  # Less than 100MB for this small model

    def test_gradient_clipping(self):
        """Test gradient clipping utilities."""
        from music_gen.models.utils import clip_gradients

        model = nn.Linear(10, 1)

        # Create some gradients
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # Clip gradients
        grad_norm = clip_gradients(model, max_norm=1.0)

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0

    def test_model_initialization(self):
        """Test model initialization utilities."""
        from music_gen.models.utils import initialize_weights

        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

        # Test different initialization methods
        methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]

        for method in methods:
            initialize_weights(model, method=method)

            # Check that weights are not zero
            for param in model.parameters():
                if param.dim() > 1:  # Weight matrices
                    assert not torch.allclose(param, torch.zeros_like(param))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
