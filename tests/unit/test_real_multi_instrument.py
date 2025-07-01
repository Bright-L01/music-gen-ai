"""
Tests for multi-instrument generation functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from music_gen.models.multi_instrument.generator import MultiInstrumentGenerator
from music_gen.models.multi_instrument.config import MultiInstrumentConfig
from music_gen.models.multi_instrument.conditioning import InstrumentConditioner


class TestMultiInstrumentGenerator:
    """Test multi-instrument generator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MultiInstrumentConfig(
            num_instruments=4,
            hidden_size=256,
            num_layers=8,
            num_heads=8,
        )

    @pytest.fixture
    def generator(self, config):
        """Create generator instance."""
        return MultiInstrumentGenerator(config)

    def test_generator_initialization(self, generator, config):
        """Test generator initialization."""
        assert generator is not None
        assert generator.config == config
        assert hasattr(generator, "generate")

    def test_generate_single_instrument(self, generator):
        """Test generating single instrument."""
        # Mock the model's generate method
        with patch.object(generator, "model") as mock_model:
            mock_model.generate.return_value = torch.randn(1, 1, 32000)
            
            result = generator.generate(
                prompts=["Soft piano melody"],
                instruments=["piano"],
                duration=10.0
            )
            
            assert result is not None
            assert isinstance(result, dict)
            assert "piano" in result
            assert result["piano"].shape == (1, 1, 32000)

    def test_generate_multiple_instruments(self, generator):
        """Test generating multiple instruments."""
        with patch.object(generator, "model") as mock_model:
            # Mock different outputs for different instruments
            mock_model.generate.side_effect = [
                torch.randn(1, 1, 32000),  # piano
                torch.randn(1, 1, 32000),  # bass
                torch.randn(1, 1, 32000),  # drums
            ]
            
            result = generator.generate(
                prompts=[
                    "Jazz piano chords",
                    "Walking bass line",
                    "Swing drums pattern"
                ],
                instruments=["piano", "bass", "drums"],
                duration=10.0
            )
            
            assert len(result) == 3
            assert all(inst in result for inst in ["piano", "bass", "drums"])
            assert all(audio.shape == (1, 1, 32000) for audio in result.values())

    def test_generate_with_conditioning(self, generator):
        """Test generation with additional conditioning."""
        with patch.object(generator, "model") as mock_model:
            mock_model.generate.return_value = torch.randn(1, 1, 32000)
            
            result = generator.generate(
                prompts=["Upbeat guitar riff"],
                instruments=["guitar"],
                duration=15.0,
                conditioning={
                    "genre": "rock",
                    "tempo": 140,
                    "key": "E minor"
                }
            )
            
            assert result is not None
            assert "guitar" in result


class TestInstrumentConditioner:
    """Test instrument conditioning."""

    @pytest.fixture
    def conditioner(self):
        """Create conditioner instance."""
        return InstrumentConditioner(
            num_instruments=8,
            embedding_dim=256
        )

    def test_conditioner_initialization(self, conditioner):
        """Test conditioner initialization."""
        assert conditioner is not None
        assert conditioner.num_instruments == 8
        assert conditioner.embedding_dim == 256

    def test_get_instrument_embedding(self, conditioner):
        """Test getting instrument embeddings."""
        # Test known instruments
        piano_emb = conditioner.get_embedding("piano")
        assert piano_emb is not None
        assert piano_emb.shape == (256,)
        
        # Test that different instruments have different embeddings
        guitar_emb = conditioner.get_embedding("guitar")
        assert not torch.allclose(piano_emb, guitar_emb)

    def test_combine_embeddings(self, conditioner):
        """Test combining multiple embeddings."""
        embeddings = [
            conditioner.get_embedding("piano"),
            conditioner.get_embedding("bass"),
        ]
        
        combined = conditioner.combine_embeddings(embeddings)
        assert combined is not None
        assert combined.shape == (256,)


class TestMultiInstrumentConfig:
    """Test multi-instrument configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiInstrumentConfig()
        
        assert config.num_instruments == 8
        assert config.max_instruments_per_track == 4
        assert config.hidden_size == 512
        assert config.num_layers == 12

    def test_custom_config(self):
        """Test custom configuration."""
        config = MultiInstrumentConfig(
            num_instruments=16,
            hidden_size=1024,
            num_heads=16,
            sample_rate=48000
        )
        
        assert config.num_instruments == 16
        assert config.hidden_size == 1024
        assert config.num_heads == 16
        assert config.sample_rate == 48000

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that hidden_size must be divisible by num_heads
        with pytest.raises(ValueError):
            MultiInstrumentConfig(
                hidden_size=513,  # Not divisible by 8
                num_heads=8
            )


@pytest.mark.integration
class TestMultiInstrumentIntegration:
    """Integration tests for multi-instrument generation."""

    def test_jazz_trio_generation(self):
        """Test generating a jazz trio."""
        config = MultiInstrumentConfig(
            num_instruments=3,
            hidden_size=256,
            num_layers=6
        )
        
        generator = MultiInstrumentGenerator(config)
        
        with patch.object(generator, "model") as mock_model:
            # Mock realistic outputs
            mock_model.generate.side_effect = [
                torch.randn(1, 1, 96000),  # 3 seconds at 32kHz
                torch.randn(1, 1, 96000),
                torch.randn(1, 1, 96000),
            ]
            
            result = generator.generate(
                prompts=[
                    "Jazz piano comping with seventh chords",
                    "Walking bass line in F major",
                    "Brush drums with swing feel"
                ],
                instruments=["piano", "bass", "drums"],
                duration=3.0,
                conditioning={
                    "genre": "jazz",
                    "tempo": 120,
                    "time_signature": "4/4"
                }
            )
            
            assert len(result) == 3
            assert all(audio.shape == (1, 1, 96000) for audio in result.values())
            
            # Verify model was called with correct parameters
            assert mock_model.generate.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])