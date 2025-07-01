"""
Tests for multi-instrument generation functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from music_gen.models.multi_instrument.generator import (
    MultiTrackGenerator,
    TrackGenerationConfig,
    GenerationResult,
)
from music_gen.models.multi_instrument.config import MultiInstrumentConfig
from music_gen.models.multi_instrument.model import MultiInstrumentMusicGen


class TestMultiTrackGenerator:
    """Test multi-track generator."""

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
    def model(self, config):
        """Create model instance."""
        return MultiInstrumentMusicGen(config)

    @pytest.fixture
    def generator(self, model, config):
        """Create generator instance."""
        return MultiTrackGenerator(model, config)

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "generate")
        assert hasattr(generator, "model")

    def test_generate_single_track(self, generator):
        """Test generating single track."""
        # Mock the model's methods
        with patch.object(generator.model, "encode_text") as mock_encode:
            with patch.object(generator.model, "generate_tokens") as mock_generate:
                mock_encode.return_value = torch.randn(1, 10, 256)
                mock_generate.return_value = torch.randint(0, 100, (1, 100))

                # Create track config
                track_config = TrackGenerationConfig(instrument="piano", volume=0.8)

                # Generate
                with patch.object(generator, "_mix_tracks"):
                    result = generator.generate(
                        prompt="Soft piano melody", track_configs=[track_config], duration=10.0
                    )

                # Result is mocked, just check the call
                assert mock_encode.called
                assert mock_generate.called

    def test_track_generation_config(self):
        """Test track generation configuration."""
        config = TrackGenerationConfig(instrument="guitar", volume=0.9, pan=0.5, reverb=0.3)

        assert config.instrument == "guitar"
        assert config.volume == 0.9
        assert config.pan == 0.5
        assert config.reverb == 0.3
        assert config.start_time == 0.0

    def test_generation_result(self):
        """Test generation result dataclass."""
        audio_tracks = {"piano": torch.randn(1, 32000), "bass": torch.randn(1, 32000)}
        mixed = torch.randn(2, 32000)  # stereo

        result = GenerationResult(
            audio_tracks=audio_tracks,
            mixed_audio=mixed,
            mixing_params={},
            track_configs=[],
            sample_rate=32000,
        )

        assert result.audio_tracks == audio_tracks
        assert result.mixed_audio.shape == (2, 32000)
        assert result.sample_rate == 32000


class TestMultiInstrumentMusicGen:
    """Test multi-instrument model."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MultiInstrumentConfig(num_instruments=8, hidden_size=256, num_layers=4, num_heads=8)

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return MultiInstrumentMusicGen(config)

    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert model is not None
        assert model.config == config
        assert hasattr(model, "encode_text")
        assert hasattr(model, "generate_tokens")

    def test_encode_text(self, model):
        """Test text encoding."""
        with patch.object(model.text_encoder, "encode") as mock_encode:
            mock_encode.return_value = torch.randn(1, 10, 256)

            embeddings = model.encode_text(["Test prompt"])

            assert mock_encode.called
            assert embeddings is not None


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
            num_instruments=16, hidden_size=1024, num_heads=16, sample_rate=48000
        )

        assert config.num_instruments == 16
        assert config.hidden_size == 1024
        assert config.num_heads == 16
        assert config.sample_rate == 48000

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that hidden_size must be divisible by num_heads
        with pytest.raises(ValueError):
            MultiInstrumentConfig(hidden_size=513, num_heads=8)  # Not divisible by 8


@pytest.mark.integration
class TestMultiInstrumentIntegration:
    """Integration tests for multi-instrument generation."""

    def test_multi_track_generation_integration(self):
        """Test multi-track generation integration."""
        config = MultiInstrumentConfig(num_instruments=3, hidden_size=256, num_layers=4)

        model = MultiInstrumentMusicGen(config)
        generator = MultiTrackGenerator(model, config)

        # Create track configs
        track_configs = [
            TrackGenerationConfig(instrument="piano", volume=0.8),
            TrackGenerationConfig(instrument="bass", volume=1.0),
            TrackGenerationConfig(instrument="drums", volume=0.7),
        ]

        # Mock the generation pipeline
        with patch.object(generator.model, "encode_text") as mock_encode:
            with patch.object(generator.model, "generate_tokens") as mock_generate:
                with patch.object(generator, "_mix_tracks") as mock_mix:
                    mock_encode.return_value = torch.randn(1, 10, 256)
                    mock_generate.return_value = torch.randint(0, 100, (1, 100))
                    mock_mix.return_value = torch.randn(2, 96000)

                    # Run generation (mocked)
                    result = generator.generate(
                        prompt="Jazz trio performance", track_configs=track_configs, duration=3.0
                    )

                    # Verify calls
                    assert mock_encode.call_count >= 1
                    assert mock_generate.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__])
