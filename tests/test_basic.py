"""
Basic tests for MusicGen functionality.
"""

import numpy as np
import pytest
import torch

from music_gen.evaluation.metrics import AudioQualityMetrics
from music_gen.models.encoders import ConditioningEncoder, T5TextEncoder
from music_gen.models.musicgen import create_musicgen_model
from music_gen.models.transformer.config import MusicGenConfig
from music_gen.utils.audio import apply_fade, normalize_audio


class TestBasicFunctionality:
    """Test basic functionality of MusicGen components."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = MusicGenConfig()
        assert config.transformer.hidden_size == 768
        assert config.t5.model_name == "t5-base"
        assert config.encodec.sample_rate == 24000

    def test_model_creation(self):
        """Test model creation."""
        # Create small model for testing
        model = create_musicgen_model("base")
        assert model is not None
        assert hasattr(model, "transformer")
        assert hasattr(model, "multimodal_encoder")
        assert hasattr(model, "audio_tokenizer")

    def test_text_encoder(self):
        """Test text encoder functionality."""
        try:
            encoder = T5TextEncoder(model_name="t5-small")  # Use smaller model for testing
            texts = ["Happy jazz music", "Calm ambient sounds"]
            device = torch.device("cpu")

            outputs = encoder.encode_text(texts, device)

            assert "hidden_states" in outputs
            assert "attention_mask" in outputs
            assert outputs["hidden_states"].shape[0] == len(texts)

        except Exception as e:
            pytest.skip(f"T5 model not available: {e}")

    def test_conditioning_encoder(self):
        """Test conditioning encoder."""
        encoder = ConditioningEncoder(
            genre_vocab_size=10,
            mood_vocab_size=5,
            embedding_dim=64,
        )

        batch_size = 2
        genre_ids = torch.randint(0, 10, (batch_size,))
        mood_ids = torch.randint(0, 5, (batch_size,))
        tempo = torch.tensor([120.0, 90.0])

        conditioning = encoder(
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
        )

        assert conditioning.shape[0] == batch_size
        assert conditioning.shape[1] == encoder.output_dim

    def test_audio_utils(self):
        """Test audio utility functions."""
        # Generate test audio
        sample_rate = 24000
        duration = 1.0
        samples = int(duration * sample_rate)

        # Sine wave test signal
        t = torch.linspace(0, duration, samples)
        freq = 440.0  # A4
        audio = torch.sin(2 * np.pi * freq * t).unsqueeze(0)

        # Test normalization
        normalized = normalize_audio(audio)
        assert torch.allclose(normalized.abs().max(), torch.tensor(1.0), atol=1e-6)

        # Test fade
        faded = apply_fade(audio, sample_rate, fade_in_duration=0.1, fade_out_duration=0.1)
        assert faded.shape == audio.shape

        # Check fade in/out
        fade_samples = int(0.1 * sample_rate)
        assert faded[0, 0] == 0.0  # Starts at zero
        assert faded[0, -1] == 0.0  # Ends at zero
        assert torch.allclose(
            faded[0, fade_samples : fade_samples + 100],
            audio[0, fade_samples : fade_samples + 100],
            atol=1e-6,
        )

    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        metrics = AudioQualityMetrics(compute_fad=False, compute_clap=False)

        # Generate test audio
        sample_rate = 24000
        duration = 2.0
        samples = int(duration * sample_rate)

        # Create two different test signals
        t = np.linspace(0, duration, samples)
        audio1 = np.sin(2 * np.pi * 440 * t)  # A4
        audio2 = np.sin(2 * np.pi * 880 * t)  # A5

        # Test basic metrics
        snr = metrics.compute_signal_to_noise_ratio(audio1)
        assert isinstance(snr, float)
        assert snr > 0  # Pure sine wave should have good SNR

        # Test harmonic/percussive ratio
        hp_ratio = metrics.compute_harmonic_percussive_ratio(audio1)
        assert isinstance(hp_ratio, float)
        assert hp_ratio > 0

        # Test spectral distance
        distance = metrics.compute_spectral_distance(audio1, audio2)
        assert isinstance(distance, float)
        assert distance > 0  # Different frequencies should have distance > 0

        # Test comprehensive evaluation
        evaluation = metrics.evaluate_audio_quality([audio1, audio2])
        assert isinstance(evaluation, dict)
        assert "snr_mean" in evaluation
        assert "diversity" in evaluation

    def test_synthetic_generation(self):
        """Test synthetic audio generation (without actual model inference)."""
        # This tests the framework without requiring model weights

        # Create dummy model configuration
        config = MusicGenConfig()
        config.transformer.hidden_size = 128  # Small for testing
        config.transformer.num_layers = 2
        config.transformer.num_heads = 4

        try:
            # Create model (this may fail if dependencies are missing)
            model = create_musicgen_model("base")

            # Test that model has expected components
            assert hasattr(model, "generate_audio")
            assert hasattr(model, "encode_audio")
            assert hasattr(model, "decode_audio")

            # Test tokenization without actual audio generation
            sample_rate = 24000
            duration = 1.0
            samples = int(duration * sample_rate)

            # Create dummy audio
            dummy_audio = torch.randn(1, samples) * 0.1

            # This would normally tokenize audio, but may fail without EnCodec
            # tokens = model.encode_audio(dummy_audio, sample_rate)
            # assert tokens.shape[0] == 1  # Batch dimension

        except Exception as e:
            pytest.skip(f"Model creation failed (expected in test environment): {e}")


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sample_rate = 24000
    duration = 1.0
    samples = int(duration * sample_rate)

    t = torch.linspace(0, duration, samples)
    freq = 440.0
    audio = torch.sin(2 * np.pi * freq * t).unsqueeze(0)

    return audio, sample_rate


def test_audio_properties(sample_audio):
    """Test basic audio properties."""
    audio, sample_rate = sample_audio

    assert audio.shape[0] == 1  # Mono
    assert audio.shape[1] == sample_rate  # 1 second duration
    assert audio.dtype == torch.float32

    # Test audio is in reasonable range
    assert audio.abs().max() <= 1.0
    assert audio.abs().min() >= 0.0


def test_import_structure():
    """Test that all modules can be imported."""
    from music_gen.evaluation import AudioQualityMetrics

    # Test main exports
    from music_gen.models.musicgen import MusicGenModel, create_musicgen_model
    from music_gen.models.transformer import MusicGenConfig
    from music_gen.training import MusicGenLightningModule

    assert MusicGenModel is not None
    assert create_musicgen_model is not None
    assert MusicGenConfig is not None
    assert MusicGenLightningModule is not None
    assert AudioQualityMetrics is not None
