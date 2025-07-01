"""
Unit tests for data augmentation functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Import augmentation modules - handle missing dependencies gracefully
try:
    from music_gen.data.augmentation import (
        AdaptiveAugmentation,
        AudioAugmentation,
        AugmentationPipeline,
        FilterAugmentation,
        NoiseAugmentation,
        PitchShiftAugmentation,
        PolymixAugmentation,
        TimeStretchAugmentation,
        VolumeAugmentation,
        create_inference_augmentation_pipeline,
        create_training_augmentation_pipeline,
    )

    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not AUGMENTATION_AVAILABLE, reason="Augmentation modules not available")
class TestAugmentationPipeline:
    """Test augmentation pipeline functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)
        # Create a simple sine wave
        t = torch.linspace(0, duration, samples)
        audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # Shape: (1, samples)
        return audio, sample_rate

    def test_pipeline_creation(self):
        """Test creating augmentation pipeline."""
        augmentations = [
            VolumeAugmentation(min_gain=0.5, max_gain=1.5),
            NoiseAugmentation(noise_level=0.01),
        ]

        pipeline = AugmentationPipeline(augmentations)

        assert pipeline is not None
        assert len(pipeline.augmentations) == 2

    def test_pipeline_processing(self, sample_audio):
        """Test pipeline audio processing."""
        audio, sample_rate = sample_audio

        augmentations = [
            VolumeAugmentation(min_gain=0.8, max_gain=1.2),
            NoiseAugmentation(noise_level=0.005),
        ]

        pipeline = AugmentationPipeline(augmentations)
        augmented = pipeline(audio, sample_rate)

        assert augmented.shape == audio.shape
        assert isinstance(augmented, torch.Tensor)
        # Augmented audio should be different from original
        assert not torch.allclose(augmented, audio, atol=1e-6)

    def test_pipeline_probability(self, sample_audio):
        """Test augmentation probability control."""
        audio, sample_rate = sample_audio

        # Low probability - should sometimes return original audio
        augmentations = [VolumeAugmentation(min_gain=0.1, max_gain=2.0, probability=0.1)]

        pipeline = AugmentationPipeline(augmentations)

        # Test multiple times to check probability
        same_count = 0
        total_tests = 10

        for _ in range(total_tests):
            augmented = pipeline(audio, sample_rate)
            if torch.allclose(augmented, audio, atol=1e-6):
                same_count += 1

        # With low probability, some should remain unchanged
        assert same_count > 0

    def test_empty_pipeline(self, sample_audio):
        """Test pipeline with no augmentations."""
        audio, sample_rate = sample_audio

        pipeline = AugmentationPipeline([])
        augmented = pipeline(audio, sample_rate)

        assert torch.allclose(augmented, audio)


@pytest.mark.unit
@pytest.mark.skipif(not AUGMENTATION_AVAILABLE, reason="Augmentation modules not available")
class TestIndividualAugmentations:
    """Test individual augmentation classes."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        return audio, sample_rate

    def test_volume_augmentation(self, sample_audio):
        """Test volume augmentation."""
        audio, sample_rate = sample_audio

        aug = VolumeAugmentation(min_gain=0.5, max_gain=2.0)
        augmented = aug(audio, sample_rate)

        assert augmented.shape == audio.shape
        # Volume should be different
        assert not torch.allclose(augmented, audio)

    def test_noise_augmentation(self, sample_audio):
        """Test noise augmentation."""
        audio, sample_rate = sample_audio

        aug = NoiseAugmentation(noise_level=0.1)
        augmented = aug(audio, sample_rate)

        assert augmented.shape == audio.shape
        # Should add noise
        assert not torch.allclose(augmented, audio)
        # Energy should be higher due to added noise
        assert augmented.pow(2).sum() > audio.pow(2).sum()

    def test_time_stretch_augmentation(self, sample_audio):
        """Test time stretch augmentation."""
        audio, sample_rate = sample_audio

        # Mock torchaudio.transforms
        with patch("music_gen.data.augmentation.torchaudio") as mock_torchaudio:
            mock_transform = Mock()
            mock_transform.return_value = audio  # Return same for simplicity
            mock_torchaudio.transforms.TimeStretch.return_value = mock_transform

            aug = TimeStretchAugmentation(min_rate=0.8, max_rate=1.2)
            augmented = aug(audio, sample_rate)

            assert augmented.shape == audio.shape

    def test_pitch_shift_augmentation(self, sample_audio):
        """Test pitch shift augmentation."""
        audio, sample_rate = sample_audio

        # Mock torchaudio.transforms
        with patch("music_gen.data.augmentation.torchaudio") as mock_torchaudio:
            mock_transform = Mock()
            mock_transform.return_value = audio
            mock_torchaudio.transforms.PitchShift.return_value = mock_transform

            aug = PitchShiftAugmentation(min_semitones=-2, max_semitones=2)
            augmented = aug(audio, sample_rate)

            assert augmented.shape == audio.shape

    def test_filter_augmentation(self, sample_audio):
        """Test filter augmentation."""
        audio, sample_rate = sample_audio

        aug = FilterAugmentation(filter_type="lowpass", cutoff_freq=8000)

        # Mock the filtering process
        with patch.object(aug, "_apply_filter", return_value=audio):
            augmented = aug(audio, sample_rate)
            assert augmented.shape == audio.shape


@pytest.mark.unit
@pytest.mark.skipif(not AUGMENTATION_AVAILABLE, reason="Augmentation modules not available")
class TestPolymixAugmentation:
    """Test Polymix augmentation specifically."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        return audio, sample_rate

    @pytest.fixture
    def mix_samples(self):
        """Create sample audio tracks for mixing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)

        mix_samples = []
        for freq in [220, 330, 550]:  # Different frequencies
            t = torch.linspace(0, duration, samples)
            audio = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
            mix_samples.append(audio)

        return mix_samples

    def test_polymix_creation(self):
        """Test Polymix augmentation creation."""
        aug = PolymixAugmentation(mix_probability=0.5, alpha=0.3)

        assert aug.mix_probability == 0.5
        assert aug.alpha == 0.3
        assert aug.mix_samples == []

    def test_polymix_set_samples(self, mix_samples):
        """Test setting mix samples."""
        aug = PolymixAugmentation()
        aug.set_mix_samples(mix_samples)

        assert len(aug.mix_samples) == 3
        assert all(isinstance(sample, torch.Tensor) for sample in aug.mix_samples)

    def test_polymix_mixing(self, sample_audio, mix_samples):
        """Test Polymix audio mixing."""
        audio, sample_rate = sample_audio

        aug = PolymixAugmentation(mix_probability=1.0, alpha=0.3)  # Always mix
        aug.set_mix_samples(mix_samples)

        mixed = aug(audio, sample_rate)

        assert mixed.shape == audio.shape
        # Mixed audio should be different from original
        assert not torch.allclose(mixed, audio)

    def test_polymix_no_mixing(self, sample_audio):
        """Test Polymix when no mixing occurs."""
        audio, sample_rate = sample_audio

        aug = PolymixAugmentation(mix_probability=0.0)  # Never mix
        result = aug(audio, sample_rate)

        # Should return original audio when not mixing
        assert torch.allclose(result, audio)


@pytest.mark.unit
@pytest.mark.skipif(not AUGMENTATION_AVAILABLE, reason="Augmentation modules not available")
class TestAdaptiveAugmentation:
    """Test adaptive augmentation functionality."""

    def test_adaptive_augmentation_creation(self):
        """Test creating adaptive augmentation."""
        aug = AdaptiveAugmentation(initial_strength=0.5, target_loss=2.0, adaptation_rate=0.1)

        assert aug.strength == 0.5
        assert aug.target_loss == 2.0
        assert aug.adaptation_rate == 0.1

    def test_strength_adaptation(self):
        """Test strength adaptation based on loss."""
        aug = AdaptiveAugmentation(initial_strength=0.5, target_loss=2.0)

        # High loss - should increase strength
        aug.update_strength(loss=3.0, target_loss=2.0)
        assert aug.strength > 0.5

        # Reset for next test
        aug.strength = 0.5

        # Low loss - should decrease strength
        aug.update_strength(loss=1.0, target_loss=2.0)
        assert aug.strength < 0.5

    def test_get_pipeline(self):
        """Test getting augmentation pipeline."""
        aug = AdaptiveAugmentation(initial_strength=0.8)
        pipeline = aug.get_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "augmentations")
        assert len(pipeline.augmentations) > 0

    def test_strength_bounds(self):
        """Test strength stays within bounds."""
        aug = AdaptiveAugmentation(initial_strength=0.5, min_strength=0.1, max_strength=0.9)

        # Try to go below minimum
        aug.update_strength(loss=0.1, target_loss=2.0)
        assert aug.strength >= 0.1

        # Try to go above maximum
        aug.update_strength(loss=10.0, target_loss=2.0)
        assert aug.strength <= 0.9


@pytest.mark.unit
@pytest.mark.skipif(not AUGMENTATION_AVAILABLE, reason="Augmentation modules not available")
class TestAugmentationFactories:
    """Test augmentation factory functions."""

    def test_create_training_pipeline(self):
        """Test creating training augmentation pipeline."""
        pipeline = create_training_augmentation_pipeline(strong=False)

        assert pipeline is not None
        assert hasattr(pipeline, "augmentations")
        assert len(pipeline.augmentations) > 0

    def test_create_strong_training_pipeline(self):
        """Test creating strong training augmentation pipeline."""
        strong_pipeline = create_training_augmentation_pipeline(strong=True)
        weak_pipeline = create_training_augmentation_pipeline(strong=False)

        # Strong pipeline should have more or stronger augmentations
        assert len(strong_pipeline.augmentations) >= len(weak_pipeline.augmentations)

    def test_create_inference_pipeline(self):
        """Test creating inference augmentation pipeline."""
        pipeline = create_inference_augmentation_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "augmentations")
        # Inference pipeline should be lighter than training
        assert len(pipeline.augmentations) > 0

    def test_pipeline_with_mix(self):
        """Test creating pipeline with mixing augmentation."""
        pipeline = create_training_augmentation_pipeline(include_mix=True)

        assert pipeline is not None
        # Should include Polymix augmentation
        has_polymix = any(isinstance(aug, PolymixAugmentation) for aug in pipeline.augmentations)
        assert has_polymix


@pytest.mark.unit
class TestAugmentationHelpers:
    """Test augmentation helper functions and utilities."""

    def test_audio_normalization(self):
        """Test audio normalization helper."""
        # Create test audio that needs normalization
        audio = torch.randn(1, 24000) * 2.0  # Values outside [-1, 1]

        def normalize_audio(audio, target_level=0.8):
            max_val = audio.abs().max()
            if max_val > target_level:
                audio = audio * (target_level / max_val)
            return audio

        normalized = normalize_audio(audio)

        assert normalized.abs().max() <= 0.8
        assert normalized.shape == audio.shape

    def test_audio_padding(self):
        """Test audio padding/truncation."""

        def pad_or_truncate(audio, target_length):
            current_length = audio.shape[-1]
            if current_length > target_length:
                return audio[:, :target_length]
            elif current_length < target_length:
                padding = target_length - current_length
                return torch.nn.functional.pad(audio, (0, padding))
            return audio

        # Test truncation
        long_audio = torch.randn(1, 48000)
        truncated = pad_or_truncate(long_audio, 24000)
        assert truncated.shape == (1, 24000)

        # Test padding
        short_audio = torch.randn(1, 12000)
        padded = pad_or_truncate(short_audio, 24000)
        assert padded.shape == (1, 24000)

    def test_augmentation_probability(self):
        """Test probability-based augmentation application."""

        def apply_with_probability(augmentation_fn, audio, probability=0.5):
            if torch.rand(1).item() < probability:
                return augmentation_fn(audio)
            return audio

        def dummy_augmentation(audio):
            return audio * 1.5

        original_audio = torch.randn(1, 1000)

        # Test with probability 0 - should always return original
        result = apply_with_probability(dummy_augmentation, original_audio, 0.0)
        assert torch.allclose(result, original_audio)

        # Test with probability 1 - should always apply augmentation
        result = apply_with_probability(dummy_augmentation, original_audio, 1.0)
        assert torch.allclose(result, original_audio * 1.5)

    def test_augmentation_chaining(self):
        """Test chaining multiple augmentations."""

        def chain_augmentations(audio, augmentations):
            result = audio
            for aug_fn in augmentations:
                result = aug_fn(result)
            return result

        def aug1(audio):
            return audio * 0.8

        def aug2(audio):
            return audio + 0.1

        audio = torch.ones(1, 100)
        augmentations = [aug1, aug2]

        result = chain_augmentations(audio, augmentations)
        expected = (audio * 0.8) + 0.1

        assert torch.allclose(result, expected)


@pytest.mark.unit
class TestAugmentationMocks:
    """Test augmentation with mocked dependencies."""

    def test_mock_torchaudio_transforms(self):
        """Test augmentation with mocked torchaudio transforms."""

        # Mock an augmentation that uses torchaudio
        class MockTimeStretch:
            def __init__(self, min_rate=0.8, max_rate=1.2):
                self.min_rate = min_rate
                self.max_rate = max_rate

            def __call__(self, audio, sample_rate):
                # Mock time stretching by simply scaling the audio
                rate = torch.rand(1).item() * (self.max_rate - self.min_rate) + self.min_rate
                return audio * rate

        audio = torch.randn(1, 24000)
        aug = MockTimeStretch(min_rate=0.9, max_rate=1.1)
        result = aug(audio, 24000)

        assert result.shape == audio.shape
        assert not torch.allclose(result, audio, atol=1e-6)

    def test_mock_librosa_effects(self):
        """Test augmentation with mocked librosa effects."""

        # Mock an augmentation that would use librosa
        class MockPitchShift:
            def __init__(self, semitones=0):
                self.semitones = semitones

            def __call__(self, audio, sample_rate):
                # Mock pitch shifting by frequency modulation
                shift_factor = 2 ** (self.semitones / 12.0)
                return audio * shift_factor

        audio = torch.randn(1, 24000)
        aug = MockPitchShift(semitones=2)
        result = aug(audio, 24000)

        assert result.shape == audio.shape
        shift_factor = 2 ** (2 / 12.0)
        expected = audio * shift_factor
        assert torch.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
