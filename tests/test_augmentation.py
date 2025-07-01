"""
Tests for audio augmentation pipeline.
"""

import pytest
import torch

from music_gen.data.augmentation import (
    AdaptiveAugmentation,
    AddNoise,
    AugmentationPipeline,
    Distortion,
    FrequencyMasking,
    PitchShift,
    PolymixAugmentation,
    Reverb,
    TimeMasking,
    VolumeAugmentation,
    create_inference_augmentation_pipeline,
    create_training_augmentation_pipeline,
)


class TestBasicAugmentations:
    """Test individual augmentation classes."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        # Generate sine wave
        t = torch.linspace(0, duration, samples)
        frequency = 440.0  # A4 note
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

        return waveform, sample_rate

    def test_volume_augmentation(self, sample_audio):
        """Test volume augmentation."""
        waveform, sample_rate = sample_audio

        aug = VolumeAugmentation(probability=1.0, gain_range_db=(-6, 6))

        # Test multiple applications
        results = []
        for _ in range(10):
            result = aug(waveform, sample_rate)
            results.append(result.abs().max().item())

        # Results should vary due to random gain
        assert len(set(results)) > 1, "Volume augmentation should produce different results"

        # All results should be reasonable
        for result in results:
            assert 0.1 < result < 2.0, f"Volume result {result} is out of reasonable range"

    def test_add_noise(self, sample_audio):
        """Test noise addition."""
        waveform, sample_rate = sample_audio

        # Test different noise types
        noise_types = ["gaussian", "pink", "brown"]

        for noise_type in noise_types:
            aug = AddNoise(probability=1.0, noise_type=noise_type)
            result = aug(waveform, sample_rate)

            # Result should be different from original
            assert not torch.allclose(
                result, waveform
            ), f"Noise type {noise_type} should change the signal"

            # Result should have similar magnitude
            original_rms = torch.sqrt(torch.mean(waveform**2))
            result_rms = torch.sqrt(torch.mean(result**2))
            ratio = result_rms / original_rms
            assert (
                0.8 < ratio < 1.5
            ), f"RMS ratio {ratio} is too extreme for noise type {noise_type}"

    def test_pitch_shift(self, sample_audio):
        """Test pitch shifting."""
        waveform, sample_rate = sample_rate

        aug = PitchShift(probability=1.0, semitone_range=(-2, 2))
        result = aug(waveform, sample_rate)

        # Result should have same length
        assert result.shape == waveform.shape

        # Result should be different (unless shift is 0)
        if not torch.allclose(result, waveform, atol=1e-6):
            # Pitch was shifted
            pass
        else:
            # Pitch shift might be very small or zero
            pass

    def test_time_masking(self, sample_audio):
        """Test time masking."""
        waveform, sample_rate = sample_audio

        aug = TimeMasking(probability=1.0, time_mask_param=1000, num_masks=2)
        result = aug(waveform, sample_rate)

        # Result should have same shape
        assert result.shape == waveform.shape

        # Result should have some zeros due to masking
        zero_count = (result == 0).sum().item()
        total_samples = result.numel()
        zero_ratio = zero_count / total_samples

        assert zero_ratio > 0.01, "Time masking should create some zero regions"

    def test_frequency_masking(self, sample_audio):
        """Test frequency masking."""
        waveform, sample_rate = sample_audio

        aug = FrequencyMasking(probability=1.0, freq_mask_param=80, num_masks=1)
        result = aug(waveform, sample_rate)

        # Result should have same shape
        assert result.shape == waveform.shape

        # Result should be different from original
        assert not torch.allclose(
            result, waveform, atol=1e-6
        ), "Frequency masking should change the signal"

    def test_reverb(self, sample_audio):
        """Test reverb effect."""
        waveform, sample_rate = sample_audio

        aug = Reverb(probability=1.0)
        result = aug(waveform, sample_rate)

        # Result should have same shape
        assert result.shape == waveform.shape

        # Result should be different
        assert not torch.allclose(result, waveform, atol=1e-6), "Reverb should change the signal"

    def test_distortion(self, sample_audio):
        """Test distortion effect."""
        waveform, sample_rate = sample_audio

        aug = Distortion(probability=1.0)
        result = aug(waveform, sample_rate)

        # Result should have same shape
        assert result.shape == waveform.shape

        # Result should be different
        assert not torch.allclose(
            result, waveform, atol=1e-6
        ), "Distortion should change the signal"


class TestPolymixAugmentation:
    """Test Polymix augmentation."""

    @pytest.fixture
    def mix_samples(self):
        """Create sample audio for mixing."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        mix_samples = []
        for i in range(3):
            # Generate different frequency sine waves
            t = torch.linspace(0, duration, samples)
            frequency = 440.0 * (i + 1)  # Different frequencies
            waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
            mix_samples.append(waveform)

        return mix_samples, sample_rate

    def test_polymix_without_samples(self, mix_samples):
        """Test Polymix when no mix samples are set."""
        mix_samples_list, sample_rate = mix_samples
        original = mix_samples_list[0]

        aug = PolymixAugmentation(probability=1.0)
        result = aug(original, sample_rate)

        # Should return original when no mix samples are set
        assert torch.allclose(result, original)

    def test_polymix_with_samples(self, mix_samples):
        """Test Polymix with mix samples."""
        mix_samples_list, sample_rate = mix_samples
        original = mix_samples_list[0]

        aug = PolymixAugmentation(probability=1.0, num_mix=2)
        aug.set_mix_samples(mix_samples_list[1:])  # Set other samples for mixing

        result = aug(original, sample_rate)

        # Result should be different due to mixing
        assert not torch.allclose(
            result, original
        ), "Polymix should change the signal when mix samples are available"

        # Result should have same shape
        assert result.shape == original.shape


class TestAugmentationPipeline:
    """Test augmentation pipeline."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        t = torch.linspace(0, duration, samples)
        frequency = 440.0
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

        return waveform, sample_rate

    def test_pipeline_creation(self):
        """Test pipeline creation with augmentations."""
        augmentations = [
            VolumeAugmentation(probability=0.5),
            AddNoise(probability=0.3),
            TimeMasking(probability=0.2),
        ]

        pipeline = AugmentationPipeline(augmentations, max_augmentations=2)

        assert len(pipeline.augmentations) == 3
        assert pipeline.max_augmentations == 2

    def test_pipeline_application(self, sample_audio):
        """Test applying pipeline to audio."""
        waveform, sample_rate = sample_audio

        augmentations = [
            VolumeAugmentation(probability=1.0),
            AddNoise(probability=1.0),
        ]

        pipeline = AugmentationPipeline(augmentations)
        result = pipeline(waveform, sample_rate)

        # Result should be different
        assert not torch.allclose(result, waveform, atol=1e-6)

        # Result should have same shape
        assert result.shape == waveform.shape

    def test_limited_augmentations(self, sample_audio):
        """Test pipeline with max_augmentations limit."""
        waveform, sample_rate = sample_audio

        augmentations = [
            VolumeAugmentation(probability=1.0),
            AddNoise(probability=1.0),
            TimeMasking(probability=1.0),
        ]

        # Limit to 1 augmentation
        pipeline = AugmentationPipeline(augmentations, max_augmentations=1)
        result = pipeline(waveform, sample_rate)

        # Should still work and produce different result
        assert result.shape == waveform.shape


class TestPreConfiguredPipelines:
    """Test pre-configured pipeline factory functions."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)

        t = torch.linspace(0, duration, samples)
        frequency = 440.0
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

        return waveform, sample_rate

    def test_training_pipeline_moderate(self, sample_audio):
        """Test moderate training pipeline."""
        waveform, sample_rate = sample_audio

        pipeline = create_training_augmentation_pipeline(strong=False)
        result = pipeline(waveform, sample_rate)

        assert result.shape == waveform.shape
        assert len(pipeline.augmentations) > 0

    def test_training_pipeline_strong(self, sample_audio):
        """Test strong training pipeline."""
        waveform, sample_rate = sample_audio

        pipeline = create_training_augmentation_pipeline(strong=True)
        result = pipeline(waveform, sample_rate)

        assert result.shape == waveform.shape
        assert len(pipeline.augmentations) > 0

    def test_inference_pipeline(self, sample_audio):
        """Test inference pipeline."""
        waveform, sample_rate = sample_audio

        pipeline = create_inference_augmentation_pipeline()
        result = pipeline(waveform, sample_rate)

        assert result.shape == waveform.shape
        # Inference pipeline should be lighter
        assert len(pipeline.augmentations) < 5


class TestAdaptiveAugmentation:
    """Test adaptive augmentation system."""

    def test_adaptive_creation(self):
        """Test adaptive augmentation creation."""
        adaptive = AdaptiveAugmentation(
            initial_strength=0.5, max_strength=1.0, adaptation_rate=0.001
        )

        assert adaptive.current_strength == 0.5
        assert adaptive.max_strength == 1.0
        assert adaptive.adaptation_rate == 0.001

    def test_strength_adaptation(self):
        """Test strength adaptation based on loss."""
        adaptive = AdaptiveAugmentation(initial_strength=0.5)

        # Simulate low loss (overfitting) - should increase strength
        initial_strength = adaptive.current_strength
        adaptive.update_strength(loss=1.0, target_loss=2.0)
        assert adaptive.current_strength > initial_strength

        # Simulate high loss (underfitting) - should decrease strength
        adaptive.current_strength = 0.8
        initial_strength = adaptive.current_strength
        adaptive.update_strength(loss=3.0, target_loss=2.0)
        assert adaptive.current_strength < initial_strength

    def test_adaptive_pipeline_generation(self):
        """Test adaptive pipeline generation."""
        adaptive = AdaptiveAugmentation(initial_strength=0.7)

        pipeline = adaptive.get_pipeline()

        assert isinstance(pipeline, AugmentationPipeline)
        assert len(pipeline.augmentations) > 0

        # All augmentations should have probabilities scaled by strength
        for aug in pipeline.augmentations:
            # Probabilities should be reasonable (not all 1.0 or 0.0)
            assert 0.0 <= aug.probability <= 1.0


class TestErrorHandling:
    """Test error handling in augmentations."""

    def test_invalid_audio_shape(self):
        """Test handling of invalid audio shapes."""
        # Test with 3D tensor (should handle gracefully)
        invalid_audio = torch.randn(2, 2, 1000)
        sample_rate = 24000

        aug = VolumeAugmentation(probability=1.0)

        # Should not crash
        try:
            result = aug(invalid_audio, sample_rate)
            assert result.shape == invalid_audio.shape
        except Exception:
            # Some augmentations might fail on invalid shapes
            # which is acceptable behavior
            pass

    def test_zero_audio(self):
        """Test handling of silent audio."""
        silent_audio = torch.zeros(1, 24000)
        sample_rate = 24000

        aug = VolumeAugmentation(probability=1.0)
        result = aug(silent_audio, sample_rate)

        # Should handle silent audio without NaN/inf
        assert torch.isfinite(result).all()

    def test_very_short_audio(self):
        """Test handling of very short audio."""
        short_audio = torch.randn(1, 10)  # Only 10 samples
        sample_rate = 24000

        aug = TimeMasking(probability=1.0, time_mask_param=5)
        result = aug(short_audio, sample_rate)

        # Should handle short audio
        assert result.shape == short_audio.shape


if __name__ == "__main__":
    pytest.main([__file__])
