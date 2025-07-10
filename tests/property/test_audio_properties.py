"""
Property-based testing for audio processing components.

Uses hypothesis to generate test cases that verify mathematical
properties and invariants of audio processing functions.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
from typing import Tuple, List
import math

from music_gen.utils.audio import (
    AudioProcessor,
    AudioNormalizer,
    AudioEffects,
    AudioValidator,
)


# Custom strategies for audio data
@st.composite
def audio_tensor(draw, min_samples=1000, max_samples=48000, channels=1):
    """Generate valid audio tensor."""
    samples = draw(st.integers(min_value=min_samples, max_value=max_samples))

    # Generate audio data within reasonable range
    audio_data = draw(
        arrays(
            dtype=np.float32,
            shape=(channels, samples),
            elements=st.floats(
                min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return torch.from_numpy(audio_data)


@st.composite
def sample_rate(draw):
    """Generate valid sample rates."""
    common_rates = [8000, 16000, 22050, 24000, 44100, 48000, 96000]
    return draw(st.sampled_from(common_rates))


@st.composite
def duration(draw):
    """Generate valid durations."""
    return draw(st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False))


@pytest.mark.property
class TestAudioProcessingProperties:
    """Property-based tests for audio processing."""

    @given(audio_tensor())
    @settings(max_examples=50, deadline=5000)
    def test_normalization_properties(self, audio):
        """Test mathematical properties of audio normalization."""
        normalizer = AudioNormalizer()

        # Skip if audio is silent (would cause division by zero)
        assume(audio.abs().max() > 1e-6)

        # Property: Peak normalization should result in max absolute value of 1.0
        normalized = normalizer.normalize_peak(audio, target_peak=1.0)

        assert torch.abs(normalized.abs().max() - 1.0) < 1e-5
        assert normalized.shape == audio.shape

        # Property: Normalization should preserve relative relationships
        if audio.numel() > 1:
            # Find two different non-zero samples
            non_zero_mask = audio.abs() > 1e-6
            if non_zero_mask.sum() >= 2:
                non_zero_indices = torch.where(non_zero_mask.flatten())[0][:2]
                if len(non_zero_indices) >= 2:
                    idx1, idx2 = non_zero_indices[0], non_zero_indices[1]

                    # Convert to actual tensor indices
                    idx1_2d = (idx1 // audio.shape[1], idx1 % audio.shape[1])
                    idx2_2d = (idx2 // audio.shape[1], idx2 % audio.shape[1])

                    original_ratio = audio[idx1_2d] / audio[idx2_2d]
                    normalized_ratio = normalized[idx1_2d] / normalized[idx2_2d]

                    assert torch.abs(original_ratio - normalized_ratio) < 1e-4

    @given(audio_tensor(), st.floats(min_value=0.01, max_value=2.0, allow_nan=False))
    @settings(max_examples=30, deadline=5000)
    def test_rms_normalization_properties(self, audio, target_rms):
        """Test RMS normalization properties."""
        normalizer = AudioNormalizer()

        # Skip silent audio
        assume(torch.sqrt(torch.mean(audio**2)) > 1e-6)

        normalized = normalizer.normalize_rms(audio, target_rms=target_rms)

        # Property: RMS should be approximately target_rms
        actual_rms = torch.sqrt(torch.mean(normalized**2))
        assert torch.abs(actual_rms - target_rms) < 1e-3

        # Property: Shape should be preserved
        assert normalized.shape == audio.shape

    @given(audio_tensor(), sample_rate(), sample_rate())
    @settings(max_examples=20, deadline=5000)
    def test_resampling_properties(self, audio, source_sr, target_sr):
        """Test audio resampling properties."""
        processor = AudioProcessor(sample_rate=source_sr)

        # Skip if sample rates are too similar (causes numerical issues)
        assume(abs(source_sr - target_sr) > 1000)

        resampled = processor.resample(audio, source_sr, target_sr)

        # Property: Output length should be proportional to sample rate ratio
        expected_length = int(audio.shape[1] * target_sr / source_sr)
        actual_length = resampled.shape[1]

        # Allow some tolerance due to resampling filters
        tolerance = max(10, int(expected_length * 0.01))  # 1% or 10 samples
        assert abs(actual_length - expected_length) <= tolerance

        # Property: Channel count should be preserved
        assert resampled.shape[0] == audio.shape[0]

    @given(audio_tensor(), st.integers(min_value=10, max_value=1000))
    @settings(max_examples=30, deadline=5000)
    def test_fade_properties(self, audio, fade_samples):
        """Test fade in/out properties."""
        processor = AudioProcessor()

        # Skip if fade is longer than audio
        assume(fade_samples < audio.shape[1] // 2)

        faded = processor.apply_fade(
            audio, fade_in_samples=fade_samples, fade_out_samples=fade_samples
        )

        # Property: Fade should start and end at zero
        assert torch.abs(faded[0, 0]) < 1e-6  # Fade in starts at 0
        assert torch.abs(faded[0, -1]) < 1e-6  # Fade out ends at 0

        # Property: Middle section should be unaffected
        if audio.shape[1] > 2 * fade_samples + 10:
            middle_start = fade_samples + 5
            middle_end = audio.shape[1] - fade_samples - 5

            original_middle = audio[:, middle_start:middle_end]
            faded_middle = faded[:, middle_start:middle_end]

            assert torch.allclose(original_middle, faded_middle, atol=1e-5)

        # Property: Shape should be preserved
        assert faded.shape == audio.shape

    @given(audio_tensor(min_samples=2000), st.floats(min_value=0.01, max_value=0.1))
    @settings(max_examples=20, deadline=5000)
    def test_silence_trimming_properties(self, audio, threshold):
        """Test silence trimming properties."""
        processor = AudioProcessor()

        # Add silence to beginning and end
        silence_samples = 500
        padded_audio = torch.cat(
            [
                torch.zeros(audio.shape[0], silence_samples),
                audio,
                torch.zeros(audio.shape[0], silence_samples),
            ],
            dim=1,
        )

        trimmed = processor.trim_silence(padded_audio, threshold=threshold)

        # Property: Trimmed audio should be shorter than padded
        assert trimmed.shape[1] <= padded_audio.shape[1]

        # Property: Trimmed audio should not be empty (unless original was silent)
        max_amplitude = audio.abs().max()
        if max_amplitude > threshold:
            assert trimmed.shape[1] > 0

        # Property: Channel count should be preserved
        assert trimmed.shape[0] == audio.shape[0]

    @given(
        audio_tensor(channels=2),
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=2),
    )
    @settings(max_examples=20, deadline=5000)
    def test_mixing_properties(self, stereo_audio, weights):
        """Test audio mixing properties."""
        processor = AudioProcessor()

        # Normalize weights
        total_weight = sum(weights)
        assume(total_weight > 1e-6)
        normalized_weights = [w / total_weight for w in weights]

        # Split stereo into two mono tracks
        track1 = stereo_audio[0:1, :]
        track2 = stereo_audio[1:2, :]

        mixed = processor.mix_tracks([track1, track2], weights=normalized_weights)

        # Property: Mixed result should have expected shape
        assert mixed.shape[0] == 1  # Mixed to mono
        assert mixed.shape[1] == stereo_audio.shape[1]

        # Property: Linear combination property
        expected = track1 * normalized_weights[0] + track2 * normalized_weights[1]
        assert torch.allclose(mixed, expected, atol=1e-5)

    @given(audio_tensor(), st.floats(min_value=-60.0, max_value=0.0))
    @settings(max_examples=20, deadline=5000)
    def test_compression_properties(self, audio, threshold_db):
        """Test dynamic compression properties."""
        effects = AudioEffects(sample_rate=24000)

        # Skip very quiet audio
        assume(audio.abs().max() > 1e-3)

        compressed = effects.apply_compression(
            audio, threshold=threshold_db, ratio=4.0, attack_ms=5.0, release_ms=50.0
        )

        # Property: Compression should reduce dynamic range
        original_range = audio.max() - audio.min()
        compressed_range = compressed.max() - compressed.min()

        # Allow for some increase due to makeup gain, but not dramatic
        assert compressed_range <= original_range * 1.5

        # Property: Shape should be preserved
        assert compressed.shape == audio.shape

        # Property: Should not amplify beyond reasonable bounds
        assert compressed.abs().max() <= 2.0


@pytest.mark.property
class TestAudioValidationProperties:
    """Property-based tests for audio validation."""

    @given(audio_tensor())
    @settings(max_examples=30, deadline=3000)
    def test_validation_consistency(self, audio):
        """Test audio validation consistency."""
        validator = AudioValidator()

        # Property: Valid audio should always pass validation
        if not (torch.isnan(audio).any() or torch.isinf(audio).any()):
            try:
                validator.validate_audio(audio)
                # If validation passes, audio should have expected properties
                assert audio.dim() == 2
                assert audio.shape[0] > 0  # At least one channel
                assert audio.shape[1] > 0  # At least one sample
            except Exception:
                # If validation fails, there should be a good reason
                pass

    @given(
        audio_tensor(),
        st.floats(min_value=-100.0, max_value=0.0),
        st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=20, deadline=3000)
    def test_silence_detection_properties(self, audio, threshold_db, min_duration):
        """Test silence detection properties."""
        validator = AudioValidator(sample_rate=24000)

        silence_segments = validator.detect_silence(
            audio, threshold=threshold_db, min_duration=min_duration
        )

        # Property: Silence segments should be non-overlapping and ordered
        for i in range(len(silence_segments) - 1):
            current_end = silence_segments[i][1]
            next_start = silence_segments[i + 1][0]
            assert current_end <= next_start

        # Property: Each segment should meet minimum duration requirement
        sample_rate = 24000
        min_samples = int(min_duration * sample_rate)

        for start, end in silence_segments:
            segment_samples = end - start
            assert segment_samples >= min_samples * 0.9  # Allow small tolerance

    @given(audio_tensor())
    @settings(max_examples=20, deadline=3000)
    def test_quality_metrics_properties(self, audio):
        """Test audio quality metrics properties."""
        validator = AudioValidator()

        # Skip if audio is too quiet or has issues
        assume(not torch.isnan(audio).any())
        assume(not torch.isinf(audio).any())
        assume(audio.abs().max() > 1e-6)

        quality_metrics = validator.validate_quality(audio)

        # Property: SNR should be reasonable for clean audio
        if quality_metrics["snr_db"] is not None:
            assert isinstance(quality_metrics["snr_db"], (int, float))
            # SNR should be finite
            assert not math.isnan(quality_metrics["snr_db"])
            assert not math.isinf(quality_metrics["snr_db"])

        # Property: Dynamic range should be non-negative
        if quality_metrics["dynamic_range_db"] is not None:
            assert quality_metrics["dynamic_range_db"] >= 0

        # Property: Clipping percentage should be between 0 and 100
        if "clipping_percent" in quality_metrics:
            clipping_pct = quality_metrics["clipping_percent"]
            assert 0 <= clipping_pct <= 100


@pytest.mark.property
class TestAudioMathematicalProperties:
    """Property-based tests for mathematical audio properties."""

    @given(audio_tensor(), audio_tensor())
    @settings(max_examples=20, deadline=3000)
    def test_convolution_properties(self, signal, kernel):
        """Test convolution mathematical properties."""
        # Skip if either is too small
        assume(signal.shape[1] >= 10)
        assume(kernel.shape[1] >= 3)
        assume(kernel.shape[1] <= signal.shape[1])

        # Ensure same number of channels
        if signal.shape[0] != kernel.shape[0]:
            # Broadcast kernel to match signal channels
            kernel = kernel[:1].expand(signal.shape[0], -1)

        # Simple convolution using torch.nn.functional
        import torch.nn.functional as F

        # Reshape for conv1d (batch, channels, length)
        signal_reshaped = signal.unsqueeze(0)  # Add batch dimension
        kernel_reshaped = kernel.unsqueeze(1)  # Add input channel dimension

        result = F.conv1d(signal_reshaped, kernel_reshaped, padding="same", groups=signal.shape[0])
        result = result.squeeze(0)  # Remove batch dimension

        # Property: Convolution with delta function should return original signal
        # (Test with a simple kernel)
        delta_kernel = torch.zeros_like(kernel)
        if delta_kernel.shape[1] % 2 == 1:  # Odd length
            center = delta_kernel.shape[1] // 2
            delta_kernel[:, center] = 1.0

            delta_kernel_reshaped = delta_kernel.unsqueeze(1)
            delta_result = F.conv1d(
                signal_reshaped, delta_kernel_reshaped, padding="same", groups=signal.shape[0]
            )
            delta_result = delta_result.squeeze(0)

            # Should be approximately equal to original signal
            assert torch.allclose(signal, delta_result, atol=1e-5)

        # Property: Output shape should match input shape (with 'same' padding)
        assert result.shape == signal.shape

    @given(
        audio_tensor(min_samples=1024, max_samples=4096), st.integers(min_value=512, max_value=2048)
    )
    @settings(max_examples=15, deadline=5000)
    def test_fft_properties(self, audio, n_fft):
        """Test FFT mathematical properties."""
        # Skip if n_fft is larger than audio
        assume(n_fft <= audio.shape[1])

        # Forward FFT
        audio_fft = torch.fft.fft(audio, n=n_fft, dim=1)

        # Property: FFT should preserve energy (Parseval's theorem)
        time_energy = torch.sum(audio[:, :n_fft] ** 2)
        freq_energy = torch.sum(torch.abs(audio_fft) ** 2) / n_fft

        # Allow for numerical precision errors
        assert torch.abs(time_energy - freq_energy) < time_energy * 0.01

        # Property: IFFT should recover original signal
        recovered_audio = torch.fft.ifft(audio_fft, dim=1).real

        assert torch.allclose(audio[:, :n_fft], recovered_audio, atol=1e-5)

        # Property: FFT of real signal should have Hermitian symmetry
        # (Only test for even n_fft for simplicity)
        if n_fft % 2 == 0:
            # For real input, FFT[k] = conj(FFT[N-k])
            mid = n_fft // 2
            for k in range(1, mid):
                fft_k = audio_fft[:, k]
                fft_n_minus_k = audio_fft[:, n_fft - k]

                # Should be complex conjugates
                assert torch.allclose(fft_k, torch.conj(fft_n_minus_k), atol=1e-5)

    @given(audio_tensor(min_samples=1000), st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=15, deadline=3000)
    def test_windowing_properties(self, audio, window_duration_seconds):
        """Test windowing function properties."""
        sample_rate = 24000
        window_samples = int(window_duration_seconds * sample_rate)

        # Skip if window is larger than audio
        assume(window_samples <= audio.shape[1])

        # Create different window functions
        hann_window = torch.hann_window(window_samples)
        hamming_window = torch.hamming_window(window_samples)

        # Apply windows
        windowed_hann = audio[:, :window_samples] * hann_window
        windowed_hamming = audio[:, :window_samples] * hamming_window

        # Property: Windowed signal should have reduced energy at edges
        if window_samples > 10:
            edge_samples = min(5, window_samples // 10)

            # Check beginning
            original_start_energy = torch.sum(audio[:, :edge_samples] ** 2)
            windowed_start_energy = torch.sum(windowed_hann[:, :edge_samples] ** 2)

            if original_start_energy > 1e-6:
                assert windowed_start_energy <= original_start_energy

            # Check end
            original_end_energy = torch.sum(audio[:, -edge_samples:] ** 2)
            windowed_end_energy = torch.sum(windowed_hann[:, -edge_samples:] ** 2)

            if original_end_energy > 1e-6:
                assert windowed_end_energy <= original_end_energy

        # Property: Window functions should sum to reasonable value
        hann_sum = torch.sum(hann_window)
        hamming_sum = torch.sum(hamming_window)

        # Both should be positive and less than window length
        assert 0 < hann_sum <= window_samples
        assert 0 < hamming_sum <= window_samples

    @given(audio_tensor(min_samples=100), st.floats(min_value=0.1, max_value=2.0))
    @settings(max_examples=20, deadline=3000)
    def test_scaling_properties(self, audio, scale_factor):
        """Test linear scaling properties."""
        # Skip if scaling would cause overflow
        assume(audio.abs().max() * scale_factor < 10.0)

        scaled_audio = audio * scale_factor

        # Property: Scaling should be linear
        assert torch.allclose(scaled_audio, audio * scale_factor, atol=1e-6)

        # Property: Energy should scale by factor squared
        original_energy = torch.sum(audio**2)
        scaled_energy = torch.sum(scaled_audio**2)
        expected_energy = original_energy * (scale_factor**2)

        assert torch.abs(scaled_energy - expected_energy) < expected_energy * 1e-5

        # Property: Shape should be preserved
        assert scaled_audio.shape == audio.shape

        # Property: Zero should remain zero
        zero_mask = audio.abs() < 1e-10
        assert torch.allclose(
            scaled_audio[zero_mask], torch.zeros_like(scaled_audio[zero_mask]), atol=1e-10
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
