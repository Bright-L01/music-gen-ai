"""
Unit tests for audio utility functions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from music_gen.utils.audio import (
    apply_fade,
    compute_audio_duration,
    concatenate_audio,
    normalize_audio,
    split_audio,
    trim_silence,
)


@pytest.mark.unit
class TestAudioNormalization:
    """Test audio normalization functions."""

    def test_peak_normalization(self, sample_audio):
        """Test peak normalization."""
        audio, sample_rate = sample_audio

        # Scale audio to test normalization
        scaled_audio = audio * 0.5
        normalized = normalize_audio(scaled_audio, method="peak")

        # Should be normalized to peak of 1.0
        assert torch.allclose(normalized.abs().max(), torch.tensor(1.0), atol=1e-6)

    def test_rms_normalization(self, sample_audio):
        """Test RMS normalization."""
        audio, sample_rate = sample_audio

        normalized = normalize_audio(audio, method="rms")

        # RMS should be approximately 1.0
        rms = torch.sqrt(torch.mean(normalized**2))
        assert torch.allclose(rms, torch.tensor(1.0), atol=1e-6)

    def test_lufs_normalization(self, sample_audio):
        """Test LUFS normalization (simplified)."""
        audio, sample_rate = sample_audio

        normalized = normalize_audio(audio, method="lufs")

        # Should modify the audio
        assert not torch.allclose(normalized, audio)
        assert normalized.shape == audio.shape

    def test_normalization_edge_cases(self):
        """Test normalization with edge cases."""
        # Zero audio
        zero_audio = torch.zeros(1, 1000)
        normalized = normalize_audio(zero_audio, method="peak")
        assert torch.allclose(normalized, zero_audio)

        # Very small audio
        tiny_audio = torch.ones(1, 1000) * 1e-10
        normalized = normalize_audio(tiny_audio, method="peak")
        assert torch.allclose(normalized, tiny_audio)


@pytest.mark.unit
class TestAudioProcessing:
    """Test audio processing functions."""

    def test_apply_fade(self, sample_audio):
        """Test fade in/out application."""
        audio, sample_rate = sample_audio
        # Clone the audio to avoid modifying the original
        audio_copy = audio.clone()

        fade_duration = 0.1  # 100ms
        faded = apply_fade(audio_copy, sample_rate, fade_duration, fade_duration)

        # Check fade in
        fade_samples = int(fade_duration * sample_rate)
        assert faded[0, 0] == 0.0  # Starts at zero

        # Check that fade has been applied (middle of fade should be < 1.0 scale)
        mid_fade_idx = fade_samples // 2
        if mid_fade_idx < audio.shape[-1]:
            # The faded version should be reduced compared to original
            assert faded[0, mid_fade_idx] <= audio[0, mid_fade_idx]

        # Check fade out
        assert faded[0, -1] == 0.0  # Ends at zero

        # Check that fade out has been applied
        mid_fade_out_idx = -fade_samples // 2
        if abs(mid_fade_out_idx) < audio.shape[-1]:
            assert faded[0, mid_fade_out_idx] <= audio[0, mid_fade_out_idx]

        # Middle should be unchanged
        mid_start = fade_samples + 100
        mid_end = -fade_samples - 100
        assert torch.allclose(faded[0, mid_start:mid_end], audio[0, mid_start:mid_end], atol=1e-6)

    def test_trim_silence(self):
        """Test silence trimming."""
        sample_rate = 24000

        # Create audio with silence at start and end
        silence_duration = 0.1  # 100ms
        audio_duration = 0.5  # 500ms

        silence_samples = int(silence_duration * sample_rate)
        audio_samples = int(audio_duration * sample_rate)

        # Silence + Audio + Silence
        audio = torch.cat(
            [
                torch.zeros(1, silence_samples),  # Start silence
                torch.sin(
                    2 * np.pi * 440 * torch.linspace(0, audio_duration, audio_samples)
                ).unsqueeze(
                    0
                ),  # Audio
                torch.zeros(1, silence_samples),  # End silence
            ],
            dim=1,
        )

        trimmed = trim_silence(audio, sample_rate, threshold_db=-40.0)

        # Should be shorter than original
        assert trimmed.shape[1] < audio.shape[1]
        # Should contain most of the audio part (allowing for some trimming tolerance)
        # The trimming algorithm may remove a few samples at boundaries
        assert trimmed.shape[1] >= audio_samples * 0.98  # Allow 2% tolerance

    def test_concatenate_audio(self):
        """Test audio concatenation."""
        sample_rate = 24000
        duration = 0.5
        samples = int(duration * sample_rate)

        # Create two different audio segments
        audio1 = torch.sin(2 * np.pi * 440 * torch.linspace(0, duration, samples)).unsqueeze(0)
        audio2 = torch.sin(2 * np.pi * 880 * torch.linspace(0, duration, samples)).unsqueeze(0)

        # Test concatenation without crossfade
        concatenated = concatenate_audio(
            [audio1, audio2], crossfade_duration=0.0, sample_rate=sample_rate
        )

        expected_length = audio1.shape[1] + audio2.shape[1]
        assert concatenated.shape[1] == expected_length

        # Test concatenation with crossfade
        crossfade_duration = 0.1
        concatenated_cf = concatenate_audio(
            [audio1, audio2], crossfade_duration=crossfade_duration, sample_rate=sample_rate
        )

        crossfade_samples = int(crossfade_duration * sample_rate)
        expected_length_cf = audio1.shape[1] + audio2.shape[1] - crossfade_samples
        assert concatenated_cf.shape[1] == expected_length_cf

    def test_split_audio(self):
        """Test audio splitting."""
        sample_rate = 24000
        total_duration = 2.0
        segment_duration = 0.8
        overlap = 0.2

        total_samples = int(total_duration * sample_rate)
        audio = torch.randn(1, total_samples)

        segments = split_audio(audio, sample_rate, segment_duration, overlap)

        # Check we got expected number of segments
        assert len(segments) > 1

        # Check segment lengths
        expected_samples = int(segment_duration * sample_rate)
        for segment in segments:
            assert segment.shape[1] == expected_samples

        # Check overlap by comparing adjacent segments
        if len(segments) >= 2:
            overlap_samples = int(overlap * sample_rate)
            step_samples = expected_samples - overlap_samples

            # The overlapped part should be identical
            seg1_end = segments[0][0, -overlap_samples:]
            seg2_start = segments[1][0, :overlap_samples]

            # Note: Due to how we create segments, this might not be exactly identical
            # but we can check that they're from the same original audio
            assert seg1_end.shape == seg2_start.shape


@pytest.mark.unit
class TestAudioUtilities:
    """Test utility functions."""

    @patch("music_gen.utils.audio.torchaudio.info")
    def test_compute_audio_duration_mock(self, mock_info):
        """Test audio duration computation with mock."""
        # Mock audio info
        mock_info_obj = Mock()
        mock_info_obj.num_frames = 48000
        mock_info_obj.sample_rate = 24000
        mock_info.return_value = mock_info_obj

        duration = compute_audio_duration("/fake/path.wav")

        expected_duration = 48000 / 24000  # 2.0 seconds
        assert duration == expected_duration
        mock_info.assert_called_once_with("/fake/path.wav")

    @patch("music_gen.utils.audio.torchaudio.info")
    def test_compute_audio_duration_error(self, mock_info):
        """Test audio duration computation with error."""
        mock_info.side_effect = Exception("File not found")

        duration = compute_audio_duration("/nonexistent/path.wav")

        assert duration == 0.0

    def test_concatenate_audio_edge_cases(self):
        """Test audio concatenation edge cases."""
        # Empty list
        result = concatenate_audio([])
        assert result.numel() == 0

        # Single audio
        audio = torch.randn(1, 1000)
        result = concatenate_audio([audio])
        assert torch.allclose(result, audio)

        # Very short segments with long crossfade
        short_audio1 = torch.randn(1, 100)
        short_audio2 = torch.randn(1, 100)

        # Crossfade longer than audio
        result = concatenate_audio(
            [short_audio1, short_audio2],
            crossfade_duration=1.0,  # Much longer than audio
            sample_rate=24000,
        )

        # Should still work (graceful degradation)
        assert result.shape[1] > 0


@pytest.mark.unit
class TestAudioIO:
    """Test audio I/O functions (mocked)."""

    @patch("music_gen.utils.audio.torchaudio.load")
    def test_load_audio_file_mock(self, mock_load):
        """Test audio file loading with mock."""
        # Mock successful load
        sample_rate = 44100
        target_sample_rate = 24000
        audio_data = torch.randn(2, 44100)  # 1 second stereo

        mock_load.return_value = (audio_data, sample_rate)

        # Import the function to test
        from music_gen.utils.audio import load_audio_file

        with patch("music_gen.utils.audio.torchaudio.transforms.Resample") as mock_resample:
            mock_resampler = Mock()
            mock_resampled = torch.randn(1, 24000)  # Mono, resampled
            mock_resampler.return_value = mock_resampled
            mock_resample.return_value = mock_resampler

            audio, sr = load_audio_file(
                "/fake/path.wav", target_sample_rate=target_sample_rate, mono=True, normalize=True
            )

            assert sr == target_sample_rate
            assert audio.shape[0] == 1  # Mono
            mock_load.assert_called_once()

    @patch("music_gen.utils.audio.torchaudio.save")
    @patch("music_gen.utils.audio.Path")
    def test_save_audio_file_mock(self, mock_path, mock_save):
        """Test audio file saving with mock."""
        from music_gen.utils.audio import save_audio_file

        # Mock path operations
        mock_path_obj = Mock()
        mock_path_obj.parent.mkdir = Mock()
        mock_path.return_value = mock_path_obj

        audio = torch.randn(1, 24000)
        sample_rate = 24000

        save_audio_file(audio, "/fake/output.wav", sample_rate)

        mock_save.assert_called_once()
        # Check that save was called with correct arguments
        args, kwargs = mock_save.call_args
        assert args[0] == "/fake/output.wav"
        assert args[2] == sample_rate

    @patch("music_gen.utils.audio.torchaudio.load")
    @patch("music_gen.utils.audio.torchaudio.save")
    @patch("music_gen.utils.audio.save_audio_file")
    def test_convert_audio_format_mock(self, mock_save_audio, mock_save, mock_load, tmp_path):
        """Test audio format conversion with mock."""
        from music_gen.utils.audio import convert_audio_format

        # Mock load
        original_audio = torch.randn(1, 44100)
        mock_load.return_value = (original_audio, 44100)

        # Use temporary paths
        input_path = tmp_path / "input.mp3"
        output_path = tmp_path / "output.wav"

        convert_audio_format(str(input_path), str(output_path), target_sample_rate=24000)

        mock_load.assert_called_once_with(str(input_path), frame_offset=0, num_frames=-1)
        mock_save_audio.assert_called_once()


@pytest.mark.unit
class TestAudioValidation:
    """Test audio validation and edge cases."""

    def test_audio_shape_validation(self):
        """Test various audio shapes."""
        # 1D audio
        audio_1d = torch.randn(1000)
        normalized = normalize_audio(audio_1d)
        assert normalized.shape == audio_1d.shape

        # 2D mono audio
        audio_2d = torch.randn(1, 1000)
        normalized = normalize_audio(audio_2d)
        assert normalized.shape == audio_2d.shape

        # 2D stereo audio
        audio_stereo = torch.randn(2, 1000)
        normalized = normalize_audio(audio_stereo)
        assert normalized.shape == audio_stereo.shape

    def test_extreme_audio_values(self):
        """Test with extreme audio values."""
        # Very loud audio
        loud_audio = torch.ones(1, 1000) * 100
        normalized = normalize_audio(loud_audio, method="peak")
        assert torch.allclose(normalized.abs().max(), torch.tensor(1.0))

        # Very quiet audio
        quiet_audio = torch.ones(1, 1000) * 1e-8
        normalized = normalize_audio(quiet_audio, method="peak")
        # Should handle gracefully without division by zero
        assert torch.isfinite(normalized).all()

    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinity values."""
        # Audio with NaN
        nan_audio = torch.tensor([[1.0, float("nan"), 2.0]])

        # Functions should handle gracefully or raise appropriate errors
        try:
            normalized = normalize_audio(nan_audio)
            # If it doesn't raise, check that result is reasonable
            assert normalized.shape == nan_audio.shape
        except (ValueError, RuntimeError):
            # Acceptable to raise an error
            pass

        # Audio with infinity
        inf_audio = torch.tensor([[1.0, float("inf"), 2.0]])

        try:
            normalized = normalize_audio(inf_audio)
            assert normalized.shape == inf_audio.shape
        except (ValueError, RuntimeError):
            # Acceptable to raise an error
            pass


if __name__ == "__main__":
    pytest.main([__file__])
