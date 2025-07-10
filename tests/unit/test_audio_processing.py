"""
Comprehensive unit tests for audio processing components.

Tests audio utilities, effects, normalization, format conversion,
and audio quality validation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import warnings

from music_gen.utils.audio import (
    AudioProcessor,
    AudioNormalizer,
    AudioEffects,
    AudioValidator,
    FormatConverter,
)
from music_gen.core.exceptions import (
    AudioProcessingError,
    ValidationError,
)


@pytest.mark.unit
class TestAudioProcessor:
    """Test core audio processing functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio tensor for testing."""
        # 1 second of 24kHz audio
        return torch.randn(1, 24000)

    @pytest.fixture
    def processor(self):
        """Create audio processor instance."""
        return AudioProcessor(sample_rate=24000)

    def test_load_audio_wav(self, processor):
        """Test loading WAV audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Create mock audio file content
            audio_data = torch.randn(1, 24000)

            with patch("torchaudio.load") as mock_load:
                mock_load.return_value = (audio_data, 24000)

                loaded_audio, sr = processor.load_audio(tmp.name)

                assert torch.equal(loaded_audio, audio_data)
                assert sr == 24000
                mock_load.assert_called_once_with(tmp.name)

    def test_load_audio_mp3(self, processor):
        """Test loading MP3 audio files."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_data = torch.randn(1, 24000)

            with patch("torchaudio.load") as mock_load:
                mock_load.return_value = (audio_data, 24000)

                loaded_audio, sr = processor.load_audio(tmp.name)

                assert isinstance(loaded_audio, torch.Tensor)
                assert sr == 24000

    def test_load_audio_unsupported_format(self, processor):
        """Test loading unsupported audio format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            with pytest.raises(AudioProcessingError, match="Unsupported audio format"):
                processor.load_audio(tmp.name)

    def test_save_audio_wav(self, processor, sample_audio):
        """Test saving audio as WAV."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with patch("torchaudio.save") as mock_save:
                processor.save_audio(sample_audio, tmp.name, sample_rate=24000)

                mock_save.assert_called_once_with(tmp.name, sample_audio, 24000)

    def test_resample_audio(self, processor, sample_audio):
        """Test audio resampling."""
        # Resample from 24kHz to 48kHz
        resampled = processor.resample(sample_audio, 24000, 48000)

        assert resampled.shape[0] == sample_audio.shape[0]  # Same channels
        assert resampled.shape[1] == 48000  # 2x length for 2x sample rate

    def test_resample_no_change(self, processor, sample_audio):
        """Test resampling with same source and target rates."""
        resampled = processor.resample(sample_audio, 24000, 24000)

        assert torch.equal(resampled, sample_audio)

    def test_convert_to_mono(self, processor):
        """Test converting stereo audio to mono."""
        stereo_audio = torch.randn(2, 24000)  # 2 channels

        mono_audio = processor.to_mono(stereo_audio)

        assert mono_audio.shape == (1, 24000)

    def test_convert_to_stereo(self, processor, sample_audio):
        """Test converting mono audio to stereo."""
        stereo_audio = processor.to_stereo(sample_audio)

        assert stereo_audio.shape == (2, 24000)
        # Both channels should be identical for mono->stereo conversion
        assert torch.equal(stereo_audio[0], stereo_audio[1])

    def test_trim_silence(self, processor):
        """Test trimming silence from audio."""
        # Create audio with silence at start and end
        audio = torch.zeros(1, 24000)
        audio[0, 5000:19000] = torch.randn(14000) * 0.5  # Audio in middle

        trimmed = processor.trim_silence(audio, threshold=0.01)

        # Should be shorter than original
        assert trimmed.shape[1] < audio.shape[1]
        assert trimmed.shape[1] > 10000  # But not too short

    def test_apply_fade(self, processor, sample_audio):
        """Test applying fade in/out."""
        faded = processor.apply_fade(sample_audio, fade_in_samples=1000, fade_out_samples=1000)

        # Check fade in: should start at 0
        assert faded[0, 0] == 0.0
        assert faded[0, 500] < sample_audio[0, 500].abs()  # Quieter in fade

        # Check fade out: should end at 0
        assert faded[0, -1] == 0.0
        assert faded[0, -500] < sample_audio[0, -500].abs()  # Quieter in fade

    def test_chunk_audio(self, processor):
        """Test chunking audio into segments."""
        # 10 seconds of audio
        long_audio = torch.randn(1, 240000)

        chunks = processor.chunk_audio(long_audio, chunk_duration=2.0, overlap=0.5)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should be ~2 seconds (48000 samples)
        for chunk in chunks:
            assert chunk.shape[1] <= 48000 + 12000  # Allow for overlap

    def test_validate_audio_tensor(self, processor, sample_audio):
        """Test audio tensor validation."""
        # Valid audio should pass
        processor.validate_audio(sample_audio)

        # Invalid shapes should fail
        with pytest.raises(ValidationError, match="Audio must be 2D"):
            processor.validate_audio(torch.randn(24000))  # 1D

        with pytest.raises(ValidationError, match="Audio must be 2D"):
            processor.validate_audio(torch.randn(1, 1, 24000))  # 3D

        # NaN/Inf should fail
        invalid_audio = sample_audio.clone()
        invalid_audio[0, 0] = float("nan")
        with pytest.raises(ValidationError, match="Audio contains NaN or Inf"):
            processor.validate_audio(invalid_audio)


@pytest.mark.unit
class TestAudioNormalizer:
    """Test audio normalization methods."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        return AudioNormalizer()

    @pytest.fixture
    def test_audio(self):
        """Create test audio with known characteristics."""
        # Audio with peak at 0.5
        audio = torch.randn(1, 24000) * 0.3
        audio[0, 1000] = 0.5  # Set peak
        return audio

    def test_peak_normalization(self, normalizer, test_audio):
        """Test peak normalization."""
        normalized = normalizer.normalize_peak(test_audio, target_peak=1.0)

        # Peak should be exactly 1.0
        assert torch.abs(normalized.abs().max() - 1.0) < 1e-6

    def test_rms_normalization(self, normalizer, test_audio):
        """Test RMS normalization."""
        normalized = normalizer.normalize_rms(test_audio, target_rms=0.5)

        # RMS should be approximately 0.5
        rms = torch.sqrt(torch.mean(normalized**2))
        assert torch.abs(rms - 0.5) < 0.01

    def test_lufs_normalization(self, normalizer, test_audio):
        """Test LUFS normalization."""
        # Mock LUFS calculation for testing
        with patch.object(normalizer, "_calculate_lufs", return_value=-20.0):
            normalized = normalizer.normalize_lufs(test_audio, target_lufs=-14.0)

            assert isinstance(normalized, torch.Tensor)
            assert normalized.shape == test_audio.shape

    def test_normalize_batch(self, normalizer):
        """Test batch normalization."""
        batch_audio = torch.randn(4, 1, 24000) * 0.3  # Batch of 4

        normalized = normalizer.normalize_batch(batch_audio, method="peak")

        assert normalized.shape == batch_audio.shape
        # Each item in batch should be peak normalized
        for i in range(4):
            peak = normalized[i].abs().max()
            assert torch.abs(peak - 1.0) < 0.01

    def test_normalization_methods(self, normalizer, test_audio):
        """Test different normalization methods."""
        methods = ["peak", "rms", "lufs"]

        for method in methods:
            if method == "lufs":
                with patch.object(normalizer, "_calculate_lufs", return_value=-20.0):
                    normalized = normalizer.normalize(test_audio, method=method)
            else:
                normalized = normalizer.normalize(test_audio, method=method)

            assert isinstance(normalized, torch.Tensor)
            assert normalized.shape == test_audio.shape

    def test_invalid_normalization_method(self, normalizer, test_audio):
        """Test invalid normalization method."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalizer.normalize(test_audio, method="invalid_method")

    def test_normalize_silent_audio(self, normalizer):
        """Test normalizing silent audio."""
        silent_audio = torch.zeros(1, 24000)

        # Should return silent audio unchanged
        normalized = normalizer.normalize_peak(silent_audio)
        assert torch.equal(normalized, silent_audio)


@pytest.mark.unit
class TestAudioEffects:
    """Test audio effects processing."""

    @pytest.fixture
    def effects(self):
        """Create effects processor."""
        return AudioEffects(sample_rate=24000)

    @pytest.fixture
    def test_audio(self):
        """Create test audio."""
        return torch.randn(1, 24000) * 0.5

    def test_apply_reverb(self, effects, test_audio):
        """Test reverb effect."""
        reverb_audio = effects.apply_reverb(test_audio, room_size=0.8, damping=0.5, wet_level=0.3)

        assert reverb_audio.shape == test_audio.shape
        # Reverb should change the audio
        assert not torch.equal(reverb_audio, test_audio)

    def test_apply_compression(self, effects, test_audio):
        """Test dynamic compression."""
        compressed = effects.apply_compression(
            test_audio, threshold=-10.0, ratio=4.0, attack_ms=5.0, release_ms=50.0  # dB
        )

        assert compressed.shape == test_audio.shape
        # Compression should reduce dynamic range
        original_range = test_audio.max() - test_audio.min()
        compressed_range = compressed.max() - compressed.min()
        assert compressed_range <= original_range

    def test_apply_eq(self, effects, test_audio):
        """Test equalizer effect."""
        eq_bands = [
            {"freq": 100, "gain": 3.0, "q": 1.0},  # Bass boost
            {"freq": 1000, "gain": -2.0, "q": 2.0},  # Mid cut
            {"freq": 8000, "gain": 1.5, "q": 1.5},  # Treble boost
        ]

        eq_audio = effects.apply_eq(test_audio, eq_bands)

        assert eq_audio.shape == test_audio.shape
        assert not torch.equal(eq_audio, test_audio)

    def test_apply_distortion(self, effects, test_audio):
        """Test distortion effect."""
        distorted = effects.apply_distortion(test_audio, drive=0.7, tone=0.5)

        assert distorted.shape == test_audio.shape
        # Distortion should clip peaks
        assert distorted.abs().max() <= 1.0

    def test_apply_delay(self, effects, test_audio):
        """Test delay effect."""
        delayed = effects.apply_delay(test_audio, delay_ms=250, feedback=0.4, wet_level=0.3)

        assert delayed.shape[0] == test_audio.shape[0]  # Same channels
        # Delay might change length slightly
        assert abs(delayed.shape[1] - test_audio.shape[1]) < 1000

    def test_apply_chorus(self, effects, test_audio):
        """Test chorus effect."""
        chorus_audio = effects.apply_chorus(test_audio, rate=1.5, depth=0.3, voices=3)

        assert chorus_audio.shape == test_audio.shape
        assert not torch.equal(chorus_audio, test_audio)

    def test_effect_chain(self, effects, test_audio):
        """Test applying multiple effects in chain."""
        effect_chain = [
            {"type": "compression", "params": {"threshold": -15.0, "ratio": 3.0}},
            {"type": "eq", "params": {"bands": [{"freq": 1000, "gain": 2.0, "q": 1.0}]}},
            {"type": "reverb", "params": {"room_size": 0.6, "wet_level": 0.2}},
        ]

        processed = effects.apply_effect_chain(test_audio, effect_chain)

        assert processed.shape == test_audio.shape
        assert not torch.equal(processed, test_audio)

    def test_invalid_effect_type(self, effects, test_audio):
        """Test invalid effect type."""
        with pytest.raises(ValueError, match="Unknown effect type"):
            effects.apply_effect(test_audio, "invalid_effect", {})


@pytest.mark.unit
class TestAudioValidator:
    """Test audio quality and content validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AudioValidator(sample_rate=24000)

    def test_validate_quality_good(self, validator):
        """Test validation of good quality audio."""
        # Create clean audio
        clean_audio = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 24000)).unsqueeze(0)

        quality_metrics = validator.validate_quality(clean_audio)

        assert quality_metrics["snr_db"] > 20  # Good SNR
        assert quality_metrics["thd_percent"] < 5  # Low distortion
        assert quality_metrics["dynamic_range_db"] > 10
        assert quality_metrics["is_clipped"] is False

    def test_validate_quality_clipped(self, validator):
        """Test validation of clipped audio."""
        # Create clipped audio
        clipped_audio = torch.ones(1, 24000)  # All samples at max

        quality_metrics = validator.validate_quality(clipped_audio)

        assert quality_metrics["is_clipped"] is True
        assert quality_metrics["clipping_percent"] > 90

    def test_validate_quality_noisy(self, validator):
        """Test validation of noisy audio."""
        # Create noisy audio
        signal = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 24000)) * 0.5
        noise = torch.randn(24000) * 0.3
        noisy_audio = (signal + noise).unsqueeze(0)

        quality_metrics = validator.validate_quality(noisy_audio)

        assert quality_metrics["snr_db"] < 10  # Poor SNR due to noise

    def test_detect_silence(self, validator):
        """Test silence detection."""
        # Audio with silence at beginning and end
        audio = torch.zeros(1, 24000)
        audio[0, 8000:16000] = torch.randn(8000) * 0.5  # Signal in middle

        silence_segments = validator.detect_silence(audio, threshold=-40)

        assert len(silence_segments) >= 2  # Start and end silence
        assert silence_segments[0][0] == 0  # Starts with silence

    def test_analyze_spectrum(self, validator):
        """Test spectral analysis."""
        # Create audio with known frequency content
        t = torch.linspace(0, 1, 24000)
        audio = (
            torch.sin(2 * np.pi * 440 * t) + torch.sin(2 * np.pi * 880 * t) * 0.5  # A4  # A5
        ).unsqueeze(0)

        spectrum_analysis = validator.analyze_spectrum(audio)

        assert "peak_frequencies" in spectrum_analysis
        assert "spectral_centroid" in spectrum_analysis
        assert "spectral_rolloff" in spectrum_analysis

        # Should detect our test frequencies
        peaks = spectrum_analysis["peak_frequencies"]
        assert any(abs(f - 440) < 10 for f in peaks)  # 440 Hz
        assert any(abs(f - 880) < 10 for f in peaks)  # 880 Hz

    def test_validate_format_compliance(self, validator):
        """Test format compliance validation."""
        # Test various audio formats
        formats_to_test = [
            {"sample_rate": 24000, "channels": 1, "bit_depth": 16},
            {"sample_rate": 48000, "channels": 2, "bit_depth": 24},
            {"sample_rate": 44100, "channels": 1, "bit_depth": 32},
        ]

        for fmt in formats_to_test:
            audio = torch.randn(fmt["channels"], fmt["sample_rate"])

            compliance = validator.validate_format_compliance(audio, fmt)

            assert compliance["sample_rate_match"] is True
            assert compliance["channel_count_match"] is True

    def test_detect_artifacts(self, validator):
        """Test audio artifact detection."""
        # Create audio with artifacts
        audio = torch.randn(1, 24000) * 0.1

        # Add click artifact
        audio[0, 12000:12010] = 1.0

        # Add DC offset
        audio += 0.1

        artifacts = validator.detect_artifacts(audio)

        assert artifacts["has_dc_offset"] is True
        assert artifacts["click_count"] > 0
        assert len(artifacts["click_locations"]) > 0


@pytest.mark.unit
class TestFormatConverter:
    """Test audio format conversion."""

    @pytest.fixture
    def converter(self):
        """Create format converter."""
        return FormatConverter()

    def test_convert_sample_rate(self, converter):
        """Test sample rate conversion."""
        audio_24k = torch.randn(1, 24000)

        # Convert to different sample rates
        audio_48k = converter.convert_sample_rate(audio_24k, 24000, 48000)
        audio_16k = converter.convert_sample_rate(audio_24k, 24000, 16000)

        assert audio_48k.shape[1] == 48000
        assert audio_16k.shape[1] == 16000

    def test_convert_bit_depth(self, converter):
        """Test bit depth conversion."""
        # Float32 audio
        audio_float = torch.randn(1, 24000)

        # Convert to int16
        audio_int16 = converter.convert_bit_depth(audio_float, "float32", "int16")

        assert audio_int16.dtype == torch.int16
        assert audio_int16.min() >= -32768
        assert audio_int16.max() <= 32767

    def test_convert_channels(self, converter):
        """Test channel conversion."""
        stereo_audio = torch.randn(2, 24000)
        mono_audio = torch.randn(1, 24000)

        # Stereo to mono
        converted_mono = converter.convert_channels(stereo_audio, "stereo", "mono")
        assert converted_mono.shape[0] == 1

        # Mono to stereo
        converted_stereo = converter.convert_channels(mono_audio, "mono", "stereo")
        assert converted_stereo.shape[0] == 2

    def test_convert_format_comprehensive(self, converter):
        """Test comprehensive format conversion."""
        source_audio = torch.randn(2, 48000)  # Stereo, 48kHz

        target_format = {"sample_rate": 24000, "channels": 1, "bit_depth": "int16"}

        converted = converter.convert_format(
            source_audio, source_sr=48000, target_format=target_format
        )

        assert converted.shape[0] == 1  # Mono
        assert converted.shape[1] == 24000  # 24kHz
        assert converted.dtype == torch.int16

    def test_batch_conversion(self, converter):
        """Test batch format conversion."""
        batch_audio = torch.randn(4, 2, 24000)  # Batch of 4 stereo tracks

        target_format = {"sample_rate": 48000, "channels": 1}

        converted_batch = converter.convert_batch(
            batch_audio, source_sr=24000, target_format=target_format
        )

        assert converted_batch.shape == (4, 1, 48000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
