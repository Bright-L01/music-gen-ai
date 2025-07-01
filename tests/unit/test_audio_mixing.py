"""
Unit tests for advanced audio mixing functionality.
"""

import numpy as np
import pytest
import torch

from music_gen.audio.advanced_mixing import (
    AdvancedMixingEngine,
    CompressorProcessor,
    EQProcessor,
    MasterBusSettings,
    ReverbProcessor,
    TrackSettings,
)


@pytest.mark.unit
class TestAdvancedMixer:
    """Test advanced audio mixing functionality."""

    @pytest.fixture
    def mixer(self):
        """Create mixer instance."""
        return AdvancedMixingEngine(sample_rate=24000)

    @pytest.fixture
    def sample_tracks(self):
        """Create sample audio tracks."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        tracks = {
            "piano": torch.sin(2 * np.pi * 440 * torch.linspace(0, duration, samples)).unsqueeze(0),
            "bass": torch.sin(2 * np.pi * 220 * torch.linspace(0, duration, samples)).unsqueeze(0),
            "drums": torch.randn(1, samples) * 0.3,
        }
        return tracks

    def test_mixer_initialization(self, mixer):
        """Test mixer initialization."""
        assert mixer.sample_rate == 24000
        assert hasattr(mixer, "reverb")
        assert hasattr(mixer, "delay")
        assert hasattr(mixer, "eq")
        assert hasattr(mixer, "compressor")

    def test_mix_tracks_basic(self, mixer, sample_tracks):
        """Test basic track mixing."""
        # Create track settings
        track_settings = {name: TrackSettings() for name in sample_tracks.keys()}

        # Convert PyTorch tensors to numpy arrays
        np_tracks = {name: audio.squeeze().numpy() for name, audio in sample_tracks.items()}

        mixed = mixer.mix_tracks(np_tracks, track_settings)

        assert isinstance(mixed, np.ndarray)
        assert len(mixed.shape) == 1  # Mono output (1D array)
        assert mixed.shape[0] == np_tracks["piano"].shape[0]  # Same length

        # Mixed signal should be different from individual tracks
        assert not np.allclose(mixed, np_tracks["piano"])

    def test_mix_with_levels(self, mixer, sample_tracks):
        """Test mixing with custom track levels."""
        track_settings = {
            "piano": TrackSettings(volume=0.8),
            "bass": TrackSettings(volume=1.2),
            "drums": TrackSettings(volume=0.5),
        }

        # Convert PyTorch tensors to numpy arrays
        np_tracks = {name: audio.squeeze().numpy() for name, audio in sample_tracks.items()}

        mixed = mixer.mix_tracks(np_tracks, track_settings)

        assert isinstance(mixed, np.ndarray)
        assert np.abs(mixed).max() <= 1.0  # Should not clip

    def test_mix_with_effects(self, mixer, sample_tracks):
        """Test mixing with effects applied."""
        track_settings = {
            "piano": TrackSettings(eq_low_gain=2.0, eq_high_gain=-1.0, reverb_send=0.3),
            "drums": TrackSettings(
                compressor_enabled=True, compressor_ratio=4.0, compressor_threshold=-12.0
            ),
        }

        # Convert PyTorch tensors to numpy arrays
        np_tracks = {name: audio.squeeze().numpy() for name, audio in sample_tracks.items()}

        mixed = mixer.mix_tracks(np_tracks, track_settings)
        assert isinstance(mixed, np.ndarray)

    def test_preset_application(self, mixer, sample_tracks):
        """Test applying mixing presets."""
        # Test different master bus settings as "presets"
        presets = {
            "jazz_club": MasterBusSettings(eq_low_shelf_gain=1.0, stereo_width=1.2),
            "rock_studio": MasterBusSettings(compressor_enabled=True, compressor_ratio=4.0),
            "ambient_hall": MasterBusSettings(eq_high_shelf_gain=-2.0, stereo_width=1.5),
        }

        track_settings = {name: TrackSettings() for name in sample_tracks.keys()}
        np_tracks = {name: audio.squeeze().numpy() for name, audio in sample_tracks.items()}

        for preset_name, master_settings in presets.items():
            mixed = mixer.mix_tracks(np_tracks, track_settings, master_settings)
            assert isinstance(mixed, np.ndarray)

    def test_mastering_chain(self, mixer, sample_tracks):
        """Test mastering chain application."""
        track_settings = {name: TrackSettings() for name in sample_tracks.keys()}
        np_tracks = {name: audio.squeeze().numpy() for name, audio in sample_tracks.items()}

        # Test master processing settings
        master_settings = MasterBusSettings(
            limiter_enabled=True, limiter_threshold=-1.0, compressor_enabled=True
        )

        mastered = mixer.mix_tracks(np_tracks, track_settings, master_settings)

        assert isinstance(mastered, np.ndarray)
        assert len(mastered.shape) == 1  # Mono output
        assert np.abs(mastered).max() <= 1.0


@pytest.mark.unit
class TestAudioEffects:
    """Test individual audio effects."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        freq = 440.0

        # Return numpy array instead of torch tensor
        t = np.linspace(0, duration, samples)
        return np.sin(2 * np.pi * freq * t)

    def test_eq_processor(self, sample_audio):
        """Test EQ processor."""
        eq = EQProcessor(sample_rate=24000)

        # Create track settings with EQ parameters
        settings = TrackSettings(eq_low_gain=3.0, eq_mid_gain=0.0, eq_high_gain=-2.0)

        processed = eq.process(sample_audio, settings)

        assert processed.shape == sample_audio.shape
        assert not np.allclose(processed, sample_audio)

    def test_compression_processor(self, sample_audio):
        """Test compression processor."""
        compressor = CompressorProcessor(sample_rate=24000)

        # Amplify signal to trigger compression
        loud_audio = sample_audio * 3.0
        compressed = compressor.process(
            loud_audio, threshold_db=-12.0, ratio=4.0, attack=0.005, release=0.05
        )

        assert compressed.shape == loud_audio.shape
        assert np.abs(compressed).max() < np.abs(loud_audio).max()

    def test_reverb_processor(self, sample_audio):
        """Test reverb processor."""
        reverb = ReverbProcessor(sample_rate=24000)

        processed = reverb.process(sample_audio, amount=0.5)

        assert processed.shape == sample_audio.shape
        assert not np.allclose(processed, sample_audio)


@pytest.mark.unit
class TestAutomation:
    """Test parameter automation."""

    def test_automation_curve(self):
        """Test automation curve generation."""
        # Simple linear interpolation test
        duration = 1.0
        sample_rate = 24000
        samples = int(duration * sample_rate)

        # Linear curve from 0 to 1
        linear = np.linspace(0.0, 1.0, samples)
        assert len(linear) == 24000
        assert linear[0] == pytest.approx(0.0, abs=1e-3)
        assert linear[-1] == pytest.approx(1.0, abs=1e-3)

        # Exponential curve
        exp = np.exp(np.linspace(np.log(0.1), np.log(1.0), samples))
        assert len(exp) == 24000
        assert exp[0] == pytest.approx(0.1, abs=1e-3)
        assert exp[-1] == pytest.approx(1.0, abs=1e-3)

    def test_parameter_automation(self):
        """Test parameter automation application."""
        # Simple interpolation between points
        points = [(0.0, 0.5), (0.5, 1.0), (1.0, 0.3)]
        duration = 1.0
        sample_rate = 24000
        samples = int(duration * sample_rate)

        # Linear interpolation
        t = np.linspace(0, 1, samples)
        volume = np.interp(t, [p[0] for p in points], [p[1] for p in points])

        assert len(volume) == 24000
        assert volume[0] == pytest.approx(0.5, abs=1e-3)
        assert volume[samples // 2] == pytest.approx(1.0, abs=1e-1)


@pytest.mark.unit
class TestMasteringChain:
    """Test mastering chain functionality."""

    @pytest.fixture
    def mastering_chain(self):
        """Create mastering chain."""
        return AdvancedMixingEngine(sample_rate=24000)

    @pytest.fixture
    def mixed_audio(self):
        """Create mixed audio for mastering."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        # Simulate mixed audio with some dynamics
        t = np.linspace(0, duration, samples)
        audio = np.random.randn(samples) * 0.3
        audio += np.sin(2 * np.pi * 440 * t) * 0.5

        return audio

    def test_loudness_normalization(self, mastering_chain, mixed_audio):
        """Test loudness normalization."""
        # Test master processing with volume control
        settings = MasterBusSettings(volume=0.8)
        stereo_audio = np.stack([mixed_audio, mixed_audio])  # Create stereo

        normalized = mastering_chain.apply_master_processing(stereo_audio, settings)

        assert normalized.shape == stereo_audio.shape
        assert isinstance(normalized, np.ndarray)

    def test_limiting(self, mastering_chain, mixed_audio):
        """Test audio limiting."""
        # Amplify to cause clipping
        loud_audio = mixed_audio * 2.0

        # Test limiting via master processing
        settings = MasterBusSettings(limiter_enabled=True, limiter_threshold=-1.0)
        stereo_audio = np.stack([loud_audio, loud_audio])  # Create stereo

        limited = mastering_chain.apply_master_processing(stereo_audio, settings)

        assert limited.shape == stereo_audio.shape
        assert np.abs(limited).max() <= 1.0

    def test_complete_mastering(self, mastering_chain, mixed_audio):
        """Test complete mastering chain."""
        settings = MasterBusSettings(
            volume=0.8,
            eq_enabled=True,
            compressor_enabled=True,
            limiter_enabled=True,
            limiter_threshold=-0.3,
        )

        stereo_audio = np.stack([mixed_audio, mixed_audio])  # Create stereo
        mastered = mastering_chain.apply_master_processing(stereo_audio, settings)

        assert mastered.shape == stereo_audio.shape
        assert np.abs(mastered).max() <= 0.99  # Accounting for headroom

    def test_stereo_imaging(self, mastering_chain):
        """Test stereo imaging processing."""
        # Create stereo audio
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)

        stereo_audio = np.random.randn(2, samples)

        # Test stereo width processing via master settings
        settings = MasterBusSettings(stereo_width=1.2)
        processed = mastering_chain.apply_master_processing(stereo_audio, settings)

        assert processed.shape == stereo_audio.shape
        assert not np.allclose(processed, stereo_audio)


@pytest.mark.unit
class TestAudioValidation:
    """Test audio validation and analysis."""

    def test_audio_quality_metrics(self):
        """Test audio quality metric calculation."""
        # Simple audio analysis without external dependencies
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 440 * t)

        # Calculate basic metrics
        rms_level = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8)
        peak_level = 20 * np.log10(np.max(np.abs(audio)) + 1e-8)
        dynamic_range = peak_level - rms_level

        assert rms_level > -10  # Should have reasonable level (sine wave ~-3dB)
        assert peak_level <= 1  # Peak should be around 0 dB
        assert dynamic_range >= 0  # Dynamic range should be positive

    def test_clipping_detection(self):
        """Test clipping detection."""
        # Create clipped audio
        sample_rate = 24000
        duration = 0.5
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)

        clean_audio = np.sin(2 * np.pi * 440 * t) * 0.8
        clipped_audio = np.clip(clean_audio * 2.0, -1.0, 1.0)

        # Simple clipping detection
        clean_clipped = np.any(np.abs(clean_audio) >= 0.99)
        has_clipping = np.any(np.abs(clipped_audio) >= 0.99)

        assert not clean_clipped
        assert has_clipping

    def test_silence_detection(self):
        """Test silence detection."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)

        # Create audio with silence in middle
        t1 = np.linspace(0, 0.5, samples // 4)
        t2 = np.linspace(0, 0.5, samples // 4)

        audio = np.concatenate(
            [
                np.sin(2 * np.pi * 440 * t1),  # First part
                np.zeros(samples // 2),  # Silence
                np.sin(2 * np.pi * 880 * t2),  # Last part
            ]
        )

        # Simple silence detection
        threshold = -40.0  # dB
        threshold_linear = 10 ** (threshold / 20)
        silence_mask = np.abs(audio) < threshold_linear

        # Find silence segments
        silence_starts = np.where(np.diff(silence_mask.astype(int)) == 1)[0]
        silence_ends = np.where(np.diff(silence_mask.astype(int)) == -1)[0]

        assert len(silence_starts) > 0  # Should find at least one silence start
        assert len(silence_ends) > 0  # Should find at least one silence end


if __name__ == "__main__":
    pytest.main([__file__])
