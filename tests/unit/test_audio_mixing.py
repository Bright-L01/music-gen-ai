"""
Unit tests for audio mixing functionality.
"""

import numpy as np
import pytest
import torch

from music_gen.audio.mixing.mixer import AudioMixer
from music_gen.audio.mixing.effects import EffectsProcessor
from music_gen.audio.mixing.mastering import MasteringChain
from music_gen.audio.mixing.automation import AutomationEngine


@pytest.mark.unit
class TestAudioMixer:
    """Test audio mixing functionality."""

    @pytest.fixture
    def mixer(self):
        """Create mixer instance."""
        return AudioMixer(sample_rate=24000)

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
        assert hasattr(mixer, "tracks")
        assert hasattr(mixer, "mix")

    def test_add_and_mix_tracks(self, mixer, sample_tracks):
        """Test adding and mixing tracks."""
        # Add tracks to mixer
        for name, audio in sample_tracks.items():
            mixer.add_track(name, audio)

        # Mix tracks
        mixed = mixer.mix()

        assert isinstance(mixed, torch.Tensor)
        assert mixed.shape == sample_tracks["piano"].shape
        
        # Mixed signal should be different from individual tracks
        assert not torch.allclose(mixed, sample_tracks["piano"])

    def test_mix_with_levels(self, mixer, sample_tracks):
        """Test mixing with custom track levels."""
        # Add tracks with different volumes
        mixer.add_track("piano", sample_tracks["piano"], volume=0.8)
        mixer.add_track("bass", sample_tracks["bass"], volume=1.2)
        mixer.add_track("drums", sample_tracks["drums"], volume=0.5)

        mixed = mixer.mix()

        assert isinstance(mixed, torch.Tensor)
        assert torch.abs(mixed).max() <= 1.5  # Check reasonable range

    def test_mix_with_panning(self, mixer, sample_tracks):
        """Test mixing with panning."""
        # Add tracks with panning
        mixer.add_track("piano", sample_tracks["piano"], pan=-0.5)  # Left
        mixer.add_track("bass", sample_tracks["bass"], pan=0.0)    # Center
        mixer.add_track("drums", sample_tracks["drums"], pan=0.5)  # Right

        mixed = mixer.mix()
        assert isinstance(mixed, torch.Tensor)

    def test_clear_tracks(self, mixer, sample_tracks):
        """Test clearing tracks from mixer."""
        # Add tracks
        for name, audio in sample_tracks.items():
            mixer.add_track(name, audio)
        
        assert len(mixer.tracks) == 3
        
        # Clear tracks
        mixer.clear()
        assert len(mixer.tracks) == 0


@pytest.mark.unit
class TestEffectsProcessor:
    """Test audio effects processing."""

    @pytest.fixture
    def processor(self):
        """Create effects processor."""
        return EffectsProcessor(sample_rate=24000)

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        freq = 440.0
        
        t = torch.linspace(0, duration, samples)
        return torch.sin(2 * np.pi * freq * t).unsqueeze(0)

    def test_reverb_effect(self, processor, sample_audio):
        """Test reverb effect."""
        processed = processor.apply_reverb(
            sample_audio,
            room_size=0.5,
            damping=0.3,
            wet_level=0.3
        )

        assert processed.shape == sample_audio.shape
        assert not torch.allclose(processed, sample_audio)
        
        # Reverb should extend the signal slightly
        assert processed.abs().sum() > sample_audio.abs().sum() * 0.9

    def test_delay_effect(self, processor, sample_audio):
        """Test delay effect."""
        processed = processor.apply_delay(
            sample_audio,
            delay_time=0.1,
            feedback=0.3,
            mix=0.5
        )

        assert processed.shape == sample_audio.shape
        assert not torch.allclose(processed, sample_audio)

    def test_eq_effect(self, processor, sample_audio):
        """Test EQ effect."""
        processed = processor.apply_eq(
            sample_audio,
            low_gain=2.0,
            mid_gain=0.0,
            high_gain=-1.0
        )

        assert processed.shape == sample_audio.shape
        assert not torch.allclose(processed, sample_audio)

    def test_compression_effect(self, processor, sample_audio):
        """Test compression effect."""
        # Amplify signal to trigger compression
        loud_audio = sample_audio * 3.0
        
        compressed = processor.apply_compression(
            loud_audio,
            threshold=-12.0,
            ratio=4.0,
            attack=0.005,
            release=0.05
        )

        assert compressed.shape == loud_audio.shape
        # Compressed signal should have lower peak than input
        assert compressed.abs().max() < loud_audio.abs().max()

    def test_limiter_effect(self, processor, sample_audio):
        """Test limiter effect."""
        # Create signal that would clip
        loud_audio = sample_audio * 5.0
        
        limited = processor.apply_limiter(
            loud_audio,
            threshold=-0.1,
            release=0.01
        )

        assert limited.shape == loud_audio.shape
        # Limited signal should not exceed threshold
        assert limited.abs().max() <= 1.0


@pytest.mark.unit
class TestMasteringChain:
    """Test mastering chain functionality."""

    @pytest.fixture
    def chain(self):
        """Create mastering chain."""
        return MasteringChain(sample_rate=24000)

    @pytest.fixture
    def mixed_audio(self):
        """Create pre-mixed audio for mastering."""
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Simulate a mixed track with multiple frequencies
        t = torch.linspace(0, duration, samples)
        audio = (
            torch.sin(2 * np.pi * 440 * t) * 0.3 +
            torch.sin(2 * np.pi * 880 * t) * 0.2 +
            torch.sin(2 * np.pi * 220 * t) * 0.4
        )
        return audio.unsqueeze(0)

    def test_mastering_process(self, chain, mixed_audio):
        """Test full mastering process."""
        mastered = chain.process(mixed_audio)

        assert mastered.shape == mixed_audio.shape
        assert not torch.allclose(mastered, mixed_audio)
        
        # Mastered audio should be louder but not clipping
        assert mastered.abs().mean() > mixed_audio.abs().mean()
        assert mastered.abs().max() <= 1.0

    def test_mastering_with_settings(self, chain, mixed_audio):
        """Test mastering with custom settings."""
        mastered = chain.process(
            mixed_audio,
            eq_settings={
                "low": 1.0,
                "mid": 0.5,
                "high": 0.8
            },
            compression_settings={
                "threshold": -15.0,
                "ratio": 3.0,
                "makeup_gain": 2.0
            },
            limiter_settings={
                "threshold": -0.1,
                "release": 0.05
            }
        )

        assert mastered.shape == mixed_audio.shape
        assert mastered.abs().max() <= 1.0


@pytest.mark.unit
class TestAutomation:
    """Test automation functionality."""

    @pytest.fixture
    def engine(self):
        """Create automation engine."""
        return AutomationEngine()

    def test_volume_automation(self, engine):
        """Test volume automation."""
        # Add automation points
        engine.add_point("volume", 0.0, 1.0)
        engine.add_point("volume", 5.0, 0.2)
        engine.add_point("volume", 10.0, 0.8)

        # Test interpolation
        assert engine.get_value("volume", 0.0) == 1.0
        assert engine.get_value("volume", 5.0) == 0.2
        assert engine.get_value("volume", 10.0) == 0.8
        
        # Test interpolation between points
        value_at_2_5 = engine.get_value("volume", 2.5)
        assert 0.2 < value_at_2_5 < 1.0

    def test_pan_automation(self, engine):
        """Test pan automation."""
        # Add automation for panning
        engine.add_point("pan", 0.0, -1.0)  # Full left
        engine.add_point("pan", 10.0, 1.0)  # Full right

        # Test smooth pan from left to right
        assert engine.get_value("pan", 0.0) == -1.0
        assert engine.get_value("pan", 5.0) == pytest.approx(0.0, rel=1e-2)
        assert engine.get_value("pan", 10.0) == 1.0

    def test_multiple_parameters(self, engine):
        """Test automating multiple parameters."""
        # Add automation for multiple parameters
        engine.add_point("volume", 0.0, 1.0)
        engine.add_point("volume", 10.0, 0.5)
        
        engine.add_point("reverb", 0.0, 0.0)
        engine.add_point("reverb", 10.0, 0.8)

        # Check that parameters are independent
        assert engine.get_value("volume", 5.0) == pytest.approx(0.75, rel=1e-2)
        assert engine.get_value("reverb", 5.0) == pytest.approx(0.4, rel=1e-2)

    def test_clear_automation(self, engine):
        """Test clearing automation data."""
        # Add some points
        engine.add_point("volume", 0.0, 1.0)
        engine.add_point("volume", 10.0, 0.0)
        
        assert "volume" in engine.parameters
        
        # Clear specific parameter
        engine.clear_parameter("volume")
        assert "volume" not in engine.parameters
        
        # Clear all
        engine.add_point("pan", 0.0, 0.0)
        engine.clear_all()
        assert len(engine.parameters) == 0


@pytest.mark.integration
class TestMixingIntegration:
    """Integration tests for complete mixing workflow."""

    def test_complete_mixing_workflow(self):
        """Test complete mixing and mastering workflow."""
        # Create mixer
        mixer = AudioMixer(sample_rate=32000)
        
        # Create test tracks
        duration = 3.0
        samples = int(32000 * duration)
        t = torch.linspace(0, duration, samples)
        
        # Add multiple instrument tracks
        tracks = {
            "piano": torch.sin(2 * np.pi * 440 * t) * 0.3,
            "bass": torch.sin(2 * np.pi * 110 * t) * 0.5,
            "lead": torch.sin(2 * np.pi * 880 * t) * 0.2,
            "drums": torch.randn(samples) * 0.1,
        }
        
        # Add tracks with different settings
        for name, audio in tracks.items():
            volume = 0.8 if name != "bass" else 1.0
            pan = {"piano": -0.3, "bass": 0.0, "lead": 0.3, "drums": 0.0}.get(name, 0.0)
            mixer.add_track(name, audio.unsqueeze(0), volume=volume, pan=pan)
        
        # Mix tracks
        mixed = mixer.mix()
        
        # Apply effects
        effects = EffectsProcessor(sample_rate=32000)
        
        # Add some reverb
        with_reverb = effects.apply_reverb(mixed, room_size=0.3, wet_level=0.2)
        
        # Apply EQ
        eq_processed = effects.apply_eq(with_reverb, low_gain=0.5, mid_gain=0.0, high_gain=0.3)
        
        # Master the mix
        mastering = MasteringChain(sample_rate=32000)
        final = mastering.process(eq_processed)
        
        # Verify output
        assert final is not None
        assert final.shape[0] == 1  # Mono/single channel
        assert final.shape[1] == samples
        assert final.abs().max() <= 1.0  # No clipping
        assert not torch.isnan(final).any()
        assert not torch.isinf(final).any()
        
        # Output should be different from input
        assert not torch.allclose(final, tracks["piano"].unsqueeze(0))


if __name__ == "__main__":
    pytest.main([__file__])