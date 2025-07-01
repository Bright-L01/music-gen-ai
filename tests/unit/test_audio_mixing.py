"""
Unit tests for audio mixing functionality.
"""

import numpy as np
import pytest
import torch

from music_gen.audio.mixing.mixer import MixingEngine, MixingConfig, TrackConfig
from music_gen.audio.mixing.effects import EffectChain
from music_gen.audio.mixing.mastering import MasteringChain
from music_gen.audio.mixing.automation import AutomationLane


@pytest.mark.unit
class TestMixingEngine:
    """Test audio mixing functionality."""

    @pytest.fixture
    def mixer(self):
        """Create mixer instance."""
        config = MixingConfig(sample_rate=24000)
        return MixingEngine(config)

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
        assert mixer.config.sample_rate == 24000
        assert hasattr(mixer, "config")
        assert hasattr(mixer, "mix")

    def test_add_and_mix_tracks(self, mixer, sample_tracks):
        """Test mixing tracks."""
        # Create track configs
        track_configs = {}
        for name in sample_tracks:
            track_configs[name] = TrackConfig(name=name)

        # Mix tracks
        mixed = mixer.mix(sample_tracks, track_configs)

        assert isinstance(mixed, torch.Tensor)
        assert mixed.shape[0] == 2  # stereo output

        # Mixed signal should be different from individual tracks
        assert not torch.allclose(mixed[:1], sample_tracks["piano"])

    def test_mix_with_levels(self, mixer, sample_tracks):
        """Test mixing with custom track levels."""
        # Create track configs with different volumes
        track_configs = {
            "piano": TrackConfig(name="piano", volume=0.8),
            "bass": TrackConfig(name="bass", volume=1.0),
            "drums": TrackConfig(name="drums", volume=0.5),
        }

        mixed = mixer.mix(sample_tracks, track_configs)

        assert isinstance(mixed, torch.Tensor)
        assert mixed.shape[0] == 2  # stereo

    def test_mix_with_panning(self, mixer, sample_tracks):
        """Test mixing with panning."""
        # Create track configs with panning
        track_configs = {
            "piano": TrackConfig(name="piano", pan=-0.5),  # Left
            "bass": TrackConfig(name="bass", pan=0.0),  # Center
            "drums": TrackConfig(name="drums", pan=0.5),  # Right
        }

        mixed = mixer.mix(sample_tracks, track_configs)
        assert isinstance(mixed, torch.Tensor)
        assert mixed.shape[0] == 2  # stereo

    def test_mix_empty_tracks(self, mixer):
        """Test mixing with no tracks raises error."""
        # Try to mix empty tracks
        with pytest.raises(ValueError, match="No tracks to mix"):
            mixer.mix({}, {})


@pytest.mark.unit
class TestEffectChain:
    """Test audio effects processing."""

    @pytest.fixture
    def effect_chain(self):
        """Create effect chain."""
        return EffectChain(sample_rate=24000)

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        freq = 440.0

        t = torch.linspace(0, duration, samples)
        return torch.sin(2 * np.pi * freq * t).unsqueeze(0)

    def test_process_empty_chain(self, effect_chain, sample_audio):
        """Test processing through empty effect chain."""
        processed = effect_chain.process(sample_audio)

        assert processed.shape == sample_audio.shape
        # Empty chain should return unchanged audio
        assert torch.allclose(processed, sample_audio)

    def test_add_and_process_effect(self, effect_chain, sample_audio):
        """Test adding effect and processing."""

        # Mock effect
        class ScaleEffect:
            def process(self, audio):
                return audio * 0.5

        effect_chain.add_effect("scale", ScaleEffect())
        processed = effect_chain.process(sample_audio)

        assert processed.shape == sample_audio.shape
        # Should be scaled down
        assert torch.allclose(processed, sample_audio * 0.5)

    def test_bypass_effect(self, effect_chain, sample_audio):
        """Test bypassing effects."""

        # Add effect but bypass it
        class ScaleEffect:
            def __init__(self):
                self.bypass = True

            def process(self, audio):
                if self.bypass:
                    return audio
                return audio * 0.5

        effect = ScaleEffect()
        effect_chain.add_effect("scale", effect)

        # Process with bypass
        processed = effect_chain.process(sample_audio)
        assert torch.allclose(processed, sample_audio)

        # Process without bypass
        effect.bypass = False
        processed = effect_chain.process(sample_audio)
        assert torch.allclose(processed, sample_audio * 0.5)


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
            torch.sin(2 * np.pi * 440 * t) * 0.3
            + torch.sin(2 * np.pi * 880 * t) * 0.2
            + torch.sin(2 * np.pi * 220 * t) * 0.4
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
            eq_settings={"low": 1.0, "mid": 0.5, "high": 0.8},
            compression_settings={"threshold": -15.0, "ratio": 3.0, "makeup_gain": 2.0},
            limiter_settings={"threshold": -0.1, "release": 0.05},
        )

        assert mastered.shape == mixed_audio.shape
        assert mastered.abs().max() <= 1.0


@pytest.mark.unit
class TestAutomation:
    """Test automation functionality."""

    @pytest.fixture
    def lane(self):
        """Create automation lane."""
        return AutomationLane()

    def test_volume_automation(self, lane):
        """Test volume automation."""
        # Add automation points
        lane.add_point(0.0, 1.0)
        lane.add_point(5.0, 0.2)
        lane.add_point(10.0, 0.8)

        # Test interpolation
        assert lane.get_value(0.0) == 1.0
        assert lane.get_value(5.0) == 0.2
        assert lane.get_value(10.0) == 0.8

        # Test interpolation between points
        value_at_2_5 = lane.get_value(2.5)
        assert 0.2 < value_at_2_5 < 1.0

    def test_pan_automation(self, lane):
        """Test pan automation."""
        # Add automation for panning
        lane.add_point(0.0, -1.0)  # Full left
        lane.add_point(10.0, 1.0)  # Full right

        # Test smooth pan from left to right
        assert lane.get_value(0.0) == -1.0
        assert lane.get_value(5.0) == pytest.approx(0.0, rel=1e-2)
        assert lane.get_value(10.0) == 1.0

    def test_clear_automation(self, lane):
        """Test clearing automation points."""
        # Add some points
        lane.add_point(0.0, 1.0)
        lane.add_point(10.0, 0.0)

        assert len(lane.points) == 2

        # Clear points
        lane.clear()
        assert len(lane.points) == 0

    def test_get_value_before_first_point(self, lane):
        """Test getting value before first automation point."""
        # Add points starting at time 5.0
        lane.add_point(5.0, 0.5)
        lane.add_point(10.0, 1.0)

        # Value before first point should be the first point's value
        assert lane.get_value(0.0) == 0.5
        assert lane.get_value(3.0) == 0.5


@pytest.mark.integration
class TestMixingIntegration:
    """Integration tests for complete mixing workflow."""

    def test_complete_mixing_workflow(self):
        """Test complete mixing and mastering workflow."""
        # Create mixer
        config = MixingConfig(sample_rate=32000)
        mixer = MixingEngine(config)

        # Create test tracks
        duration = 3.0
        samples = int(32000 * duration)
        t = torch.linspace(0, duration, samples)

        # Create tracks (all mono)
        tracks = {
            "piano": torch.sin(2 * np.pi * 440 * t).unsqueeze(0) * 0.3,
            "bass": torch.sin(2 * np.pi * 110 * t).unsqueeze(0) * 0.5,
            "lead": torch.sin(2 * np.pi * 880 * t).unsqueeze(0) * 0.2,
            "drums": torch.randn(1, samples) * 0.1,
        }

        # Create track configs
        track_configs = {
            "piano": TrackConfig(name="piano", volume=0.8, pan=-0.3, reverb_send=0.2),
            "bass": TrackConfig(name="bass", volume=1.0, pan=0.0),
            "lead": TrackConfig(name="lead", volume=0.8, pan=0.3),
            "drums": TrackConfig(name="drums", volume=0.8, pan=0.0),
        }

        # Mix tracks
        mixed = mixer.mix(tracks, track_configs)

        # Master the mix
        mastering = MasteringChain(sample_rate=32000)
        final = mastering.process(mixed)

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
