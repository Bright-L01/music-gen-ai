"""
Tests for audio mixing functionality.
"""

import pytest
import numpy as np
import torch

from music_gen.audio.mixing.mixer import MixingEngine, MixingConfig, TrackConfig
from music_gen.audio.mixing.effects import EffectChain
from music_gen.audio.mixing.automation import AutomationLane
from music_gen.audio.mixing.mastering import MasteringChain


class TestMixingEngine:
    """Test cases for MixingEngine class."""

    def test_mixer_initialization(self):
        """Test MixingEngine initialization."""
        config = MixingConfig(sample_rate=32000, channels=1)
        mixer = MixingEngine(config)
        assert mixer.config.sample_rate == 32000
        assert mixer.config.channels == 1

    def test_mix_tracks(self):
        """Test mixing multiple tracks."""
        config = MixingConfig(sample_rate=32000)
        mixer = MixingEngine(config)

        # Create tracks
        tracks = {"track1": torch.randn(1, 32000), "track2": torch.randn(1, 32000)}

        # Create track configs
        track_configs = {
            "track1": TrackConfig(name="track1", volume=0.8),
            "track2": TrackConfig(name="track2", volume=0.6),
        }

        # Mix tracks
        mixed = mixer.mix(tracks, track_configs)
        assert mixed is not None
        assert mixed.shape[0] == 2  # stereo output


class TestEffectChain:
    """Test cases for EffectChain."""

    def test_effect_chain_initialization(self):
        """Test EffectChain initialization."""
        chain = EffectChain(sample_rate=32000)
        assert chain is not None
        assert chain.sample_rate == 32000

    def test_add_effect(self):
        """Test adding effects to chain."""
        chain = EffectChain(sample_rate=32000)

        # Mock effect
        class MockEffect:
            def process(self, audio):
                return audio * 0.5

        chain.add_effect("mock", MockEffect())
        assert "mock" in chain.effects

    def test_process_chain(self):
        """Test processing audio through effect chain."""
        chain = EffectChain(sample_rate=32000)
        audio = torch.randn(1, 32000)

        # Process through empty chain
        processed = chain.process(audio)
        assert processed is not None
        assert processed.shape == audio.shape


class TestAutomationLane:
    """Test cases for AutomationLane."""

    def test_automation_lane_initialization(self):
        """Test AutomationLane initialization."""
        lane = AutomationLane()
        assert lane is not None

    def test_add_automation_point(self):
        """Test adding automation points."""
        lane = AutomationLane()

        # Add automation points
        lane.add_point(0.0, 1.0)
        lane.add_point(5.0, 0.5)
        lane.add_point(10.0, 0.8)

        assert len(lane.points) == 3

    def test_get_value_at_time(self):
        """Test getting interpolated values."""
        lane = AutomationLane()

        # Add automation points
        lane.add_point(0.0, -1.0)
        lane.add_point(10.0, 1.0)

        # Test interpolation
        value = lane.get_value(5.0)
        assert value == pytest.approx(0.0, rel=1e-3)


class TestMasteringChain:
    """Test cases for MasteringChain."""

    def test_mastering_chain_initialization(self):
        """Test MasteringChain initialization."""
        chain = MasteringChain()
        assert chain is not None

    def test_apply_mastering(self):
        """Test applying mastering chain."""
        chain = MasteringChain()
        audio = torch.randn(1, 32000)

        # Apply mastering
        mastered = chain.process(
            audio,
            eq_settings={"low": 0.0, "mid": 0.2, "high": 0.1},
            compression_settings={"threshold": -15.0, "ratio": 3.0},
            limiter_settings={"threshold": -0.3},
        )

        assert mastered is not None
        assert mastered.shape == audio.shape


@pytest.mark.integration
class TestMixingIntegration:
    """Integration tests for mixing components."""

    def test_full_mixing_pipeline(self):
        """Test complete mixing pipeline."""
        # Create mixer
        config = MixingConfig(sample_rate=32000)
        mixer = MixingEngine(config)

        # Create tracks
        tracks = {}
        track_configs = {}
        for i in range(3):
            tracks[f"track_{i}"] = torch.randn(1, 32000)
            track_configs[f"track_{i}"] = TrackConfig(
                name=f"track_{i}", volume=0.7, reverb_send=0.2 if i == 0 else 0.0
            )

        # Mix tracks
        mixed = mixer.mix(tracks, track_configs)

        # Master the final mix
        mastering = MasteringChain(sample_rate=32000)
        final = mastering.process(mixed)

        assert final is not None
        assert final.shape[0] == 2  # stereo
        assert not torch.isnan(final).any()
        assert not torch.isinf(final).any()


if __name__ == "__main__":
    pytest.main([__file__])
