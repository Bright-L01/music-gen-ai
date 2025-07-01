"""
Tests for audio mixing functionality.
"""

import pytest
import numpy as np
import torch

from music_gen.audio.mixing.mixer import AudioMixer
from music_gen.audio.mixing.effects import EffectsProcessor
from music_gen.audio.mixing.automation import AutomationEngine
from music_gen.audio.mixing.mastering import MasteringChain


class TestAudioMixer:
    """Test cases for AudioMixer class."""

    def test_mixer_initialization(self):
        """Test AudioMixer initialization."""
        mixer = AudioMixer(sample_rate=32000, channels=1)
        assert mixer.sample_rate == 32000
        assert mixer.channels == 1

    def test_add_track(self):
        """Test adding tracks to mixer."""
        mixer = AudioMixer()
        
        # Create dummy audio
        audio = torch.randn(1, 32000)
        
        # Add track
        mixer.add_track("track1", audio)
        assert "track1" in mixer.tracks
        assert mixer.tracks["track1"].shape == audio.shape

    def test_mix_tracks(self):
        """Test mixing multiple tracks."""
        mixer = AudioMixer()
        
        # Add multiple tracks
        track1 = torch.randn(1, 32000)
        track2 = torch.randn(1, 32000)
        
        mixer.add_track("track1", track1, volume=0.8)
        mixer.add_track("track2", track2, volume=0.6)
        
        # Mix tracks
        mixed = mixer.mix()
        assert mixed is not None
        assert mixed.shape == track1.shape


class TestEffectsProcessor:
    """Test cases for EffectsProcessor."""

    def test_effects_processor_initialization(self):
        """Test EffectsProcessor initialization."""
        processor = EffectsProcessor()
        assert processor is not None

    def test_apply_reverb(self):
        """Test reverb effect."""
        processor = EffectsProcessor()
        audio = torch.randn(1, 32000)
        
        # Apply reverb
        processed = processor.apply_reverb(audio, room_size=0.5, damping=0.3)
        assert processed is not None
        assert processed.shape == audio.shape

    def test_apply_compression(self):
        """Test compression effect."""
        processor = EffectsProcessor()
        audio = torch.randn(1, 32000)
        
        # Apply compression
        compressed = processor.apply_compression(
            audio, 
            threshold=-20.0,
            ratio=4.0,
            attack=0.001,
            release=0.1
        )
        assert compressed is not None
        assert compressed.shape == audio.shape


class TestAutomationEngine:
    """Test cases for AutomationEngine."""

    def test_automation_engine_initialization(self):
        """Test AutomationEngine initialization."""
        engine = AutomationEngine()
        assert engine is not None

    def test_add_automation_point(self):
        """Test adding automation points."""
        engine = AutomationEngine()
        
        # Add volume automation
        engine.add_point("volume", time=0.0, value=1.0)
        engine.add_point("volume", time=5.0, value=0.5)
        engine.add_point("volume", time=10.0, value=0.8)
        
        assert "volume" in engine.parameters
        assert len(engine.parameters["volume"]) == 3

    def test_get_value_at_time(self):
        """Test getting interpolated values."""
        engine = AutomationEngine()
        
        # Add automation points
        engine.add_point("pan", time=0.0, value=-1.0)
        engine.add_point("pan", time=10.0, value=1.0)
        
        # Test interpolation
        value = engine.get_value("pan", time=5.0)
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
            limiter_settings={"threshold": -0.3}
        )
        
        assert mastered is not None
        assert mastered.shape == audio.shape


@pytest.mark.integration
class TestMixingIntegration:
    """Integration tests for mixing components."""

    def test_full_mixing_pipeline(self):
        """Test complete mixing pipeline."""
        # Create mixer
        mixer = AudioMixer(sample_rate=32000)
        
        # Add tracks
        for i in range(3):
            track = torch.randn(1, 32000)
            mixer.add_track(f"track_{i}", track, volume=0.7)
        
        # Create effects processor
        effects = EffectsProcessor()
        
        # Mix tracks
        mixed = mixer.mix()
        
        # Apply effects
        with_reverb = effects.apply_reverb(mixed, room_size=0.3)
        compressed = effects.apply_compression(with_reverb)
        
        # Master the final mix
        mastering = MasteringChain()
        final = mastering.process(compressed)
        
        assert final is not None
        assert final.shape == (1, 32000)
        assert not torch.isnan(final).any()
        assert not torch.isinf(final).any()


if __name__ == "__main__":
    pytest.main([__file__])