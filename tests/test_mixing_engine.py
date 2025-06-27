"""Tests for professional audio mixing engine."""

import pytest
import torch
import numpy as np

from music_gen.audio.mixing import (
    MixingEngine,
    MixingConfig,
    TrackConfig,
    EffectChain,
    AutomationLane,
    AutomationPoint,
    InterpolationType
)
from music_gen.audio.mixing.effects import (
    EQ,
    Compressor,
    Reverb,
    Delay,
    Chorus,
    Limiter,
    Gate,
    Distortion
)


class TestMixingEngine:
    """Test the main mixing engine."""
    
    @pytest.fixture
    def mixing_engine(self):
        config = MixingConfig(sample_rate=44100, channels=2)
        return MixingEngine(config)
    
    def test_simple_mix(self, mixing_engine):
        """Test basic mixing of multiple tracks."""
        # Create test tracks
        duration = 1.0  # 1 second
        samples = int(44100 * duration)
        
        tracks = {
            "track1": torch.sin(2 * np.pi * 440 * torch.linspace(0, duration, samples)),  # A4
            "track2": torch.sin(2 * np.pi * 880 * torch.linspace(0, duration, samples)),  # A5
        }
        
        track_configs = {
            "track1": TrackConfig(name="track1", volume=0.5),
            "track2": TrackConfig(name="track2", volume=0.3),
        }
        
        mixed = mixing_engine.mix(tracks, track_configs)
        
        assert mixed.shape == (2, samples)  # Stereo output
        assert mixed.abs().max() <= 1.0  # No clipping
        
    def test_panning(self, mixing_engine):
        """Test stereo panning."""
        samples = 44100
        mono_track = torch.ones(samples)
        
        tracks = {
            "left": mono_track.clone(),
            "center": mono_track.clone(),
            "right": mono_track.clone(),
        }
        
        track_configs = {
            "left": TrackConfig(name="left", volume=1.0, pan=-1.0),
            "center": TrackConfig(name="center", volume=1.0, pan=0.0),
            "right": TrackConfig(name="right", volume=1.0, pan=1.0),
        }
        
        mixed = mixing_engine.mix(tracks, track_configs)
        
        # Check panning worked correctly
        left_channel = mixed[0]
        right_channel = mixed[1]
        
        # Left-panned track should be louder in left channel
        assert left_channel[:100].mean() > right_channel[:100].mean()
        
    def test_mute_solo(self, mixing_engine):
        """Test mute and solo functionality."""
        samples = 1000
        
        tracks = {
            "track1": torch.ones(samples),
            "track2": torch.ones(samples) * 0.5,
            "track3": torch.ones(samples) * 0.3,
        }
        
        # Test mute
        track_configs = {
            "track1": TrackConfig(name="track1", volume=1.0, mute=True),
            "track2": TrackConfig(name="track2", volume=1.0),
            "track3": TrackConfig(name="track3", volume=1.0),
        }
        
        mixed = mixing_engine.mix(tracks, track_configs)
        # Muted track1 shouldn't contribute
        assert mixed.mean() < 1.0
        
        # Test solo
        track_configs["track2"].solo = True
        mixed_solo = mixing_engine.mix(tracks, track_configs)
        # Only track2 should be heard
        assert torch.allclose(mixed_solo.mean(), torch.tensor(0.5), atol=0.1)
        
    def test_automation(self, mixing_engine):
        """Test parameter automation."""
        samples = 44100
        track = torch.ones(samples)
        
        # Create volume automation (fade in)
        volume_automation = AutomationLane("volume", default_value=0.0)
        volume_automation.add_point(0.0, 0.0, InterpolationType.LINEAR)
        volume_automation.add_point(1.0, 1.0, InterpolationType.LINEAR)
        
        track_config = TrackConfig(name="track", volume=1.0)
        track_config.automation["volume"] = volume_automation
        
        tracks = {"track": track}
        track_configs = {"track": track_config}
        
        mixed = mixing_engine.mix(tracks, track_configs)
        
        # Volume should increase over time
        assert mixed[:, 0].mean() < mixed[:, -1].mean()


class TestAudioEffects:
    """Test individual audio effects."""
    
    @pytest.fixture
    def test_audio(self):
        """Create test audio signal."""
        duration = 1.0
        sample_rate = 44100
        t = torch.linspace(0, duration, int(sample_rate * duration))
        # Mix of frequencies
        signal = (
            torch.sin(2 * np.pi * 440 * t) * 0.5 +  # A4
            torch.sin(2 * np.pi * 880 * t) * 0.3 +  # A5
            torch.sin(2 * np.pi * 110 * t) * 0.2    # A2 (bass)
        )
        return signal.unsqueeze(0).repeat(2, 1)  # Stereo
    
    def test_eq(self, test_audio):
        """Test equalizer effect."""
        eq = EQ(
            sample_rate=44100,
            bands=[
                {"freq": 100, "gain": 6, "q": 0.7, "type": "low_shelf"},
                {"freq": 1000, "gain": -3, "q": 1.0, "type": "bell"},
                {"freq": 10000, "gain": 3, "q": 0.7, "type": "high_shelf"},
            ]
        )
        
        processed = eq.process(test_audio)
        
        assert processed.shape == test_audio.shape
        # EQ should change the signal
        assert not torch.allclose(processed, test_audio)
        
    def test_compressor(self, test_audio):
        """Test compressor effect."""
        # Add some dynamics
        test_audio = test_audio.clone()
        test_audio[:, :1000] *= 0.1  # Quiet section
        test_audio[:, 1000:] *= 2.0  # Loud section
        
        compressor = Compressor(
            sample_rate=44100,
            threshold=-10,
            ratio=4,
            attack=0.01,
            release=0.1
        )
        
        processed = compressor.process(test_audio)
        
        # Compressor should reduce dynamic range
        orig_range = test_audio.max() - test_audio.min()
        comp_range = processed.max() - processed.min()
        assert comp_range < orig_range
        
    def test_reverb(self, test_audio):
        """Test reverb effect."""
        # Short impulse for reverb testing
        impulse = torch.zeros_like(test_audio)
        impulse[:, 1000] = 1.0
        
        reverb = Reverb(
            sample_rate=44100,
            room_size=0.8,
            damping=0.5,
            wet_mix=0.5
        )
        
        processed = reverb.process(impulse)
        
        # Reverb should create decay tail
        assert processed[:, 1100:].abs().sum() > 0  # Energy after impulse
        assert processed.abs().max() <= 1.0  # No clipping
        
    def test_delay(self, test_audio):
        """Test delay effect."""
        # Short click for delay testing
        click = torch.zeros_like(test_audio)
        click[:, 1000] = 1.0
        
        delay = Delay(
            sample_rate=44100,
            delay_time=0.1,  # 100ms
            feedback=0.5,
            wet_mix=1.0
        )
        
        processed = delay.process(click)
        
        # Should have delayed copy at ~4410 samples (100ms at 44.1kHz)
        delay_samples = int(0.1 * 44100)
        assert processed[:, 1000 + delay_samples].abs().mean() > 0.4
        
    def test_chorus(self, test_audio):
        """Test chorus effect."""
        chorus = Chorus(
            sample_rate=44100,
            num_voices=4,
            depth=0.02,
            rate=1.5,
            mix=0.5
        )
        
        processed = chorus.process(test_audio)
        
        assert processed.shape == test_audio.shape
        # Chorus should create slight variations
        assert not torch.allclose(processed, test_audio)
        
    def test_limiter(self, test_audio):
        """Test limiter effect."""
        # Create clipping signal
        loud_audio = test_audio * 3.0
        
        limiter = Limiter(
            sample_rate=44100,
            threshold=-0.3,
            release=0.05
        )
        
        processed = limiter.process(loud_audio)
        
        # Limiter should prevent clipping
        assert processed.abs().max() <= 10 ** (-0.3 / 20) + 0.01  # Small tolerance
        
    def test_gate(self, test_audio):
        """Test noise gate effect."""
        # Add noise floor
        noisy_audio = test_audio.clone()
        noise = torch.randn_like(test_audio) * 0.01
        noisy_audio[:, :10000] = noise[:, :10000]  # Noise only section
        noisy_audio[:, 10000:] += noise[:, 10000:]  # Signal + noise
        
        gate = Gate(
            sample_rate=44100,
            threshold=-30,
            attack=0.001,
            release=0.1
        )
        
        processed = gate.process(noisy_audio)
        
        # Gate should reduce noise in quiet section
        assert processed[:, :5000].abs().mean() < noisy_audio[:, :5000].abs().mean()
        
    def test_distortion(self, test_audio):
        """Test distortion effect."""
        distortion = Distortion(
            sample_rate=44100,
            drive=5.0,
            tone=0.7,
            output_gain=0.5,
            mode="soft"
        )
        
        processed = distortion.process(test_audio)
        
        # Distortion should add harmonics
        assert processed.abs().max() <= 1.0
        # Check that signal is modified
        assert not torch.allclose(processed, test_audio)


class TestEffectChain:
    """Test effect chaining."""
    
    def test_chain_processing(self):
        """Test processing through multiple effects."""
        chain = EffectChain(sample_rate=44100)
        
        # Add effects to chain
        chain.add_effect("eq", EQ(44100, [{"freq": 1000, "gain": 3, "q": 1, "type": "bell"}]))
        chain.add_effect("comp", Compressor(44100, threshold=-15, ratio=3))
        chain.add_effect("reverb", Reverb(44100, room_size=0.3, wet_mix=0.2))
        
        # Process audio
        test_audio = torch.randn(2, 44100)
        processed = chain.process(test_audio)
        
        assert processed.shape == test_audio.shape
        assert not torch.allclose(processed, test_audio)
        
    def test_chain_management(self):
        """Test adding/removing effects from chain."""
        chain = EffectChain(sample_rate=44100)
        
        # Add effects
        chain.add_effect("effect1", Compressor(44100))
        chain.add_effect("effect2", Reverb(44100))
        
        assert len(chain.effects) == 2
        
        # Remove effect
        chain.remove_effect("effect1")
        assert len(chain.effects) == 1
        assert chain.effects[0][0] == "effect2"


class TestAutomation:
    """Test automation system."""
    
    def test_linear_automation(self):
        """Test linear interpolation."""
        lane = AutomationLane("volume", default_value=0.5, min_value=0.0, max_value=1.0)
        
        lane.add_point(0.0, 0.0)
        lane.add_point(1.0, 1.0)
        
        # Test interpolation
        assert lane.get_value(0.0) == 0.0
        assert lane.get_value(0.5) == 0.5
        assert lane.get_value(1.0) == 1.0
        
    def test_exponential_automation(self):
        """Test exponential interpolation."""
        lane = AutomationLane("volume")
        
        lane.add_point(0.0, 0.1, InterpolationType.EXPONENTIAL)
        lane.add_point(1.0, 1.0, InterpolationType.LINEAR)
        
        # Exponential should curve upward
        mid_value = lane.get_value(0.5)
        assert 0.1 < mid_value < 0.5  # Less than linear interpolation
        
    def test_automation_array(self):
        """Test getting automation as array."""
        lane = AutomationLane("pan", default_value=0.0, min_value=-1.0, max_value=1.0)
        
        lane.add_point(0.0, -1.0)
        lane.add_point(1.0, 1.0)
        
        values = lane.get_values(44100, 44100)  # 1 second at 44.1kHz
        
        assert len(values) == 44100
        assert values[0] == -1.0
        assert values[-1] == 1.0
        assert -0.1 < values[22050] < 0.1  # Middle should be near 0


class TestMastering:
    """Test mastering chain."""
    
    def test_mastering_chain(self):
        """Test default mastering chain."""
        from music_gen.audio.mixing.mastering import MasteringChain
        
        mastering = MasteringChain(sample_rate=44100)
        
        # Test with full-range audio
        test_audio = torch.randn(2, 44100) * 0.8
        processed = mastering.process(test_audio)
        
        assert processed.shape == test_audio.shape
        assert processed.abs().max() <= 1.0  # Should be limited