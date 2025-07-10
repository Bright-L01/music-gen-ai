"""
Tests for audio/mixing/effects.py
"""

import pytest
import torch

from music_gen.audio.mixing.effects import *


class TestEffects:
    """Test cases for effects module."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        # Create 1 second of stereo audio at 44.1kHz
        sample_rate = 44100
        duration = 1.0
        samples = int(duration * sample_rate)

        # Generate stereo sine wave
        t = torch.linspace(0, duration, samples)
        freq = 440.0  # A4
        audio = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # Add channel dim
        audio = audio.repeat(2, 1)  # Make stereo
        return audio

    def test_effect_chain_creation(self):
        """Test EffectChain creation."""
        chain = EffectChain(sample_rate=44100)
        assert chain.sample_rate == 44100
        assert len(chain.effects) == 0

    def test_effect_chain_add_remove(self):
        """Test adding and removing effects from chain."""
        chain = EffectChain()
        eq = EQ(sample_rate=44100)

        # Add effect
        chain.add_effect("eq", eq)
        assert len(chain.effects) == 1
        assert chain.effects[0][0] == "eq"

        # Remove effect
        chain.remove_effect("eq")
        assert len(chain.effects) == 0

    def test_effect_chain_processing(self, sample_audio):
        """Test audio processing through effect chain."""
        chain = EffectChain()
        eq = EQ(sample_rate=44100)
        chain.add_effect("eq", eq)

        output = chain.process(sample_audio)
        assert output.shape == sample_audio.shape
        assert isinstance(output, torch.Tensor)

    def test_eq_creation(self):
        """Test EQ effect creation."""
        eq = EQ(sample_rate=44100)
        assert eq.sample_rate == 44100
        assert len(eq.bands) == 0

    def test_eq_with_bands(self):
        """Test EQ with frequency bands."""
        bands = [
            {"freq": 1000, "gain": 3.0, "q": 1.0, "type": "bell"},
            {"freq": 100, "gain": -2.0, "q": 0.7, "type": "high_pass"},
        ]
        eq = EQ(sample_rate=44100, bands=bands)
        assert len(eq.bands) == 2
        assert len(eq.filters) == 2

    def test_eq_processing(self, sample_audio):
        """Test EQ audio processing."""
        bands = [{"freq": 1000, "gain": 3.0, "q": 1.0, "type": "bell"}]
        eq = EQ(sample_rate=44100, bands=bands)

        output = eq.process(sample_audio)
        assert output.shape == sample_audio.shape
        assert isinstance(output, torch.Tensor)
