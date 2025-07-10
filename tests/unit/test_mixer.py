"""
Tests for audio/mixing/mixer.py
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from music_gen.audio.mixing.mixer import *


class TestMixer:
    """Test cases for mixer module."""

    def test_track_config_creation(self):
        """Test TrackConfig dataclass creation."""
        # Test default values
        track = TrackConfig(name="vocals")
        assert track.name == "vocals"
        assert track.volume == 0.7
        assert track.pan == 0.0
        assert not track.mute
        assert not track.solo
        assert track.reverb_send == 0.0
        assert track.delay_send == 0.0

        # Test custom values
        track_custom = TrackConfig(name="guitar", volume=0.8, pan=-0.5, mute=True, eq_low_gain=3.0)
        assert track_custom.name == "guitar"
        assert track_custom.volume == 0.8
        assert track_custom.pan == -0.5
        assert track_custom.mute
        assert track_custom.eq_low_gain == 3.0

    def test_mixing_config_creation(self):
        """Test MixingConfig dataclass creation."""
        # Test default values
        config = MixingConfig()
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.bit_depth == 24
        assert config.master_volume == 0.8
        assert config.master_limiter
        assert config.reverb_bus_enabled
        assert config.delay_bus_enabled

        # Test custom values
        config_custom = MixingConfig(
            sample_rate=48000, channels=1, master_volume=0.9, use_gpu=False
        )
        assert config_custom.sample_rate == 48000
        assert config_custom.channels == 1
        assert config_custom.master_volume == 0.9
        assert not config_custom.use_gpu

    @patch("music_gen.audio.mixing.mixer.EffectChain")
    def test_mixing_engine_creation(self, mock_effect_chain):
        """Test MixingEngine creation."""
        # Mock EffectChain to avoid complex initialization
        mock_chain = MagicMock()
        mock_effect_chain.return_value = mock_chain

        config = MixingConfig(use_gpu=False)  # Force CPU for testing
        engine = MixingEngine(config)

        assert engine.config == config
        assert engine.device.type == "cpu"
        assert hasattr(engine, "master_chain")

        # Test GPU detection
        with patch("torch.cuda.is_available", return_value=True):
            config_gpu = MixingConfig(use_gpu=True)
            engine_gpu = MixingEngine(config_gpu)
            # Should use cuda if available and requested
            assert engine_gpu.device.type in ["cuda", "cpu"]

    @patch("music_gen.audio.mixing.mixer.EffectChain")
    @patch("music_gen.audio.mixing.effects.Reverb")
    @patch("music_gen.audio.mixing.effects.Delay")
    def test_mixing_engine_buses_init(self, mock_delay, mock_reverb, mock_effect_chain):
        """Test mixing engine bus initialization."""
        # Mock dependencies
        mock_effect_chain.return_value = MagicMock()
        mock_reverb.return_value = MagicMock()
        mock_delay.return_value = MagicMock()

        # Test with buses enabled
        config = MixingConfig(reverb_bus_enabled=True, delay_bus_enabled=True, use_gpu=False)
        engine = MixingEngine(config)

        # Should initialize reverb and delay buses
        mock_reverb.assert_called_once()
        mock_delay.assert_called_once()

        # Test with buses disabled
        config_no_buses = MixingConfig(
            reverb_bus_enabled=False, delay_bus_enabled=False, use_gpu=False
        )
        mock_reverb.reset_mock()
        mock_delay.reset_mock()

        engine_no_buses = MixingEngine(config_no_buses)

        # Should not initialize disabled buses
        assert engine_no_buses.reverb_bus is None
        assert engine_no_buses.delay_bus is None

    def test_track_config_eq_settings(self):
        """Test EQ settings in TrackConfig."""
        track = TrackConfig(
            name="bass",
            eq_low_gain=6.0,
            eq_mid_gain=-2.0,
            eq_high_gain=1.5,
            eq_low_freq=80.0,
            eq_mid_freq=800.0,
            eq_high_freq=8000.0,
        )

        assert track.eq_low_gain == 6.0
        assert track.eq_mid_gain == -2.0
        assert track.eq_high_gain == 1.5
        assert track.eq_low_freq == 80.0
        assert track.eq_mid_freq == 800.0
        assert track.eq_high_freq == 8000.0

    def test_track_config_compressor_settings(self):
        """Test compressor settings in TrackConfig."""
        track = TrackConfig(
            name="drums",
            compressor_threshold=-12.0,
            compressor_ratio=6.0,
            compressor_attack=0.003,
            compressor_release=0.05,
        )

        assert track.compressor_threshold == -12.0
        assert track.compressor_ratio == 6.0
        assert track.compressor_attack == 0.003
        assert track.compressor_release == 0.05

    def test_mixing_config_validation(self):
        """Test mixing config value validation."""
        # Test valid configurations
        config = MixingConfig(
            sample_rate=48000, channels=2, bit_depth=24, master_volume=0.85, headroom=-3.0
        )

        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.bit_depth == 24
        assert config.master_volume == 0.85
        assert config.headroom == -3.0
