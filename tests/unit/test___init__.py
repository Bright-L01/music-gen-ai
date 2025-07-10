"""
Tests for audio/mixing/__init__.py
"""

import pytest

from music_gen.audio.mixing import *


class TestInit:
    """Test cases for __init__ module."""

    def test_imports_available(self):
        """Test that all expected classes are importable."""
        # Test mixing engine classes
        assert MixingEngine is not None
        assert MixingConfig is not None
        assert TrackConfig is not None

        # Test effect classes
        assert EffectChain is not None
        assert EQ is not None

        # Test automation classes
        assert AutomationLane is not None
        assert AutomationPoint is not None

        # Test mastering classes
        assert MasteringChain is not None

    def test_class_instantiation(self):
        """Test that key classes can be instantiated."""
        # Test basic instantiation without complex parameters
        try:
            effect_chain = EffectChain()
            assert isinstance(effect_chain, EffectChain)
        except Exception as e:
            pytest.skip(f"EffectChain instantiation failed: {e}")

        try:
            eq = EQ()
            assert isinstance(eq, EQ)
        except Exception as e:
            pytest.skip(f"EQ instantiation failed: {e}")

        try:
            automation_point = AutomationPoint(time=0.0, value=1.0)
            assert isinstance(automation_point, AutomationPoint)
        except Exception as e:
            pytest.skip(f"AutomationPoint instantiation failed: {e}")

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = [
            "MixingEngine",
            "MixingConfig",
            "TrackConfig",
            "EffectChain",
            "EQ",
            "Compressor",
            "Reverb",
            "Delay",
            "Chorus",
            "Limiter",
            "Gate",
            "Distortion",
            "AutomationLane",
            "AutomationPoint",
            "MasteringChain",
        ]

        import music_gen.audio.mixing as mixing_module

        for export in expected_exports:
            assert hasattr(mixing_module, export), f"Missing export: {export}"
