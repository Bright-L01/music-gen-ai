"""
Basic tests for musicgen-minimal.
These tests verify the package actually works, not just that imports succeed.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from musicgen import MusicGenerator


class TestMusicGenerator:
    """Test the core MusicGenerator functionality."""

    @pytest.fixture(scope="class")
    def generator(self):
        """Create a generator instance for testing."""
        # Use the smallest model for testing
        return MusicGenerator("facebook/musicgen-small")

    def test_generator_creation(self, generator):
        """Test that generator can be created successfully."""
        assert generator is not None
        assert generator.model_name == "facebook/musicgen-small"
        assert generator.device in ["cpu", "cuda"]

    def test_model_info(self, generator):
        """Test that model info is available."""
        info = generator.get_model_info()

        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "sample_rate" in info
        assert info["sample_rate"] > 0

    def test_basic_generation(self, generator):
        """Test basic music generation."""
        audio, sample_rate = generator.generate(
            "test music",
            duration=1.0,  # Short duration for testing
            temperature=1.0,
            guidance_scale=3.0,
        )

        # Check audio output
        assert isinstance(audio, np.ndarray)
        assert len(audio.shape) == 1  # Should be 1D audio
        assert len(audio) > 0
        assert isinstance(sample_rate, int)
        assert sample_rate > 0

        # Check duration is approximately correct (within 10%)
        expected_samples = int(sample_rate * 1.0)
        assert abs(len(audio) - expected_samples) < expected_samples * 0.1

    def test_audio_saving(self, generator):
        """Test saving audio to file."""
        # Generate short audio
        audio, sample_rate = generator.generate("test", duration=0.5)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            generator.save_audio(audio, sample_rate, temp_path)

            # Check file was created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0

            # Check it's a valid WAV file (basic check)
            with open(temp_path, "rb") as f:
                header = f.read(12)
                assert header[:4] == b"RIFF"
                assert header[8:12] == b"WAVE"

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_different_parameters(self, generator):
        """Test generation with different parameters."""
        # Test different temperature
        audio1, sr1 = generator.generate("test", duration=0.5, temperature=0.5)
        audio2, sr2 = generator.generate("test", duration=0.5, temperature=1.5)

        assert len(audio1) > 0
        assert len(audio2) > 0
        assert sr1 == sr2  # Sample rate should be same

        # Test different guidance scale
        audio3, sr3 = generator.generate("test", duration=0.5, guidance_scale=1.0)
        audio4, sr4 = generator.generate("test", duration=0.5, guidance_scale=5.0)

        assert len(audio3) > 0
        assert len(audio4) > 0
        assert sr3 == sr4

    def test_invalid_parameters(self, generator):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative duration
        with pytest.raises((ValueError, Exception)):
            generator.generate("test", duration=-1.0)

        # Test zero duration
        with pytest.raises((ValueError, Exception)):
            generator.generate("test", duration=0.0)


class TestPackageStructure:
    """Test package structure and imports."""

    def test_imports(self):
        """Test that package imports work correctly."""
        # Test main import
        from musicgen import MusicGenerator

        assert MusicGenerator is not None

        # Test version
        import musicgen

        assert hasattr(musicgen, "__version__")
        assert isinstance(musicgen.__version__, str)

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from musicgen import cli

        assert cli is not None
        assert hasattr(cli, "main")


class TestRequiredDependencies:
    """Test that required dependencies are available."""

    def test_torch(self):
        """Test PyTorch is available."""
        import torch

        assert hasattr(torch, "__version__")

    def test_transformers(self):
        """Test transformers library is available."""
        import transformers

        assert hasattr(transformers, "__version__")

    def test_scipy(self):
        """Test scipy is available."""
        import scipy

        assert hasattr(scipy, "__version__")

    def test_numpy(self):
        """Test numpy is available."""
        import numpy as np

        assert hasattr(np, "__version__")

    def test_rich(self):
        """Test rich is available."""
        import rich

        assert hasattr(rich, "__version__")

    def test_typer(self):
        """Test typer is available."""
        import typer

        assert hasattr(typer, "__version__")


# Skip slow tests by default
@pytest.mark.slow
class TestSlowOperations:
    """Tests that take a long time to run."""

    def test_large_model_loading(self):
        """Test loading larger models (skipped by default)."""
        generator = MusicGenerator("facebook/musicgen-medium")
        assert generator is not None

    def test_long_generation(self):
        """Test generating longer audio clips (skipped by default)."""
        generator = MusicGenerator("facebook/musicgen-small")
        audio, sr = generator.generate("test music", duration=10.0)

        expected_samples = int(sr * 10.0)
        assert abs(len(audio) - expected_samples) < expected_samples * 0.1


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
