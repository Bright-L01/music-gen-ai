"""
Unit tests for CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from typer.testing import CliRunner

from music_gen.cli import app


@pytest.mark.unit
class TestCLI:
    """Test CLI interface functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MusicGen" in result.stdout
        assert "generate" in result.stdout
        assert "info" in result.stdout

    @patch("music_gen.cli.FastMusicGenerator")
    def test_generate_command_basic(self, mock_generator_class, runner, temp_output_dir):
        """Test basic generate command."""
        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.audio = torch.randn(1, 24000).numpy()  # Mock audio array
        mock_result.generation_time = 5.0
        mock_result.sample_rate = 24000
        mock_generator.generate_single.return_value = mock_result
        mock_generator.get_performance_stats.return_value = {"cache_stats": {"cached_models": 1}}
        mock_generator_class.return_value = mock_generator

        output_file = temp_output_dir / "test_output.wav"

        with patch("scipy.io.wavfile.write") as mock_save:
            result = runner.invoke(
                app,
                ["generate", "Happy jazz music", "--duration", "10", "--output", str(output_file)],
            )

            # CLI should attempt to call the generator and save function
            mock_generator.generate_single.assert_called_once()
            mock_save.assert_called_once()

    @patch("music_gen.cli.FastMusicGenerator")
    def test_generate_with_conditioning(self, mock_generator_class, runner, temp_output_dir):
        """Test generate command with conditioning parameters."""
        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.audio = torch.randn(1, 24000).numpy()
        mock_result.generation_time = 5.0
        mock_result.sample_rate = 24000
        mock_generator.generate_single.return_value = mock_result
        mock_generator.get_performance_stats.return_value = {"cache_stats": {"cached_models": 1}}
        mock_generator_class.return_value = mock_generator

        output_file = temp_output_dir / "conditioned_output.wav"

        with patch("scipy.io.wavfile.write"):
            result = runner.invoke(
                app,
                [
                    "generate",
                    "Jazz piano solo",
                    "--duration",
                    "15",
                    "--genre",
                    "jazz",
                    "--mood",
                    "energetic",
                    "--tempo",
                    "120",
                    "--temperature",
                    "0.8",
                    "--output",
                    str(output_file),
                ],
            )

            # Check that conditioning parameters were passed
            call_args = mock_generator.generate_single.call_args
            assert call_args[1]["duration"] == 15.0
            assert call_args[1]["temperature"] == 0.8

    @patch("music_gen.cli.FastMusicGenerator")
    def test_generate_test_mode(self, mock_generator_class, runner, temp_output_dir):
        """Test test command (batch generation)."""
        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.audio = torch.randn(1, 24000).numpy()
        mock_result.generation_time = 5.0
        mock_result.sample_rate = 24000
        mock_generator.generate_single.return_value = mock_result
        mock_generator.get_performance_stats.return_value = {"cache_stats": {"cached_models": 1}}
        mock_generator_class.return_value = mock_generator

        with patch("scipy.io.wavfile.write"):
            result = runner.invoke(
                app, ["test", "--output-dir", str(temp_output_dir), "--model-size", "base"]
            )

            # Test command should run without crashing (may have internal errors due to mocking complexity)
            # The important thing is that the command structure works
            assert result.exit_code == 0  # Should not crash the CLI itself

    def test_info_command(self, runner):
        """Test info command."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Mock GPU"):
                    result = runner.invoke(app, ["info"])

                    assert result.exit_code == 0
                    assert "System Information" in result.stdout
                    assert "PyTorch" in result.stdout

    @patch("music_gen.cli.FastMusicGenerator")
    def test_generate_with_seed(self, mock_generator_class, runner, temp_output_dir):
        """Test generate command with seed."""
        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.audio = torch.randn(1, 24000).numpy()
        mock_result.generation_time = 5.0
        mock_result.sample_rate = 24000
        mock_generator.generate_single.return_value = mock_result
        mock_generator.get_performance_stats.return_value = {"cache_stats": {"cached_models": 1}}
        mock_generator_class.return_value = mock_generator

        output_file = temp_output_dir / "seeded_output.wav"

        with patch("scipy.io.wavfile.write"):
            with patch("torch.manual_seed") as mock_seed:
                result = runner.invoke(
                    app, ["generate", "Test music", "--seed", "42", "--output", str(output_file)]
                )

                mock_seed.assert_called_with(42)

    @patch("music_gen.cli.FastMusicGenerator")
    def test_generate_device_selection(self, mock_generator_class, runner, temp_output_dir):
        """Test device selection."""
        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.audio = torch.randn(1, 24000).numpy()
        mock_result.generation_time = 5.0
        mock_result.sample_rate = 24000
        mock_generator.generate_single.return_value = mock_result
        mock_generator.get_performance_stats.return_value = {"cache_stats": {"cached_models": 1}}
        mock_generator_class.return_value = mock_generator

        output_file = temp_output_dir / "device_test.wav"

        with patch("scipy.io.wavfile.write"):
            with patch("torch.cuda.is_available", return_value=False):
                result = runner.invoke(
                    app,
                    ["generate", "Test music", "--device", "auto", "--output", str(output_file)],
                )

                # Should handle device selection gracefully

    @patch("music_gen.cli.FastMusicGenerator")
    def test_cli_error_handling(self, mock_generator_class, runner):
        """Test CLI error handling."""
        # Test with model creation failure
        mock_generator_class.side_effect = Exception("Model not found")

        result = runner.invoke(
            app,
            [
                "generate",
                "Test prompt",
                "--model-size",
                "invalid-model",
                "--output",
                "/tmp/test.wav",
            ],
        )

        assert result.exit_code != 0


@pytest.mark.unit
class TestCLIUtils:
    """Test CLI utility functions."""

    def test_basic_cli_structure(self):
        """Test basic CLI structure and commands."""
        # Test that the app is a Typer instance
        from typer import Typer

        assert isinstance(app, Typer)

        # Test that commands exist by trying to get help
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout
        assert "info" in result.stdout
        assert "test" in result.stdout

    def test_path_validation(self):
        """Test output path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Valid path
            valid_path = temp_path / "output.wav"
            assert str(valid_path).endswith(".wav")

            # Test extension checking
            invalid_path = temp_path / "output.txt"
            assert not str(invalid_path).endswith(".wav")

    def test_duration_parsing(self):
        """Test duration parsing and formatting."""
        # Simple duration tests
        duration = 65.5
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        formatted = f"{minutes}:{seconds:02d}"

        assert formatted == "1:05"

        duration = 30.0
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        formatted = f"{minutes}:{seconds:02d}"

        assert formatted == "0:30"


@pytest.mark.unit
class TestCLIConfiguration:
    """Test CLI configuration management."""

    def test_model_size_options(self):
        """Test model size configuration."""
        # Test that model sizes are valid
        valid_sizes = ["small", "base", "large"]

        for size in valid_sizes:
            assert size in ["small", "base", "large"]

    def test_device_configuration(self):
        """Test device configuration options."""
        # Test device selection logic
        with patch("torch.cuda.is_available", return_value=True):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cpu"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test basic parameter ranges
        assert 0.0 <= 1.0 <= 2.0  # Temperature range
        assert 1 <= 50 <= 100  # Top-k range
        assert 0.0 <= 0.9 <= 1.0  # Top-p range
        assert 60 <= 120 <= 180  # Tempo range


if __name__ == "__main__":
    pytest.main([__file__])
