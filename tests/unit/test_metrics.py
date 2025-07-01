"""
Tests for music_gen.evaluation.metrics
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from music_gen.evaluation.metrics import *


class TestMetrics:
    """Test metrics module."""

    @patch('music_gen.evaluation.metrics.librosa.filters.mel')
    def test_audioqualitymetrics_creation(self, mock_mel_filters):
        """Test AudioQualityMetrics creation."""
        # Mock the mel filterbank
        mock_mel_filters.return_value = np.random.randn(128, 1025)
        
        # Test creation with minimal dependencies
        metrics = AudioQualityMetrics(
            sample_rate=24000,
            compute_fad=False,  # Disable to avoid torchvggish dependency
            compute_clap=False,  # Disable to avoid CLAP dependency
            compute_inception_score=False
        )
        
        assert metrics.sample_rate == 24000
        assert metrics.hop_length == 512
        assert metrics.n_fft == 2048
        assert metrics.n_mels == 128
        assert not metrics.compute_fad
        assert not metrics.compute_clap
        assert not metrics.compute_inception_score

    @patch('music_gen.evaluation.metrics.librosa.filters.mel')
    @patch('music_gen.evaluation.metrics.librosa.stft')
    def test_extract_mel_spectrogram(self, mock_stft, mock_mel_filters):
        """Test mel spectrogram extraction."""
        # Mock dependencies
        mock_mel_filters.return_value = np.random.randn(128, 1025)
        mock_stft.return_value = np.random.randn(1025, 100) + 1j * np.random.randn(1025, 100)
        
        metrics = AudioQualityMetrics(
            compute_fad=False,
            compute_clap=False,
            compute_inception_score=False
        )
        
        # Create test audio
        audio = np.random.randn(24000)  # 1 second at 24kHz
        
        # Extract mel spectrogram
        mel_spec = metrics.extract_mel_spectrogram(audio, log_scale=True)
        
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.shape[0] == 128  # n_mels
        mock_stft.assert_called_once()

    @patch('music_gen.evaluation.metrics.librosa.filters.mel')
    def test_metrics_with_disabled_features(self, mock_mel_filters):
        """Test metrics creation with all advanced features disabled."""
        mock_mel_filters.return_value = np.random.randn(128, 1025)
        
        # This should work without heavy dependencies
        metrics = AudioQualityMetrics(
            sample_rate=44100,
            n_mels=64,
            compute_fad=False,
            compute_clap=False,
            compute_inception_score=False
        )
        
        assert metrics.sample_rate == 44100
        assert metrics.n_mels == 64
        assert hasattr(metrics, 'mel_basis')

    @patch('music_gen.evaluation.metrics.librosa.filters.mel')
    def test_metrics_edge_cases(self, mock_mel_filters):
        """Test metrics with edge cases."""
        mock_mel_filters.return_value = np.random.randn(64, 513)
        
        # Test with different parameters
        metrics = AudioQualityMetrics(
            sample_rate=16000,
            hop_length=256,
            n_fft=1024,
            n_mels=64,
            compute_fad=False,
            compute_clap=False,
            compute_inception_score=False
        )
        
        assert metrics.sample_rate == 16000
        assert metrics.hop_length == 256
        assert metrics.n_fft == 1024
        assert metrics.n_mels == 64
