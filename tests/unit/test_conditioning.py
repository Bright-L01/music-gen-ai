"""
Tests for music_gen.models.multi_instrument.conditioning
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from music_gen.models.multi_instrument.conditioning import *


class TestConditioningModel:
    """Test conditioning model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        from music_gen.models.multi_instrument.config import MultiInstrumentConfig

        return MultiInstrumentConfig()

    def test_instrumentembedding_creation(self, device, mock_config):
        """Test InstrumentEmbedding model creation."""
        model = InstrumentEmbedding(mock_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "instrument_names")
        assert hasattr(model, "instrument_embeddings")

    def test_instrumentembedding_forward(self, device, mock_config):
        """Test InstrumentEmbedding forward pass."""
        model = InstrumentEmbedding(mock_config).to(device)

        # Test with instrument names
        instrument_names = ["piano", "drums"]
        embeddings = model(instrument_names=instrument_names)
        assert embeddings.shape[0] == len(instrument_names)
        assert embeddings.shape[1] == mock_config.instrument_embedding_dim

    def test_instrumentconditioner_creation(self, device, mock_config):
        """Test InstrumentConditioner model creation."""
        model = InstrumentConditioner(mock_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "instrument_embedding")
        assert hasattr(model, "projection")

    def test_instrumentconditioner_forward(self, device, mock_config):
        """Test InstrumentConditioner forward pass."""
        model = InstrumentConditioner(mock_config).to(device)

        # Create test input
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, mock_config.hidden_size).to(device)
        instrument_names = [["piano", "drums"], ["guitar", "bass"]]

        # Test forward pass
        output, mixing_params = model(hidden_states, instrument_names=instrument_names)
        assert output.shape == hidden_states.shape
        assert isinstance(mixing_params, dict)

    def test_instrumentclassifier_creation(self, device, mock_config):
        """Test InstrumentClassifier model creation."""
        model = InstrumentClassifier(mock_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "classifier")

    def test_instrumentclassifier_forward(self, device, mock_config):
        """Test InstrumentClassifier forward pass."""
        model = InstrumentClassifier(mock_config).to(device)

        # Create test audio input (need large input for Conv1d layers)
        batch_size = 2
        num_samples = 100000  # Larger input to handle the conv layers
        audio = torch.randn(batch_size, num_samples).to(device)

        # Test forward pass
        logits = model(audio)
        expected_classes = len(mock_config.get_instrument_names()) + 1
        assert logits.shape == (batch_size, expected_classes)
