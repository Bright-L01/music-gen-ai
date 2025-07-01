"""
Tests for music_gen.training.lightning_module
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from music_gen.training.lightning_module import *


class TestLightningModule:
    """Test lightning_module training components."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for training."""
        model = MagicMock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.train.return_value = model
        model.eval.return_value = model
        return model

    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader."""
        dataloader = MagicMock()
        dataloader.__iter__.return_value = iter(
            [{"input": torch.randn(4, 128), "target": torch.randn(4, 10)} for _ in range(2)]
        )
        dataloader.__len__.return_value = 2
        return dataloader

    # TODO: Add specific training tests
