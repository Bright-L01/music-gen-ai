"""
Training infrastructure for MusicGen.
"""

from .lightning_module import MusicGenLightningModule, ProgressiveTrainingModule

__all__ = [
    "MusicGenLightningModule",
    "ProgressiveTrainingModule",
]
