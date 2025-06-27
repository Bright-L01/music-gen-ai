"""Mastering chain for final audio processing."""

import torch
from typing import Optional

from music_gen.audio.mixing.effects import EffectChain, EQ, Compressor, Limiter


class MasteringChain:
    """Professional mastering chain."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.chain = EffectChain(sample_rate)
        self._setup_default_chain()
        
    def _setup_default_chain(self):
        """Setup default mastering chain."""
        # EQ for tonal balance
        eq = EQ(
            self.sample_rate,
            bands=[
                {"freq": 40, "gain": -2, "q": 0.7, "type": "high_pass"},
                {"freq": 100, "gain": 1, "q": 0.7, "type": "low_shelf"},
                {"freq": 3000, "gain": 2, "q": 0.5, "type": "bell"},
                {"freq": 10000, "gain": 1, "q": 0.7, "type": "high_shelf"},
            ]
        )
        self.chain.add_effect("eq", eq)
        
        # Multiband compression
        comp = Compressor(
            self.sample_rate,
            threshold=-15,
            ratio=3,
            attack=0.01,
            release=0.1,
            makeup_gain=3
        )
        self.chain.add_effect("compressor", comp)
        
        # Final limiter
        limiter = Limiter(
            self.sample_rate,
            threshold=-0.3,
            release=0.05,
            lookahead=0.005
        )
        self.chain.add_effect("limiter", limiter)
        
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio through mastering chain."""
        return self.chain.process(audio)