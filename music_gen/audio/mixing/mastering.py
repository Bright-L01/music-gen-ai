"""Mastering chain for final audio processing."""

import torch

from music_gen.audio.mixing.effects import EQ, Compressor, EffectChain, Limiter


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
            ],
        )
        self.chain.add_effect("eq", eq)

        # Multiband compression
        comp = Compressor(
            self.sample_rate, threshold=-15, ratio=3, attack=0.01, release=0.1, makeup_gain=3
        )
        self.chain.add_effect("compressor", comp)

        # Final limiter
        limiter = Limiter(self.sample_rate, threshold=-0.3, release=0.05, lookahead=0.005)
        self.chain.add_effect("limiter", limiter)

    def process(
        self,
        audio: torch.Tensor,
        eq_settings: dict = None,
        compression_settings: dict = None,
        limiter_settings: dict = None,
    ) -> torch.Tensor:
        """Process audio through mastering chain with optional settings."""
        # Apply custom settings if provided
        if eq_settings or compression_settings or limiter_settings:
            # Create temporary chain with custom settings
            temp_chain = EffectChain(self.sample_rate)

            # Add EQ with custom settings
            if eq_settings:
                eq = EQ(self.sample_rate)
                # Apply EQ settings (simplified implementation)
                temp_chain.add_effect("eq", eq)

            # Add compressor with custom settings
            if compression_settings:
                comp = Compressor(
                    self.sample_rate,
                    threshold=compression_settings.get("threshold", -15),
                    ratio=compression_settings.get("ratio", 3),
                    attack=compression_settings.get("attack", 0.01),
                    release=compression_settings.get("release", 0.1),
                    makeup_gain=compression_settings.get("makeup_gain", 3),
                )
                temp_chain.add_effect("compressor", comp)

            # Add limiter with custom settings
            if limiter_settings:
                limiter = Limiter(
                    self.sample_rate,
                    threshold=limiter_settings.get("threshold", -0.3),
                    release=limiter_settings.get("release", 0.05),
                    lookahead=limiter_settings.get("lookahead", 0.005),
                )
                temp_chain.add_effect("limiter", limiter)

            return temp_chain.process(audio)

        return self.chain.process(audio)
