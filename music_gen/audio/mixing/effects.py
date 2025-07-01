"""Professional audio effects for mixing engine."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class BaseEffect(ABC):
    """Base class for audio effects."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    @abstractmethod
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio through effect.

        Args:
            audio: Input audio [channels, samples]

        Returns:
            Processed audio [channels, samples]
        """

    def reset(self):
        """Reset effect state."""


class EffectChain:
    """Chain multiple effects together."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.effects: List[Tuple[str, BaseEffect]] = []

    def add_effect(self, name: str, effect: BaseEffect):
        """Add effect to chain."""
        self.effects.append((name, effect))

    def remove_effect(self, name: str):
        """Remove effect from chain."""
        self.effects = [(n, e) for n, e in self.effects if n != name]

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio through effect chain."""
        output = audio
        for name, effect in self.effects:
            output = effect.process(output)
        return output

    def reset(self):
        """Reset all effects in chain."""
        for _, effect in self.effects:
            effect.reset()


class EQ(BaseEffect):
    """Parametric equalizer with multiple bands."""

    def __init__(self, sample_rate: int = 44100, bands: Optional[List[Dict[str, Any]]] = None):
        super().__init__(sample_rate)
        self.bands = bands or []
        self._init_filters()

    def _init_filters(self):
        """Initialize filter coefficients for each band."""
        self.filters = []

        for band in self.bands:
            freq = band.get("freq", 1000)
            gain = band.get("gain", 0)
            q = band.get("q", 1.0)
            filter_type = band.get("type", "bell")

            # Calculate filter coefficients
            omega = 2 * np.pi * freq / self.sample_rate
            sin_omega = np.sin(omega)
            cos_omega = np.cos(omega)
            alpha = sin_omega / (2 * q)
            A = 10 ** (gain / 40)

            if filter_type == "bell":
                b0 = 1 + alpha * A
                b1 = -2 * cos_omega
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cos_omega
                a2 = 1 - alpha / A
            elif filter_type == "low_shelf":
                S = 1  # Shelf slope
                beta = np.sqrt(A) / q

                b0 = A * ((A + 1) - (A - 1) * cos_omega + beta * sin_omega)
                b1 = 2 * A * ((A - 1) - (A + 1) * cos_omega)
                b2 = A * ((A + 1) - (A - 1) * cos_omega - beta * sin_omega)
                a0 = (A + 1) + (A - 1) * cos_omega + beta * sin_omega
                a1 = -2 * ((A - 1) + (A + 1) * cos_omega)
                a2 = (A + 1) + (A - 1) * cos_omega - beta * sin_omega
            elif filter_type == "high_shelf":
                S = 1  # Shelf slope
                beta = np.sqrt(A) / q

                b0 = A * ((A + 1) + (A - 1) * cos_omega + beta * sin_omega)
                b1 = -2 * A * ((A - 1) + (A + 1) * cos_omega)
                b2 = A * ((A + 1) + (A - 1) * cos_omega - beta * sin_omega)
                a0 = (A + 1) - (A - 1) * cos_omega + beta * sin_omega
                a1 = 2 * ((A - 1) - (A + 1) * cos_omega)
                a2 = (A + 1) - (A - 1) * cos_omega - beta * sin_omega
            elif filter_type == "high_pass":
                # High pass filter
                b0 = 1
                b1 = -2 * cos_omega
                b2 = 1
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha
            elif filter_type == "low_pass":
                # Low pass filter  
                b0 = (1 - cos_omega) / 2
                b1 = 1 - cos_omega
                b2 = (1 - cos_omega) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha
            else:
                # Default to bell filter for unknown types
                b0 = 1 + alpha * A
                b1 = -2 * cos_omega
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cos_omega
                a2 = 1 - alpha / A

            # Normalize coefficients
            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0

            self.filters.append(
                {
                    "b": torch.tensor([b0, b1, b2], dtype=torch.float32),
                    "a": torch.tensor([1.0, a1, a2], dtype=torch.float32),
                    "state": None,
                }
            )

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply EQ to audio."""
        output = audio.clone()

        for filter_data in self.filters:
            output = self._apply_biquad(output, filter_data)

        return output

    def _apply_biquad(self, audio: torch.Tensor, filter_data: Dict) -> torch.Tensor:
        """Apply biquad filter to audio."""
        b = filter_data["b"].to(audio.device)
        a = filter_data["a"].to(audio.device)

        # Initialize state if needed
        if filter_data["state"] is None:
            filter_data["state"] = torch.zeros(audio.shape[0], 2, device=audio.device)

        state = filter_data["state"]
        output = torch.zeros_like(audio)

        # Process sample by sample (simplified for clarity)
        # In production, use more efficient batch processing
        for i in range(audio.shape[-1]):
            x = audio[:, i]

            # Direct form II transposed
            y = b[0] * x + state[:, 0]
            state[:, 0] = b[1] * x - a[1] * y + state[:, 1]
            state[:, 1] = b[2] * x - a[2] * y

            output[:, i] = y

        filter_data["state"] = state
        return output


class Compressor(BaseEffect):
    """Dynamic range compressor."""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold: float = -10.0,  # dB
        ratio: float = 4.0,
        attack: float = 0.005,  # seconds
        release: float = 0.1,  # seconds
        knee: float = 2.0,  # dB
        makeup_gain: float = 0.0,  # dB
        lookahead: float = 0.005,  # seconds
    ):
        super().__init__(sample_rate)
        self.threshold = threshold
        self.ratio = ratio
        self.knee = knee
        self.makeup_gain = makeup_gain

        # Time constants
        self.attack_samples = int(attack * sample_rate)
        self.release_samples = int(release * sample_rate)
        self.lookahead_samples = int(lookahead * sample_rate)

        # State
        self.envelope = 0.0
        self.delay_buffer = None
        self.delay_index = 0

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply compression to audio."""
        device = audio.device
        channels, num_samples = audio.shape

        # Initialize delay buffer for lookahead
        if self.delay_buffer is None and self.lookahead_samples > 0:
            self.delay_buffer = torch.zeros(channels, self.lookahead_samples, device=device)

        # Convert to dB
        epsilon = 1e-8
        audio_db = 20 * torch.log10(audio.abs() + epsilon)

        # Calculate gain reduction
        gain_reduction = torch.zeros_like(audio_db)

        for i in range(num_samples):
            # Get current sample level (max across channels)
            level_db = audio_db[:, i].max().item()

            # Soft knee compression
            if level_db > self.threshold - self.knee / 2:
                if level_db < self.threshold + self.knee / 2:
                    # Soft knee region
                    knee_factor = (level_db - self.threshold + self.knee / 2) / self.knee
                    gain_reduction[:, i] = (
                        knee_factor**2 * (self.threshold - level_db) * (1 - 1 / self.ratio) / 2
                    )
                else:
                    # Above knee
                    gain_reduction[:, i] = (self.threshold - level_db) * (1 - 1 / self.ratio)

        # Apply attack/release envelope
        smoothed_gain = torch.zeros_like(gain_reduction)

        for i in range(num_samples):
            target_gain = gain_reduction[:, i].mean().item()

            if target_gain < self.envelope:
                # Attack
                self.envelope += (target_gain - self.envelope) / self.attack_samples
            else:
                # Release
                self.envelope += (target_gain - self.envelope) / self.release_samples

            smoothed_gain[:, i] = self.envelope

        # Convert gain reduction to linear
        gain_linear = 10 ** ((smoothed_gain + self.makeup_gain) / 20)

        # Apply gain with lookahead delay
        if self.lookahead_samples > 0:
            output = torch.zeros_like(audio)

            for i in range(num_samples):
                # Get delayed sample
                delayed_sample = self.delay_buffer[:, self.delay_index]

                # Store current sample in delay buffer
                self.delay_buffer[:, self.delay_index] = audio[:, i]
                self.delay_index = (self.delay_index + 1) % self.lookahead_samples

                # Apply gain to delayed sample
                output[:, i] = delayed_sample * gain_linear[:, i]
        else:
            output = audio * gain_linear

        return output

    def reset(self):
        """Reset compressor state."""
        self.envelope = 0.0
        self.delay_buffer = None
        self.delay_index = 0


class Reverb(BaseEffect):
    """Reverb effect using Freeverb algorithm."""

    def __init__(
        self,
        sample_rate: int = 44100,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_mix: float = 0.3,
        pre_delay: float = 0.0,  # seconds
        width: float = 1.0,
    ):
        super().__init__(sample_rate)
        self.room_size = room_size
        self.damping = damping
        self.wet_mix = wet_mix
        self.pre_delay = pre_delay
        self.width = width

        # Freeverb constants
        self.num_combs = 8
        self.num_allpass = 4

        # Initialize delays
        self._init_delays()

    def _init_delays(self):
        """Initialize comb and allpass filters."""
        # Comb filter delays (in samples)
        comb_delays = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        self.comb_delays = [int(d * self.sample_rate / 44100) for d in comb_delays]

        # Allpass filter delays
        allpass_delays = [556, 441, 341, 225]
        self.allpass_delays = [int(d * self.sample_rate / 44100) for d in allpass_delays]

        # Pre-delay
        self.pre_delay_samples = int(self.pre_delay * self.sample_rate)

        # Initialize buffers
        self.comb_buffers = None
        self.allpass_buffers = None
        self.pre_delay_buffer = None

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply reverb to audio."""
        device = audio.device
        channels, num_samples = audio.shape

        # Initialize buffers on first use
        if self.comb_buffers is None:
            self._init_buffers(channels, device)

        # Convert stereo to mono for reverb processing
        if channels == 2:
            mono = (audio[0] + audio[1]) / 2
        else:
            mono = audio[0]

        # Apply pre-delay
        if self.pre_delay_samples > 0:
            delayed = self._apply_pre_delay(mono)
        else:
            delayed = mono

        # Process through comb filters
        comb_sum = torch.zeros_like(delayed)

        for i in range(self.num_combs):
            comb_out = self._process_comb(
                delayed, self.comb_buffers[i], self.comb_delays[i], self.room_size, self.damping
            )
            comb_sum += comb_out

        # Scale comb output
        comb_sum /= self.num_combs

        # Process through allpass filters
        allpass_out = comb_sum

        for i in range(self.num_allpass):
            allpass_out = self._process_allpass(
                allpass_out, self.allpass_buffers[i], self.allpass_delays[i]
            )

        # Create stereo output
        if channels == 2:
            # Apply width control
            wet_left = allpass_out
            wet_right = allpass_out * self.width
            wet = torch.stack([wet_left, wet_right])
        else:
            wet = allpass_out.unsqueeze(0)

        # Mix wet and dry signals
        output = audio * (1 - self.wet_mix) + wet * self.wet_mix

        return output

    def _init_buffers(self, channels: int, device: torch.device):
        """Initialize delay buffers."""
        # Comb filters
        self.comb_buffers = []
        for delay in self.comb_delays:
            buffer = {"data": torch.zeros(delay, device=device), "index": 0, "filter_state": 0.0}
            self.comb_buffers.append(buffer)

        # Allpass filters
        self.allpass_buffers = []
        for delay in self.allpass_delays:
            buffer = {"data": torch.zeros(delay, device=device), "index": 0}
            self.allpass_buffers.append(buffer)

        # Pre-delay buffer
        if self.pre_delay_samples > 0:
            self.pre_delay_buffer = {
                "data": torch.zeros(self.pre_delay_samples, device=device),
                "index": 0,
            }

    def _apply_pre_delay(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pre-delay to audio."""
        output = torch.zeros_like(audio)

        for i in range(audio.shape[0]):
            # Read from delay buffer
            output[i] = self.pre_delay_buffer["data"][self.pre_delay_buffer["index"]]

            # Write to delay buffer
            self.pre_delay_buffer["data"][self.pre_delay_buffer["index"]] = audio[i]

            # Update index
            self.pre_delay_buffer["index"] = (
                self.pre_delay_buffer["index"] + 1
            ) % self.pre_delay_samples

        return output

    def _process_comb(
        self, input_signal: torch.Tensor, buffer: Dict, delay: int, feedback: float, damping: float
    ) -> torch.Tensor:
        """Process signal through comb filter."""
        output = torch.zeros_like(input_signal)

        for i in range(input_signal.shape[0]):
            # Read from delay line
            delayed = buffer["data"][buffer["index"]]

            # Apply damping (simple lowpass filter)
            buffer["filter_state"] = delayed * (1 - damping) + buffer["filter_state"] * damping

            # Calculate output
            output[i] = delayed

            # Write to delay line with feedback
            buffer["data"][buffer["index"]] = input_signal[i] + buffer["filter_state"] * feedback

            # Update index
            buffer["index"] = (buffer["index"] + 1) % delay

        return output

    def _process_allpass(
        self, input_signal: torch.Tensor, buffer: Dict, delay: int, gain: float = 0.5
    ) -> torch.Tensor:
        """Process signal through allpass filter."""
        output = torch.zeros_like(input_signal)

        for i in range(input_signal.shape[0]):
            # Read from delay line
            delayed = buffer["data"][buffer["index"]]

            # Calculate output
            output[i] = -input_signal[i] + delayed

            # Write to delay line
            buffer["data"][buffer["index"]] = input_signal[i] + delayed * gain

            # Update index
            buffer["index"] = (buffer["index"] + 1) % delay

        return output


class Delay(BaseEffect):
    """Delay effect with feedback."""

    def __init__(
        self,
        sample_rate: int = 44100,
        delay_time: float = 0.5,  # seconds
        feedback: float = 0.5,
        wet_mix: float = 0.5,
        modulation_rate: float = 0.0,  # Hz
        modulation_depth: float = 0.0,  # seconds
    ):
        super().__init__(sample_rate)
        self.delay_time = delay_time
        self.feedback = feedback
        self.wet_mix = wet_mix
        self.modulation_rate = modulation_rate
        self.modulation_depth = modulation_depth

        # Calculate delay in samples
        self.delay_samples = int(delay_time * sample_rate)
        self.mod_depth_samples = int(modulation_depth * sample_rate)

        # Initialize buffers
        self.delay_buffer = None
        self.write_index = 0
        self.mod_phase = 0.0

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply delay effect."""
        device = audio.device
        channels, num_samples = audio.shape

        # Initialize buffer on first use
        if self.delay_buffer is None:
            buffer_size = self.delay_samples + self.mod_depth_samples + 1
            self.delay_buffer = torch.zeros(channels, buffer_size, device=device)

        output = torch.zeros_like(audio)

        for i in range(num_samples):
            # Calculate modulated delay time
            if self.modulation_rate > 0:
                mod = np.sin(2 * np.pi * self.mod_phase) * self.mod_depth_samples
                delay = int(self.delay_samples + mod)
                self.mod_phase += self.modulation_rate / self.sample_rate
                if self.mod_phase > 1.0:
                    self.mod_phase -= 1.0
            else:
                delay = self.delay_samples

            # Calculate read position
            read_index = (self.write_index - delay) % self.delay_buffer.shape[1]

            # Read delayed signal
            delayed = self.delay_buffer[:, read_index]

            # Calculate output
            output[:, i] = audio[:, i] * (1 - self.wet_mix) + delayed * self.wet_mix

            # Write to delay buffer with feedback
            self.delay_buffer[:, self.write_index] = audio[:, i] + delayed * self.feedback

            # Update write position
            self.write_index = (self.write_index + 1) % self.delay_buffer.shape[1]

        return output


class Chorus(BaseEffect):
    """Chorus effect using multiple modulated delays."""

    def __init__(
        self,
        sample_rate: int = 44100,
        num_voices: int = 4,
        depth: float = 0.02,  # seconds
        rate: float = 1.5,  # Hz
        mix: float = 0.5,
        spread: float = 0.5,
    ):
        super().__init__(sample_rate)
        self.num_voices = num_voices
        self.depth = depth
        self.rate = rate
        self.mix = mix
        self.spread = spread

        # Create delay lines for each voice
        self.delays = []
        for i in range(num_voices):
            # Vary parameters for each voice
            voice_rate = rate * (1 + i * 0.1 * spread)
            voice_depth = depth * (1 + i * 0.05 * spread)
            base_delay = 0.02 + i * 0.005  # Base delay time

            delay = Delay(
                sample_rate=sample_rate,
                delay_time=base_delay,
                feedback=0.0,
                wet_mix=1.0,
                modulation_rate=voice_rate,
                modulation_depth=voice_depth,
            )
            self.delays.append(delay)

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply chorus effect."""
        # Sum all voices
        wet = torch.zeros_like(audio)

        for delay in self.delays:
            wet += delay.process(audio)

        # Average voices
        wet /= self.num_voices

        # Mix with dry signal
        return audio * (1 - self.mix) + wet * self.mix


class Limiter(BaseEffect):
    """Brickwall limiter for mastering."""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold: float = -0.3,  # dB
        release: float = 0.05,  # seconds
        lookahead: float = 0.005,  # seconds
    ):
        super().__init__(sample_rate)
        self.threshold = threshold
        self.threshold_linear = 10 ** (threshold / 20)

        self.release_samples = int(release * sample_rate)
        self.lookahead_samples = int(lookahead * sample_rate)

        # State
        self.envelope = 1.0
        self.delay_buffer = None
        self.delay_index = 0

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply limiting."""
        device = audio.device
        channels, num_samples = audio.shape

        # Initialize delay buffer
        if self.delay_buffer is None and self.lookahead_samples > 0:
            self.delay_buffer = torch.zeros(channels, self.lookahead_samples, device=device)

        output = torch.zeros_like(audio)

        for i in range(num_samples):
            # Find peak across channels
            peak = audio[:, i].abs().max().item()

            # Calculate gain reduction
            if peak > self.threshold_linear:
                target_gain = self.threshold_linear / peak
            else:
                target_gain = 1.0

            # Smooth envelope
            if target_gain < self.envelope:
                self.envelope = target_gain  # Instant attack
            else:
                self.envelope += (1.0 - self.envelope) / self.release_samples  # Release

            # Apply gain with lookahead
            if self.lookahead_samples > 0:
                # Get delayed sample
                delayed_sample = self.delay_buffer[:, self.delay_index]

                # Store current sample
                self.delay_buffer[:, self.delay_index] = audio[:, i]
                self.delay_index = (self.delay_index + 1) % self.lookahead_samples

                output[:, i] = delayed_sample * self.envelope
            else:
                output[:, i] = audio[:, i] * self.envelope

        return output


class Gate(BaseEffect):
    """Noise gate."""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold: float = -40.0,  # dB
        attack: float = 0.001,  # seconds
        hold: float = 0.01,  # seconds
        release: float = 0.1,  # seconds
        range_db: float = -60.0,  # dB
    ):
        super().__init__(sample_rate)
        self.threshold = threshold
        self.threshold_linear = 10 ** (threshold / 20)
        self.range_linear = 10 ** (range_db / 20)

        self.attack_samples = int(attack * sample_rate)
        self.hold_samples = int(hold * sample_rate)
        self.release_samples = int(release * sample_rate)

        # State
        self.envelope = 0.0
        self.hold_counter = 0
        self.gate_state = "closed"  # "closed", "opening", "open", "holding", "closing"

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply noise gate."""
        output = torch.zeros_like(audio)

        for i in range(audio.shape[1]):
            # Get input level
            level = audio[:, i].abs().max().item()

            # State machine
            if self.gate_state == "closed":
                if level > self.threshold_linear:
                    self.gate_state = "opening"

            elif self.gate_state == "opening":
                self.envelope += (1.0 - self.envelope) / self.attack_samples
                if self.envelope >= 0.99:
                    self.gate_state = "open"
                    self.hold_counter = self.hold_samples

            elif self.gate_state == "open":
                if level < self.threshold_linear:
                    self.gate_state = "holding"

            elif self.gate_state == "holding":
                self.hold_counter -= 1
                if self.hold_counter <= 0:
                    self.gate_state = "closing"
                elif level > self.threshold_linear:
                    self.gate_state = "open"
                    self.hold_counter = self.hold_samples

            elif self.gate_state == "closing":
                self.envelope -= self.envelope / self.release_samples
                if self.envelope <= self.range_linear:
                    self.envelope = self.range_linear
                    self.gate_state = "closed"
                if level > self.threshold_linear:
                    self.gate_state = "opening"

            # Apply gain
            gain = self.range_linear + (1.0 - self.range_linear) * self.envelope
            output[:, i] = audio[:, i] * gain

        return output


class Distortion(BaseEffect):
    """Distortion/saturation effect."""

    def __init__(
        self,
        sample_rate: int = 44100,
        drive: float = 5.0,  # 1.0 to 20.0
        tone: float = 0.5,  # 0.0 to 1.0
        output_gain: float = 0.5,
        mode: str = "soft",  # "soft", "hard", "tube"
    ):
        super().__init__(sample_rate)
        self.drive = drive
        self.tone = tone
        self.output_gain = output_gain
        self.mode = mode

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply distortion."""
        # Pre-gain
        signal = audio * self.drive

        # Apply distortion
        if self.mode == "soft":
            # Soft clipping (tanh)
            distorted = torch.tanh(signal)

        elif self.mode == "hard":
            # Hard clipping
            distorted = torch.clamp(signal, -1.0, 1.0)

        elif self.mode == "tube":
            # Tube-like saturation
            distorted = torch.sign(signal) * (1 - torch.exp(-torch.abs(signal)))

        # Tone control (simple high-frequency emphasis/reduction)
        if self.tone != 0.5:
            # Simple first-order highpass/lowpass
            cutoff = 1000 * (2 ** (2 * (self.tone - 0.5)))  # 250Hz to 4kHz
            omega = 2 * np.pi * cutoff / self.sample_rate

            if self.tone > 0.5:
                # Emphasize highs
                alpha = 1 / (1 + omega)
                filtered = torch.zeros_like(distorted)

                for i in range(1, distorted.shape[1]):
                    filtered[:, i] = (
                        alpha * (distorted[:, i] - distorted[:, i - 1]) + alpha * filtered[:, i - 1]
                    )

                distorted = distorted + filtered * (self.tone - 0.5) * 2
            else:
                # Reduce highs
                alpha = omega / (1 + omega)
                filtered = torch.zeros_like(distorted)

                for i in range(1, distorted.shape[1]):
                    filtered[:, i] = alpha * distorted[:, i] + (1 - alpha) * filtered[:, i - 1]

                distorted = filtered + (distorted - filtered) * (self.tone * 2)

        # Output gain
        return distorted * self.output_gain
