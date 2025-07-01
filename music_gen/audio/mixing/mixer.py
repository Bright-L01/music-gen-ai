"""Professional multi-track mixing engine."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from music_gen.audio.mixing.automation import AutomationLane
from music_gen.audio.mixing.effects import EffectChain


@dataclass
class TrackConfig:
    """Configuration for a single track in the mix."""

    name: str
    volume: float = 0.7  # 0.0 to 1.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    mute: bool = False
    solo: bool = False

    # Sends
    reverb_send: float = 0.0  # 0.0 to 1.0
    delay_send: float = 0.0  # 0.0 to 1.0

    # EQ settings
    eq_low_gain: float = 0.0  # -20 to +20 dB
    eq_mid_gain: float = 0.0  # -20 to +20 dB
    eq_high_gain: float = 0.0  # -20 to +20 dB
    eq_low_freq: float = 100.0  # Hz
    eq_mid_freq: float = 1000.0  # Hz
    eq_high_freq: float = 10000.0  # Hz

    # Dynamics
    compressor_threshold: float = -10.0  # dB
    compressor_ratio: float = 4.0  # :1
    compressor_attack: float = 0.005  # seconds
    compressor_release: float = 0.1  # seconds

    # Effects chain
    effects: Optional[EffectChain] = None

    # Automation
    automation: Dict[str, AutomationLane] = field(default_factory=dict)


@dataclass
class MixingConfig:
    """Configuration for the mixing engine."""

    sample_rate: int = 44100
    channels: int = 2  # 1=mono, 2=stereo
    bit_depth: int = 24

    # Master bus settings
    master_volume: float = 0.8
    master_limiter: bool = True
    master_limiter_threshold: float = -0.3  # dB

    # Bus sends
    reverb_bus_enabled: bool = True
    delay_bus_enabled: bool = True

    # Mix settings
    auto_gain_staging: bool = True
    headroom: float = -6.0  # dB

    # Processing
    use_gpu: bool = True
    buffer_size: int = 512


class MixingEngine:
    """Professional multi-track mixing engine with effects processing."""

    def __init__(self, config: MixingConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Initialize buses
        self.reverb_bus = None
        self.delay_bus = None
        self._init_buses()

        # Master effects
        self.master_chain = EffectChain(sample_rate=config.sample_rate)
        if config.master_limiter:
            self._init_master_limiter()

    def _init_buses(self):
        """Initialize send buses."""
        from music_gen.audio.mixing.effects import Delay, Reverb

        if self.config.reverb_bus_enabled:
            self.reverb_bus = Reverb(
                sample_rate=self.config.sample_rate,
                room_size=0.8,
                damping=0.7,
                wet_mix=1.0,  # 100% wet for send bus
            )

        if self.config.delay_bus_enabled:
            self.delay_bus = Delay(
                sample_rate=self.config.sample_rate,
                delay_time=0.25,
                feedback=0.4,
                wet_mix=1.0,  # 100% wet for send bus
            )

    def _init_master_limiter(self):
        """Initialize master bus limiter."""
        from music_gen.audio.mixing.effects import Limiter

        limiter = Limiter(
            sample_rate=self.config.sample_rate,
            threshold=self.config.master_limiter_threshold,
            release=0.05,
        )
        self.master_chain.add_effect("limiter", limiter)

    def mix(
        self,
        tracks: Dict[str, torch.Tensor],
        track_configs: Dict[str, TrackConfig],
        duration: Optional[float] = None,
    ) -> torch.Tensor:
        """Mix multiple tracks into stereo output.

        Args:
            tracks: Dictionary of track_name -> audio tensor [channels, samples]
            track_configs: Dictionary of track_name -> TrackConfig
            duration: Optional duration in seconds (crops/pads all tracks)

        Returns:
            Mixed stereo audio [2, samples]
        """
        if not tracks:
            raise ValueError("No tracks to mix")

        # Determine output length
        if duration is not None:
            output_samples = int(duration * self.config.sample_rate)
        else:
            output_samples = max(t.shape[-1] for t in tracks.values())

        # Initialize mix buses
        mix_bus = torch.zeros(2, output_samples, device=self.device)
        reverb_send = torch.zeros(2, output_samples, device=self.device)
        delay_send = torch.zeros(2, output_samples, device=self.device)

        # Check for soloed tracks
        soloed_tracks = [name for name, cfg in track_configs.items() if cfg.solo]

        # Process each track
        for track_name, audio in tracks.items():
            config = track_configs.get(track_name, TrackConfig(name=track_name))

            # Skip muted tracks (unless soloed)
            if config.mute and track_name not in soloed_tracks:
                continue

            # Skip non-soloed tracks if any track is soloed
            if soloed_tracks and track_name not in soloed_tracks:
                continue

            # Process track
            processed = self._process_track(audio, config, output_samples)

            # Apply to buses
            mix_bus += processed

            # Send to effect buses
            if config.reverb_send > 0 and self.reverb_bus:
                reverb_send += processed * config.reverb_send

            if config.delay_send > 0 and self.delay_bus:
                delay_send += processed * config.delay_send

        # Process effect buses and add to mix
        if self.reverb_bus and reverb_send.abs().max() > 0:
            reverb_return = self.reverb_bus.process(reverb_send)
            mix_bus += reverb_return * 0.3  # Return level

        if self.delay_bus and delay_send.abs().max() > 0:
            delay_return = self.delay_bus.process(delay_send)
            mix_bus += delay_return * 0.2  # Return level

        # Apply master processing
        if self.config.auto_gain_staging:
            mix_bus = self._auto_gain_stage(mix_bus)

        # Apply master volume
        mix_bus *= self.config.master_volume

        # Apply master effects chain
        mix_bus = self.master_chain.process(mix_bus)

        return mix_bus

    def _process_track(
        self, audio: torch.Tensor, config: TrackConfig, output_samples: int
    ) -> torch.Tensor:
        """Process a single track with effects and panning.

        Args:
            audio: Input audio [channels, samples]
            config: Track configuration
            output_samples: Target number of samples

        Returns:
            Processed stereo audio [2, samples]
        """
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Convert to stereo if needed
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]  # Take first 2 channels

        # Pad or crop to target length
        if audio.shape[-1] < output_samples:
            audio = F.pad(audio, (0, output_samples - audio.shape[-1]))
        else:
            audio = audio[..., :output_samples]

        # Move to device
        audio = audio.to(self.device)

        # Apply volume automation if present
        if "volume" in config.automation:
            volume_curve = config.automation["volume"].get_values(
                output_samples, self.config.sample_rate
            )
            audio *= torch.from_numpy(volume_curve).to(self.device).unsqueeze(0)
        else:
            audio *= config.volume

        # Apply EQ
        audio = self._apply_eq(audio, config)

        # Apply compression
        audio = self._apply_compression(audio, config)

        # Apply custom effects chain if present
        if config.effects:
            audio = config.effects.process(audio)

        # Apply panning
        audio = self._apply_panning(audio, config)

        return audio

    def _apply_eq(self, audio: torch.Tensor, config: TrackConfig) -> torch.Tensor:
        """Apply parametric EQ to audio."""
        from music_gen.audio.mixing.effects import EQ

        if config.eq_low_gain == 0 and config.eq_mid_gain == 0 and config.eq_high_gain == 0:
            return audio

        eq = EQ(
            sample_rate=self.config.sample_rate,
            bands=[
                {
                    "freq": config.eq_low_freq,
                    "gain": config.eq_low_gain,
                    "q": 0.7,
                    "type": "low_shelf",
                },
                {"freq": config.eq_mid_freq, "gain": config.eq_mid_gain, "q": 1.0, "type": "bell"},
                {
                    "freq": config.eq_high_freq,
                    "gain": config.eq_high_gain,
                    "q": 0.7,
                    "type": "high_shelf",
                },
            ],
        )

        return eq.process(audio)

    def _apply_compression(self, audio: torch.Tensor, config: TrackConfig) -> torch.Tensor:
        """Apply compression to audio."""
        from music_gen.audio.mixing.effects import Compressor

        if config.compressor_ratio <= 1.0:
            return audio

        compressor = Compressor(
            sample_rate=self.config.sample_rate,
            threshold=config.compressor_threshold,
            ratio=config.compressor_ratio,
            attack=config.compressor_attack,
            release=config.compressor_release,
        )

        return compressor.process(audio)

    def _apply_panning(self, audio: torch.Tensor, config: TrackConfig) -> torch.Tensor:
        """Apply stereo panning using constant power law."""
        if config.pan == 0:
            return audio

        # Constant power panning
        pan_rad = config.pan * np.pi / 4  # -45° to +45°
        left_gain = np.cos(pan_rad)
        right_gain = np.sin(pan_rad)

        # Apply pan automation if present
        if "pan" in config.automation:
            pan_curve = config.automation["pan"].get_values(
                audio.shape[-1], self.config.sample_rate
            )
            pan_curve = torch.from_numpy(pan_curve).to(self.device)

            # Calculate time-varying gains
            pan_rad = pan_curve * np.pi / 4
            left_gain = torch.cos(pan_rad)
            right_gain = torch.sin(pan_rad)

            audio[0] *= left_gain
            audio[1] *= right_gain
        else:
            # Static panning
            audio[0] *= left_gain
            audio[1] *= right_gain

        return audio

    def _auto_gain_stage(self, audio: torch.Tensor) -> torch.Tensor:
        """Automatically adjust gain to target headroom."""
        # Find peak level
        peak = audio.abs().max()

        if peak > 0:
            # Calculate target peak based on headroom
            target_peak = 10 ** (self.config.headroom / 20)

            # Apply gain to reach target
            gain = target_peak / peak
            audio *= gain

        return audio

    def bounce_stems(
        self,
        tracks: Dict[str, torch.Tensor],
        track_configs: Dict[str, TrackConfig],
        stem_groups: Dict[str, List[str]],
        duration: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Bounce tracks to stem groups.

        Args:
            tracks: Individual tracks
            track_configs: Track configurations
            stem_groups: Dictionary of stem_name -> list of track names
            duration: Optional duration

        Returns:
            Dictionary of stem_name -> stereo audio
        """
        stems = {}

        for stem_name, track_names in stem_groups.items():
            # Get tracks for this stem
            stem_tracks = {name: tracks[name] for name in track_names if name in tracks}
            stem_configs = {
                name: track_configs.get(name, TrackConfig(name=name)) for name in track_names
            }

            # Mix stem
            if stem_tracks:
                stem_audio = self.mix(stem_tracks, stem_configs, duration)
                stems[stem_name] = stem_audio

        return stems

    def get_metering(
        self, audio: torch.Tensor, window_size: int = 1024
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Get metering information for audio.

        Args:
            audio: Input audio [channels, samples]
            window_size: Window size for analysis

        Returns:
            Dictionary with metering data
        """
        # RMS level
        rms = torch.sqrt((audio**2).mean())
        rms_db = 20 * torch.log10(rms + 1e-8)

        # Peak level
        peak = audio.abs().max()
        peak_db = 20 * torch.log10(peak + 1e-8)

        # LUFS (simplified)
        # Real LUFS calculation would use ITU-R BS.1770 algorithm
        lufs = rms_db - 10  # Rough approximation

        # Spectral centroid
        fft = torch.fft.rfft(audio, dim=-1)
        freqs = torch.fft.rfftfreq(audio.shape[-1], 1 / self.config.sample_rate)
        magnitude = fft.abs()
        spectral_centroid = (freqs * magnitude).sum() / magnitude.sum()

        return {
            "rms": float(rms),
            "rms_db": float(rms_db),
            "peak": float(peak),
            "peak_db": float(peak_db),
            "lufs": float(lufs),
            "spectral_centroid": float(spectral_centroid),
            "crest_factor": float(peak / (rms + 1e-8)),
        }
