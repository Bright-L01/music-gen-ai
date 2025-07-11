"""
Advanced audio augmentation pipeline for music generation training.
"""

import logging
import random
from typing import List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioAugmentation:
    """Base class for audio augmentations."""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() < self.probability:
            return self.apply(waveform, sample_rate)
        return waveform

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise NotImplementedError


class TimeStretch(AudioAugmentation):
    """Time stretching (tempo change) augmentation."""

    def __init__(
        self,
        probability: float = 0.3,
        rate_range: tuple = (0.8, 1.2),
        fixed_rate: Optional[float] = None,
    ):
        super().__init__(probability)
        self.rate_range = rate_range
        self.fixed_rate = fixed_rate

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            rate = self.fixed_rate if self.fixed_rate else random.uniform(*self.rate_range)

            # Use torchaudio's time stretch
            stretch = torchaudio.transforms.TimeStretch(
                hop_length=512,
                n_freq=201,  # For 16kHz audio with 512 hop
            )

            # Convert to spectrogram, stretch, then back
            spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=400,
                hop_length=512,
            )

            # Apply to each channel
            stretched_channels = []
            for i in range(waveform.shape[0]):
                spec = spec_transform(waveform[i])
                stretched_spec = stretch(spec, rate)

                # Convert back to waveform (this is simplified)
                # In practice, you'd use a more sophisticated reconstruction
                stretched_waveform = torch.istft(
                    torch.view_as_complex(stretched_spec.unsqueeze(-1).repeat(1, 1, 1, 2)),
                    n_fft=400,
                    hop_length=512,
                )
                stretched_channels.append(stretched_waveform)

            result = torch.stack(stretched_channels, dim=0)

            # Pad or trim to original length
            if result.shape[-1] != waveform.shape[-1]:
                if result.shape[-1] > waveform.shape[-1]:
                    result = result[..., : waveform.shape[-1]]
                else:
                    padding = waveform.shape[-1] - result.shape[-1]
                    result = torch.nn.functional.pad(result, (0, padding))

            return result

        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return waveform


class PitchShift(AudioAugmentation):
    """Pitch shifting augmentation."""

    def __init__(
        self,
        probability: float = 0.3,
        semitone_range: tuple = (-2, 2),
        fixed_semitones: Optional[float] = None,
    ):
        super().__init__(probability)
        self.semitone_range = semitone_range
        self.fixed_semitones = fixed_semitones

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            n_steps = (
                self.fixed_semitones
                if self.fixed_semitones
                else random.uniform(*self.semitone_range)
            )

            pitch_shift = torchaudio.transforms.PitchShift(
                sample_rate=sample_rate,
                n_steps=n_steps,
            )

            return pitch_shift(waveform)

        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return waveform


class AddNoise(AudioAugmentation):
    """Add various types of noise."""

    def __init__(
        self,
        probability: float = 0.2,
        noise_level_range: tuple = (0.001, 0.01),
        noise_type: str = "gaussian",  # gaussian, pink, brown
    ):
        super().__init__(probability)
        self.noise_level_range = noise_level_range
        self.noise_type = noise_type

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        noise_level = random.uniform(*self.noise_level_range)

        if self.noise_type == "gaussian":
            noise = torch.randn_like(waveform) * noise_level
        elif self.noise_type == "pink":
            noise = self._generate_pink_noise(waveform.shape, waveform.device) * noise_level
        elif self.noise_type == "brown":
            noise = self._generate_brown_noise(waveform.shape, waveform.device) * noise_level
        else:
            noise = torch.randn_like(waveform) * noise_level

        return waveform + noise

    def _generate_pink_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate pink (1/f) noise."""
        # Simplified pink noise generation
        white_noise = torch.randn(shape, device=device)

        # Apply frequency domain filtering for pink noise
        # This is a simplified approach
        fft = torch.fft.fft(white_noise)
        freqs = torch.fft.fftfreq(shape[-1], device=device)

        # Pink noise has 1/f characteristic
        filter_response = 1.0 / torch.sqrt(torch.abs(freqs) + 1e-8)
        filtered_fft = fft * filter_response.unsqueeze(0)

        pink_noise = torch.fft.ifft(filtered_fft).real
        return pink_noise

    def _generate_brown_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate brown (1/f²) noise."""
        # Simplified brown noise generation
        white_noise = torch.randn(shape, device=device)

        # Apply frequency domain filtering for brown noise
        fft = torch.fft.fft(white_noise)
        freqs = torch.fft.fftfreq(shape[-1], device=device)

        # Brown noise has 1/f² characteristic
        filter_response = 1.0 / (torch.abs(freqs) + 1e-8)
        filtered_fft = fft * filter_response.unsqueeze(0)

        brown_noise = torch.fft.ifft(filtered_fft).real
        return brown_noise


class VolumeAugmentation(AudioAugmentation):
    """Volume/gain augmentation."""

    def __init__(
        self,
        probability: float = 0.4,
        gain_range_db: tuple = (-6, 6),
    ):
        super().__init__(probability)
        self.gain_range_db = gain_range_db

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        gain_db = random.uniform(*self.gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear


class FrequencyMasking(AudioAugmentation):
    """Frequency masking in spectrogram domain."""

    def __init__(
        self,
        probability: float = 0.3,
        freq_mask_param: int = 80,
        num_masks: int = 1,
    ):
        super().__init__(probability)
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            # Convert to spectrogram
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)
            inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

            # Process each channel
            masked_channels = []
            for i in range(waveform.shape[0]):
                spec = spec_transform(waveform[i])

                # Apply frequency masking
                for _ in range(self.num_masks):
                    freq_mask_size = random.randint(0, self.freq_mask_param)
                    freq_start = random.randint(0, max(1, spec.shape[0] - freq_mask_size))
                    spec[freq_start : freq_start + freq_mask_size, :] = 0

                # Convert back to waveform
                masked_waveform = inverse_transform(spec, length=waveform.shape[-1])
                masked_channels.append(masked_waveform)

            return torch.stack(masked_channels, dim=0)

        except Exception as e:
            logger.warning(f"Frequency masking failed: {e}")
            return waveform


class TimeMasking(AudioAugmentation):
    """Time masking (silence random segments)."""

    def __init__(
        self,
        probability: float = 0.2,
        time_mask_param: int = 100,
        num_masks: int = 1,
    ):
        super().__init__(probability)
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        result = waveform.clone()

        for _ in range(self.num_masks):
            mask_size = random.randint(0, self.time_mask_param)
            mask_start = random.randint(0, max(1, waveform.shape[-1] - mask_size))
            result[..., mask_start : mask_start + mask_size] = 0

        return result


class Reverb(AudioAugmentation):
    """Add reverb effect."""

    def __init__(
        self,
        probability: float = 0.25,
        room_size_range: tuple = (0.1, 0.9),
        damping_range: tuple = (0.1, 0.9),
    ):
        super().__init__(probability)
        self.room_size_range = room_size_range
        self.damping_range = damping_range

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            # Simple reverb using convolution with impulse response
            room_size = random.uniform(*self.room_size_range)
            damping = random.uniform(*self.damping_range)

            # Generate simple impulse response
            ir_length = int(sample_rate * room_size * 0.5)  # Up to 0.5 seconds
            ir = torch.randn(ir_length) * torch.exp(-torch.arange(ir_length) * damping / ir_length)
            ir = ir / ir.abs().max()  # Normalize

            # Apply convolution
            result = torch.nn.functional.conv1d(
                waveform.unsqueeze(1), ir.unsqueeze(0).unsqueeze(0), padding=ir_length // 2
            ).squeeze(1)

            # Mix with original
            mix_ratio = 0.3
            result = (1 - mix_ratio) * waveform + mix_ratio * result

            return result

        except Exception as e:
            logger.warning(f"Reverb failed: {e}")
            return waveform


class Distortion(AudioAugmentation):
    """Add harmonic distortion."""

    def __init__(
        self,
        probability: float = 0.15,
        drive_range: tuple = (1.0, 3.0),
    ):
        super().__init__(probability)
        self.drive_range = drive_range

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        drive = random.uniform(*self.drive_range)

        # Apply soft clipping distortion
        distorted = torch.tanh(waveform * drive) / drive

        # Mix with original
        mix_ratio = 0.3
        return (1 - mix_ratio) * waveform + mix_ratio * distorted


class PolymixAugmentation(AudioAugmentation):
    """Mix multiple audio samples (Polymix technique)."""

    def __init__(
        self,
        probability: float = 0.2,
        mix_ratio_range: tuple = (0.1, 0.5),
        num_mix: int = 2,
    ):
        super().__init__(probability)
        self.mix_ratio_range = mix_ratio_range
        self.num_mix = num_mix
        self.mix_samples = []  # Will be populated externally

    def set_mix_samples(self, samples: List[torch.Tensor]):
        """Set samples to mix with."""
        self.mix_samples = samples

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.mix_samples:
            return waveform

        result = waveform.clone()

        for _ in range(min(self.num_mix, len(self.mix_samples))):
            # Choose random sample to mix
            mix_sample = random.choice(self.mix_samples)

            # Ensure same shape
            if mix_sample.shape != waveform.shape:
                # Resize to match
                if mix_sample.shape[-1] > waveform.shape[-1]:
                    start_idx = random.randint(0, mix_sample.shape[-1] - waveform.shape[-1])
                    mix_sample = mix_sample[..., start_idx : start_idx + waveform.shape[-1]]
                else:
                    # Pad with repetition or silence
                    padding = waveform.shape[-1] - mix_sample.shape[-1]
                    mix_sample = torch.nn.functional.pad(mix_sample, (0, padding))

                # Match channels
                if mix_sample.shape[0] != waveform.shape[0]:
                    if mix_sample.shape[0] == 1:
                        mix_sample = mix_sample.repeat(waveform.shape[0], 1)
                    else:
                        mix_sample = mix_sample.mean(dim=0, keepdim=True).repeat(
                            waveform.shape[0], 1
                        )

            # Mix with random ratio
            mix_ratio = random.uniform(*self.mix_ratio_range)
            result = (1 - mix_ratio) * result + mix_ratio * mix_sample

        return result


class AugmentationPipeline:
    """Pipeline of audio augmentations."""

    def __init__(
        self,
        augmentations: List[AudioAugmentation],
        max_augmentations: Optional[int] = None,
        always_apply: Optional[List[str]] = None,
    ):
        self.augmentations = augmentations
        self.max_augmentations = max_augmentations
        self.always_apply = always_apply or []

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply augmentation pipeline."""

        # Determine which augmentations to apply
        if self.max_augmentations is not None:
            # Apply random subset
            num_to_apply = random.randint(1, min(self.max_augmentations, len(self.augmentations)))
            augs_to_apply = random.sample(self.augmentations, num_to_apply)
        else:
            # Apply all (each with its own probability)
            augs_to_apply = self.augmentations

        # Apply augmentations
        result = waveform
        for aug in augs_to_apply:
            result = aug(result, sample_rate)

        return result

    def add_augmentation(self, aug: AudioAugmentation):
        """Add augmentation to pipeline."""
        self.augmentations.append(aug)

    def remove_augmentation(self, aug_type: type):
        """Remove augmentation by type."""
        self.augmentations = [aug for aug in self.augmentations if not isinstance(aug, aug_type)]


def create_training_augmentation_pipeline(
    strong: bool = False,
    include_mix: bool = True,
) -> AugmentationPipeline:
    """Create pre-configured augmentation pipeline for training."""

    if strong:
        # Strong augmentation for robust training
        augmentations = [
            VolumeAugmentation(probability=0.5, gain_range_db=(-8, 8)),
            AddNoise(probability=0.3, noise_level_range=(0.001, 0.02)),
            PitchShift(probability=0.4, semitone_range=(-3, 3)),
            TimeStretch(probability=0.4, rate_range=(0.7, 1.3)),
            FrequencyMasking(probability=0.4, freq_mask_param=120),
            TimeMasking(probability=0.3, time_mask_param=200),
            Reverb(probability=0.3),
            Distortion(probability=0.2, drive_range=(1.0, 4.0)),
        ]
    else:
        # Moderate augmentation
        augmentations = [
            VolumeAugmentation(probability=0.4, gain_range_db=(-4, 4)),
            AddNoise(probability=0.2, noise_level_range=(0.001, 0.01)),
            PitchShift(probability=0.3, semitone_range=(-2, 2)),
            TimeStretch(probability=0.3, rate_range=(0.85, 1.15)),
            FrequencyMasking(probability=0.3, freq_mask_param=80),
            TimeMasking(probability=0.2, time_mask_param=100),
            Reverb(probability=0.2),
        ]

    # Add polymix if requested
    if include_mix:
        augmentations.append(PolymixAugmentation(probability=0.2))

    return AugmentationPipeline(
        augmentations=augmentations,
        max_augmentations=3 if strong else 2,
    )


def create_inference_augmentation_pipeline() -> AugmentationPipeline:
    """Create light augmentation pipeline for inference diversity."""

    augmentations = [
        VolumeAugmentation(probability=0.3, gain_range_db=(-2, 2)),
        AddNoise(probability=0.1, noise_level_range=(0.001, 0.005)),
    ]

    return AugmentationPipeline(augmentations=augmentations, max_augmentations=1)


class AdaptiveAugmentation:
    """Adaptive augmentation that adjusts based on training progress."""

    def __init__(
        self,
        initial_strength: float = 0.5,
        max_strength: float = 1.0,
        adaptation_rate: float = 0.001,
    ):
        self.initial_strength = initial_strength
        self.max_strength = max_strength
        self.current_strength = initial_strength
        self.adaptation_rate = adaptation_rate
        self.step_count = 0

    def update_strength(self, loss: float, target_loss: float = 2.0):
        """Update augmentation strength based on training loss."""
        self.step_count += 1

        # Increase strength if loss is too low (overfitting)
        # Decrease strength if loss is too high (underfitting)
        if loss < target_loss:
            self.current_strength = min(
                self.max_strength, self.current_strength + self.adaptation_rate
            )
        else:
            self.current_strength = max(0.1, self.current_strength - self.adaptation_rate)

    def get_pipeline(self) -> AugmentationPipeline:
        """Get augmentation pipeline with current strength."""

        # Scale probabilities by current strength
        augmentations = [
            VolumeAugmentation(probability=0.4 * self.current_strength),
            AddNoise(probability=0.2 * self.current_strength),
            PitchShift(probability=0.3 * self.current_strength),
            TimeStretch(probability=0.3 * self.current_strength),
            FrequencyMasking(probability=0.3 * self.current_strength),
            TimeMasking(probability=0.2 * self.current_strength),
        ]

        max_augs = max(1, int(3 * self.current_strength))

        return AugmentationPipeline(
            augmentations=augmentations,
            max_augmentations=max_augs,
        )
