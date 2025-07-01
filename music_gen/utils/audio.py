"""
Audio utility functions for MusicGen.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio_file(
    file_path: Union[str, Path],
    target_sample_rate: Optional[int] = None,
    normalize: bool = True,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return the waveform and sample rate.

    Args:
        file_path: Path to the audio file
        target_sample_rate: Target sample rate for resampling
        normalize: Whether to normalize the audio
        mono: Whether to convert to mono
        duration: Maximum duration in seconds
        offset: Start offset in seconds

    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        # Load audio file
        # Temporarily load to get sample rate, then reload with offset/duration
        temp_waveform, temp_sample_rate = torchaudio.load(str(file_path))

        waveform, sample_rate = torchaudio.load(
            str(file_path),
            frame_offset=int(offset * temp_sample_rate) if offset > 0 else 0,
            num_frames=int(duration * temp_sample_rate) if duration else -1,
        )

        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if target sample rate is specified
        if target_sample_rate is not None and sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        # Normalize if requested
        if normalize:
            waveform = normalize_audio(waveform)

        return waveform, sample_rate

    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise


def save_audio_file(
    waveform: torch.Tensor,
    file_path: Union[str, Path],
    sample_rate: int = 24000,
    normalize: bool = True,
    format: Optional[str] = None,
) -> None:
    """
    Save an audio waveform to a file.

    Args:
        waveform: Audio waveform tensor
        file_path: Output file path
        sample_rate: Sample rate of the audio
        normalize: Whether to normalize the audio
        format: Audio format (inferred from extension if None)
    """
    try:
        # Ensure waveform is on CPU
        if waveform.is_cuda:
            waveform = waveform.cpu()

        # Normalize if requested
        if normalize:
            waveform = normalize_audio(waveform)

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save audio file
        torchaudio.save(
            str(file_path),
            waveform,
            sample_rate,
            format=format,
        )

    except Exception as e:
        logger.error(f"Failed to save audio file {file_path}: {e}")
        raise


def normalize_audio(waveform: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """
    Normalize audio waveform.

    Args:
        waveform: Input waveform
        method: Normalization method ("peak", "rms", "lufs")

    Returns:
        Normalized waveform
    """
    if method == "peak":
        # Peak normalization
        max_val = waveform.abs().max()
        if max_val > 1e-8:
            waveform = waveform / max_val
    elif method == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 1e-8:
            waveform = waveform / rms
    elif method == "lufs":
        # LUFS normalization (simplified)
        # This is a basic implementation - for production use proper LUFS measurement
        rms = torch.sqrt(torch.mean(waveform**2))
        target_lufs = -23.0  # EBU R128 standard
        current_lufs = 20 * torch.log10(rms + 1e-8)
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        waveform = waveform * gain_linear

    return waveform


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration: float = 0.1,
) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio.

    Args:
        waveform: Input waveform
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        min_duration: Minimum duration to keep in seconds

    Returns:
        Trimmed waveform
    """
    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20)

    # Find non-silent regions
    energy = waveform.abs()
    non_silent = energy > threshold_linear

    if waveform.dim() > 1:
        # For multi-channel audio, consider any channel active
        non_silent = non_silent.any(dim=0)

    # Find first and last non-silent samples
    non_silent_indices = torch.where(non_silent)[0]

    if len(non_silent_indices) == 0:
        # All silence - return minimum duration
        min_samples = int(min_duration * sample_rate)
        return waveform[..., :min_samples]

    start_idx = non_silent_indices[0].item()
    end_idx = non_silent_indices[-1].item() + 1

    # Ensure minimum duration
    min_samples = int(min_duration * sample_rate)
    if end_idx - start_idx < min_samples:
        center = (start_idx + end_idx) // 2
        start_idx = max(0, center - min_samples // 2)
        end_idx = min(waveform.shape[-1], start_idx + min_samples)

    return waveform[..., start_idx:end_idx]


def apply_fade(
    waveform: torch.Tensor,
    sample_rate: int,
    fade_in_duration: float = 0.01,
    fade_out_duration: float = 0.01,
) -> torch.Tensor:
    """
    Apply fade in/out to audio waveform.

    Args:
        waveform: Input waveform
        sample_rate: Sample rate
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds

    Returns:
        Waveform with fades applied
    """
    seq_len = waveform.shape[-1]
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)

    # Create fade curves
    if fade_in_samples > 0 and fade_in_samples < seq_len:
        fade_in = torch.linspace(0, 1, fade_in_samples)
        waveform[..., :fade_in_samples] *= fade_in

    if fade_out_samples > 0 and fade_out_samples < seq_len:
        fade_out = torch.linspace(1, 0, fade_out_samples)
        waveform[..., -fade_out_samples:] *= fade_out

    return waveform


def convert_audio_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_sample_rate: Optional[int] = None,
    target_format: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """
    Convert audio file to different format/sample rate.

    Args:
        input_path: Input file path
        output_path: Output file path
        target_sample_rate: Target sample rate
        target_format: Target format
        normalize: Whether to normalize
    """
    # Load audio
    waveform, sample_rate = load_audio_file(
        input_path,
        target_sample_rate=target_sample_rate,
        normalize=normalize,
    )

    # Save in new format
    save_audio_file(
        waveform,
        output_path,
        sample_rate=target_sample_rate or sample_rate,
        format=target_format,
    )


def concatenate_audio(
    audio_list: List[torch.Tensor],
    crossfade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """
    Concatenate multiple audio segments with crossfading.

    Args:
        audio_list: List of audio tensors
        crossfade_duration: Crossfade duration in seconds
        sample_rate: Sample rate

    Returns:
        Concatenated audio
    """
    if not audio_list:
        return torch.empty(0)

    if len(audio_list) == 1:
        return audio_list[0]

    crossfade_samples = int(crossfade_duration * sample_rate)

    # Start with first segment
    result = audio_list[0].clone()

    for i in range(1, len(audio_list)):
        current_segment = audio_list[i]

        if crossfade_samples > 0 and result.shape[-1] >= crossfade_samples:
            # Apply crossfade
            fade_out = torch.linspace(1, 0, crossfade_samples)
            fade_in = torch.linspace(0, 1, crossfade_samples)

            # Fade out end of previous segment
            result[..., -crossfade_samples:] *= fade_out

            # Fade in beginning of current segment and add
            if current_segment.shape[-1] >= crossfade_samples:
                current_segment = current_segment.clone()
                current_segment[..., :crossfade_samples] *= fade_in

                # Overlap and add
                result[..., -crossfade_samples:] += current_segment[..., :crossfade_samples]

                # Append remaining part
                if current_segment.shape[-1] > crossfade_samples:
                    result = torch.cat([result, current_segment[..., crossfade_samples:]], dim=-1)
            else:
                # Current segment too short for crossfade
                result = torch.cat([result, current_segment], dim=-1)
        else:
            # No crossfade
            result = torch.cat([result, current_segment], dim=-1)

    return result


def save_audio_sample(
    audio: torch.Tensor,
    sample_rate: int,
    output_dir: Union[str, Path],
    prefix: str = "sample",
    format: str = "wav",
) -> str:
    """
    Save audio sample with automatic filename generation.

    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        output_dir: Output directory
        prefix: Filename prefix
        format: Audio format

    Returns:
        Path to saved file
    """
    import time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.{format}"
    output_path = output_dir / filename

    save_audio_file(audio, output_path, sample_rate=sample_rate)

    return str(output_path)


def compute_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Compute duration of audio file without loading the entire file.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        info = torchaudio.info(str(file_path))
        duration = info.num_frames / info.sample_rate
        return float(duration)
    except Exception as e:
        logger.error(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0


def split_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_duration: float,
    overlap: float = 0.0,
) -> List[torch.Tensor]:
    """
    Split audio into segments.

    Args:
        waveform: Input waveform
        sample_rate: Sample rate
        segment_duration: Segment duration in seconds
        overlap: Overlap between segments in seconds

    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step_samples = segment_samples - overlap_samples

    segments = []
    start = 0

    while start + segment_samples <= waveform.shape[-1]:
        segment = waveform[..., start : start + segment_samples]
        segments.append(segment)
        start += step_samples

    # Add final segment if there's remaining audio
    if start < waveform.shape[-1]:
        final_segment = waveform[..., start:]
        # Pad to segment length if needed
        if final_segment.shape[-1] < segment_samples:
            padding = segment_samples - final_segment.shape[-1]
            final_segment = torch.nn.functional.pad(final_segment, (0, padding))
        segments.append(final_segment)

    return segments
