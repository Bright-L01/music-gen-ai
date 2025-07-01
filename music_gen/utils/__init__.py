"""
Utility functions for MusicGen.
"""

from .audio import (
    apply_fade,
    compute_audio_duration,
    concatenate_audio,
    convert_audio_format,
    load_audio_file,
    normalize_audio,
    save_audio_file,
    save_audio_sample,
    split_audio,
    trim_silence,
)

__all__ = [
    "load_audio_file",
    "save_audio_file",
    "normalize_audio",
    "trim_silence",
    "apply_fade",
    "convert_audio_format",
    "concatenate_audio",
    "save_audio_sample",
    "compute_audio_duration",
    "split_audio",
]
