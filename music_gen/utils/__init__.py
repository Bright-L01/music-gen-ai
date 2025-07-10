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
from .optional_imports import (
    OptionalDependencyError,
    check_system_requirements,
    get_available_backends,
    is_torch_available,
    is_torchaudio_available,
    is_transformers_available,
    optional_import,
    requires_optional,
    suggest_installations,
)

__all__ = [
    # Audio utilities
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
    # Optional imports
    "optional_import",
    "requires_optional",
    "get_available_backends",
    "check_system_requirements",
    "suggest_installations",
    "is_torch_available",
    "is_torchaudio_available",
    "is_transformers_available",
    "OptionalDependencyError",
]
