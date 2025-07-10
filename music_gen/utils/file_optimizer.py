"""
File handling optimization for Music Gen AI.

This module provides optimized file handling for audio processing,
including streaming, chunked processing, and memory-efficient operations.
"""

import asyncio
import hashlib
import io
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Optional, Union

import aiofiles
import numpy as np
from pydub import AudioSegment

from ..core.exceptions import AudioProcessingError


class FileStreamHandler:
    """Handles streaming file operations for large audio files."""

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.temp_dir = Path(tempfile.gettempdir()) / "musicgen_temp"
        self.temp_dir.mkdir(exist_ok=True)

    async def stream_file_upload(
        self, file_stream: AsyncIterator[bytes], max_size: Optional[int] = None
    ) -> Path:
        """Stream file upload with size validation."""

        # Create temporary file
        temp_file = self.temp_dir / f"upload_{os.urandom(16).hex()}.tmp"

        total_size = 0
        hasher = hashlib.sha256()

        try:
            async with aiofiles.open(temp_file, "wb") as f:
                async for chunk in file_stream:
                    if max_size and total_size + len(chunk) > max_size:
                        raise AudioProcessingError(f"File exceeds maximum size of {max_size} bytes")

                    await f.write(chunk)
                    hasher.update(chunk)
                    total_size += len(chunk)

            # Rename to content-based name
            file_hash = hasher.hexdigest()[:16]
            final_path = self.temp_dir / f"audio_{file_hash}.tmp"

            # Use atomic rename
            temp_file.rename(final_path)

            return final_path

        except Exception as e:
            # Clean up on error
            if temp_file.exists():
                temp_file.unlink()
            raise AudioProcessingError(f"File upload failed: {e}")

    async def stream_file_download(
        self, file_path: Path, chunk_size: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """Stream file download in chunks."""

        chunk_size = chunk_size or self.chunk_size

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(file_path, "rb") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @asynccontextmanager
    async def temporary_file(self, suffix: str = ".tmp"):
        """Context manager for temporary files with automatic cleanup."""

        temp_file = self.temp_dir / f"temp_{os.urandom(16).hex()}{suffix}"

        try:
            yield temp_file
        finally:
            # Clean up
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""

        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0
        for file in self.temp_dir.glob("*.tmp"):
            try:
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_seconds:
                    file.unlink()
                    cleaned_count += 1
            except Exception:
                pass

        return cleaned_count


class AudioStreamProcessor:
    """Processes audio files in a streaming manner to minimize memory usage."""

    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.chunk_duration = 10  # Process 10 seconds at a time

    async def process_audio_stream(
        self, input_file: Path, output_file: Path, process_func: callable, format: str = "wav"
    ):
        """Process audio file in chunks to minimize memory usage."""

        # Load audio in chunks
        audio = AudioSegment.from_file(str(input_file))
        chunk_ms = self.chunk_duration * 1000

        # Create output stream
        output_segments = []

        for i in range(0, len(audio), chunk_ms):
            # Extract chunk
            chunk = audio[i : i + chunk_ms]

            # Convert to numpy array
            samples = np.array(chunk.get_array_of_samples())

            # Process chunk
            processed_samples = await asyncio.to_thread(process_func, samples)

            # Convert back to AudioSegment
            processed_chunk = AudioSegment(
                processed_samples.tobytes(),
                frame_rate=chunk.frame_rate,
                sample_width=chunk.sample_width,
                channels=chunk.channels,
            )

            output_segments.append(processed_chunk)

            # Allow other tasks to run
            await asyncio.sleep(0)

        # Combine all chunks
        output_audio = sum(output_segments)

        # Export with optimized settings
        export_kwargs = {
            "format": format,
            "bitrate": "192k" if format == "mp3" else None,
            "parameters": ["-q:a", "0"] if format == "mp3" else None,
        }

        await asyncio.to_thread(
            output_audio.export,
            str(output_file),
            **{k: v for k, v in export_kwargs.items() if v is not None},
        )

    async def convert_audio_format(
        self, input_file: Path, output_format: str, optimize: bool = True
    ) -> Path:
        """Convert audio format with optimization."""

        output_file = input_file.with_suffix(f".{output_format}")

        # Load audio
        audio = await asyncio.to_thread(AudioSegment.from_file, str(input_file))

        # Optimize if requested
        if optimize:
            # Normalize audio
            audio = audio.normalize()

            # Apply compression for smaller file size
            if output_format in ["mp3", "ogg"]:
                audio = audio.compress_dynamic_range()

        # Export with format-specific optimizations
        export_params = {
            "mp3": {"bitrate": "192k", "parameters": ["-q:a", "0"]},
            "ogg": {"bitrate": "192k", "codec": "libvorbis"},
            "flac": {"parameters": ["-compression_level", "8"]},
            "wav": {"parameters": ["-sample_fmt", "s16"]},
        }

        params = export_params.get(output_format, {})

        await asyncio.to_thread(audio.export, str(output_file), format=output_format, **params)

        return output_file


class MemoryEfficientAudioLoader:
    """Memory-efficient audio loading using memory mapping and lazy loading."""

    @staticmethod
    async def load_audio_lazy(
        file_path: Path, start_time: Optional[float] = None, duration: Optional[float] = None
    ) -> np.ndarray:
        """Load only a portion of the audio file."""

        import soundfile as sf

        # Open file without loading into memory
        async def _load():
            with sf.SoundFile(str(file_path)) as audio_file:
                sample_rate = audio_file.samplerate

                # Calculate frame positions
                start_frame = int(start_time * sample_rate) if start_time else 0
                num_frames = int(duration * sample_rate) if duration else -1

                # Seek to start position
                audio_file.seek(start_frame)

                # Read only requested portion
                audio_data = audio_file.read(frames=num_frames, dtype="float32")

                return audio_data, sample_rate

        audio_data, sample_rate = await asyncio.to_thread(_load)
        return audio_data

    @staticmethod
    def create_memory_mapped_array(
        shape: tuple, dtype: np.dtype = np.float32, mode: str = "w+"
    ) -> np.ndarray:
        """Create a memory-mapped array for large audio data."""

        # Create temporary file for memory mapping
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        # Create memory-mapped array
        mmap_array = np.memmap(temp_file.name, dtype=dtype, mode=mode, shape=shape)

        return mmap_array


class AudioCacheManager:
    """Manages audio file caching for frequently accessed files."""

    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.cache_index = {}
        self._load_cache_index()

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / ".cache_index"
        if index_file.exists():
            import json

            with open(index_file) as f:
                self.cache_index = json.load(f)

    def _save_cache_index(self):
        """Save cache index to disk."""
        import json

        index_file = self.cache_dir / ".cache_index"
        with open(index_file, "w") as f:
            json.dump(self.cache_index, f)

    def _get_cache_key(self, file_path: Path, params: dict) -> str:
        """Generate cache key from file path and processing parameters."""

        # Include file modification time in key
        file_stat = file_path.stat()
        key_parts = [str(file_path), str(file_stat.st_mtime), str(file_stat.st_size)]

        # Add processing parameters
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")

        # Generate hash
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get_cached_audio(self, file_path: Path, params: dict = {}) -> Optional[Path]:
        """Get cached audio file if available."""

        cache_key = self._get_cache_key(file_path, params)

        if cache_key in self.cache_index:
            cached_path = self.cache_dir / self.cache_index[cache_key]["filename"]
            if cached_path.exists():
                # Update access time
                self.cache_index[cache_key]["last_access"] = asyncio.get_event_loop().time()
                self._save_cache_index()
                return cached_path
            else:
                # Remove stale entry
                del self.cache_index[cache_key]

        return None

    async def cache_audio(
        self, original_path: Path, processed_path: Path, params: dict = {}
    ) -> Path:
        """Cache processed audio file."""

        cache_key = self._get_cache_key(original_path, params)

        # Check cache size
        await self._enforce_cache_limit()

        # Copy to cache
        cache_filename = f"{cache_key[:16]}{processed_path.suffix}"
        cache_path = self.cache_dir / cache_filename

        await asyncio.to_thread(shutil.copy2, processed_path, cache_path)

        # Update index
        self.cache_index[cache_key] = {
            "filename": cache_filename,
            "original_path": str(original_path),
            "params": params,
            "size": cache_path.stat().st_size,
            "created": asyncio.get_event_loop().time(),
            "last_access": asyncio.get_event_loop().time(),
        }

        self._save_cache_index()
        return cache_path

    async def _enforce_cache_limit(self):
        """Remove old cache entries if cache size exceeds limit."""

        # Calculate total cache size
        total_size = sum(entry["size"] for entry in self.cache_index.values())

        if total_size > self.max_cache_size:
            # Sort by last access time (LRU)
            sorted_entries = sorted(self.cache_index.items(), key=lambda x: x[1]["last_access"])

            # Remove oldest entries until under limit
            while total_size > self.max_cache_size * 0.9:  # Keep 10% buffer
                if not sorted_entries:
                    break

                cache_key, entry = sorted_entries.pop(0)
                cache_path = self.cache_dir / entry["filename"]

                if cache_path.exists():
                    cache_path.unlink()

                total_size -= entry["size"]
                del self.cache_index[cache_key]

            self._save_cache_index()


# Global instances
file_stream_handler = FileStreamHandler()
audio_stream_processor = AudioStreamProcessor()
memory_efficient_loader = MemoryEfficientAudioLoader()


# Utility functions for common operations
async def optimize_audio_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    target_format: str = "mp3",
    target_bitrate: str = "192k",
    normalize: bool = True,
) -> Path:
    """Optimize audio file for size and quality."""

    if output_path is None:
        output_path = input_path.with_suffix(f".{target_format}")

    # Load audio
    audio = await asyncio.to_thread(AudioSegment.from_file, str(input_path))

    # Optimize
    if normalize:
        audio = audio.normalize()

    # Apply format-specific optimizations
    if target_format == "mp3":
        # Remove silence
        audio = audio.strip_silence(silence_thresh=-50)

        # Export with VBR for better quality/size ratio
        await asyncio.to_thread(
            audio.export,
            str(output_path),
            format="mp3",
            parameters=["-q:a", "0", "-b:a", target_bitrate],
        )
    elif target_format == "ogg":
        await asyncio.to_thread(
            audio.export, str(output_path), format="ogg", codec="libvorbis", bitrate=target_bitrate
        )
    else:
        await asyncio.to_thread(audio.export, str(output_path), format=target_format)

    return output_path


async def cleanup_temp_files(max_age_hours: int = 24):
    """Clean up old temporary files."""

    cleaned = file_stream_handler.cleanup_old_files(max_age_hours)
    print(f"Cleaned up {cleaned} temporary files")
