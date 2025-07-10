"""
Audio repository implementations.

Provides concrete implementations for audio file storage and retrieval.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio

from music_gen.core.exceptions import AudioNotFoundError, AudioSaveError
from music_gen.core.interfaces.repositories import AudioRepository

logger = logging.getLogger(__name__)


class FileSystemAudioRepository(AudioRepository):
    """File system based audio repository."""

    def __init__(self, base_path: Path):
        """Initialize repository with base path.

        Args:
            base_path: Base directory for audio storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_audio_path(self, audio_id: str, format: str = "wav") -> Path:
        """Get path for an audio file."""
        # Organize by date for better file management
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_path = self.base_path / date_str
        dir_path.mkdir(parents=True, exist_ok=True)

        safe_id = audio_id.replace("/", "_").replace(":", "_")
        return dir_path / f"{safe_id}.{format}"

    async def save_audio(
        self, audio_id: str, audio_data: torch.Tensor, sample_rate: int, format: str = "wav"
    ) -> str:
        """Save audio data to filesystem."""
        try:
            audio_path = self._get_audio_path(audio_id, format)

            # Ensure audio is on CPU
            if audio_data.is_cuda:
                audio_data = audio_data.cpu()

            # Ensure correct shape (channels, samples)
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)
            elif audio_data.dim() == 3:
                # Handle batch dimension
                audio_data = audio_data.squeeze(0)

            # Save audio file
            torchaudio.save(str(audio_path), audio_data, sample_rate, format=format)

            logger.info(f"Audio saved: {audio_id} at {audio_path}")
            return str(audio_path)

        except Exception as e:
            logger.error(f"Failed to save audio {audio_id}: {e}")
            raise AudioSaveError(f"Failed to save audio: {e}")

    async def load_audio(self, audio_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio data from filesystem."""
        # Try different formats
        for format in ["wav", "mp3", "flac"]:
            audio_path = self._get_audio_path(audio_id, format)
            if audio_path.exists():
                try:
                    audio_data, sample_rate = torchaudio.load(str(audio_path))
                    logger.info(f"Audio loaded: {audio_id}")
                    return audio_data, sample_rate
                except Exception as e:
                    logger.error(f"Failed to load audio {audio_id}: {e}")
                    raise AudioNotFoundError(f"Failed to load audio: {e}")

        # Also check in all date directories
        for date_dir in self.base_path.iterdir():
            if date_dir.is_dir():
                for format in ["wav", "mp3", "flac"]:
                    safe_id = audio_id.replace("/", "_").replace(":", "_")
                    audio_path = date_dir / f"{safe_id}.{format}"
                    if audio_path.exists():
                        try:
                            audio_data, sample_rate = torchaudio.load(str(audio_path))
                            logger.info(f"Audio loaded: {audio_id}")
                            return audio_data, sample_rate
                        except Exception as e:
                            logger.error(f"Failed to load audio {audio_id}: {e}")
                            raise AudioNotFoundError(f"Failed to load audio: {e}")

        raise AudioNotFoundError(f"Audio not found: {audio_id}")

    async def exists(self, audio_id: str) -> bool:
        """Check if audio exists."""
        # Check in today's directory
        for format in ["wav", "mp3", "flac"]:
            audio_path = self._get_audio_path(audio_id, format)
            if audio_path.exists():
                return True

        # Check in all date directories
        safe_id = audio_id.replace("/", "_").replace(":", "_")
        for date_dir in self.base_path.iterdir():
            if date_dir.is_dir():
                for format in ["wav", "mp3", "flac"]:
                    audio_path = date_dir / f"{safe_id}.{format}"
                    if audio_path.exists():
                        return True

        return False

    async def delete_audio(self, audio_id: str) -> None:
        """Delete audio file."""
        deleted = False

        # Try to delete from today's directory
        for format in ["wav", "mp3", "flac"]:
            audio_path = self._get_audio_path(audio_id, format)
            if audio_path.exists():
                audio_path.unlink()
                deleted = True
                logger.info(f"Audio deleted: {audio_id}")
                break

        # If not found, check all date directories
        if not deleted:
            safe_id = audio_id.replace("/", "_").replace(":", "_")
            for date_dir in self.base_path.iterdir():
                if date_dir.is_dir():
                    for format in ["wav", "mp3", "flac"]:
                        audio_path = date_dir / f"{safe_id}.{format}"
                        if audio_path.exists():
                            audio_path.unlink()
                            logger.info(f"Audio deleted: {audio_id}")
                            return

        if not deleted:
            logger.warning(f"Audio not found for deletion: {audio_id}")

    async def get_audio_url(self, audio_id: str) -> str:
        """Get URL for audio file access."""
        # In filesystem implementation, return file path
        # In production, this would return a proper URL
        for format in ["wav", "mp3", "flac"]:
            audio_path = self._get_audio_path(audio_id, format)
            if audio_path.exists():
                return f"file://{audio_path.absolute()}"

        # Check all date directories
        safe_id = audio_id.replace("/", "_").replace(":", "_")
        for date_dir in self.base_path.iterdir():
            if date_dir.is_dir():
                for format in ["wav", "mp3", "flac"]:
                    audio_path = date_dir / f"{safe_id}.{format}"
                    if audio_path.exists():
                        return f"file://{audio_path.absolute()}"

        raise AudioNotFoundError(f"Audio not found: {audio_id}")

    async def list_audio(self, prefix: Optional[str] = None, limit: int = 100) -> List[str]:
        """List available audio IDs."""
        audio_ids = []

        for date_dir in sorted(self.base_path.iterdir(), reverse=True):
            if date_dir.is_dir():
                for audio_file in date_dir.iterdir():
                    if audio_file.suffix in [".wav", ".mp3", ".flac"]:
                        # Extract audio ID from filename
                        audio_id = audio_file.stem.replace("_", "/", 1)

                        # Apply prefix filter if provided
                        if prefix is None or audio_id.startswith(prefix):
                            audio_ids.append(audio_id)

                            if len(audio_ids) >= limit:
                                return audio_ids

        return audio_ids
