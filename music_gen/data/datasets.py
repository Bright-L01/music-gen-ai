"""
Dataset implementations for music generation training.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .augmentation import (
    AdaptiveAugmentation,
    create_inference_augmentation_pipeline,
    create_training_augmentation_pipeline,
)

logger = logging.getLogger(__name__)


class MusicCapsDataset(Dataset):
    """
    Dataset for MusicCaps: A Dataset Composed of Music Clips and Text Descriptions.

    This dataset contains music clips with natural language descriptions,
    perfect for training text-to-music generation models.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_audio_length: float = 30.0,
        sample_rate: int = 24000,
        max_text_length: int = 512,
        audio_tokenizer=None,
        conditioning_vocab: Optional[Dict[str, Dict]] = None,
        augment_audio: bool = True,
        augment_text: bool = False,
        cache_audio_tokens: bool = True,
        augmentation_strength: str = "moderate",  # "light", "moderate", "strong"
        use_adaptive_augmentation: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.audio_tokenizer = audio_tokenizer
        self.conditioning_vocab = conditioning_vocab or {}
        self.augment_audio = augment_audio
        self.augment_text = augment_text
        self.cache_audio_tokens = cache_audio_tokens
        self.augmentation_strength = augmentation_strength
        self.use_adaptive_augmentation = use_adaptive_augmentation

        # Load metadata
        self.metadata = self._load_metadata()

        # Audio cache for tokenized audio
        self.audio_cache = {} if cache_audio_tokens else None

        # Initialize audio augmentation pipeline
        if augment_audio and split == "train":
            if use_adaptive_augmentation:
                self.adaptive_augmentation = AdaptiveAugmentation()
                self.audio_pipeline = None  # Will be created dynamically
            else:
                strong_aug = augmentation_strength == "strong"
                self.audio_pipeline = create_training_augmentation_pipeline(
                    strong=strong_aug, include_mix=True
                )
                self.adaptive_augmentation = None
        else:
            # Light augmentation for validation/inference
            self.audio_pipeline = (
                create_inference_augmentation_pipeline() if augment_audio else None
            )
            self.adaptive_augmentation = None

        # Initialize mix samples for Polymix augmentation
        self.mix_samples = []
        if self.audio_pipeline and hasattr(self.audio_pipeline, "augmentations"):
            # Find PolymixAugmentation and set up mix samples
            for aug in self.audio_pipeline.augmentations:
                if hasattr(aug, "set_mix_samples"):
                    # Load a few random samples for mixing
                    self._setup_mix_samples(aug)

        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load dataset metadata."""

        metadata_file = self.data_dir / f"{self.split}.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Filter samples that have audio files
        valid_metadata = []
        for item in metadata:
            audio_path = self.data_dir / "audio" / f"{item['id']}.wav"
            if audio_path.exists():
                item["audio_path"] = str(audio_path)
                valid_metadata.append(item)
            else:
                logger.warning(f"Audio file not found: {audio_path}")

        return valid_metadata

    def _setup_mix_samples(self, polymix_aug, num_samples: int = 10):
        """Set up mix samples for Polymix augmentation."""

        if len(self.metadata) < num_samples:
            return

        # Select random samples for mixing
        mix_metadata = random.sample(self.metadata, num_samples)
        mix_samples = []

        for meta in mix_metadata:
            try:
                waveform = self._load_audio(meta["audio_path"])
                mix_samples.append(waveform)
            except Exception as e:
                logger.warning(f"Failed to load mix sample {meta['audio_path']}: {e}")

        if mix_samples:
            polymix_aug.set_mix_samples(mix_samples)
            logger.info(f"Set up {len(mix_samples)} mix samples for Polymix augmentation")

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations using the new pipeline."""

        if self.adaptive_augmentation is not None:
            # Use adaptive augmentation
            pipeline = self.adaptive_augmentation.get_pipeline()
            return pipeline(waveform, self.sample_rate)
        elif self.audio_pipeline is not None:
            # Use pre-configured pipeline
            return self.audio_pipeline(waveform, self.sample_rate)
        else:
            return waveform

    def update_augmentation_strength(self, loss: float, target_loss: float = 2.0):
        """Update adaptive augmentation strength based on training loss."""
        if self.adaptive_augmentation is not None:
            self.adaptive_augmentation.update_strength(loss, target_loss)

    def _augment_text(self, text: str) -> str:
        """Apply text augmentations."""

        if not self.augment_text:
            return text

        # Simple text augmentations
        augmentations = []

        # Add random musical descriptors
        descriptors = [
            "melodic",
            "rhythmic",
            "harmonic",
            "dynamic",
            "expressive",
            "energetic",
            "calm",
            "peaceful",
            "intense",
            "dramatic",
        ]
        if random.random() < 0.3:
            descriptor = random.choice(descriptors)
            augmentations.append(f"{descriptor} {text}")

        # Add tempo/style hints
        tempo_hints = ["slow", "medium tempo", "fast", "upbeat", "downtempo"]
        if random.random() < 0.2:
            tempo = random.choice(tempo_hints)
            augmentations.append(f"{tempo} {text}")

        if augmentations:
            return random.choice(augmentations)

        return text

    def _extract_conditioning(self, metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract conditioning information from metadata."""

        conditioning = {}

        # Genre conditioning
        if "genre" in metadata and "genre" in self.conditioning_vocab:
            genre = metadata["genre"].lower()
            genre_vocab = self.conditioning_vocab["genre"]
            genre_id = genre_vocab.get(genre, genre_vocab.get("unknown", 0))
            conditioning["genre_ids"] = torch.tensor(genre_id, dtype=torch.long)

        # Mood conditioning
        if "mood" in metadata and "mood" in self.conditioning_vocab:
            mood = metadata["mood"].lower()
            mood_vocab = self.conditioning_vocab["mood"]
            mood_id = mood_vocab.get(mood, mood_vocab.get("unknown", 0))
            conditioning["mood_ids"] = torch.tensor(mood_id, dtype=torch.long)

        # Tempo conditioning
        if "tempo" in metadata:
            tempo = float(metadata["tempo"])
            conditioning["tempo"] = torch.tensor(tempo, dtype=torch.float32)

        # Duration conditioning
        if "duration" in metadata:
            duration = float(metadata["duration"])
            conditioning["duration"] = torch.tensor(duration, dtype=torch.float32)

        return conditioning

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""

        try:
            # Load audio
            waveform, orig_sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if orig_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sample_rate,
                    new_freq=self.sample_rate,
                )
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Truncate or pad to max length
            max_samples = int(self.max_audio_length * self.sample_rate)
            if waveform.shape[1] > max_samples:
                # Random crop during training, center crop during validation
                if self.split == "train":
                    start_idx = random.randint(0, waveform.shape[1] - max_samples)
                else:
                    start_idx = (waveform.shape[1] - max_samples) // 2
                waveform = waveform[:, start_idx : start_idx + max_samples]
            elif waveform.shape[1] < max_samples:
                # Pad with zeros
                padding = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-8)

            return waveform

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            # Return silence as fallback
            max_samples = int(self.max_audio_length * self.sample_rate)
            return torch.zeros(1, max_samples)

    def _tokenize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Tokenize audio using the provided tokenizer."""

        if self.audio_tokenizer is None:
            return waveform

        try:
            # Check cache first
            if self.audio_cache is not None:
                audio_hash = hash(waveform.numpy().tobytes())
                if audio_hash in self.audio_cache:
                    return self.audio_cache[audio_hash]

            # Tokenize audio
            with torch.no_grad():
                tokens = self.audio_tokenizer.tokenize(waveform.unsqueeze(0))
                tokens = tokens.squeeze(0)  # Remove batch dimension

            # Cache result
            if self.audio_cache is not None:
                self.audio_cache[audio_hash] = tokens

            return tokens

        except Exception as e:
            logger.error(f"Failed to tokenize audio: {e}")
            # Return dummy tokens
            seq_len = int(self.max_audio_length * 50)  # Rough estimate
            return torch.zeros(seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""

        metadata = self.metadata[idx]

        # Load and process audio
        waveform = self._load_audio(metadata["audio_path"])

        # Apply audio augmentations
        if self.augment_audio and self.split == "train":
            waveform = self._augment_audio(waveform)

        # Tokenize audio
        audio_tokens = self._tokenize_audio(waveform)

        # Process text
        text = metadata.get("caption", metadata.get("description", ""))
        text = self._augment_text(text)

        # Extract conditioning
        conditioning = self._extract_conditioning(metadata)

        sample = {
            "audio_tokens": audio_tokens,
            "text": text,
            "audio_path": metadata["audio_path"],
            "id": metadata.get("id", idx),
            **conditioning,
        }

        return sample


class FreeMusicArchiveDataset(Dataset):
    """Dataset for Free Music Archive (FMA) dataset."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        subset: str = "medium",  # small, medium, large, full
        **kwargs,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.subset = subset

        # Load FMA metadata
        self.metadata = self._load_fma_metadata(split)

        # Initialize parent class
        super().__init__(**kwargs)

    def _load_fma_metadata(self, split: str) -> List[Dict[str, Any]]:
        """Load FMA dataset metadata."""

        # Load tracks metadata
        tracks_file = self.data_dir / "fma_metadata" / "tracks.csv"
        tracks_df = pd.read_csv(tracks_file, header=[0, 1])

        # Filter by subset
        subset_mask = tracks_df[("set", "subset")] <= self.subset
        tracks_df = tracks_df[subset_mask]

        # Split data
        if split == "train":
            split_mask = tracks_df[("set", "split")] == "training"
        elif split == "val":
            split_mask = tracks_df[("set", "split")] == "validation"
        elif split == "test":
            split_mask = tracks_df[("set", "split")] == "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        tracks_df = tracks_df[split_mask]

        # Convert to list of dictionaries
        metadata = []
        for idx, row in tracks_df.iterrows():
            track_id = str(idx).zfill(6)
            audio_path = self.data_dir / "fma_medium" / track_id[:3] / f"{track_id}.mp3"

            if audio_path.exists():
                metadata.append(
                    {
                        "id": track_id,
                        "audio_path": str(audio_path),
                        "genre": row[("track", "genre_top")],
                        "title": row[("track", "title")],
                        "duration": row[("track", "duration")],
                        "caption": f"Music track titled '{row[('track', 'title')]}'",
                    }
                )

        return metadata


class SyntheticMusicDataset(Dataset):
    """Synthetic dataset for testing and development."""

    def __init__(
        self,
        num_samples: int = 1000,
        max_audio_length: float = 10.0,
        sample_rate: int = 24000,
        **kwargs,
    ):
        self.num_samples = num_samples
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate

        # Generate synthetic metadata
        self.metadata = self._generate_synthetic_metadata()

    def _generate_synthetic_metadata(self) -> List[Dict[str, Any]]:
        """Generate synthetic metadata for testing."""

        genres = ["jazz", "classical", "rock", "electronic", "ambient", "folk"]
        moods = ["happy", "sad", "energetic", "calm", "dramatic", "peaceful"]
        instruments = ["piano", "guitar", "violin", "drums", "synthesizer", "saxophone"]

        metadata = []
        for i in range(self.num_samples):
            genre = random.choice(genres)
            mood = random.choice(moods)
            instrument = random.choice(instruments)
            tempo = random.randint(60, 180)

            caption = f"{mood} {genre} music with {instrument}"

            metadata.append(
                {
                    "id": f"synthetic_{i:06d}",
                    "caption": caption,
                    "genre": genre,
                    "mood": mood,
                    "tempo": tempo,
                    "duration": self.max_audio_length,
                }
            )

        return metadata

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")

        metadata = self.metadata[idx]

        # Generate synthetic audio
        waveform = self._load_audio("")  # Path not used for synthetic

        sample = {
            "audio_tokens": waveform,  # For synthetic, we use the raw audio
            "text": metadata["caption"],
            "id": metadata["id"],
            "genre": metadata["genre"],
            "mood": metadata["mood"],
            "tempo": metadata["tempo"],
            "duration": metadata["duration"],
        }

        return sample

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Generate synthetic audio."""

        # Generate simple sine wave with noise
        duration = self.max_audio_length
        samples = int(duration * self.sample_rate)

        # Base frequency
        freq = random.uniform(200, 800)
        t = torch.linspace(0, duration, samples)

        # Generate complex waveform
        waveform = (
            torch.sin(2 * np.pi * freq * t) * 0.3
            + torch.sin(2 * np.pi * freq * 1.5 * t) * 0.2
            + torch.sin(2 * np.pi * freq * 2 * t) * 0.1
            + torch.randn(samples) * 0.05
        )

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform.unsqueeze(0)


def create_dataset(dataset_name: str, data_dir: str, split: str = "train", **kwargs) -> Dataset:
    """Factory function to create datasets."""

    if dataset_name == "musiccaps":
        return MusicCapsDataset(data_dir, split, **kwargs)
    elif dataset_name == "fma":
        return FreeMusicArchiveDataset(data_dir, split, **kwargs)
    elif dataset_name == "synthetic":
        return SyntheticMusicDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""

    # Separate different types of data
    audio_tokens = [item["audio_tokens"] for item in batch]
    texts = [item["text"] for item in batch]

    # Check if audio_tokens are waveforms or tokenized
    # If waveforms (2D: channels x samples), we need to handle differently
    if len(audio_tokens[0].shape) == 2:  # Waveform format
        # For waveforms, we pad on the time dimension (last dimension)
        max_audio_len = max(tokens.shape[-1] for tokens in audio_tokens)
        num_channels = audio_tokens[0].shape[0]
        padded_audio = torch.zeros(len(batch), num_channels, max_audio_len, dtype=torch.float32)
        audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.bool)

        for i, tokens in enumerate(audio_tokens):
            length = tokens.shape[-1]
            padded_audio[i, :, :length] = tokens
            audio_mask[i, :length] = True
    else:  # Tokenized format (1D)
        max_audio_len = max(tokens.shape[0] for tokens in audio_tokens)
        padded_audio = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
        audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.bool)

        for i, tokens in enumerate(audio_tokens):
            length = tokens.shape[0]
            padded_audio[i, :length] = tokens
            audio_mask[i, :length] = True

    # Create labels (shifted audio tokens for language modeling)
    # Only create labels for tokenized data, not raw waveforms
    if len(audio_tokens[0].shape) == 1:  # Tokenized format
        labels = padded_audio.clone()
        labels[~audio_mask] = -100  # Ignore padding in loss calculation
    else:  # Waveform format - no labels needed
        labels = None

    # Collect conditioning information
    conditioning = {}
    for key in ["genre_ids", "mood_ids", "tempo", "duration"]:
        if key in batch[0]:
            values = [item[key] for item in batch]
            # Convert to tensor if needed
            if isinstance(values[0], (int, float)):
                conditioning[key] = torch.tensor(values)
            else:
                conditioning[key] = torch.stack(values)

    # Create batch dictionary
    batch_dict = {
        "input_ids": padded_audio,
        "attention_mask": audio_mask,
        "texts": texts,
        **conditioning,
    }

    # Only add labels if they exist (for tokenized data)
    if labels is not None:
        batch_dict["labels"] = labels

    return batch_dict


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """Create DataLoader with appropriate settings."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
