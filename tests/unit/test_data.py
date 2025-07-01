"""
Unit tests for data loading and preprocessing.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from music_gen.data.datasets import (
    MusicCapsDataset,
    SyntheticMusicDataset,
    collate_fn,
    create_dataloader,
    create_dataset,
)


@pytest.mark.unit
class TestSyntheticDataset:
    """Test synthetic dataset (no external dependencies)."""

    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        num_samples = 100
        dataset = SyntheticMusicDataset(
            num_samples=num_samples,
            max_audio_length=5.0,
            sample_rate=24000,
        )

        assert len(dataset) == num_samples
        assert len(dataset.metadata) == num_samples

    def test_synthetic_dataset_metadata(self):
        """Test synthetic dataset metadata generation."""
        dataset = SyntheticMusicDataset(num_samples=10)

        for item in dataset.metadata:
            assert "id" in item
            assert "caption" in item
            assert "genre" in item
            assert "mood" in item
            assert "tempo" in item
            assert "duration" in item

            # Check data types and ranges
            assert isinstance(item["tempo"], int)
            assert 60 <= item["tempo"] <= 180

    def test_synthetic_dataset_getitem(self):
        """Test synthetic dataset item retrieval."""
        dataset = SyntheticMusicDataset(num_samples=5, max_audio_length=2.0)

        # Mock the tokenizer to avoid dependencies
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = torch.randint(0, 256, (100,))
        dataset.audio_tokenizer = mock_tokenizer

        item = dataset[0]

        assert "audio_tokens" in item
        assert "text" in item
        assert "id" in item
        assert isinstance(item["text"], str)
        assert isinstance(item["audio_tokens"], torch.Tensor)


@pytest.mark.unit
class TestMusicCapsDataset:
    """Test MusicCaps dataset functionality."""

    def test_dataset_metadata_loading(self, temp_dir, dataset_metadata):
        """Test metadata loading."""
        # Create fake metadata file
        metadata_file = temp_dir / "train.json"
        import json

        with open(metadata_file, "w") as f:
            json.dump(dataset_metadata, f)

        # Create fake audio directory
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        # Create fake audio files
        for item in dataset_metadata:
            audio_file = audio_dir / f"{item['id']}.wav"
            audio_file.touch()

        try:
            dataset = MusicCapsDataset(
                data_dir=str(temp_dir),
                split="train",
                max_audio_length=10.0,
            )

            assert len(dataset) == len(dataset_metadata)
            assert len(dataset.metadata) == len(dataset_metadata)

        except ImportError as e:
            pytest.skip(f"Missing audio dependencies: {e}")

    def test_conditioning_extraction(self, temp_dir, dataset_metadata):
        """Test conditioning information extraction."""
        # Setup test environment
        metadata_file = temp_dir / "train.json"
        import json

        with open(metadata_file, "w") as f:
            json.dump(dataset_metadata, f)

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        for item in dataset_metadata:
            (audio_dir / f"{item['id']}.wav").touch()

        try:
            conditioning_vocab = {
                "genre": {"jazz": 0, "ambient": 1, "electronic": 2},
                "mood": {"happy": 0, "calm": 1, "energetic": 2},
            }

            dataset = MusicCapsDataset(
                data_dir=str(temp_dir),
                split="train",
                conditioning_vocab=conditioning_vocab,
            )

            # Test conditioning extraction
            item_metadata = dataset_metadata[0]  # jazz, happy
            conditioning = dataset._extract_conditioning(item_metadata)

            assert "genre_ids" in conditioning
            assert "mood_ids" in conditioning
            assert conditioning["genre_ids"].item() == 0  # jazz
            assert conditioning["mood_ids"].item() == 0  # happy

        except ImportError as e:
            pytest.skip(f"Missing audio dependencies: {e}")

    @patch("music_gen.data.datasets.torchaudio.load")
    def test_audio_loading_mock(self, mock_load, temp_dir, dataset_metadata):
        """Test audio loading with mocked torchaudio."""
        # Setup
        metadata_file = temp_dir / "train.json"
        import json

        with open(metadata_file, "w") as f:
            json.dump(dataset_metadata, f)

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        for item in dataset_metadata:
            (audio_dir / f"{item['id']}.wav").touch()

        # Mock audio loading
        sample_rate = 24000
        duration = 10.0
        samples = int(duration * sample_rate)
        mock_waveform = torch.randn(1, samples)
        mock_load.return_value = (mock_waveform, sample_rate)

        dataset = MusicCapsDataset(
            data_dir=str(temp_dir),
            split="train",
            max_audio_length=duration,
            sample_rate=sample_rate,
        )

        # Test audio loading
        audio_path = dataset.metadata[0]["audio_path"]
        processed_audio = dataset._load_audio(audio_path)

        assert processed_audio.shape[0] == 1  # Mono
        assert processed_audio.shape[1] == samples  # Correct length
        mock_load.assert_called_once()


@pytest.mark.unit
class TestDatasetFactory:
    """Test dataset factory functions."""

    def test_create_dataset_synthetic(self):
        """Test creating synthetic dataset."""
        dataset = create_dataset("synthetic", data_dir="", num_samples=50)

        assert isinstance(dataset, SyntheticMusicDataset)
        assert len(dataset) == 50

    def test_create_dataset_unknown(self):
        """Test creating unknown dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataset("unknown_dataset", data_dir="")

    def test_create_dataset_musiccaps(self, temp_dir):
        """Test creating MusicCaps dataset."""
        # Create minimal setup
        metadata_file = temp_dir / "train.json"
        with open(metadata_file, "w") as f:
            import json

            json.dump([], f)  # Empty dataset

        try:
            dataset = create_dataset("musiccaps", data_dir=str(temp_dir), split="train")
            assert isinstance(dataset, MusicCapsDataset)

        except ImportError as e:
            pytest.skip(f"Missing dependencies: {e}")


@pytest.mark.unit
class TestCollateFunction:
    """Test data collation functionality."""

    def test_collate_basic(self):
        """Test basic collation functionality."""
        # Create sample batch items
        batch_items = [
            {
                "audio_tokens": torch.tensor([1, 2, 3, 4, 5]),
                "text": "Happy jazz music",
                "genre_ids": torch.tensor(0),
                "mood_ids": torch.tensor(1),
                "tempo": torch.tensor(120.0),
            },
            {
                "audio_tokens": torch.tensor([6, 7, 8]),
                "text": "Calm ambient music",
                "genre_ids": torch.tensor(2),
                "mood_ids": torch.tensor(0),
                "tempo": torch.tensor(90.0),
            },
        ]

        batch = collate_fn(batch_items)

        # Check required keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "texts" in batch

        # Check shapes
        assert batch["input_ids"].shape[0] == 2  # batch size
        assert batch["input_ids"].shape[1] == 5  # max sequence length
        assert batch["attention_mask"].shape == batch["input_ids"].shape

        # Check conditioning
        assert "genre_ids" in batch
        assert "mood_ids" in batch
        assert "tempo" in batch
        assert batch["genre_ids"].shape[0] == 2

    def test_collate_padding(self):
        """Test sequence padding in collation."""
        batch_items = [
            {
                "audio_tokens": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                "text": "Long sequence",
            },
            {
                "audio_tokens": torch.tensor([1, 2]),
                "text": "Short sequence",
            },
        ]

        batch = collate_fn(batch_items)

        # Check padding
        assert batch["input_ids"].shape[1] == 8  # Padded to max length
        assert batch["attention_mask"][0].sum() == 8  # First item fully attended
        assert batch["attention_mask"][1].sum() == 2  # Second item only first 2 tokens

        # Check labels have padding ignored
        assert (batch["labels"][1, 2:] == -100).all()  # Padding tokens ignored

    def test_collate_empty_conditioning(self):
        """Test collation with missing conditioning."""
        batch_items = [
            {
                "audio_tokens": torch.tensor([1, 2, 3]),
                "text": "Music without conditioning",
            },
        ]

        batch = collate_fn(batch_items)

        # Should still work without conditioning
        assert "input_ids" in batch
        assert "texts" in batch
        assert len(batch["texts"]) == 1


@pytest.mark.unit
class TestDataLoader:
    """Test DataLoader functionality."""

    def test_create_dataloader(self):
        """Test DataLoader creation."""
        dataset = SyntheticMusicDataset(num_samples=20)

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = torch.randint(0, 256, (50,))
        dataset.audio_tokenizer = mock_tokenizer

        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        assert dataloader.batch_size == 4
        assert dataloader.dataset == dataset

        # Test iteration
        batch = next(iter(dataloader))
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 4  # batch size

    def test_dataloader_iteration(self):
        """Test full DataLoader iteration."""
        dataset = SyntheticMusicDataset(num_samples=10)

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = torch.randint(0, 256, (30,))
        dataset.audio_tokenizer = mock_tokenizer

        dataloader = create_dataloader(
            dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
        )

        batches = list(dataloader)

        # Should have 4 batches: 3+3+3+1
        assert len(batches) == 4
        assert batches[0]["input_ids"].shape[0] == 3
        assert batches[-1]["input_ids"].shape[0] == 1  # Last batch smaller


@pytest.mark.unit
class TestDataAugmentation:
    """Test data augmentation functionality."""

    @patch("music_gen.data.datasets.torchaudio.transforms.TimeStretch")
    @patch("music_gen.data.datasets.torchaudio.transforms.PitchShift")
    def test_audio_augmentation_creation(self, mock_pitch, mock_stretch, temp_dir):
        """Test audio augmentation setup."""
        # Create minimal dataset setup
        metadata_file = temp_dir / "train.json"
        with open(metadata_file, "w") as f:
            import json

            json.dump([{"id": "test", "caption": "test"}], f)

        try:
            dataset = MusicCapsDataset(
                data_dir=str(temp_dir),
                split="train",
                augment_audio=True,
            )

            # Should create audio pipeline for augmentation
            assert dataset.audio_pipeline is not None

        except ImportError as e:
            pytest.skip(f"Missing audio dependencies: {e}")

    def test_text_augmentation(self, temp_dir):
        """Test text augmentation functionality."""
        # Create minimal setup
        metadata_file = temp_dir / "train.json"
        with open(metadata_file, "w") as f:
            import json

            json.dump([{"id": "test", "caption": "test music"}], f)

        try:
            dataset = MusicCapsDataset(
                data_dir=str(temp_dir),
                split="train",
                augment_text=True,
            )

            original_text = "jazz piano music"
            augmented_text = dataset._augment_text(original_text)

            # Augmented text should be different (sometimes)
            # or the same (if no augmentation applied this time)
            assert isinstance(augmented_text, str)
            assert len(augmented_text) > 0

        except ImportError as e:
            pytest.skip(f"Missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
