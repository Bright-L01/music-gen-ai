"""
PyTest configuration and shared fixtures.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Make AudioQualityMetrics import optional to avoid librosa dependency in tests
try:
    from music_gen.evaluation.metrics import AudioQualityMetrics

    AUDIO_METRICS_AVAILABLE = True
except ImportError:
    AudioQualityMetrics = None
    AUDIO_METRICS_AVAILABLE = False
# Make optional imports for testing without heavy dependencies
try:
    from music_gen.models.encoders import ConditioningEncoder

    CONDITIONING_AVAILABLE = True
except ImportError:
    ConditioningEncoder = None
    CONDITIONING_AVAILABLE = False

try:
    from music_gen.models.musicgen import create_musicgen_model

    MUSICGEN_AVAILABLE = True
except ImportError:
    create_musicgen_model = None
    MUSICGEN_AVAILABLE = False

try:
    from music_gen.models.transformer.config import MusicGenConfig, TransformerConfig

    CONFIG_AVAILABLE = True
except ImportError:
    MusicGenConfig = None
    TransformerConfig = None
    CONFIG_AVAILABLE = False


@pytest.fixture(scope="session")
def test_config():
    """Create a minimal config for testing."""
    if not CONFIG_AVAILABLE:
        pytest.skip("MusicGenConfig not available (dependencies missing)")

    config = MusicGenConfig()

    # Use small model for testing
    config.transformer.hidden_size = 128
    config.transformer.num_layers = 2
    config.transformer.num_heads = 4
    config.transformer.intermediate_size = 256
    config.transformer.vocab_size = 256
    config.transformer.max_sequence_length = 512

    # Small conditioning vocab
    config.conditioning.genre_vocab_size = 10
    config.conditioning.mood_vocab_size = 5
    config.conditioning.tempo_bins = 20

    return config


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for CI/CD compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sample_rate = 24000
    duration = 1.0
    samples = int(duration * sample_rate)

    # Generate sine wave with some harmonics
    t = torch.linspace(0, duration, samples)
    freq = 440.0  # A4
    audio = (
        torch.sin(2 * np.pi * freq * t) * 0.5
        + torch.sin(2 * np.pi * freq * 2 * t) * 0.3
        + torch.sin(2 * np.pi * freq * 3 * t) * 0.2
    )

    # Add to mono channel
    audio = audio.unsqueeze(0)

    return audio, sample_rate


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 2
    seq_len = 100
    vocab_size = 256

    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "texts": ["Happy jazz music", "Calm ambient sounds"],
        "genre_ids": torch.randint(0, 10, (batch_size,)),
        "mood_ids": torch.randint(0, 5, (batch_size,)),
        "tempo": torch.tensor([120.0, 90.0]),
    }

    return batch


@pytest.fixture
def sample_texts():
    """Sample text prompts for testing."""
    return [
        "Happy jazz music with piano",
        "Calm ambient music with nature sounds",
        "Energetic electronic dance music",
        "Melancholic classical piano piece",
        "Upbeat rock song with guitar solo",
    ]


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def conditioning_encoder():
    """Create conditioning encoder for testing."""
    if not CONDITIONING_AVAILABLE:
        pytest.skip("ConditioningEncoder not available (dependencies missing)")

    return ConditioningEncoder(
        genre_vocab_size=10,
        mood_vocab_size=5,
        tempo_bins=20,
        embedding_dim=64,
        use_genre=True,
        use_mood=True,
        use_tempo=True,
        use_duration=True,
        fusion_method="concat",
    )


@pytest.fixture
def audio_metrics():
    """Create audio metrics evaluator."""
    if not AUDIO_METRICS_AVAILABLE:
        pytest.skip("AudioQualityMetrics not available (librosa dependency missing)")
    return AudioQualityMetrics(
        sample_rate=24000,
        compute_fad=False,  # Disable heavy computations for testing
        compute_clap=False,
        compute_inception_score=False,
    )


@pytest.fixture
def mock_model_config():
    """Configuration for mock model testing."""
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "vocab_size": 100,
        "max_sequence_length": 256,
    }


@pytest.fixture
def dataset_metadata():
    """Sample dataset metadata for testing."""
    return [
        {
            "id": "test_001",
            "caption": "Happy jazz music with piano and drums",
            "genre": "jazz",
            "mood": "happy",
            "tempo": 120,
            "duration": 10.0,
            "audio_path": "/fake/path/test_001.wav",
        },
        {
            "id": "test_002",
            "caption": "Calm ambient music with nature sounds",
            "genre": "ambient",
            "mood": "calm",
            "tempo": 80,
            "duration": 15.0,
            "audio_path": "/fake/path/test_002.wav",
        },
        {
            "id": "test_003",
            "caption": "Energetic electronic dance music",
            "genre": "electronic",
            "mood": "energetic",
            "tempo": 128,
            "duration": 8.0,
            "audio_path": "/fake/path/test_003.wav",
        },
    ]


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests that test individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests that test complete workflows")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "model: Tests that require model weights")


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        # Run slow tests if explicitly requested
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="run GPU tests")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
