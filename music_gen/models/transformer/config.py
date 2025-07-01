"""
Configuration classes for the MusicGen transformer model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TransformerConfig:
    """Configuration for the main transformer model."""

    # Model architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 2048
    max_sequence_length: int = 8192

    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    use_cache: bool = True

    # Cross-attention configuration
    cross_attention_layers: Optional[List[int]] = None  # If None, all layers have cross-attention
    text_hidden_size: int = 768  # T5-Base hidden size

    # Audio tokenization
    audio_vocab_size: int = 2048
    num_quantizers: int = 8

    # Positional encoding
    max_position_embeddings: int = 8192
    use_learned_positional_encoding: bool = True
    use_rotary_positional_encoding: bool = False

    # Conditioning
    use_conditioning: bool = True
    conditioning_dim: int = 512
    genre_vocab_size: int = 50
    mood_vocab_size: int = 20
    tempo_bins: int = 100

    # Training specifics
    gradient_checkpointing: bool = False
    use_scaled_dot_product_attention: bool = True

    # Generation
    max_generation_length: int = 4096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cross_attention_layers is None:
            self.cross_attention_layers = list(range(self.num_layers))

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@dataclass
class EnCodecConfig:
    """Configuration for EnCodec audio tokenizer."""

    # Model selection
    model_name: str = "facebook/encodec_24khz"
    sample_rate: int = 24000

    # Tokenization
    num_quantizers: int = 8
    bandwidth: float = 6.0  # kbps

    # Audio processing
    normalize: bool = True
    chunk_length: Optional[float] = None  # seconds, None for full length
    overlap: float = 0.01  # overlap between chunks

    # Compression
    use_compression: bool = True
    compression_model: str = "facebook/encodec_24khz"


@dataclass
class T5Config:
    """Configuration for T5 text encoder."""

    # Model selection
    model_name: str = "t5-base"
    hidden_size: int = 768

    # Text processing
    max_text_length: int = 512
    tokenizer_name: str = "t5-base"

    # Fine-tuning
    freeze_encoder: bool = True
    dropout: float = 0.1

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class ConditioningConfig:
    """Configuration for conditioning inputs."""

    # Genre conditioning
    use_genre: bool = True
    genre_vocab_size: int = 50
    genre_embedding_dim: int = 128

    # Mood conditioning
    use_mood: bool = True
    mood_vocab_size: int = 20
    mood_embedding_dim: int = 128

    # Tempo conditioning
    use_tempo: bool = True
    tempo_bins: int = 100
    tempo_min: int = 60
    tempo_max: int = 200
    tempo_embedding_dim: int = 128

    # Duration conditioning
    use_duration: bool = True
    duration_max: float = 120.0  # seconds
    duration_embedding_dim: int = 128

    # Instrumentation
    use_instruments: bool = False
    instrument_vocab_size: int = 128
    instrument_embedding_dim: int = 128

    # Combined conditioning
    conditioning_dropout: float = 0.1
    conditioning_fusion: str = "concat"  # concat, add, attention


@dataclass
class MusicGenConfig:
    """Complete configuration for MusicGen model."""

    # Sub-configurations
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    encodec: EnCodecConfig = field(default_factory=EnCodecConfig)
    t5: T5Config = field(default_factory=T5Config)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)

    # Model metadata
    model_name: str = "musicgen-base"
    version: str = "1.0.0"

    # Training configuration
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # Inference configuration
    default_generation_params: Dict[str, Any] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.default_generation_params is None:
            self.default_generation_params = {
                "max_length": 1024,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
                "do_sample": True,
                "num_beams": 1,
                "repetition_penalty": 1.1,
            }

        # Convert dictionaries to config objects if needed
        if isinstance(self.transformer, dict):
            self.transformer = TransformerConfig(**self.transformer)
        if isinstance(self.encodec, dict):
            self.encodec = EnCodecConfig(**self.encodec)
        if isinstance(self.t5, dict):
            self.t5 = T5Config(**self.t5)
        if isinstance(self.conditioning, dict):
            self.conditioning = ConditioningConfig(**self.conditioning)

        # Ensure consistency between configs
        self.transformer.text_hidden_size = self.t5.hidden_size
        self.transformer.audio_vocab_size = 2**self.encodec.num_quantizers

        if self.conditioning.use_genre:
            self.transformer.conditioning_dim += self.conditioning.genre_embedding_dim
        if self.conditioning.use_mood:
            self.transformer.conditioning_dim += self.conditioning.mood_embedding_dim
        if self.conditioning.use_tempo:
            self.transformer.conditioning_dim += self.conditioning.tempo_embedding_dim
        if self.conditioning.use_duration:
            self.transformer.conditioning_dim += self.conditioning.duration_embedding_dim
