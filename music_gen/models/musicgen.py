"""
Main MusicGen model combining text encoding, conditioning, and transformer generation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation.beam_search import BeamSearchConfig, beam_search_generate
from .encodec.audio_tokenizer import EnCodecTokenizer
from .encoders import MultiModalEncoder
from .transformer.config import MusicGenConfig
from .transformer.model import MusicGenTransformer

logger = logging.getLogger(__name__)


class MusicGenModel(nn.Module):
    """
    Complete MusicGen model for text-to-music generation.

    This model combines:
    - T5 text encoder for understanding text prompts
    - Conditioning encoder for musical attributes (genre, mood, tempo)
    - Transformer decoder with cross-attention for music generation
    - EnCodec tokenizer for audio representation
    """

    def __init__(self, config: MusicGenConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self._init_encoders()
        self._init_transformer()
        self._init_audio_tokenizer()

        # Special tokens
        self.pad_token_id = config.transformer.pad_token_id
        self.bos_token_id = config.transformer.bos_token_id
        self.eos_token_id = config.transformer.eos_token_id

        # Generation parameters
        self.generation_config = config.default_generation_params

    def _init_encoders(self):
        """Initialize text and conditioning encoders."""
        conditioning_config = {
            "genre_vocab_size": self.config.conditioning.genre_vocab_size,
            "mood_vocab_size": self.config.conditioning.mood_vocab_size,
            "tempo_bins": self.config.conditioning.tempo_bins,
            "tempo_min": self.config.conditioning.tempo_min,
            "tempo_max": self.config.conditioning.tempo_max,
            "duration_max": self.config.conditioning.duration_max,
            "instrument_vocab_size": self.config.conditioning.instrument_vocab_size,
            "embedding_dim": self.config.conditioning.genre_embedding_dim,
            "dropout": self.config.conditioning.conditioning_dropout,
            "use_genre": self.config.conditioning.use_genre,
            "use_mood": self.config.conditioning.use_mood,
            "use_tempo": self.config.conditioning.use_tempo,
            "use_duration": self.config.conditioning.use_duration,
            "use_instruments": self.config.conditioning.use_instruments,
            "fusion_method": self.config.conditioning.conditioning_fusion,
        }

        self.multimodal_encoder = MultiModalEncoder(
            t5_model_name=self.config.t5.model_name,
            freeze_t5=self.config.t5.freeze_encoder,
            max_text_length=self.config.t5.max_text_length,
            conditioning_config=conditioning_config,
            output_projection_dim=self.config.transformer.hidden_size,
        )

    def _init_transformer(self):
        """Initialize the main transformer model."""
        self.transformer = MusicGenTransformer(self.config.transformer)

    def _init_audio_tokenizer(self):
        """Initialize the audio tokenizer."""
        self.audio_tokenizer = EnCodecTokenizer(
            model_name=self.config.encodec.model_name,
            sample_rate=self.config.encodec.sample_rate,
            bandwidth=self.config.encodec.bandwidth,
            normalize=self.config.encodec.normalize,
        )

        # Update transformer vocab size to match audio tokenizer
        if hasattr(self.audio_tokenizer, "codebook_size"):
            self.config.transformer.vocab_size = self.audio_tokenizer.codebook_size

    def encode_audio(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode audio to discrete tokens."""
        with torch.no_grad():
            tokens = self.audio_tokenizer.tokenize(audio, sample_rate)
        return tokens

    def decode_audio(
        self,
        tokens: torch.Tensor,
        time_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode tokens back to audio."""
        with torch.no_grad():
            if time_frames is None:
                # Infer time frames from sequence length
                seq_len = tokens.shape[-1]
                time_frames = seq_len // self.audio_tokenizer.num_quantizers

            audio = self.audio_tokenizer.detokenize(tokens, time_frames)
        return audio

    def prepare_inputs(
        self,
        texts: List[str],
        device: torch.device,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare and encode inputs for the model."""

        # Encode multimodal inputs
        encoder_outputs = self.multimodal_encoder(
            texts=texts,
            device=device,
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
            duration=duration,
            instrument_ids=instrument_ids,
        )

        return encoder_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """Forward pass of the complete model."""

        device = input_ids.device
        input_ids.shape[0]

        # Prepare encoder inputs if texts are provided
        encoder_hidden_states = None
        encoder_attention_mask = None
        conditioning_embeddings = None

        if texts is not None:
            encoder_outputs = self.prepare_inputs(
                texts=texts,
                device=device,
                genre_ids=genre_ids,
                mood_ids=mood_ids,
                tempo=tempo,
                duration=duration,
                instrument_ids=instrument_ids,
            )

            encoder_hidden_states = encoder_outputs["text_hidden_states"]
            encoder_attention_mask = encoder_outputs["text_attention_mask"]
            conditioning_embeddings = encoder_outputs["conditioning_embeddings"]

        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            conditioning_embeddings=conditioning_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = transformer_outputs["logits"]

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + tuple(transformer_outputs.values())[1:]
            if loss is not None:
                output = (loss,) + output
            return output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": transformer_outputs.get("past_key_values"),
            "hidden_states": transformer_outputs.get("hidden_states"),
            "all_hidden_states": transformer_outputs.get("all_hidden_states"),
        }

    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.1,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate music tokens from text prompts.

        Args:
            texts: List of text prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty factor
            genre_ids: Genre conditioning
            mood_ids: Mood conditioning
            tempo: Tempo conditioning
            duration: Duration conditioning
            instrument_ids: Instrument conditioning
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            device: Device to run generation on

        Returns:
            Generated token sequences
        """

        if device is None:
            device = next(self.parameters()).device

        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        batch_size = len(texts)

        # Prepare encoder inputs
        encoder_outputs = self.prepare_inputs(
            texts=texts,
            device=device,
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
            duration=duration,
            instrument_ids=instrument_ids,
        )

        encoder_hidden_states = encoder_outputs["text_hidden_states"]
        encoder_attention_mask = encoder_outputs["text_attention_mask"]
        conditioning_embeddings = encoder_outputs["conditioning_embeddings"]

        # Initialize generation with BOS token
        input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        # Use beam search if num_beams > 1
        if num_beams > 1:
            return self._beam_search_generate(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                conditioning_embeddings=conditioning_embeddings,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        # Generation loop for greedy/sampling
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            # Forward pass
            outputs = self.transformer(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                conditioning_embeddings=conditioning_embeddings if step == 0 else None,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]  # Get last token logits
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        if logits[i, previous_token] < 0:
                            logits[i, previous_token] *= repetition_penalty
                        else:
                            logits[i, previous_token] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Sampling
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                    logits = logits_filtered

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i][indices_to_remove] = float("-inf")

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Check for EOS tokens
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return input_ids

    @torch.no_grad()
    def generate_audio(
        self, texts: List[str], duration: float = 10.0, **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate audio from text prompts.

        Args:
            texts: Text prompts
            duration: Target duration in seconds
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated audio tensor
        """

        # Calculate target sequence length
        target_length = self.audio_tokenizer.get_sequence_length(duration)
        generation_kwargs.setdefault("max_length", target_length)

        # Generate tokens
        tokens = self.generate(texts, **generation_kwargs)

        # Convert to audio
        audio = self.decode_audio(tokens)

        return audio

    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        conditioning_embeddings: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        num_beams: int = 4,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        length_penalty: float = 1.0,
        diversity_penalty: float = 0.0,
        early_stopping: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate using beam search."""

        # Create beam search configuration
        beam_config = BeamSearchConfig(
            num_beams=num_beams,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            diversity_penalty=diversity_penalty,
            early_stopping=early_stopping,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=self.bos_token_id,
        )

        # Perform beam search
        generated_sequences, scores = beam_search_generate(
            model=self,
            input_ids=input_ids,
            config=beam_config,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            conditioning_embeddings=conditioning_embeddings,
        )

        return generated_sequences

    @torch.no_grad()
    def generate_with_beam_search(
        self,
        texts: List[str],
        num_beams: int = 4,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        length_penalty: float = 1.0,
        diversity_penalty: float = 0.0,
        early_stopping: bool = True,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate music tokens using beam search.

        Args:
            texts: List of text prompts
            num_beams: Number of beams for beam search
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Repetition penalty factor
            length_penalty: Length penalty factor for beam search
            diversity_penalty: Diversity penalty for diverse beam search
            early_stopping: Whether to stop early when EOS is found
            genre_ids: Genre conditioning
            mood_ids: Mood conditioning
            tempo: Tempo conditioning
            duration: Duration conditioning
            instrument_ids: Instrument conditioning
            device: Device to run generation on
            **kwargs: Additional arguments

        Returns:
            Generated token sequences
        """

        if device is None:
            device = next(self.parameters()).device

        batch_size = len(texts)

        # Prepare encoder inputs
        encoder_outputs = self.prepare_inputs(
            texts=texts,
            device=device,
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
            duration=duration,
            instrument_ids=instrument_ids,
        )

        encoder_hidden_states = encoder_outputs["text_hidden_states"]
        encoder_attention_mask = encoder_outputs["text_attention_mask"]
        conditioning_embeddings = encoder_outputs["conditioning_embeddings"]

        # Initialize generation with BOS token
        input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        # Generate using beam search
        return self._beam_search_generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            conditioning_embeddings=conditioning_embeddings,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            length_penalty=length_penalty,
            diversity_penalty=diversity_penalty,
            early_stopping=early_stopping,
            **kwargs,
        )

    @torch.no_grad()
    def generate_audio_with_beam_search(
        self, texts: List[str], duration: float = 10.0, num_beams: int = 4, **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate audio from text prompts using beam search.

        Args:
            texts: Text prompts
            duration: Target duration in seconds
            num_beams: Number of beams for beam search
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated audio tensor
        """

        # Calculate target sequence length
        target_length = self.audio_tokenizer.get_sequence_length(duration)
        generation_kwargs.setdefault("max_length", target_length)

        # Generate tokens using beam search
        tokens = self.generate_with_beam_search(texts, num_beams=num_beams, **generation_kwargs)

        # Convert to audio
        audio = self.decode_audio(tokens)

        return audio

    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration."""
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save configuration
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from saved weights and configuration."""
        import json
        import os

        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = MusicGenConfig(**config_dict)

        # Create model
        model = cls(config, **kwargs)

        # Load weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model


def create_musicgen_model(model_size: str = "base", **config_overrides) -> MusicGenModel:
    """
    Factory function to create MusicGen model with predefined configurations.

    Args:
        model_size: Model size ("small", "base", "large")
        **config_overrides: Configuration overrides

    Returns:
        MusicGenModel instance
    """

    if model_size == "small":
        base_config = {
            "transformer": {
                "hidden_size": 512,
                "num_layers": 8,
                "num_heads": 8,
                "intermediate_size": 2048,
            }
        }
    elif model_size == "base":
        base_config = {
            "transformer": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
            }
        }
    elif model_size == "large":
        base_config = {
            "transformer": {
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "intermediate_size": 4096,
            }
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Merge with overrides
    config_dict = {**base_config, **config_overrides}
    config = MusicGenConfig(**config_dict)

    return MusicGenModel(config)


# Performance optimization methods
def compile_model(
    model: MusicGenModel, mode: str = "max-autotune", dynamic: bool = False
) -> MusicGenModel:
    """Compile the model for better performance using PyTorch 2.0+."""
    if not hasattr(torch, "compile"):
        logger.warning("PyTorch compilation not available, requires PyTorch 2.0+")
        return model

    try:
        logger.info(f"Compiling model with mode: {mode}")

        # Compile transformer
        model.transformer = torch.compile(
            model.transformer,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,  # Allow partial compilation
        )

        # Compile multimodal encoder if possible
        try:
            model.multimodal_encoder = torch.compile(
                model.multimodal_encoder, mode=mode, dynamic=dynamic, fullgraph=False
            )
        except Exception as e:
            logger.warning(f"Could not compile multimodal encoder: {e}")

        # Compile audio tokenizer if possible
        try:
            model.audio_tokenizer = torch.compile(
                model.audio_tokenizer, mode=mode, dynamic=dynamic, fullgraph=False
            )
        except Exception as e:
            logger.warning(f"Could not compile audio tokenizer: {e}")

        logger.info("Model compilation completed successfully")
        return model

    except Exception as e:
        logger.error(f"Model compilation failed: {e}")
        return model


def enable_mixed_precision(model: MusicGenModel) -> MusicGenModel:
    """Enable mixed precision training/inference."""
    logger.info("Enabling mixed precision")

    # Convert model to half precision where possible
    for module in [model.transformer, model.multimodal_encoder]:
        try:
            # Convert to half precision but keep embeddings in float32
            for name, param in module.named_parameters():
                if "embed" not in name.lower() and "norm" not in name.lower():
                    param.data = param.data.half()
        except Exception as e:
            logger.warning(f"Could not enable mixed precision for {type(module).__name__}: {e}")

    return model


def optimize_for_inference(model: MusicGenModel) -> MusicGenModel:
    """Optimize model for inference performance."""
    logger.info("Optimizing model for inference")

    # Set to eval mode
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Enable optimized attention if available
    try:
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Use optimized attention implementation
            for module in model.modules():
                if hasattr(module, "use_optimized_attention"):
                    module.use_optimized_attention = True
    except Exception as e:
        logger.warning(f"Could not enable optimized attention: {e}")

    return model


def optimize_memory_usage(model: MusicGenModel) -> MusicGenModel:
    """Optimize model for memory efficiency."""
    logger.info("Optimizing memory usage")

    # Enable gradient checkpointing if available
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
        elif hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True

    # Clear unused caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model
