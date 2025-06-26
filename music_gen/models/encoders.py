"""
Text and conditioning encoders for MusicGen.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import T5EncoderModel, T5Tokenizer
import logging

logger = logging.getLogger(__name__)


class T5TextEncoder(nn.Module):
    """T5-based text encoder for converting text prompts to embeddings."""
    
    def __init__(
        self,
        model_name: str = "t5-base",
        freeze_encoder: bool = True,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.max_length = max_length
        
        # Load T5 encoder and tokenizer
        try:
            self.encoder = T5EncoderModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error(f"Failed to load T5 model {model_name}: {e}")
            raise
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        self.hidden_size = self.encoder.config.d_model
        
    def encode_text(
        self,
        texts: List[str],
        device: torch.device,
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode text inputs into embeddings."""
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Encode with T5
        with torch.set_grad_enabled(not self.freeze_encoder):
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        output = {
            "hidden_states": hidden_states,
            "input_ids": input_ids,
        }
        
        if return_attention_mask:
            output["attention_mask"] = attention_mask
        
        return output
    
    def forward(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for text encoding."""
        return self.encode_text(texts, device)


class ConditioningEncoder(nn.Module):
    """Encoder for various conditioning inputs (genre, mood, tempo, etc.)."""
    
    def __init__(
        self,
        genre_vocab_size: int = 50,
        mood_vocab_size: int = 20,
        tempo_bins: int = 100,
        tempo_min: int = 60,
        tempo_max: int = 200,
        duration_max: float = 120.0,
        instrument_vocab_size: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        use_genre: bool = True,
        use_mood: bool = True,
        use_tempo: bool = True,
        use_duration: bool = True,
        use_instruments: bool = False,
        fusion_method: str = "concat",  # concat, add, attention
    ):
        super().__init__()
        
        self.use_genre = use_genre
        self.use_mood = use_mood
        self.use_tempo = use_tempo
        self.use_duration = use_duration
        self.use_instruments = use_instruments
        self.fusion_method = fusion_method
        self.embedding_dim = embedding_dim
        
        # Conditioning embeddings
        if use_genre:
            self.genre_embedding = nn.Embedding(genre_vocab_size, embedding_dim)
        
        if use_mood:
            self.mood_embedding = nn.Embedding(mood_vocab_size, embedding_dim)
        
        if use_tempo:
            self.tempo_bins = tempo_bins
            self.tempo_min = tempo_min
            self.tempo_max = tempo_max
            self.tempo_embedding = nn.Embedding(tempo_bins, embedding_dim)
        
        if use_duration:
            self.duration_max = duration_max
            self.duration_embedding = nn.Linear(1, embedding_dim)
        
        if use_instruments:
            self.instrument_embedding = nn.Embedding(instrument_vocab_size, embedding_dim)
        
        # Calculate output dimension
        self.output_dim = self._calculate_output_dim()
        
        # Fusion layers
        if fusion_method == "attention":
            self.attention_weights = nn.Linear(embedding_dim, 1)
        elif fusion_method == "add":
            # All embeddings must have same dimension for addition
            self.output_dim = embedding_dim
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def _calculate_output_dim(self) -> int:
        """Calculate the output dimension based on active conditioning."""
        if self.fusion_method == "add":
            return self.embedding_dim
        
        # For concat fusion
        total_dim = 0
        if self.use_genre:
            total_dim += self.embedding_dim
        if self.use_mood:
            total_dim += self.embedding_dim
        if self.use_tempo:
            total_dim += self.embedding_dim
        if self.use_duration:
            total_dim += self.embedding_dim
        if self.use_instruments:
            total_dim += self.embedding_dim
        
        return total_dim
    
    def _tempo_to_bin(self, tempo: torch.Tensor) -> torch.Tensor:
        """Convert tempo values to discrete bins."""
        # Clamp tempo to valid range
        tempo = torch.clamp(tempo, self.tempo_min, self.tempo_max)
        
        # Convert to bin indices
        normalized = (tempo - self.tempo_min) / (self.tempo_max - self.tempo_min)
        bins = (normalized * (self.tempo_bins - 1)).long()
        
        return bins
    
    def _duration_to_normalized(self, duration: torch.Tensor) -> torch.Tensor:
        """Normalize duration values."""
        return torch.clamp(duration / self.duration_max, 0.0, 1.0).unsqueeze(-1)
    
    def forward(
        self,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for conditioning encoding."""
        
        embeddings = []
        
        # Genre embedding
        if self.use_genre and genre_ids is not None:
            genre_emb = self.genre_embedding(genre_ids)
            embeddings.append(genre_emb)
        
        # Mood embedding
        if self.use_mood and mood_ids is not None:
            mood_emb = self.mood_embedding(mood_ids)
            embeddings.append(mood_emb)
        
        # Tempo embedding
        if self.use_tempo and tempo is not None:
            tempo_bins = self._tempo_to_bin(tempo)
            tempo_emb = self.tempo_embedding(tempo_bins)
            embeddings.append(tempo_emb)
        
        # Duration embedding
        if self.use_duration and duration is not None:
            duration_norm = self._duration_to_normalized(duration)
            duration_emb = self.duration_embedding(duration_norm)
            embeddings.append(duration_emb)
        
        # Instrument embedding
        if self.use_instruments and instrument_ids is not None:
            instrument_emb = self.instrument_embedding(instrument_ids)
            embeddings.append(instrument_emb)
        
        if not embeddings:
            # No conditioning provided
            batch_size = 1
            if genre_ids is not None:
                batch_size = genre_ids.shape[0]
            elif mood_ids is not None:
                batch_size = mood_ids.shape[0]
            elif tempo is not None:
                batch_size = tempo.shape[0]
            elif duration is not None:
                batch_size = duration.shape[0]
            
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # Fuse embeddings
        if self.fusion_method == "concat":
            conditioning = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == "add":
            conditioning = sum(embeddings) / len(embeddings)
        elif self.fusion_method == "attention":
            # Stack embeddings and apply attention
            stacked = torch.stack(embeddings, dim=1)  # (batch, num_conditions, embed_dim)
            attention_weights = F.softmax(self.attention_weights(stacked), dim=1)
            conditioning = (stacked * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Apply dropout and layer norm
        conditioning = self.dropout(conditioning)
        conditioning = self.layer_norm(conditioning)
        
        return conditioning


class MultiModalEncoder(nn.Module):
    """Combined encoder for text and conditioning inputs."""
    
    def __init__(
        self,
        t5_model_name: str = "t5-base",
        freeze_t5: bool = True,
        max_text_length: int = 512,
        conditioning_config: Optional[Dict[str, Any]] = None,
        output_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Text encoder
        self.text_encoder = T5TextEncoder(
            model_name=t5_model_name,
            freeze_encoder=freeze_t5,
            max_length=max_text_length,
        )
        
        # Conditioning encoder
        if conditioning_config is None:
            conditioning_config = {}
        
        self.conditioning_encoder = ConditioningEncoder(**conditioning_config)
        
        # Optional projection layer to match transformer hidden size
        self.text_projection = None
        self.conditioning_projection = None
        
        if output_projection_dim is not None:
            if self.text_encoder.hidden_size != output_projection_dim:
                self.text_projection = nn.Linear(
                    self.text_encoder.hidden_size,
                    output_projection_dim
                )
            
            if self.conditioning_encoder.output_dim != output_projection_dim:
                self.conditioning_projection = nn.Linear(
                    self.conditioning_encoder.output_dim,
                    output_projection_dim
                )
    
    def forward(
        self,
        texts: List[str],
        device: torch.device,
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal encoding."""
        
        # Encode text
        text_outputs = self.text_encoder(texts, device)
        text_hidden_states = text_outputs["hidden_states"]
        
        # Project text embeddings if needed
        if self.text_projection is not None:
            text_hidden_states = self.text_projection(text_hidden_states)
        
        # Encode conditioning
        conditioning_embeddings = self.conditioning_encoder(
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
            duration=duration,
            instrument_ids=instrument_ids,
        )
        
        # Project conditioning embeddings if needed
        if self.conditioning_projection is not None:
            conditioning_embeddings = self.conditioning_projection(conditioning_embeddings)
        
        return {
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_outputs.get("attention_mask"),
            "conditioning_embeddings": conditioning_embeddings,
            "text_input_ids": text_outputs.get("input_ids"),
        }