"""Instrument conditioning system for multi-instrument generation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from music_gen.models.multi_instrument.config import MultiInstrumentConfig


class InstrumentEmbedding(nn.Module):
    """Learnable embeddings for different instruments."""

    def __init__(self, config: MultiInstrumentConfig):
        super().__init__()
        self.config = config

        # Create instrument name to index mapping
        self.instrument_names = config.get_instrument_names()
        self.name_to_idx = {name: idx for idx, name in enumerate(self.instrument_names)}

        # Instrument embeddings
        self.instrument_embeddings = nn.Embedding(
            len(self.instrument_names) + 1, config.instrument_embedding_dim  # +1 for unknown/mixed
        )

        # Learnable features for each instrument
        self.register_buffer("frequency_ranges", self._create_frequency_features())
        self.register_buffer("instrument_properties", self._create_property_features())

        # Feature projection
        feature_dim = self.frequency_ranges.size(1) + self.instrument_properties.size(1)
        self.feature_projection = nn.Linear(feature_dim, config.instrument_embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(config.instrument_dropout)

    def _create_frequency_features(self) -> torch.Tensor:
        """Create frequency range features for instruments."""
        features = []
        for name in self.instrument_names:
            inst_config = self.config.get_instrument_config(name)
            freq_min, freq_max = inst_config.frequency_range
            oct_min, oct_max = inst_config.typical_octave_range

            # Normalize frequency to log scale
            log_freq_min = np.log(freq_min + 1e-6)
            log_freq_max = np.log(freq_max + 1e-6)

            features.append(
                [
                    log_freq_min / 10.0,  # Normalized log min freq
                    log_freq_max / 10.0,  # Normalized log max freq
                    (oct_max - oct_min) / 10.0,  # Octave range
                    (oct_min + oct_max) / 20.0,  # Center octave
                ]
            )

        # Add features for unknown/mixed
        features.append([0.0, 1.0, 1.0, 0.5])

        return torch.tensor(features, dtype=torch.float32)

    def _create_property_features(self) -> torch.Tensor:
        """Create property features for instruments."""
        features = []
        for name in self.instrument_names:
            inst_config = self.config.get_instrument_config(name)
            features.append(
                [
                    float(inst_config.polyphonic),
                    float(inst_config.percussion),
                    float(inst_config.sustained),
                    inst_config.default_volume,
                    (inst_config.default_pan + 1.0) / 2.0,  # Normalize to [0, 1]
                ]
            )

        # Add features for unknown/mixed
        features.append([0.5, 0.0, 0.5, 0.7, 0.5])

        return torch.tensor(features, dtype=torch.float32)

    def forward(
        self,
        instrument_names: Optional[List[str]] = None,
        instrument_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get instrument embeddings.

        Args:
            instrument_names: List of instrument names
            instrument_indices: Tensor of instrument indices

        Returns:
            Instrument embeddings [batch_size, num_instruments, embedding_dim]
        """
        if instrument_names is not None:
            # Convert names to indices
            indices = []
            for name in instrument_names:
                idx = self.name_to_idx.get(name.lower(), len(self.instrument_names))
                indices.append(idx)
            instrument_indices = torch.tensor(
                indices, device=self.instrument_embeddings.weight.device
            )

        if instrument_indices is None:
            raise ValueError("Either instrument_names or instrument_indices must be provided")

        # Get base embeddings
        embeddings = self.instrument_embeddings(instrument_indices)

        # Get instrument features
        freq_features = self.frequency_ranges[instrument_indices]
        prop_features = self.instrument_properties[instrument_indices]

        # Combine features
        all_features = torch.cat([freq_features, prop_features], dim=-1)
        feature_embeddings = self.feature_projection(all_features)

        # Combine embeddings
        combined = embeddings + feature_embeddings
        return self.dropout(combined)


class InstrumentConditioner(nn.Module):
    """Conditioning module for multi-instrument generation."""

    def __init__(self, config: MultiInstrumentConfig):
        super().__init__()
        self.config = config

        # Instrument embedding
        self.instrument_embedding = InstrumentEmbedding(config)

        # Track-level conditioning
        self.track_embeddings = nn.Embedding(config.max_tracks, config.hidden_size)

        # Mixing parameters prediction
        if config.use_automatic_mixing:
            self.mixing_predictor = nn.Sequential(
                nn.Linear(config.instrument_embedding_dim, config.mixing_latent_dim),
                nn.ReLU(),
                nn.Linear(config.mixing_latent_dim, config.mixing_latent_dim),
                nn.ReLU(),
                nn.Linear(config.mixing_latent_dim, 5),  # volume, pan, reverb, eq_low, eq_high
            )

        # Cross-instrument attention
        if config.use_instrument_attention:
            self.instrument_attention = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )

        # Projection to model dimension
        self.projection = nn.Linear(
            config.instrument_embedding_dim + config.hidden_size, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        instrument_names: Optional[List[List[str]]] = None,
        instrument_indices: Optional[torch.Tensor] = None,
        track_indices: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply instrument conditioning to hidden states.

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            instrument_names: List of instrument names per track per batch
            instrument_indices: Instrument indices [batch_size, num_tracks]
            track_indices: Track indices [batch_size, num_tracks]
            layer_idx: Current transformer layer index

        Returns:
            Conditioned hidden states and mixing parameters
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Get instrument embeddings
        if instrument_names is not None:
            # Flatten instrument names for embedding lookup
            flat_names = [name for batch_names in instrument_names for name in batch_names]
            inst_embeddings = self.instrument_embedding(instrument_names=flat_names)
            num_tracks = len(instrument_names[0])
            inst_embeddings = inst_embeddings.view(batch_size, num_tracks, -1)
        else:
            inst_embeddings = self.instrument_embedding(instrument_indices=instrument_indices)
            num_tracks = instrument_indices.size(1)

        # Get track embeddings
        if track_indices is None:
            track_indices = (
                torch.arange(num_tracks, device=device).unsqueeze(0).expand(batch_size, -1)
            )
        track_embeddings = self.track_embeddings(track_indices)

        # Combine instrument and track embeddings
        combined_embeddings = torch.cat([inst_embeddings, track_embeddings], dim=-1)
        conditioning = self.projection(combined_embeddings)

        # Apply instrument attention if at specified layers
        if (
            self.config.use_instrument_attention
            and layer_idx is not None
            and layer_idx in self.config.instrument_cross_attention_layers
        ):

            # Expand conditioning to sequence length
            conditioning_expanded = conditioning.unsqueeze(2).expand(-1, -1, seq_len, -1)
            conditioning_flat = conditioning_expanded.reshape(
                batch_size * num_tracks, seq_len, hidden_size
            )

            # Expand hidden states for each track
            hidden_expanded = hidden_states.unsqueeze(1).expand(-1, num_tracks, -1, -1)
            hidden_flat = hidden_expanded.reshape(batch_size * num_tracks, seq_len, hidden_size)

            # Apply cross-attention
            attended, _ = self.instrument_attention(
                hidden_flat, conditioning_flat, conditioning_flat
            )

            # Reshape and average across tracks
            attended = attended.reshape(batch_size, num_tracks, seq_len, hidden_size)
            hidden_states = hidden_states + attended.mean(dim=1)

        # Apply conditioning
        hidden_states = self.layer_norm(
            hidden_states + self.dropout(conditioning.mean(dim=1, keepdim=True))
        )

        # Predict mixing parameters if enabled
        mixing_params = {}
        if self.config.use_automatic_mixing:
            mix_pred = self.mixing_predictor(inst_embeddings)  # [batch_size, num_tracks, 5]
            mixing_params = {
                "volume": torch.sigmoid(mix_pred[..., 0]),  # [0, 1]
                "pan": torch.tanh(mix_pred[..., 1]),  # [-1, 1]
                "reverb": torch.sigmoid(mix_pred[..., 2]),  # [0, 1]
                "eq_low": torch.tanh(mix_pred[..., 3]),  # [-1, 1]
                "eq_high": torch.tanh(mix_pred[..., 4]),  # [-1, 1]
            }

        return hidden_states, mixing_params


class InstrumentClassifier(nn.Module):
    """Classify instruments from audio features."""

    def __init__(self, config: MultiInstrumentConfig):
        super().__init__()
        self.config = config

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=256, stride=128),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=64, stride=32),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=16, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(config.get_instrument_names()) + 1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Classify instruments in audio.

        Args:
            audio: Audio tensor [batch_size, num_samples]

        Returns:
            Instrument logits [batch_size, num_instruments]
        """
        # Add channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Extract features
        features = self.feature_extractor(audio)
        features = features.flatten(1)

        # Classify
        logits = self.classifier(features)
        return logits
