"""
Core transformer model for MusicGen with cross-attention to text.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from torch.utils.checkpoint import checkpoint

from .config import TransformerConfig


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional encoding."""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional cross-attention and RoPE."""
    
    def __init__(self, config: TransformerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Query projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Key and value projections
        if is_cross_attention:
            # Cross-attention: K,V from text encoder
            kv_dim = config.text_hidden_size
        else:
            # Self-attention: K,V from same sequence
            kv_dim = self.hidden_size
            
        self.k_proj = nn.Linear(kv_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(kv_dim, self.hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Rotary positional encoding (only for self-attention)
        if config.use_rotary_positional_encoding and not is_cross_attention:
            self.rotary_emb = RotaryPositionalEncoding(self.head_dim, config.max_sequence_length)
        else:
            self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of multi-head attention."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query projection
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        
        # Key and value projections
        if self.is_cross_attention:
            # Cross-attention: use provided key_value_states
            if key_value_states is None:
                raise ValueError("key_value_states must be provided for cross-attention")
            
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            kv_seq_len = key_value_states.shape[1]
        else:
            # Self-attention: use hidden_states
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            kv_seq_len = seq_len
        
        key_states = key_states.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Handle past key values for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            kv_seq_len = key_states.shape[2]
        
        # Apply rotary positional encoding (self-attention only)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(query_states, max(seq_len, kv_seq_len))
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Compute attention
        if self.config.use_scaled_dot_product_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's optimized implementation when available
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # Manual implementation
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Prepare key-value cache for next step
        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)
        
        return attn_output, present_key_value


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using SwiGLU activation."""
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        output = self.down_proj(intermediate)
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention, cross-attention, and feed-forward."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Self-attention
        self.self_attn = MultiHeadAttention(config, is_cross_attention=False)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Cross-attention (if this layer supports it)
        self.has_cross_attention = layer_idx in config.cross_attention_layers
        if self.has_cross_attention:
            self.cross_attn = MultiHeadAttention(config, is_cross_attention=True)
            self.cross_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of transformer layer."""
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # Cross-attention
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)
            hidden_states, _ = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                use_cache=False,  # Don't cache cross-attention
            )
            hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class MusicGenTransformer(nn.Module):
    """Main transformer model for music generation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional encoding
        if config.use_learned_positional_encoding and not config.use_rotary_positional_encoding:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = None
        
        # Conditioning projection
        if config.use_conditioning and config.conditioning_dim > 0:
            self.conditioning_proj = nn.Linear(config.conditioning_dim, config.hidden_size)
        else:
            self.conditioning_proj = None
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx) for layer_idx in range(config.num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following standard practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module):
        """Set input embedding layer."""
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        conditioning_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass of the transformer."""
        
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Positional embeddings
        if self.embed_positions is not None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.embed_positions(position_ids)
            inputs_embeds = inputs_embeds + position_embeddings
        
        # Add conditioning embeddings
        if self.conditioning_proj is not None and conditioning_embeddings is not None:
            conditioning_projected = self.conditioning_proj(conditioning_embeddings)
            # Add conditioning to the first token (similar to BERT's [CLS])
            inputs_embeds[:, 0] = inputs_embeds[:, 0] + conditioning_projected
        
        hidden_states = self.dropout(inputs_embeds)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        
        # Convert attention mask to bias format
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Process through transformer layers
        all_hidden_states = []
        present_key_values = []
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Get past key value for this layer
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.config.gradient_checkpointing and self.training:
                hidden_states, present_key_value = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    use_cache,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if output_hidden_states:
            output["all_hidden_states"] = all_hidden_states
        
        if use_cache:
            output["past_key_values"] = tuple(present_key_values)
        
        return output