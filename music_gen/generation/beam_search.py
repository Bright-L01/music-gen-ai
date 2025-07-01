"""
Beam search implementation for MusicGen generation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchConfig:
    """Configuration for beam search generation."""

    num_beams: int = 4
    max_length: int = 1024
    min_length: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    diversity_penalty: float = 0.0
    num_beam_groups: int = 1
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1


class BeamHypothesis:
    """Single beam hypothesis for beam search."""

    def __init__(
        self,
        tokens: torch.Tensor,
        score: float,
        past_key_values: Optional[Tuple] = None,
    ):
        self.tokens = tokens
        self.score = score
        self.past_key_values = past_key_values

    def __len__(self):
        return len(self.tokens)

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def add_token(
        self,
        token_id: int,
        log_prob: float,
        past_key_values: Optional[Tuple] = None,
    ) -> "BeamHypothesis":
        """Add a token to this hypothesis."""
        new_tokens = torch.cat([self.tokens, torch.tensor([token_id], device=self.tokens.device)])
        new_score = self.score + log_prob
        return BeamHypothesis(new_tokens, new_score, past_key_values)

    def get_length_normalized_score(self, length_penalty: float) -> float:
        """Get length-normalized score."""
        length = len(self.tokens)
        if length_penalty == 0.0:
            return self.score
        return self.score / (length**length_penalty)


class BeamSearcher:
    """Beam search implementation for sequence generation."""

    def __init__(self, config: BeamSearchConfig):
        self.config = config
        self.num_beams = config.num_beams
        self.max_length = config.max_length
        self.min_length = config.min_length
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.repetition_penalty = config.repetition_penalty
        self.length_penalty = config.length_penalty
        self.diversity_penalty = config.diversity_penalty
        self.early_stopping = config.early_stopping
        self.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id

        # For diverse beam search
        self.num_beam_groups = config.num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        if self.num_beams % self.num_beam_groups != 0:
            raise ValueError(
                f"num_beams ({self.num_beams}) must be divisible by num_beam_groups ({self.num_beam_groups})"
            )

    @torch.no_grad()
    def search(
        self,
        model,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        conditioning_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform beam search generation.

        Args:
            model: The model to use for generation
            input_ids: Initial input token IDs [batch_size, seq_len]
            encoder_hidden_states: Encoder outputs for cross-attention
            encoder_attention_mask: Encoder attention mask
            conditioning_embeddings: Conditioning embeddings
            attention_mask: Decoder attention mask
            **model_kwargs: Additional model arguments

        Returns:
            Tuple of (generated_sequences, scores)
        """

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Expand inputs for beam search
        input_ids = self._expand_for_beams(input_ids, self.num_beams)

        if encoder_hidden_states is not None:
            encoder_hidden_states = self._expand_for_beams(encoder_hidden_states, self.num_beams)
        if encoder_attention_mask is not None:
            encoder_attention_mask = self._expand_for_beams(encoder_attention_mask, self.num_beams)
        if conditioning_embeddings is not None:
            conditioning_embeddings = self._expand_for_beams(
                conditioning_embeddings, self.num_beams
            )
        if attention_mask is not None:
            attention_mask = self._expand_for_beams(attention_mask, self.num_beams)

        # Initialize beam search state
        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)  # [batch_size * num_beams]

        # Track finished sequences
        done = [False for _ in range(batch_size)]
        generated_hyps = [[] for _ in range(batch_size)]

        # Generation loop
        cur_len = input_ids.shape[-1]
        past_key_values = None

        while cur_len < self.max_length:
            # Get model outputs
            model_inputs = self._prepare_model_inputs(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                conditioning_embeddings=(
                    conditioning_embeddings if cur_len == input_ids.shape[-1] else None
                ),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **model_kwargs,
            )

            outputs = model(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]  # [batch_size * num_beams, vocab_size]
            past_key_values = outputs.get("past_key_values")

            # Process logits
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Apply penalties and filtering
            next_token_scores = self._postprocess_next_token_scores(
                scores=next_token_scores,
                input_ids=input_ids,
                cur_len=cur_len,
                batch_size=batch_size,
            )

            # Beam search step
            if self.num_beam_groups > 1:
                # Diverse beam search
                next_token_scores, next_tokens, beam_indices = self._diverse_beam_search_step(
                    next_token_scores, beam_scores, cur_len, batch_size
                )
            else:
                # Standard beam search
                next_token_scores, next_tokens, beam_indices = self._beam_search_step(
                    next_token_scores, beam_scores, batch_size
                )

            # Update beam scores
            beam_scores = next_token_scores

            # Reorder sequences based on beam indices
            input_ids = input_ids[beam_indices]
            if past_key_values is not None:
                past_key_values = self._reorder_cache(past_key_values, beam_indices)

            # Append tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype
                        ),
                    ],
                    dim=-1,
                )

            cur_len += 1

            # Check for EOS and early stopping
            if self.early_stopping:
                done, generated_hyps = self._check_early_stopping(
                    input_ids, beam_scores, cur_len, batch_size, done, generated_hyps
                )
                if all(done):
                    break

        # Finalize generation
        return self._finalize_generation(
            input_ids, beam_scores, batch_size, done, generated_hyps, cur_len
        )

    def _expand_for_beams(self, tensor: torch.Tensor, num_beams: int) -> torch.Tensor:
        """Expand tensor for beam search."""
        if tensor is None:
            return None

        batch_size = tensor.shape[0]
        expanded_tensor = (
            tensor.unsqueeze(1)
            .expand(batch_size, num_beams, *tensor.shape[1:])
            .contiguous()
            .view(batch_size * num_beams, *tensor.shape[1:])
        )

        return expanded_tensor

    def _prepare_model_inputs(self, **kwargs) -> Dict:
        """Prepare inputs for model forward pass."""
        model_inputs = {}
        for key, value in kwargs.items():
            if value is not None:
                model_inputs[key] = value
        return model_inputs

    def _postprocess_next_token_scores(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
        cur_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Apply various penalties and filtering to logits."""

        # Apply temperature
        if self.temperature != 1.0:
            scores = scores / self.temperature

        # Apply repetition penalty
        if self.repetition_penalty != 1.0:
            scores = self._apply_repetition_penalty(scores, input_ids)

        # Enforce minimum length
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")

        # Apply n-gram repetition penalty
        if self.no_repeat_ngram_size > 0:
            scores = self._apply_ngram_penalty(scores, input_ids, cur_len)

        # Apply top-k filtering
        if self.top_k > 0:
            scores = self._apply_top_k_filtering(scores)

        # Apply top-p filtering
        if self.top_p < 1.0:
            scores = self._apply_top_p_filtering(scores)

        return scores

    def _apply_repetition_penalty(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply repetition penalty."""
        for i in range(scores.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                if scores[i, previous_token] < 0:
                    scores[i, previous_token] *= self.repetition_penalty
                else:
                    scores[i, previous_token] /= self.repetition_penalty
        return scores

    def _apply_ngram_penalty(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
        cur_len: int,
    ) -> torch.Tensor:
        """Apply n-gram repetition penalty."""
        if cur_len + 1 < self.no_repeat_ngram_size:
            return scores

        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            # Get the last n-1 tokens
            ngram_prefix = input_ids[batch_idx, -(self.no_repeat_ngram_size - 1) :].tolist()

            # Find all n-grams in the sequence
            banned_tokens = set()
            for i in range(len(input_ids[batch_idx]) - self.no_repeat_ngram_size + 1):
                ngram = input_ids[batch_idx, i : i + self.no_repeat_ngram_size].tolist()
                if ngram[:-1] == ngram_prefix:
                    banned_tokens.add(ngram[-1])

            # Ban repeated tokens
            for token in banned_tokens:
                scores[batch_idx, token] = -float("inf")

        return scores

    def _apply_top_k_filtering(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply top-k filtering."""
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        scores_filtered = torch.full_like(scores, -float("inf"))
        scores_filtered.scatter_(-1, top_k_indices, top_k_scores)
        return scores_filtered

    def _apply_top_p_filtering(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply top-p (nucleus) filtering."""
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set filtered logits to -inf
        for i in range(scores.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            scores[i][indices_to_remove] = -float("inf")

        return scores

    def _beam_search_step(
        self,
        next_token_scores: torch.Tensor,
        beam_scores: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single beam search step."""

        vocab_size = next_token_scores.shape[-1]

        # Add beam scores to next token scores
        next_scores = next_token_scores + beam_scores[:, None]

        # Reshape to [batch_size, num_beams * vocab_size]
        next_scores = next_scores.view(batch_size, self.num_beams * vocab_size)

        # Get top 2*num_beams scores
        next_scores, next_tokens = torch.topk(
            next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
        )

        # Convert token indices back to beam and token indices
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # Select best beams
        beam_outputs = []
        for batch_idx in range(batch_size):
            beam_idx = 0
            for beam_score, beam_token, beam_idx_candidate in zip(
                next_scores[batch_idx], next_tokens[batch_idx], next_indices[batch_idx]
            ):
                batch_beam_idx = batch_idx * self.num_beams + beam_idx_candidate

                if beam_token.item() == self.eos_token_id:
                    # End of sequence
                    continue

                beam_outputs.append((beam_score, beam_token, batch_beam_idx))
                beam_idx += 1

                if beam_idx >= self.num_beams:
                    break

        # Extract outputs
        beam_scores_new = torch.tensor(
            [x[0] for x in beam_outputs], device=next_token_scores.device
        )
        beam_tokens = torch.tensor([x[1] for x in beam_outputs], device=next_token_scores.device)
        beam_indices = torch.tensor([x[2] for x in beam_outputs], device=next_token_scores.device)

        return beam_scores_new, beam_tokens, beam_indices

    def _diverse_beam_search_step(
        self,
        next_token_scores: torch.Tensor,
        beam_scores: torch.Tensor,
        cur_len: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform diverse beam search step."""
        # For simplicity, fall back to standard beam search
        # Full diverse beam search implementation would be more complex
        return self._beam_search_step(next_token_scores, beam_scores, batch_size)

    def _reorder_cache(
        self,
        past_key_values: Tuple,
        beam_indices: torch.Tensor,
    ) -> Tuple:
        """Reorder past key values according to beam indices."""
        reordered_past = []
        for layer_past in past_key_values:
            if isinstance(layer_past, tuple):
                # (key, value) tuple
                reordered_layer = tuple(
                    past_state.index_select(0, beam_indices) for past_state in layer_past
                )
            else:
                # Single tensor
                reordered_layer = layer_past.index_select(0, beam_indices)
            reordered_past.append(reordered_layer)
        return tuple(reordered_past)

    def _check_early_stopping(
        self,
        input_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        cur_len: int,
        batch_size: int,
        done: List[bool],
        generated_hyps: List[List[BeamHypothesis]],
    ) -> Tuple[List[bool], List[List[BeamHypothesis]]]:
        """Check for early stopping conditions."""

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # Check if any beam ended with EOS
            for beam_id in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_id
                beam_tokens = input_ids[effective_beam_id]

                if beam_tokens[-1] == self.eos_token_id:
                    # This beam finished
                    score = beam_scores[effective_beam_id].item()
                    hyp = BeamHypothesis(beam_tokens[:-1], score)  # Remove EOS
                    generated_hyps[batch_idx].append(hyp)

            # Check if we should stop for this batch
            if len(generated_hyps[batch_idx]) >= self.num_beams:
                done[batch_idx] = True

        return done, generated_hyps

    def _finalize_generation(
        self,
        input_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        batch_size: int,
        done: List[bool],
        generated_hyps: List[List[BeamHypothesis]],
        cur_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finalize generation and return best sequences."""

        # Add remaining beams to hypotheses
        for batch_idx in range(batch_size):
            for beam_id in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_id
                beam_tokens = input_ids[effective_beam_id]
                score = beam_scores[effective_beam_id].item()

                hyp = BeamHypothesis(beam_tokens, score)
                generated_hyps[batch_idx].append(hyp)

        # Select best hypothesis for each batch
        output_batch_size = batch_size
        sent_lengths = torch.zeros(output_batch_size, dtype=torch.long)
        best_scores = torch.zeros(output_batch_size)

        # Sort hypotheses by score
        for batch_idx, hyps in enumerate(generated_hyps):
            sorted_hyps = sorted(
                hyps, key=lambda x: x.get_length_normalized_score(self.length_penalty), reverse=True
            )
            best_hyp = sorted_hyps[0]
            sent_lengths[batch_idx] = len(best_hyp.tokens)
            best_scores[batch_idx] = best_hyp.score

        # Create output tensor
        max_len = sent_lengths.max().item()
        decoded = torch.full(
            (output_batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        # Fill with best sequences
        for batch_idx, hyps in enumerate(generated_hyps):
            sorted_hyps = sorted(
                hyps, key=lambda x: x.get_length_normalized_score(self.length_penalty), reverse=True
            )
            best_hyp = sorted_hyps[0]
            decoded[batch_idx, : len(best_hyp.tokens)] = best_hyp.tokens

        return decoded, best_scores


def beam_search_generate(
    model,
    input_ids: torch.Tensor,
    config: BeamSearchConfig,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    conditioning_embeddings: Optional[torch.Tensor] = None,
    **model_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for beam search generation.

    Args:
        model: The model to use for generation
        input_ids: Initial input token IDs
        config: Beam search configuration
        encoder_hidden_states: Encoder outputs
        encoder_attention_mask: Encoder attention mask
        conditioning_embeddings: Conditioning embeddings
        **model_kwargs: Additional model arguments

    Returns:
        Tuple of (generated_sequences, scores)
    """

    searcher = BeamSearcher(config)
    return searcher.search(
        model=model,
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        conditioning_embeddings=conditioning_embeddings,
        **model_kwargs,
    )
