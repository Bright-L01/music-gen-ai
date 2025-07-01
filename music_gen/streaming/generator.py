"""
Streaming generation engine for real-time music generation.
"""

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""

    # Chunk parameters
    chunk_duration: float = 1.0  # Duration of each audio chunk in seconds
    overlap_duration: float = 0.25  # Overlap between chunks for smooth transitions
    lookahead_chunks: int = 2  # Number of chunks to generate ahead

    # Generation parameters
    temperature: float = 0.9  # Lower for more stable streaming
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # Quality vs latency trade-offs
    max_latency_ms: int = 500  # Maximum acceptable latency
    quality_mode: str = "balanced"  # "fast", "balanced", "quality"

    # Memory management
    max_context_length: int = 2048  # Maximum tokens to keep in context
    context_window_overlap: int = 256  # Overlap when sliding context window

    # Streaming controls
    enable_interruption: bool = True  # Allow real-time interruption/modification
    adaptive_quality: bool = True  # Adjust quality based on network conditions

    # Buffer management
    buffer_size: int = 8  # Number of chunks to buffer
    min_buffer_size: int = 2  # Minimum buffer before starting playback


class StreamingState:
    """Manages state for streaming generation."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset streaming state."""
        self.generated_tokens = []
        self.past_key_values = None
        self.current_chunk_idx = 0
        self.total_generated_duration = 0.0
        self.generation_start_time = None
        self.last_chunk_time = None
        self.encoder_outputs = None
        self.conditioning_embeddings = None
        self.is_active = False
        self.interrupt_requested = False

    def update_context(self, new_tokens: torch.Tensor, new_past_key_values: Optional[Tuple]):
        """Update generation context with new tokens and cache."""

        # Add new tokens
        if isinstance(new_tokens, torch.Tensor):
            if new_tokens.dim() == 1:
                new_tokens = new_tokens.tolist()
            elif new_tokens.dim() == 2:
                new_tokens = new_tokens[0].tolist()  # Take first batch

        self.generated_tokens.extend(new_tokens)

        # Update past key values
        self.past_key_values = new_past_key_values

        # Manage context window size to prevent memory explosion
        if len(self.generated_tokens) > self.config.max_context_length:
            # Slide the context window
            tokens_to_remove = (
                len(self.generated_tokens)
                - self.config.max_context_length
                + self.config.context_window_overlap
            )
            self.generated_tokens = self.generated_tokens[tokens_to_remove:]

            # Note: In practice, you'd also need to adjust past_key_values accordingly
            # This is a simplified implementation
            logger.debug(f"Sliding context window, removed {tokens_to_remove} tokens")

    def get_current_tokens(self) -> torch.Tensor:
        """Get current token sequence as tensor."""
        if not self.generated_tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(self.generated_tokens, dtype=torch.long)


class StreamingGenerator:
    """Real-time streaming generator for music generation."""

    def __init__(self, model, config: StreamingConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Calculate token parameters based on audio tokenizer
        self.frame_rate = model.audio_tokenizer.frame_rate
        self.num_quantizers = model.audio_tokenizer.num_quantizers

        # Calculate chunk sizes in tokens
        self.chunk_frames = int(config.chunk_duration * self.frame_rate)
        self.chunk_tokens = self.chunk_frames * self.num_quantizers
        self.overlap_frames = int(config.overlap_duration * self.frame_rate)
        self.overlap_tokens = self.overlap_frames * self.num_quantizers

        logger.info(
            f"Streaming config: {config.chunk_duration}s chunks = {self.chunk_tokens} tokens, "
            f"overlap = {self.overlap_tokens} tokens"
        )

        # State management
        self.current_state = StreamingState(config)
        self.generation_thread = None
        self.stop_generation = threading.Event()
        self.chunk_queue = Queue(maxsize=config.buffer_size)

        # Performance tracking
        self.generation_stats = {
            "chunks_generated": 0,
            "total_generation_time": 0.0,
            "average_chunk_time": 0.0,
            "buffer_underruns": 0,
        }

    def prepare_streaming(
        self,
        texts: List[str],
        genre_ids: Optional[torch.Tensor] = None,
        mood_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare for streaming generation."""

        logger.info(f"Preparing streaming for texts: {texts}")

        # Reset state
        self.current_state.reset()
        self.stop_generation.clear()

        # Prepare encoder inputs
        encoder_outputs = self.model.prepare_inputs(
            texts=texts,
            device=self.device,
            genre_ids=genre_ids,
            mood_ids=mood_ids,
            tempo=tempo,
            duration=duration,
            instrument_ids=instrument_ids,
        )

        # Store encoder outputs in state
        self.current_state.encoder_outputs = encoder_outputs
        self.current_state.generation_start_time = time.time()
        self.current_state.is_active = True

        # Initialize generation with BOS token
        initial_tokens = torch.full(
            (1, 1), self.model.bos_token_id, dtype=torch.long, device=self.device
        )

        self.current_state.update_context(initial_tokens[0], None)

        return {
            "status": "prepared",
            "chunk_duration": self.config.chunk_duration,
            "frame_rate": self.frame_rate,
            "expected_latency_ms": self._estimate_latency(),
        }

    def _estimate_latency(self) -> float:
        """Estimate generation latency based on configuration."""
        # Rough estimation based on token generation speed
        # This would be calibrated based on actual hardware performance
        base_latency = 100  # Base model inference latency
        token_latency = self.chunk_tokens * 2  # ~2ms per token (rough estimate)
        return base_latency + token_latency

    def start_streaming(self) -> Iterator[Dict[str, Any]]:
        """Start streaming generation."""

        if not self.current_state.is_active:
            raise RuntimeError("Must call prepare_streaming() first")

        logger.info("Starting streaming generation")

        # Start generation in background thread
        self.generation_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.generation_thread.start()

        # Yield chunks as they become available
        while self.current_state.is_active:
            try:
                # Wait for next chunk with timeout
                chunk_data = self.chunk_queue.get(timeout=self.config.max_latency_ms / 1000.0)

                if chunk_data.get("type") == "chunk":
                    yield chunk_data
                elif chunk_data.get("type") == "end":
                    break
                elif chunk_data.get("type") == "error":
                    raise RuntimeError(chunk_data.get("error", "Generation failed"))

            except Empty:
                # Timeout waiting for chunk
                self.generation_stats["buffer_underruns"] += 1
                logger.warning("Buffer underrun - generation not keeping up with real-time")
                yield {
                    "type": "buffer_underrun",
                    "timestamp": time.time(),
                    "stats": self.generation_stats.copy(),
                }
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self.stop_streaming()
                yield {
                    "type": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                }
                break

    def _generation_worker(self):
        """Background worker for chunk generation."""

        try:
            chunk_idx = 0

            while not self.stop_generation.is_set() and self.current_state.is_active:
                start_time = time.time()

                # Generate next chunk
                chunk_tokens, audio_chunk = self._generate_next_chunk()

                if chunk_tokens is None:
                    # End of generation
                    break

                generation_time = time.time() - start_time
                self.generation_stats["total_generation_time"] += generation_time
                self.generation_stats["chunks_generated"] += 1
                self.generation_stats["average_chunk_time"] = (
                    self.generation_stats["total_generation_time"]
                    / self.generation_stats["chunks_generated"]
                )

                # Create chunk data
                chunk_data = {
                    "type": "chunk",
                    "chunk_idx": chunk_idx,
                    "tokens": chunk_tokens,
                    "audio": audio_chunk,
                    "duration": self.config.chunk_duration,
                    "timestamp": time.time(),
                    "generation_time_ms": generation_time * 1000,
                    "total_duration": self.current_state.total_generated_duration,
                    "stats": self.generation_stats.copy(),
                }

                # Add to queue (this may block if buffer is full)
                if not self.stop_generation.is_set():
                    self.chunk_queue.put(chunk_data)

                chunk_idx += 1
                self.current_state.current_chunk_idx = chunk_idx
                self.current_state.total_generated_duration += self.config.chunk_duration
                self.current_state.last_chunk_time = time.time()

        except Exception as e:
            logger.error(f"Generation worker error: {e}")
            error_data = {
                "type": "error",
                "error": str(e),
                "timestamp": time.time(),
            }
            try:
                self.chunk_queue.put(error_data)
            except:
                pass
        finally:
            # Signal end of generation
            try:
                self.chunk_queue.put({"type": "end", "timestamp": time.time()})
            except:
                pass

    def _generate_next_chunk(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate the next chunk of tokens and audio."""

        # Get current context
        current_tokens = self.current_state.get_current_tokens()

        if len(current_tokens) == 0:
            logger.error("No tokens in context for generation")
            return None, None

        # Prepare input for generation
        input_ids = current_tokens.unsqueeze(0).to(self.device)  # Add batch dimension

        # Generate tokens for this chunk
        chunk_tokens = []
        past_key_values = self.current_state.past_key_values

        for step in range(self.chunk_tokens):
            if self.stop_generation.is_set():
                break

            # Forward pass
            model_inputs = {
                "input_ids": input_ids if past_key_values is None else input_ids[:, -1:],
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            # Add encoder outputs on first step
            if past_key_values is None and self.current_state.encoder_outputs:
                model_inputs.update(
                    {
                        "encoder_hidden_states": self.current_state.encoder_outputs[
                            "text_hidden_states"
                        ],
                        "encoder_attention_mask": self.current_state.encoder_outputs[
                            "text_attention_mask"
                        ],
                        "conditioning_embeddings": self.current_state.encoder_outputs[
                            "conditioning_embeddings"
                        ],
                    }
                )

            with torch.no_grad():
                outputs = self.model.transformer(**model_inputs)

            logits = outputs["logits"][:, -1, :]  # Get last token logits
            past_key_values = outputs["past_key_values"]

            # Apply generation parameters
            logits = self._apply_generation_params(logits, input_ids)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            chunk_tokens.append(next_token.item())

            # Update input for next step
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == self.model.eos_token_id:
                logger.info("Generated EOS token, ending streaming")
                self.current_state.is_active = False
                break

        if not chunk_tokens:
            return None, None

        # Update state with new tokens
        chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=self.device)
        self.current_state.update_context(chunk_tensor, past_key_values)

        # Convert tokens to audio
        try:
            audio_chunk = self._tokens_to_audio_chunk(chunk_tensor)
        except Exception as e:
            logger.error(f"Failed to convert tokens to audio: {e}")
            audio_chunk = None

        return chunk_tensor, audio_chunk

    def _apply_generation_params(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply generation parameters to logits."""

        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply repetition penalty
        if self.config.repetition_penalty != 1.0:
            for previous_token in set(input_ids[0].tolist()):
                if logits[0, previous_token] < 0:
                    logits[0, previous_token] *= self.config.repetition_penalty
                else:
                    logits[0, previous_token] /= self.config.repetition_penalty

        # Apply top-k filtering
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k, dim=-1)
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered

        # Apply top-p filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
            logits[0][indices_to_remove] = float("-inf")

        return logits

    def _tokens_to_audio_chunk(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to audio chunk."""

        # Calculate time frames for this chunk
        chunk_frames = len(tokens) // self.num_quantizers

        if chunk_frames == 0:
            logger.warning("Not enough tokens for audio conversion")
            return torch.zeros(
                1, 1, int(self.config.chunk_duration * self.model.audio_tokenizer.sample_rate)
            )

        # Reshape tokens to codes format
        tokens_reshaped = tokens[: chunk_frames * self.num_quantizers]
        tokens_batch = tokens_reshaped.unsqueeze(0)  # Add batch dimension

        # Convert to audio using the model's audio tokenizer
        audio = self.model.audio_tokenizer.detokenize(tokens_batch, chunk_frames)

        return audio

    def stop_streaming(self):
        """Stop streaming generation."""
        logger.info("Stopping streaming generation")

        self.current_state.is_active = False
        self.stop_generation.set()

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)

        # Clear remaining chunks
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except Empty:
                break

    def pause_streaming(self):
        """Pause streaming generation."""
        self.current_state.is_active = False
        logger.info("Streaming paused")

    def resume_streaming(self):
        """Resume streaming generation."""
        if not self.stop_generation.is_set():
            self.current_state.is_active = True
            logger.info("Streaming resumed")

    def interrupt_and_modify(self, new_prompt: str, **kwargs) -> bool:
        """Interrupt current generation and modify parameters."""
        if not self.config.enable_interruption:
            return False

        logger.info(f"Interrupting generation with new prompt: {new_prompt}")

        # Prepare new encoder outputs
        try:
            new_encoder_outputs = self.model.prepare_inputs(
                texts=[new_prompt], device=self.device, **kwargs
            )

            # Update state
            self.current_state.encoder_outputs = new_encoder_outputs
            self.current_state.interrupt_requested = True

            return True

        except Exception as e:
            logger.error(f"Failed to interrupt generation: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics."""
        current_time = time.time()

        stats = self.generation_stats.copy()
        stats.update(
            {
                "is_active": self.current_state.is_active,
                "current_chunk": self.current_state.current_chunk_idx,
                "total_duration": self.current_state.total_generated_duration,
                "context_length": len(self.current_state.generated_tokens),
                "buffer_size": self.chunk_queue.qsize(),
                "uptime": current_time - (self.current_state.generation_start_time or current_time),
            }
        )

        if stats["chunks_generated"] > 0:
            stats["real_time_factor"] = stats["total_duration"] / stats["total_generation_time"]

        return stats


def create_streaming_generator(
    model, chunk_duration: float = 1.0, quality_mode: str = "balanced", **config_kwargs
) -> StreamingGenerator:
    """Factory function to create a streaming generator with sensible defaults."""

    # Quality presets
    quality_presets = {
        "fast": {
            "chunk_duration": 0.5,
            "temperature": 1.0,
            "top_k": 50,
            "lookahead_chunks": 1,
            "max_latency_ms": 200,
        },
        "balanced": {
            "chunk_duration": 1.0,
            "temperature": 0.9,
            "top_k": 40,
            "lookahead_chunks": 2,
            "max_latency_ms": 500,
        },
        "quality": {
            "chunk_duration": 2.0,
            "temperature": 0.8,
            "top_k": 30,
            "lookahead_chunks": 3,
            "max_latency_ms": 1000,
        },
    }

    preset = quality_presets.get(quality_mode, quality_presets["balanced"])
    preset.update(config_kwargs)
    preset["chunk_duration"] = chunk_duration  # Override with user preference

    config = StreamingConfig(**preset)
    return StreamingGenerator(model, config)
