"""
Optimized MusicGen generator with caching, batching, and performance improvements.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model_cache import get_cache_stats, get_cached_model

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request for music generation."""

    prompt: str
    duration: float = 10.0
    temperature: float = 1.0
    guidance_scale: float = 3.0
    top_k: int = 250
    top_p: float = 0.0
    request_id: str = None


@dataclass
class GenerationResult:
    """Result of music generation."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    generation_time: float
    request_id: str = None
    metadata: Dict = None


class FastMusicGenerator:
    """
    Optimized MusicGen generator with performance improvements.
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: str = None,
        max_concurrent: int = 3,
        warmup: bool = True,
    ):
        """
        Initialize the fast generator.

        Args:
            model_name: MusicGen model to use
            device: Device to run on (auto-detect if None)
            max_concurrent: Maximum concurrent generations
            warmup: Whether to warmup the model cache
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_concurrent = max_concurrent

        # Thread safety
        self._generation_lock = threading.Semaphore(max_concurrent)
        self._stats = {
            "total_generations": 0,
            "total_generation_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(f"FastMusicGenerator initialized: {model_name} on {device}")
        logger.info(f"Max concurrent generations: {max_concurrent}")

        # Warmup if requested
        if warmup:
            self.warmup()

    def warmup(self):
        """Warmup the generator by preloading the model."""
        logger.info("Warming up FastMusicGenerator...")
        start_time = time.time()

        # Load model into cache
        model = get_cached_model(self.model_name, self.device)

        # Test generation to warm up CUDA kernels
        if self.device == "cuda":
            logger.info("Warming up CUDA kernels...")
            try:
                self._generate_single_optimized(model, "test warmup", duration=1.0, temperature=1.0)
                logger.info("✓ CUDA kernels warmed up")
            except Exception as e:
                logger.warning(f"CUDA warmup failed: {e}")

        warmup_time = time.time() - start_time
        logger.info(f"✓ Warmup complete in {warmup_time:.2f}s")

    def generate_single(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        top_k: int = 250,
        top_p: float = 0.0,
    ) -> GenerationResult:
        """
        Generate a single audio clip with optimizations.

        Args:
            prompt: Text prompt for generation
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance scale
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            GenerationResult with audio and metadata
        """
        with self._generation_lock:
            return self._generate_single_thread_safe(
                prompt, duration, temperature, guidance_scale, top_k, top_p
            )

    def _generate_single_thread_safe(
        self,
        prompt: str,
        duration: float,
        temperature: float,
        guidance_scale: float,
        top_k: int,
        top_p: float,
    ) -> GenerationResult:
        """Thread-safe single generation."""
        start_time = time.time()

        # Get cached model
        cache_stats_before = get_cache_stats()
        model = get_cached_model(self.model_name, self.device)
        cache_stats_after = get_cache_stats()

        # Track cache performance
        if cache_stats_after["total_accesses"] > cache_stats_before["total_accesses"]:
            cached_before = cache_stats_before.get("cached_models", 0)
            cached_after = cache_stats_after.get("cached_models", 0)
            if cached_before == cached_after:
                self._stats["cache_hits"] += 1
            else:
                self._stats["cache_misses"] += 1

        # Generate audio
        audio_np, sample_rate = self._generate_single_optimized(
            model, prompt, duration, temperature, guidance_scale, top_k, top_p
        )

        generation_time = time.time() - start_time

        # Update stats
        self._stats["total_generations"] += 1
        self._stats["total_generation_time"] += generation_time

        # Create result
        result = GenerationResult(
            audio=audio_np,
            sample_rate=sample_rate,
            duration=len(audio_np) / sample_rate,
            generation_time=generation_time,
            metadata={
                "prompt": prompt,
                "model": self.model_name,
                "device": self.device,
                "parameters": {
                    "temperature": temperature,
                    "guidance_scale": guidance_scale,
                    "top_k": top_k,
                    "top_p": top_p,
                },
            },
        )

        logger.info(f"Generated audio in {generation_time:.2f}s (target: {duration:.1f}s)")
        return result

    def _generate_single_optimized(
        self,
        model,
        prompt: str,
        duration: float,
        temperature: float,
        guidance_scale: float = 3.0,
        top_k: int = 250,
        top_p: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """Optimized single generation with memory management."""

        # Clear GPU memory before generation
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Process input efficiently
        inputs = model.processor(text=[prompt], padding=True, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.device)

        # Calculate tokens needed
        max_new_tokens = int(duration * 50)  # 50Hz frame rate

        # Optimized generation with memory management
        with torch.no_grad():
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, "compile") and self.device == "cuda":
                try:
                    if not hasattr(model.model, "_compiled"):
                        model.model = torch.compile(model.model, mode="reduce-overhead")
                        model.model._compiled = True
                        logger.info("✓ Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"torch.compile failed: {e}")

            # Generate with optimized parameters
            audio_values = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                guidance_scale=guidance_scale,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p > 0 else None,
                pad_token_id=model.model.config.pad_token_id,
                use_cache=True,  # Enable KV caching
            )

        # Efficient post-processing
        audio = audio_values[0, 0].cpu().numpy()

        # Memory cleanup
        del audio_values, inputs
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return audio, model.sample_rate

    def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Generate multiple audio clips concurrently.

        Args:
            requests: List of generation requests

        Returns:
            List of generation results
        """
        if not requests:
            return []

        logger.info(f"Generating batch of {len(requests)} requests")
        start_time = time.time()

        results = []

        # Use ThreadPoolExecutor for concurrent generation
        with ThreadPoolExecutor(max_workers=min(len(requests), self.max_concurrent)) as executor:
            # Submit all requests
            future_to_request = {}
            for req in requests:
                future = executor.submit(
                    self.generate_single,
                    req.prompt,
                    req.duration,
                    req.temperature,
                    req.guidance_scale,
                    req.top_k,
                    req.top_p,
                )
                future_to_request[future] = req

            # Collect results as they complete
            for future in as_completed(future_to_request):
                req = future_to_request[future]
                try:
                    result = future.result()
                    result.request_id = req.request_id
                    results.append(result)
                except Exception as e:
                    logger.error(f"Generation failed for request {req.request_id}: {e}")
                    # Create error result
                    error_result = GenerationResult(
                        audio=np.zeros(int(req.duration * 32000)),  # Silent audio
                        sample_rate=32000,
                        duration=req.duration,
                        generation_time=0,
                        request_id=req.request_id,
                        metadata={"error": str(e)},
                    )
                    results.append(error_result)

        total_time = time.time() - start_time
        logger.info(f"Batch generation complete in {total_time:.2f}s")

        # Sort results by request order
        request_ids = [req.request_id for req in requests]
        results.sort(
            key=lambda r: request_ids.index(r.request_id) if r.request_id in request_ids else 999
        )

        return results

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        avg_generation_time = (
            self._stats["total_generation_time"] / self._stats["total_generations"]
            if self._stats["total_generations"] > 0
            else 0
        )

        cache_hit_rate = (
            self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
            if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
            else 0
        )

        return {
            **self._stats,
            "average_generation_time": avg_generation_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": get_cache_stats(),
        }

    def clear_cache(self):
        """Clear the model cache."""
        from .model_cache import clear_cache

        clear_cache()
        logger.info("Model cache cleared")


# Convenience functions
def generate_music_fast(
    prompt: str,
    duration: float = 10.0,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    model_name: str = "facebook/musicgen-small",
    device: str = None,
) -> GenerationResult:
    """
    Generate music with optimized performance.

    Args:
        prompt: Text prompt
        duration: Duration in seconds
        temperature: Sampling temperature
        guidance_scale: Guidance scale
        model_name: Model to use
        device: Device to use

    Returns:
        GenerationResult
    """
    generator = FastMusicGenerator(model_name, device, warmup=False)
    return generator.generate_single(prompt, duration, temperature, guidance_scale)
