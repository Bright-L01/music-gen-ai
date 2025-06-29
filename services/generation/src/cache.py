"""
Generation Cache System

Intelligent caching for generated audio to reduce redundant processing.
Uses similarity matching for prompts and stores results in Redis/S3.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

from .models import GenerationRequest


logger = logging.getLogger(__name__)


class GenerationCache:
    """
    Multi-level caching system for generation results
    
    Features:
    - Exact prompt matching
    - Semantic similarity matching
    - Popularity-based retention
    - Automatic cache warming
    """
    
    def __init__(
        self,
        redis_url: str = None,
        similarity_threshold: float = 0.95,
        max_cache_size: int = 10000
    ):
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.client: Optional[redis.Redis] = None
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Sentence encoder for semantic similarity
        self.encoder = None
        self._encoder_loaded = False
        
        # Cache keys
        self.CACHE_PREFIX = "cache:generation"
        self.EMBEDDINGS_PREFIX = "cache:embeddings"
        self.STATS_PREFIX = "cache:stats"
        self.POPULAR_PREFIX = "cache:popular"
        
        # Cache TTL (7 days for regular, 30 days for popular)
        self.DEFAULT_TTL = 7 * 24 * 60 * 60
        self.POPULAR_TTL = 30 * 24 * 60 * 60
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            logger.info("Connected to Redis cache")
            
            # Load sentence encoder in background
            await self._load_encoder()
            
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            
    async def _load_encoder(self):
        """Load sentence encoder for similarity matching"""
        try:
            # Use a lightweight model for speed
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self._encoder_loaded = True
            logger.info("Loaded sentence encoder for cache similarity matching")
        except Exception as e:
            logger.warning(f"Failed to load sentence encoder: {e}")
            self._encoder_loaded = False
            
    def get_cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key from request"""
        # Create normalized key from request parameters
        key_parts = [
            request.prompt.lower().strip(),
            str(request.duration),
            str(request.temperature),
            request.genre or "",
            request.mood or "",
            json.dumps(sorted(request.instruments) if request.instruments else [])
        ]
        
        # Add structure if present
        if request.structure:
            structure_str = json.dumps({
                "sections": [
                    {
                        "type": s.type,
                        "duration": s.duration,
                        "energy": s.energy
                    }
                    for s in request.structure.sections
                ],
                "tempo": request.structure.tempo,
                "key": request.structure.key
            }, sort_keys=True)
            key_parts.append(structure_str)
            
        # Create hash
        key_str = "|".join(key_parts)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"{self.CACHE_PREFIX}:{key_hash}"
        
    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        try:
            # Get from cache
            data = await self.client.get(cache_key)
            if not data:
                return None
                
            result = json.loads(data)
            
            # Update access stats
            await self._increment_stat("hits")
            await self._update_access_time(cache_key)
            
            # Increment popularity
            popularity_key = f"{self.POPULAR_PREFIX}:{cache_key}"
            await self.client.zincrby("cache:popular:audio", 1, cache_key)
            
            logger.info(f"Cache hit for key: {cache_key}")
            return result
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            await self._increment_stat("errors")
            return None
            
    async def set(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Store result in cache"""
        try:
            # Check cache size
            size = await self.client.dbsize()
            if size >= self.max_cache_size:
                await self._evict_old_entries()
                
            # Store data
            ttl = ttl or self.DEFAULT_TTL
            await self.client.setex(
                cache_key,
                ttl,
                json.dumps(data)
            )
            
            # Store embedding if encoder is available
            if self._encoder_loaded and "prompt" in data:
                await self._store_embedding(cache_key, data["prompt"])
                
            # Update stats
            await self._increment_stat("sets")
            
            logger.info(f"Cached result with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            await self._increment_stat("errors")
            
    async def find_similar(
        self,
        prompt: str,
        threshold: float = None
    ) -> Optional[Tuple[str, float]]:
        """Find similar cached prompts using semantic search"""
        if not self._encoder_loaded:
            return None
            
        threshold = threshold or self.similarity_threshold
        
        try:
            # Encode query prompt
            query_embedding = self.encoder.encode([prompt])[0]
            
            # Get all cached embeddings
            embedding_keys = await self.client.keys(f"{self.EMBEDDINGS_PREFIX}:*")
            if not embedding_keys:
                return None
                
            best_match = None
            best_score = 0.0
            
            # Compare with cached embeddings
            for emb_key in embedding_keys[:1000]:  # Limit to prevent overload
                emb_data = await self.client.get(emb_key)
                if not emb_data:
                    continue
                    
                cached_embedding = np.array(json.loads(emb_data))
                
                # Calculate similarity
                similarity = cosine_similarity(
                    [query_embedding],
                    [cached_embedding]
                )[0][0]
                
                if similarity > best_score and similarity >= threshold:
                    # Extract cache key from embedding key
                    cache_key = emb_key.replace(
                        f"{self.EMBEDDINGS_PREFIX}:",
                        f"{self.CACHE_PREFIX}:"
                    )
                    best_match = cache_key
                    best_score = similarity
                    
            if best_match:
                logger.info(f"Found similar cache entry with score {best_score:.3f}")
                await self._increment_stat("similarity_hits")
                return best_match, best_score
                
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            
        return None
        
    async def _store_embedding(self, cache_key: str, prompt: str):
        """Store prompt embedding for similarity search"""
        try:
            # Generate embedding
            embedding = self.encoder.encode([prompt])[0]
            
            # Store with same TTL as cache entry
            ttl = await self.client.ttl(cache_key)
            if ttl > 0:
                emb_key = cache_key.replace(
                    self.CACHE_PREFIX,
                    self.EMBEDDINGS_PREFIX
                )
                await self.client.setex(
                    emb_key,
                    ttl,
                    json.dumps(embedding.tolist())
                )
        except Exception as e:
            logger.warning(f"Failed to store embedding: {e}")
            
    async def warm_cache(self, popular_prompts: List[str]):
        """Pre-populate cache with popular prompts"""
        logger.info(f"Warming cache with {len(popular_prompts)} popular prompts")
        
        # This would trigger generation for popular prompts
        # Implementation depends on generation service
        pass
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = await self.client.hgetall(self.STATS_PREFIX)
        
        # Calculate hit rate
        hits = int(stats.get("hits", 0))
        misses = int(stats.get("misses", 0))
        total = hits + misses
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            "hits": hits,
            "misses": misses,
            "sets": int(stats.get("sets", 0)),
            "errors": int(stats.get("errors", 0)),
            "similarity_hits": int(stats.get("similarity_hits", 0)),
            "hit_rate": hit_rate,
            "cache_size": await self.client.dbsize()
        }
        
    async def clear(self):
        """Clear all cache entries"""
        pattern = f"{self.CACHE_PREFIX}:*"
        keys = await self.client.keys(pattern)
        
        if keys:
            await self.client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
            
        # Clear embeddings
        emb_pattern = f"{self.EMBEDDINGS_PREFIX}:*"
        emb_keys = await self.client.keys(emb_pattern)
        if emb_keys:
            await self.client.delete(*emb_keys)
            
        # Clear stats
        await self.client.delete(self.STATS_PREFIX)
        
    async def _evict_old_entries(self, count: int = 100):
        """Evict oldest cache entries"""
        # Get all cache keys with access times
        pattern = f"{self.CACHE_PREFIX}:*"
        keys = await self.client.keys(pattern)
        
        if not keys:
            return
            
        # Get TTLs and sort by remaining time
        key_ttls = []
        for key in keys:
            ttl = await self.client.ttl(key)
            if ttl > 0:
                key_ttls.append((key, ttl))
                
        # Sort by TTL (ascending) and delete oldest
        key_ttls.sort(key=lambda x: x[1])
        
        for key, _ in key_ttls[:count]:
            await self.client.delete(key)
            
            # Delete corresponding embedding
            emb_key = key.replace(self.CACHE_PREFIX, self.EMBEDDINGS_PREFIX)
            await self.client.delete(emb_key)
            
        logger.info(f"Evicted {min(count, len(key_ttls))} old cache entries")
        
    async def _increment_stat(self, stat_name: str):
        """Increment a statistic counter"""
        await self.client.hincrby(self.STATS_PREFIX, stat_name, 1)
        
    async def _update_access_time(self, cache_key: str):
        """Update last access time for cache entry"""
        # Extend TTL for frequently accessed items
        popularity_score = await self.client.zscore("cache:popular:audio", cache_key)
        
        if popularity_score and popularity_score > 10:
            # Popular item, extend TTL
            await self.client.expire(cache_key, self.POPULAR_TTL)
            
            # Update embedding TTL too
            emb_key = cache_key.replace(self.CACHE_PREFIX, self.EMBEDDINGS_PREFIX)
            await self.client.expire(emb_key, self.POPULAR_TTL)