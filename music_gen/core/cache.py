"""
Production-ready caching layer for Music Gen AI.

This module provides a comprehensive caching solution with Redis support,
intelligent invalidation strategies, and performance monitoring.
"""

import asyncio
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Union

import redis.asyncio as redis
from pydantic import BaseModel

from .config import get_config

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration settings."""

    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 10
    key_prefix: str = "musicgen"
    compression_threshold: int = 1024  # Compress values larger than 1KB
    serialization: str = "json"  # "json" or "pickle"


class CacheKey:
    """Utility for generating consistent cache keys."""

    @staticmethod
    def user_profile(user_id: str) -> str:
        return f"user:profile:{user_id}"

    @staticmethod
    def user_generations(user_id: str, page: int = 1) -> str:
        return f"user:generations:{user_id}:page:{page}"

    @staticmethod
    def model_metadata(model_id: str) -> str:
        return f"model:metadata:{model_id}"

    @staticmethod
    def generation_result(task_id: str) -> str:
        return f"generation:result:{task_id}"

    @staticmethod
    def trending_tracks(limit: int = 10) -> str:
        return f"trending:tracks:limit:{limit}"

    @staticmethod
    def track_metadata(track_id: str) -> str:
        return f"track:metadata:{track_id}"

    @staticmethod
    def user_tracks(user_id: str, page: int = 1) -> str:
        return f"user:tracks:{user_id}:page:{page}"

    @staticmethod
    def search_results(query: str, filters: Dict, page: int = 1) -> str:
        # Create deterministic key from search parameters
        filter_str = "&".join(f"{k}={v}" for k, v in sorted(filters.items()))
        return f"search:results:{hash(query)}:{hash(filter_str)}:page:{page}"

    @staticmethod
    def model_inference_cache(model_id: str, prompt_hash: str, params_hash: str) -> str:
        return f"inference:cache:{model_id}:{prompt_hash}:{params_hash}"


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        pass


class RedisBackend(CacheBackend):
    """Redis cache backend with advanced features."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            self.redis_client = redis.Redis(connection_pool=self.redis_pool)

            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info("Redis cache backend connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
        self._connected = False
        logger.info("Redis cache backend disconnected")

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.config.serialization == "pickle":
            data = pickle.dumps(value)
        else:
            # JSON serialization
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                data = json.dumps(value).encode()
            else:
                # Fallback to pickle for complex objects
                data = pickle.dumps(value)

        # Compress large values
        if len(data) > self.config.compression_threshold:
            import gzip

            data = gzip.compress(data)
            # Add compression marker
            data = b"GZIP:" + data

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        # Check for compression
        if data.startswith(b"GZIP:"):
            import gzip

            data = gzip.decompress(data[5:])

        # Try JSON first, fallback to pickle
        try:
            if self.config.serialization == "json":
                return json.loads(data.decode())
            else:
                return pickle.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback to pickle
            return pickle.loads(data)

    def _get_key(self, key: str) -> str:
        """Get prefixed cache key."""
        return f"{self.config.key_prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._connected:
            return None

        try:
            data = await self.redis_client.get(self._get_key(key))
            if data is None:
                return None

            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._connected:
            return False

        try:
            data = self._serialize(value)
            ttl = ttl or self.config.default_ttl

            await self.redis_client.setex(self._get_key(key), ttl, data)
            return True

        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self._connected:
            return False

        try:
            result = await self.redis_client.delete(self._get_key(key))
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._connected:
            return False

        try:
            result = await self.redis_client.exists(self._get_key(key))
            return result > 0

        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self._connected:
            return 0

        try:
            pattern_key = self._get_key(pattern)
            keys = await self.redis_client.keys(pattern_key)

            if keys:
                result = await self.redis_client.delete(*keys)
                return result
            return 0

        except Exception as e:
            logger.error(f"Cache clear pattern error for pattern '{pattern}': {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._connected:
            return {}

        try:
            info = await self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0),
                "total_keys": len(await self.redis_client.keys(f"{self.config.key_prefix}:*")),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


class MemoryBackend(CacheBackend):
    """In-memory cache backend for development/testing."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.data: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = asyncio.Lock()

    async def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.data.items() if expiry and expiry < current_time
        ]
        for key in expired_keys:
            del self.data[key]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            await self._cleanup_expired()

            if key not in self.data:
                return None

            value, expiry = self.data[key]
            if expiry and expiry < time.time():
                del self.data[key]
                return None

            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            ttl = ttl or self.config.default_ttl
            expiry = time.time() + ttl if ttl > 0 else None
            self.data[key] = (value, expiry)
            return True

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self.data:
                del self.data[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        return await self.get(key) is not None

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern (simple wildcard support)."""
        async with self._lock:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                matching_keys = [key for key in self.data.keys() if key.startswith(prefix)]
            else:
                matching_keys = [key for key in self.data.keys() if key == pattern]

            for key in matching_keys:
                del self.data[key]

            return len(matching_keys)


class CacheManager:
    """Main cache manager with invalidation strategies."""

    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self.hit_count = 0
        self.miss_count = 0
        self.invalidation_tags: Dict[str, Set[str]] = {}  # tag -> set of keys

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with hit/miss tracking."""
        value = await self.backend.get(key)

        if value is not None:
            self.hit_count += 1
            logger.debug(f"Cache hit for key: {key}")
            return value
        else:
            self.miss_count += 1
            logger.debug(f"Cache miss for key: {key}")
            return default

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with optional tags for invalidation."""
        success = await self.backend.set(key, value, ttl)

        if success and tags:
            # Associate key with tags for invalidation
            for tag in tags:
                if tag not in self.invalidation_tags:
                    self.invalidation_tags[tag] = set()
                self.invalidation_tags[tag].add(key)

        return success

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return await self.backend.delete(key)

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all cache entries associated with given tags."""
        keys_to_delete = set()

        for tag in tags:
            if tag in self.invalidation_tags:
                keys_to_delete.update(self.invalidation_tags[tag])
                del self.invalidation_tags[tag]

        if not keys_to_delete:
            return 0

        # Delete keys
        deleted_count = 0
        for key in keys_to_delete:
            if await self.backend.delete(key):
                deleted_count += 1

        logger.info(f"Invalidated {deleted_count} cache entries for tags: {tags}")
        return deleted_count

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        return await self.backend.clear_pattern(pattern)

    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0.0

    def reset_stats(self):
        """Reset hit/miss statistics."""
        self.hit_count = 0
        self.miss_count = 0


# Global cache instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager

    if _cache_manager is None:
        config = get_config()
        cache_config = CacheConfig(
            redis_url=getattr(config, "redis_url", "redis://localhost:6379"),
            default_ttl=getattr(config, "cache_ttl", 3600),
            key_prefix=getattr(config, "cache_prefix", "musicgen"),
        )

        # Try Redis first, fallback to memory
        try:
            backend = RedisBackend(cache_config)
            await backend.connect()
        except Exception as e:
            logger.warning(f"Redis unavailable, using memory cache: {e}")
            backend = MemoryBackend(cache_config)

        _cache_manager = CacheManager(backend)

    return _cache_manager


def cache_result(
    key_func: callable,
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    serialize_args: bool = True,
):
    """Decorator for caching function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache_manager()

            # Generate cache key
            if serialize_args:
                # Create deterministic key from arguments
                args_str = str(hash((args, tuple(sorted(kwargs.items())))))
                cache_key = f"{func.__name__}:{args_str}"
            else:
                cache_key = key_func(*args, **kwargs)

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl, tags=tags)

            return result

        return wrapper

    return decorator


class CacheInvalidationService:
    """Service for managing cache invalidation strategies."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def invalidate_user_data(self, user_id: str):
        """Invalidate all cached data for a user."""
        patterns = [
            f"user:profile:{user_id}",
            f"user:generations:{user_id}:*",
            f"user:tracks:{user_id}:*",
        ]

        total_deleted = 0
        for pattern in patterns:
            deleted = await self.cache_manager.clear_pattern(pattern)
            total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} cache entries for user {user_id}")
        return total_deleted

    async def invalidate_track_data(self, track_id: str):
        """Invalidate cached data for a track."""
        keys = [
            CacheKey.track_metadata(track_id),
            "trending:tracks:*",  # Track might appear in trending
            "search:results:*",  # Track might appear in search results
        ]

        total_deleted = 0
        for key in keys:
            if "*" in key:
                deleted = await self.cache_manager.clear_pattern(key)
            else:
                deleted = 1 if await self.cache_manager.delete(key) else 0
            total_deleted += deleted

        logger.info(f"Invalidated {total_deleted} cache entries for track {track_id}")
        return total_deleted

    async def invalidate_search_results(self):
        """Invalidate all search result caches."""
        deleted = await self.cache_manager.clear_pattern("search:results:*")
        logger.info(f"Invalidated {deleted} search result cache entries")
        return deleted

    async def invalidate_trending_data(self):
        """Invalidate trending data caches."""
        deleted = await self.cache_manager.clear_pattern("trending:*")
        logger.info(f"Invalidated {deleted} trending data cache entries")
        return deleted


# Convenience functions
async def cache_user_profile(user_id: str, profile_data: Dict, ttl: int = 3600):
    """Cache user profile data."""
    cache = await get_cache_manager()
    key = CacheKey.user_profile(user_id)
    return await cache.set(key, profile_data, ttl=ttl, tags=["user_data"])


async def get_cached_user_profile(user_id: str) -> Optional[Dict]:
    """Get cached user profile data."""
    cache = await get_cache_manager()
    key = CacheKey.user_profile(user_id)
    return await cache.get(key)


async def cache_trending_tracks(tracks: List[Dict], limit: int = 10, ttl: int = 600):
    """Cache trending tracks (shorter TTL for freshness)."""
    cache = await get_cache_manager()
    key = CacheKey.trending_tracks(limit)
    return await cache.set(key, tracks, ttl=ttl, tags=["trending"])


async def get_cached_trending_tracks(limit: int = 10) -> Optional[List[Dict]]:
    """Get cached trending tracks."""
    cache = await get_cache_manager()
    key = CacheKey.trending_tracks(limit)
    return await cache.get(key)
