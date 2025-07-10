"""
Task repository implementations.

Provides concrete implementations for task storage and retrieval.
"""

import asyncio
import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

from music_gen.core.exceptions import TaskNotFoundError
from music_gen.core.interfaces.repositories import TaskRepository
from music_gen.utils.optional_imports import optional_import

logger = logging.getLogger(__name__)


class InMemoryTaskRepository(TaskRepository):
    """In-memory task repository for development/testing."""

    def __init__(self, max_tasks: int = 1000):
        """Initialize repository.

        Args:
            max_tasks: Maximum number of tasks to keep in memory
        """
        self._tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_tasks = max_tasks
        self._lock = asyncio.Lock()

    async def create_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Create a new task."""
        async with self._lock:
            # Add timestamp if not present
            if "created_at" not in task_data:
                task_data["created_at"] = datetime.utcnow().isoformat()

            # Enforce maximum tasks limit
            if len(self._tasks) >= self._max_tasks:
                # Remove oldest task
                self._tasks.popitem(last=False)

            self._tasks[task_id] = task_data.copy()
            logger.debug(f"Task created: {task_id}")

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        async with self._lock:
            task = self._tasks.get(task_id)
            return task.copy() if task else None

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """Update task data."""
        async with self._lock:
            if task_id not in self._tasks:
                raise TaskNotFoundError(f"Task not found: {task_id}")

            # Add update timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()

            self._tasks[task_id].update(updates)
            logger.debug(f"Task updated: {task_id}")

    async def delete_task(self, task_id: str) -> None:
        """Delete a task."""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.debug(f"Task deleted: {task_id}")

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        async with self._lock:
            tasks = list(self._tasks.values())

            # Filter by status if provided
            if status:
                tasks = [t for t in tasks if t.get("status") == status]

            # Apply pagination
            return tasks[offset : offset + limit]


class RedisTaskRepository(TaskRepository):
    """Redis-based task repository for production use."""

    def __init__(self, redis_url: str, key_prefix: str = "musicgen:task:"):
        """Initialize repository with Redis connection.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all Redis keys
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            with optional_import(
                "aioredis",
                fallback_message="Redis support requires aioredis",
                install_command="pip install aioredis",
            ) as aioredis:
                if aioredis is None:
                    raise ImportError(
                        "aioredis is required for RedisTaskRepository. Install with: pip install aioredis"
                    )

                self._redis = await aioredis.from_url(self._redis_url)
        return self._redis

    def _get_key(self, task_id: str) -> str:
        """Get Redis key for a task."""
        return f"{self._key_prefix}{task_id}"

    async def create_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Create a new task in Redis."""
        redis = await self._get_redis()

        # Add timestamp
        task_data["created_at"] = datetime.utcnow().isoformat()

        # Store as JSON
        key = self._get_key(task_id)
        await redis.set(key, json.dumps(task_data))

        # Set expiration (24 hours by default)
        await redis.expire(key, 86400)

        logger.debug(f"Task created in Redis: {task_id}")

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from Redis."""
        redis = await self._get_redis()
        key = self._get_key(task_id)

        data = await redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """Update task in Redis."""
        redis = await self._get_redis()
        key = self._get_key(task_id)

        # Get existing data
        data = await redis.get(key)
        if not data:
            raise TaskNotFoundError(f"Task not found: {task_id}")

        task_data = json.loads(data)
        task_data.update(updates)
        task_data["updated_at"] = datetime.utcnow().isoformat()

        # Save updated data
        await redis.set(key, json.dumps(task_data))

        logger.debug(f"Task updated in Redis: {task_id}")

    async def delete_task(self, task_id: str) -> None:
        """Delete task from Redis."""
        redis = await self._get_redis()
        key = self._get_key(task_id)

        await redis.delete(key)
        logger.debug(f"Task deleted from Redis: {task_id}")

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks from Redis."""
        redis = await self._get_redis()

        # Get all task keys
        pattern = f"{self._key_prefix}*"
        keys = []
        async for key in redis.scan_iter(pattern):
            keys.append(key)

        # Get task data
        tasks = []
        for key in keys[offset : offset + limit]:
            data = await redis.get(key)
            if data:
                task = json.loads(data)
                if status is None or task.get("status") == status:
                    tasks.append(task)

        return tasks


class PostgreSQLTaskRepository(TaskRepository):
    """PostgreSQL-based task repository for production use."""

    def __init__(self, database_url: str, table_name: str = "tasks"):
        """Initialize repository with PostgreSQL connection.

        Args:
            database_url: PostgreSQL connection URL
            table_name: Name of the tasks table
        """
        self._database_url = database_url
        self._table_name = table_name
        self._pool = None

    async def _get_pool(self):
        """Get connection pool (lazy initialization)."""
        if self._pool is None:
            with optional_import(
                "asyncpg",
                fallback_message="PostgreSQL support requires asyncpg",
                install_command="pip install asyncpg",
            ) as asyncpg:
                if asyncpg is None:
                    raise ImportError(
                        "asyncpg is required for PostgreSQLTaskRepository. Install with: pip install asyncpg"
                    )

                self._pool = await asyncpg.create_pool(self._database_url)
                await self._ensure_table_exists()
        return self._pool

    async def _ensure_table_exists(self):
        """Create tasks table if it doesn't exist."""
        if self._pool is None:
            return

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            id VARCHAR(255) PRIMARY KEY,
            data JSONB NOT NULL,
            status VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_status
        ON {self._table_name}(status);

        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_created_at
        ON {self._table_name}(created_at);
        """

        async with self._pool.acquire() as connection:
            await connection.execute(create_table_sql)

    async def create_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Create a new task in PostgreSQL."""
        pool = await self._get_pool()

        # Add timestamp
        task_data["created_at"] = datetime.utcnow().isoformat()
        status = task_data.get("status", "pending")

        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self._table_name} (id, data, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $4)
                """,
                task_id,
                json.dumps(task_data),
                status,
                datetime.utcnow(),
            )

        logger.debug(f"Task created in PostgreSQL: {task_id}")

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT data FROM {self._table_name} WHERE id = $1", task_id
            )

            if row:
                return json.loads(row["data"])
            return None

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """Update task in PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as connection:
            # Get existing data
            row = await connection.fetchrow(
                f"SELECT data FROM {self._table_name} WHERE id = $1", task_id
            )

            if not row:
                raise TaskNotFoundError(f"Task not found: {task_id}")

            task_data = json.loads(row["data"])
            task_data.update(updates)
            task_data["updated_at"] = datetime.utcnow().isoformat()

            new_status = task_data.get("status", "pending")

            # Update the record
            await connection.execute(
                f"""
                UPDATE {self._table_name}
                SET data = $2, status = $3, updated_at = $4
                WHERE id = $1
                """,
                task_id,
                json.dumps(task_data),
                new_status,
                datetime.utcnow(),
            )

        logger.debug(f"Task updated in PostgreSQL: {task_id}")

    async def delete_task(self, task_id: str) -> None:
        """Delete task from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as connection:
            await connection.execute(f"DELETE FROM {self._table_name} WHERE id = $1", task_id)

        logger.debug(f"Task deleted from PostgreSQL: {task_id}")

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as connection:
            if status:
                query = f"""
                SELECT data FROM {self._table_name}
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """
                rows = await connection.fetch(query, status, limit, offset)
            else:
                query = f"""
                SELECT data FROM {self._table_name}
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """
                rows = await connection.fetch(query, limit, offset)

        return [json.loads(row["data"]) for row in rows]
