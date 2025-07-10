"""
Production-ready database connection pooling for Music Gen AI.

Implements asyncpg connection pooling with health checks, monitoring,
and failover capabilities based on 2024 best practices.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg
from asyncpg import Pool, Connection
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.ext.asyncio.session import async_sessionmaker
from sqlalchemy.pool import QueuePool

from ...core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for database connection pool."""

    # Connection settings
    database_url: str
    min_size: int = 10
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes

    # Timeout settings
    command_timeout: float = 60.0
    server_settings: Dict[str, str] = None

    # Health check settings
    health_check_interval: float = 30.0  # 30 seconds
    max_failed_health_checks: int = 3

    # Monitoring
    enable_monitoring: bool = True
    log_slow_queries: bool = True
    slow_query_threshold: float = 1.0  # 1 second

    def __post_init__(self):
        if self.server_settings is None:
            self.server_settings = {
                "application_name": "musicgen_api",
                "jit": "off",  # Disable JIT for faster connection startup
            }


@dataclass
class PoolStats:
    """Connection pool statistics."""

    size: int
    free_size: int
    total_connections_created: int
    total_queries_executed: int
    avg_query_time: float
    failed_health_checks: int
    last_health_check: float


class HealthCheckError(Exception):
    """Raised when database health checks fail."""

    pass


class DatabaseConnectionPool:
    """Production-ready database connection pool with monitoring and health checks."""

    def __init__(self, config: PoolConfig):
        self.config = config
        self._pool: Optional[Pool] = None
        self._sqlalchemy_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

        # Health check tracking
        self._failed_health_checks = 0
        self._last_health_check = 0.0
        self._health_check_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_connections_created = 0
        self._total_queries_executed = 0
        self._query_times: List[float] = []
        self._start_time = time.time()

        # Connection validation query
        self._health_check_query = "SELECT 1"

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            # Create asyncpg pool
            self._pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings,
                init=self._init_connection,
            )

            # Create SQLAlchemy async engine for ORM operations
            self._sqlalchemy_engine = create_async_engine(
                self.config.database_url.replace("postgresql://", "postgresql+asyncpg://"),
                poolclass=QueuePool,
                pool_size=self.config.min_size,
                max_overflow=self.config.max_size - self.config.min_size,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,  # Enable connection validation
                echo=False,  # Set to True for SQL logging in development
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._sqlalchemy_engine, class_=AsyncSession, expire_on_commit=False
            )

            # Start health check task
            if self.config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(
                f"Database pool initialized - "
                f"min_size: {self.config.min_size}, "
                f"max_size: {self.config.max_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def _init_connection(self, connection: Connection) -> None:
        """Initialize each new connection."""
        self._total_connections_created += 1

        # Set connection-specific settings
        await connection.execute("SET timezone TO 'UTC'")
        await connection.execute("SET statement_timeout TO '60s'")

        if self.config.log_slow_queries:
            await connection.execute("SET log_min_duration_statement TO '1000'")  # 1 second

        logger.debug(
            f"Initialized new database connection (total: {self._total_connections_created})"
        )

    @asynccontextmanager
    async def acquire_connection(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool with monitoring."""
        if not self._pool:
            raise RuntimeError("Pool not initialized")

        start_time = time.time()

        try:
            async with self._pool.acquire() as connection:
                yield connection

        except Exception as e:
            logger.error(f"Error with pooled connection: {e}")
            raise

        finally:
            # Record query time for monitoring
            query_time = time.time() - start_time
            self._query_times.append(query_time)
            self._total_queries_executed += 1

            # Keep only recent query times for average calculation
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-500:]

            if query_time > self.config.slow_query_threshold:
                logger.warning(f"Slow database operation: {query_time:.3f}s")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy async session with automatic transaction management."""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query with monitoring."""
        async with self.acquire_connection() as conn:
            return await conn.fetch(query, *args)

    async def execute_scalar(self, query: str, *args) -> Any:
        """Execute a query returning a single value."""
        async with self.acquire_connection() as conn:
            return await conn.fetchval(query, *args)

    async def execute_transaction(self, queries: List[tuple]) -> None:
        """Execute multiple queries in a transaction."""
        async with self.acquire_connection() as conn:
            async with conn.transaction():
                for query, args in queries:
                    await conn.execute(query, *args)

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _perform_health_check(self) -> None:
        """Perform health check on the pool."""
        try:
            start_time = time.time()

            # Test connection with simple query
            async with self.acquire_connection() as conn:
                await conn.fetchval(self._health_check_query)

            check_time = time.time() - start_time
            self._last_health_check = time.time()
            self._failed_health_checks = 0

            if check_time > 1.0:  # Health check taking too long
                logger.warning(f"Slow health check: {check_time:.3f}s")
            else:
                logger.debug(f"Health check passed in {check_time:.3f}s")

        except Exception as e:
            self._failed_health_checks += 1
            logger.error(
                f"Health check failed ({self._failed_health_checks}/{self.config.max_failed_health_checks}): {e}"
            )

            if self._failed_health_checks >= self.config.max_failed_health_checks:
                logger.critical("Maximum health check failures reached - pool may be unhealthy")
                raise HealthCheckError("Database pool failed multiple health checks")

    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        if not self._pool:
            raise RuntimeError("Pool not initialized")

        avg_query_time = (
            sum(self._query_times) / len(self._query_times) if self._query_times else 0.0
        )

        return PoolStats(
            size=self._pool.get_size(),
            free_size=self._pool.get_idle_size(),
            total_connections_created=self._total_connections_created,
            total_queries_executed=self._total_queries_executed,
            avg_query_time=avg_query_time,
            failed_health_checks=self._failed_health_checks,
            last_health_check=self._last_health_check,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        logger.info("Closing database connection pool...")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close SQLAlchemy engine
        if self._sqlalchemy_engine:
            await self._sqlalchemy_engine.dispose()

        # Close asyncpg pool
        if self._pool:
            await self._pool.close()

        logger.info("Database connection pool closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global pool instance
_db_pool: Optional[DatabaseConnectionPool] = None


async def get_database_pool() -> DatabaseConnectionPool:
    """Get or create global database pool."""
    global _db_pool

    if _db_pool is None:
        config = get_config()

        pool_config = PoolConfig(
            database_url=config.database_url,
            min_size=getattr(config, "db_pool_min_size", 10),
            max_size=getattr(config, "db_pool_max_size", 20),
            health_check_interval=getattr(config, "db_health_check_interval", 30.0),
            log_slow_queries=getattr(config, "db_log_slow_queries", True),
            slow_query_threshold=getattr(config, "db_slow_query_threshold", 1.0),
        )

        _db_pool = DatabaseConnectionPool(pool_config)
        await _db_pool.initialize()

    return _db_pool


async def close_database_pool() -> None:
    """Close global database pool."""
    global _db_pool

    if _db_pool:
        await _db_pool.close()
        _db_pool = None


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database session."""
    pool = await get_database_pool()
    async with pool.get_session() as session:
        yield session


async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """FastAPI dependency for getting raw database connection."""
    pool = await get_database_pool()
    async with pool.acquire_connection() as connection:
        yield connection


# Health check functions for monitoring
async def check_database_health() -> Dict[str, Any]:
    """Check database health and return status."""
    try:
        pool = await get_database_pool()
        stats = pool.get_stats()

        # Perform test query
        start_time = time.time()
        await pool.execute_scalar("SELECT 1")
        response_time = time.time() - start_time

        health_status = "healthy"
        if stats.failed_health_checks > 0:
            health_status = "degraded"
        if stats.failed_health_checks >= 3:
            health_status = "unhealthy"

        return {
            "status": health_status,
            "response_time_ms": round(response_time * 1000, 2),
            "pool_size": stats.size,
            "free_connections": stats.free_size,
            "total_connections_created": stats.total_connections_created,
            "total_queries": stats.total_queries_executed,
            "avg_query_time_ms": round(stats.avg_query_time * 1000, 2),
            "failed_health_checks": stats.failed_health_checks,
            "last_health_check": stats.last_health_check,
        }

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
