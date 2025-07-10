"""
Database query optimization for Music Gen AI.

This module provides query optimization strategies, indexing recommendations,
and performance monitoring for database operations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from ...core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics for a database query."""

    query: str
    execution_time: float
    row_count: int
    plan: Optional[str] = None
    index_hits: int = 0
    index_misses: int = 0
    sequential_scans: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class IndexRecommendation:
    """Recommendation for creating a database index."""

    table_name: str
    columns: List[str]
    index_type: str = "btree"
    reason: str = ""
    estimated_improvement: float = 0.0
    query_patterns: List[str] = None

    def __post_init__(self):
        if self.query_patterns is None:
            self.query_patterns = []

    def to_sql(self) -> str:
        """Generate SQL for creating the index."""
        columns_str = ", ".join(self.columns)
        index_name = f"idx_{self.table_name}_{'_'.join(self.columns)}"

        if self.index_type == "gin":
            return f"CREATE INDEX {index_name} ON {self.table_name} USING gin({columns_str});"
        elif self.index_type == "gist":
            return f"CREATE INDEX {index_name} ON {self.table_name} USING gist({columns_str});"
        else:
            return f"CREATE INDEX {index_name} ON {self.table_name}({columns_str});"


class QueryOptimizer:
    """Optimizes database queries and provides performance recommendations."""

    def __init__(self):
        self.slow_query_threshold = 0.1  # 100ms
        self.query_stats: List[QueryStats] = []
        self.index_recommendations: List[IndexRecommendation] = []
        self._analyzed_tables: Set[str] = set()

    async def analyze_query(self, session: AsyncSession, query: str) -> QueryStats:
        """Analyze a query and collect performance statistics."""

        # Execute EXPLAIN ANALYZE
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        start_time = time.time()
        try:
            result = await session.execute(text(explain_query))
            explain_data = result.scalar()
            execution_time = time.time() - start_time

            # Parse explain output
            stats = self._parse_explain_output(explain_data[0] if explain_data else {})
            stats.query = query
            stats.execution_time = execution_time

            # Store stats for analysis
            self.query_stats.append(stats)

            # Check if query is slow
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected ({execution_time:.3f}s): {query[:100]}...")
                await self._analyze_slow_query(session, query, stats)

            return stats

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return QueryStats(query=query, execution_time=0, row_count=0)

    def _parse_explain_output(self, explain_data: Dict) -> QueryStats:
        """Parse EXPLAIN output to extract statistics."""

        plan = explain_data.get("Plan", {})

        # Extract key metrics
        total_time = plan.get("Actual Total Time", 0)
        row_count = plan.get("Actual Rows", 0)

        # Count index usage
        index_hits = 0
        index_misses = 0
        sequential_scans = 0

        def count_node_types(node):
            nonlocal index_hits, index_misses, sequential_scans

            node_type = node.get("Node Type", "")
            if "Index Scan" in node_type:
                index_hits += 1
            elif "Seq Scan" in node_type:
                sequential_scans += 1
                index_misses += 1

            # Recurse into child nodes
            for child in node.get("Plans", []):
                count_node_types(child)

        count_node_types(plan)

        return QueryStats(
            query="",
            execution_time=total_time / 1000,  # Convert to seconds
            row_count=row_count,
            plan=str(plan),
            index_hits=index_hits,
            index_misses=index_misses,
            sequential_scans=sequential_scans,
        )

    async def _analyze_slow_query(self, session: AsyncSession, query: str, stats: QueryStats):
        """Analyze a slow query and generate optimization recommendations."""

        # Extract table names from query
        tables = self._extract_table_names(query)

        for table in tables:
            if table not in self._analyzed_tables:
                await self._analyze_table_indexes(session, table)
                self._analyzed_tables.add(table)

        # Generate index recommendations based on query patterns
        if stats.sequential_scans > 0:
            # Look for WHERE clauses that could benefit from indexes
            where_columns = self._extract_where_columns(query)
            join_columns = self._extract_join_columns(query)

            for table, columns in where_columns.items():
                if columns:
                    self.index_recommendations.append(
                        IndexRecommendation(
                            table_name=table,
                            columns=list(columns),
                            reason=f"Sequential scan detected on WHERE clause",
                            estimated_improvement=stats.execution_time * 0.7,
                            query_patterns=[query],
                        )
                    )

            for table, columns in join_columns.items():
                if columns:
                    self.index_recommendations.append(
                        IndexRecommendation(
                            table_name=table,
                            columns=list(columns),
                            reason=f"Join operation without index",
                            estimated_improvement=stats.execution_time * 0.5,
                            query_patterns=[query],
                        )
                    )

    async def _analyze_table_indexes(self, session: AsyncSession, table_name: str):
        """Analyze existing indexes for a table."""

        # Query to get table statistics
        stats_query = """
        SELECT 
            schemaname,
            tablename,
            attname,
            n_distinct,
            most_common_vals,
            correlation
        FROM pg_stats
        WHERE tablename = :table_name
        """

        # Query to get existing indexes
        index_query = """
        SELECT 
            indexname,
            indexdef,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch
        FROM pg_indexes
        JOIN pg_stat_user_indexes USING (indexname)
        WHERE tablename = :table_name
        """

        try:
            # Get column statistics
            stats_result = await session.execute(text(stats_query), {"table_name": table_name})
            column_stats = stats_result.fetchall()

            # Get index statistics
            index_result = await session.execute(text(index_query), {"table_name": table_name})
            existing_indexes = index_result.fetchall()

            # Analyze for missing indexes
            high_cardinality_columns = [
                row["attname"] for row in column_stats if row["n_distinct"] > 100
            ]

            indexed_columns = set()
            for index in existing_indexes:
                # Extract column names from index definition
                indexdef = index["indexdef"]
                # Simple extraction - would need more robust parsing in production
                if "(" in indexdef and ")" in indexdef:
                    cols = indexdef.split("(")[1].split(")")[0]
                    indexed_columns.update(c.strip() for c in cols.split(","))

            # Recommend indexes for high cardinality columns without indexes
            for column in high_cardinality_columns:
                if column not in indexed_columns:
                    self.index_recommendations.append(
                        IndexRecommendation(
                            table_name=table_name,
                            columns=[column],
                            reason=f"High cardinality column without index",
                            estimated_improvement=0.3,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")

    def _extract_table_names(self, query: str) -> Set[str]:
        """Extract table names from a query."""
        # Simple extraction - in production, use proper SQL parser
        tables = set()
        query_upper = query.upper()

        # Look for FROM and JOIN clauses
        for keyword in ["FROM", "JOIN"]:
            if keyword in query_upper:
                parts = query_upper.split(keyword)
                for i in range(1, len(parts)):
                    # Extract next word as table name
                    words = parts[i].strip().split()
                    if words:
                        table = words[0].strip("(),")
                        if not any(kw in table for kw in ["SELECT", "WHERE", "GROUP", "ORDER"]):
                            tables.add(table.lower())

        return tables

    def _extract_where_columns(self, query: str) -> Dict[str, Set[str]]:
        """Extract columns used in WHERE clauses."""
        # Simple extraction - in production, use proper SQL parser
        result = {}
        query_upper = query.upper()

        if "WHERE" in query_upper:
            where_part = query_upper.split("WHERE")[1].split("GROUP BY")[0].split("ORDER BY")[0]

            # Look for column = value patterns
            import re

            pattern = r"(\w+)\.(\w+)\s*[=<>]"
            matches = re.findall(pattern, where_part)

            for table, column in matches:
                table = table.lower()
                column = column.lower()
                if table not in result:
                    result[table] = set()
                result[table].add(column)

        return result

    def _extract_join_columns(self, query: str) -> Dict[str, Set[str]]:
        """Extract columns used in JOIN conditions."""
        # Simple extraction - in production, use proper SQL parser
        result = {}
        query_upper = query.upper()

        if "JOIN" in query_upper:
            import re

            # Look for JOIN ... ON patterns
            pattern = r"JOIN\s+(\w+)\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)"
            matches = re.findall(pattern, query_upper)

            for match in matches:
                if len(match) >= 5:
                    table1, col1, table2, col2 = match[1], match[2], match[3], match[4]

                    for table, col in [(table1, col1), (table2, col2)]:
                        table = table.lower()
                        col = col.lower()
                        if table not in result:
                            result[table] = set()
                        result[table].add(col)

        return result

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""

        # Analyze query statistics
        total_queries = len(self.query_stats)
        slow_queries = [q for q in self.query_stats if q.execution_time > self.slow_query_threshold]

        # Calculate averages
        avg_execution_time = (
            sum(q.execution_time for q in self.query_stats) / total_queries
            if total_queries > 0
            else 0
        )
        total_sequential_scans = sum(q.sequential_scans for q in self.query_stats)

        # Group recommendations by table
        recommendations_by_table = {}
        for rec in self.index_recommendations:
            if rec.table_name not in recommendations_by_table:
                recommendations_by_table[rec.table_name] = []
            recommendations_by_table[rec.table_name].append(
                {
                    "columns": rec.columns,
                    "type": rec.index_type,
                    "reason": rec.reason,
                    "estimated_improvement": rec.estimated_improvement,
                    "sql": rec.to_sql(),
                }
            )

        return {
            "summary": {
                "total_queries_analyzed": total_queries,
                "slow_queries": len(slow_queries),
                "average_execution_time": avg_execution_time,
                "total_sequential_scans": total_sequential_scans,
                "index_recommendations": len(self.index_recommendations),
            },
            "slow_queries": [
                {
                    "query": q.query[:200] + "..." if len(q.query) > 200 else q.query,
                    "execution_time": q.execution_time,
                    "row_count": q.row_count,
                    "sequential_scans": q.sequential_scans,
                    "timestamp": q.timestamp.isoformat(),
                }
                for q in slow_queries[:10]  # Top 10 slowest
            ],
            "index_recommendations": recommendations_by_table,
            "optimization_sql": [rec.to_sql() for rec in self.index_recommendations],
        }


class QueryCache:
    """Simple query result cache for read operations."""

    def __init__(self, ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached query result."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Cache query result."""
        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


# Optimized query patterns
class OptimizedQueries:
    """Collection of optimized query patterns."""

    @staticmethod
    def get_user_generations_optimized(user_id: str, limit: int = 10, offset: int = 0) -> str:
        """Optimized query for fetching user generations with proper indexing."""
        return f"""
        SELECT 
            g.id,
            g.prompt,
            g.status,
            g.created_at,
            g.metadata,
            g.audio_url,
            g.duration
        FROM generations g
        WHERE g.user_id = '{user_id}'
        ORDER BY g.created_at DESC
        LIMIT {limit} OFFSET {offset}
        """

    @staticmethod
    def get_trending_tracks_optimized(hours: int = 24, limit: int = 10) -> str:
        """Optimized query for trending tracks using materialized view."""
        return f"""
        SELECT 
            t.id,
            t.title,
            t.user_id,
            t.audio_url,
            t.duration,
            COUNT(DISTINCT p.id) as play_count,
            COUNT(DISTINCT l.id) as like_count
        FROM tracks t
        LEFT JOIN plays p ON t.id = p.track_id 
            AND p.created_at > NOW() - INTERVAL '{hours} hours'
        LEFT JOIN likes l ON t.id = l.track_id 
            AND l.created_at > NOW() - INTERVAL '{hours} hours'
        WHERE t.is_public = true
        GROUP BY t.id
        ORDER BY play_count DESC, like_count DESC
        LIMIT {limit}
        """

    @staticmethod
    def search_tracks_optimized(
        query: str, filters: Dict[str, Any], limit: int = 20, offset: int = 0
    ) -> str:
        """Optimized full-text search query using GIN indexes."""
        conditions = ["t.is_public = true"]

        # Full-text search
        if query:
            conditions.append(f"t.search_vector @@ plainto_tsquery('english', '{query}')")

        # Apply filters
        if filters.get("genre"):
            conditions.append(f"t.genre = '{filters['genre']}'")
        if filters.get("user_id"):
            conditions.append(f"t.user_id = '{filters['user_id']}'")
        if filters.get("min_duration"):
            conditions.append(f"t.duration >= {filters['min_duration']}")

        where_clause = " AND ".join(conditions)

        return f"""
        SELECT 
            t.id,
            t.title,
            t.description,
            t.user_id,
            t.audio_url,
            t.duration,
            t.genre,
            ts_rank(t.search_vector, plainto_tsquery('english', '{query}')) as rank
        FROM tracks t
        WHERE {where_clause}
        ORDER BY rank DESC, t.created_at DESC
        LIMIT {limit} OFFSET {offset}
        """


# Database migration for performance indexes
PERFORMANCE_INDEXES = [
    # User-related indexes
    "CREATE INDEX idx_users_email ON users(email);",
    "CREATE INDEX idx_users_created_at ON users(created_at DESC);",
    # Generation indexes
    "CREATE INDEX idx_generations_user_id_created_at ON generations(user_id, created_at DESC);",
    "CREATE INDEX idx_generations_status ON generations(status) WHERE status != 'completed';",
    # Track indexes
    "CREATE INDEX idx_tracks_user_id ON tracks(user_id);",
    "CREATE INDEX idx_tracks_is_public_created_at ON tracks(is_public, created_at DESC) WHERE is_public = true;",
    "CREATE INDEX idx_tracks_genre ON tracks(genre) WHERE is_public = true;",
    # Full-text search
    "CREATE INDEX idx_tracks_search_vector ON tracks USING gin(search_vector);",
    # Play/Like indexes for trending
    "CREATE INDEX idx_plays_track_id_created_at ON plays(track_id, created_at DESC);",
    "CREATE INDEX idx_likes_track_id_created_at ON likes(track_id, created_at DESC);",
    # Composite indexes for common queries
    "CREATE INDEX idx_tracks_public_genre_created ON tracks(is_public, genre, created_at DESC);",
]
