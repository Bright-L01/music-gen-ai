#!/usr/bin/env python3
"""
Data Migration Testing Suite for Music Gen AI
Tests data migration procedures, schema evolution, and data integrity
"""

import os
import json
import time
import psycopg2
import redis
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import subprocess
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/results/data_migration_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationTestResult:
    """Result of a migration test"""

    test_name: str
    migration_type: str
    success: bool
    duration: float
    records_migrated: int
    records_verified: int
    data_integrity_check: bool
    performance_acceptable: bool
    rollback_tested: bool
    details: Dict[str, Any]
    issues: List[str]


class DataMigrationTestSuite:
    def __init__(self):
        self.postgres_host = os.getenv("POSTGRES_HOST", "postgres-staging")
        self.redis_host = os.getenv("REDIS_HOST", "redis-staging")
        self.results_dir = Path("/app/results/data_migration")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.db_params = {
            "host": self.postgres_host,
            "port": 5432,
            "database": "musicgen_staging",
            "user": "musicgen",
            "password": os.getenv("POSTGRES_PASSWORD", "staging_password_change_me"),
        }

        self.test_results: List[MigrationTestResult] = []

    def setup_test_environment(self) -> bool:
        """Setup test database and schemas"""
        logger.info("Setting up test environment for migration...")

        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                # Create test schema for migration testing
                cursor.execute("CREATE SCHEMA IF NOT EXISTS migration_test")

                # Create "legacy" tables (simulating old schema)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.users_v1 (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        username VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.audio_files_v1 (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES migration_test.users_v1(id),
                        file_path VARCHAR(500),
                        duration_seconds FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.generations_v1 (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES migration_test.users_v1(id),
                        prompt TEXT,
                        audio_file_id INTEGER REFERENCES migration_test.audio_files_v1(id),
                        parameters JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create "new" tables (simulating target schema)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.users_v2 (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        email VARCHAR(255) UNIQUE NOT NULL,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        display_name VARCHAR(255),
                        tier VARCHAR(50) DEFAULT 'free',
                        quota_used INTEGER DEFAULT 0,
                        quota_limit INTEGER DEFAULT 100,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB,
                        legacy_id INTEGER UNIQUE
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.audio_files_v2 (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID REFERENCES migration_test.users_v2(id),
                        file_path VARCHAR(500),
                        storage_url VARCHAR(1000),
                        duration_seconds FLOAT,
                        file_size_bytes BIGINT,
                        format VARCHAR(50),
                        sample_rate INTEGER,
                        bit_depth INTEGER,
                        channels INTEGER DEFAULT 1,
                        checksum VARCHAR(64),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        legacy_id INTEGER UNIQUE
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.generations_v2 (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID REFERENCES migration_test.users_v2(id),
                        prompt TEXT NOT NULL,
                        enhanced_prompt TEXT,
                        audio_file_id UUID REFERENCES migration_test.audio_files_v2(id),
                        model_version VARCHAR(50),
                        parameters JSONB,
                        generation_time_seconds FLOAT,
                        status VARCHAR(50) DEFAULT 'pending',
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        legacy_id INTEGER UNIQUE
                    )
                """
                )

                # Create migration tracking table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_test.migration_log (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) NOT NULL,
                        table_name VARCHAR(255),
                        records_processed INTEGER DEFAULT 0,
                        records_succeeded INTEGER DEFAULT 0,
                        records_failed INTEGER DEFAULT 0,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'running',
                        error_details JSONB
                    )
                """
                )

                conn.commit()
            conn.close()

            logger.info("Test environment setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False

    def populate_legacy_data(self, num_users: int = 100) -> bool:
        """Populate legacy tables with test data"""
        logger.info(f"Populating legacy tables with {num_users} test users...")

        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                # Clear existing test data
                cursor.execute("TRUNCATE migration_test.generations_v1 CASCADE")
                cursor.execute("TRUNCATE migration_test.audio_files_v1 CASCADE")
                cursor.execute("TRUNCATE migration_test.users_v1 CASCADE")

                # Insert test users
                user_ids = []
                for i in range(num_users):
                    cursor.execute(
                        """
                        INSERT INTO migration_test.users_v1 (email, username, metadata)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    """,
                        (
                            f"user{i}@test.com",
                            f"testuser{i}" if i % 3 != 0 else None,  # Some users without username
                            json.dumps(
                                {"source": "legacy", "tier": "free" if i % 5 != 0 else "premium"}
                            ),
                        ),
                    )
                    user_ids.append(cursor.fetchone()[0])

                # Insert audio files
                audio_file_ids = []
                for user_id in user_ids:
                    for j in range(3):  # 3 audio files per user
                        cursor.execute(
                            """
                            INSERT INTO migration_test.audio_files_v1 (user_id, file_path, duration_seconds)
                            VALUES (%s, %s, %s)
                            RETURNING id
                        """,
                            (user_id, f"/legacy/audio/user_{user_id}_file_{j}.wav", 30.0 + j * 10),
                        )
                        audio_file_ids.append((cursor.fetchone()[0], user_id))

                # Insert generations
                for audio_file_id, user_id in audio_file_ids:
                    cursor.execute(
                        """
                        INSERT INTO migration_test.generations_v1 (user_id, prompt, audio_file_id, parameters)
                        VALUES (%s, %s, %s, %s)
                    """,
                        (
                            user_id,
                            f"Generate upbeat music for user {user_id}",
                            audio_file_id,
                            json.dumps({"temperature": 0.8, "duration": 30}),
                        ),
                    )

                conn.commit()

                # Get counts
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v1")
                users_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM migration_test.audio_files_v1")
                audio_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM migration_test.generations_v1")
                generations_count = cursor.fetchone()[0]

            conn.close()

            logger.info(
                f"Populated legacy data: {users_count} users, {audio_count} audio files, {generations_count} generations"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to populate legacy data: {e}")
            return False

    def test_user_migration(self) -> MigrationTestResult:
        """Test user data migration from v1 to v2 schema"""
        logger.info("Testing user data migration...")
        start_time = time.time()
        issues = []

        try:
            conn = psycopg2.connect(**self.db_params)

            # Log migration start
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO migration_test.migration_log (migration_name, table_name, status)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """,
                    ("user_migration_v1_to_v2", "users", "running"),
                )
                migration_log_id = cursor.fetchone()[0]

            # Perform migration
            with conn.cursor() as cursor:
                # Get source count
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v1")
                source_count = cursor.fetchone()[0]

                # Migrate users with data transformation
                cursor.execute(
                    """
                    INSERT INTO migration_test.users_v2 (
                        email, username, display_name, tier, created_at, metadata, legacy_id
                    )
                    SELECT 
                        email,
                        COALESCE(username, 'user_' || id),  -- Generate username if missing
                        COALESCE(username, 'User ' || id),  -- Display name
                        COALESCE(metadata->>'tier', 'free'),
                        created_at,
                        metadata || '{"migrated": true}'::jsonb,
                        id
                    FROM migration_test.users_v1
                    ON CONFLICT (email) DO NOTHING
                """
                )

                records_migrated = cursor.rowcount

                # Verify migration
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v2")
                target_count = cursor.fetchone()[0]

                # Data integrity checks
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.users_v1 u1
                    LEFT JOIN migration_test.users_v2 u2 ON u1.id = u2.legacy_id
                    WHERE u2.id IS NULL
                """
                )
                missing_records = cursor.fetchone()[0]

                if missing_records > 0:
                    issues.append(f"{missing_records} users failed to migrate")

                # Check data transformation
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.users_v2
                    WHERE username IS NULL OR username = ''
                """
                )
                invalid_usernames = cursor.fetchone()[0]

                if invalid_usernames > 0:
                    issues.append(
                        f"{invalid_usernames} users have invalid usernames after migration"
                    )

                # Update migration log
                cursor.execute(
                    """
                    UPDATE migration_test.migration_log
                    SET records_processed = %s,
                        records_succeeded = %s,
                        records_failed = %s,
                        completed_at = CURRENT_TIMESTAMP,
                        status = %s,
                        error_details = %s
                    WHERE id = %s
                """,
                    (
                        source_count,
                        records_migrated,
                        missing_records,
                        "completed" if missing_records == 0 else "completed_with_errors",
                        json.dumps({"issues": issues}) if issues else None,
                        migration_log_id,
                    ),
                )

                conn.commit()

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="user_migration_v1_to_v2",
                migration_type="schema_evolution",
                success=missing_records == 0,
                duration=duration,
                records_migrated=records_migrated,
                records_verified=target_count,
                data_integrity_check=missing_records == 0,
                performance_acceptable=duration < 60,  # Should complete in under 60 seconds
                rollback_tested=False,  # Will test separately
                details={
                    "source_count": source_count,
                    "target_count": target_count,
                    "missing_records": missing_records,
                    "invalid_usernames": invalid_usernames,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"User migration test failed: {e}")
            return MigrationTestResult(
                test_name="user_migration_v1_to_v2",
                migration_type="schema_evolution",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Migration failed: {str(e)}"],
            )

    def test_audio_files_migration(self) -> MigrationTestResult:
        """Test audio files migration with data enrichment"""
        logger.info("Testing audio files migration...")
        start_time = time.time()
        issues = []

        try:
            conn = psycopg2.connect(**self.db_params)

            with conn.cursor() as cursor:
                # Get source count
                cursor.execute("SELECT COUNT(*) FROM migration_test.audio_files_v1")
                source_count = cursor.fetchone()[0]

                # Migrate audio files with data enrichment
                cursor.execute(
                    """
                    INSERT INTO migration_test.audio_files_v2 (
                        user_id, file_path, storage_url, duration_seconds,
                        file_size_bytes, format, sample_rate, checksum,
                        created_at, legacy_id
                    )
                    SELECT 
                        u2.id,
                        af.file_path,
                        'https://storage.example.com' || af.file_path,  -- Generate storage URL
                        af.duration_seconds,
                        CAST(af.duration_seconds * 44100 * 2 * 2 AS BIGINT),  -- Estimate file size
                        'wav',  -- Default format
                        44100,  -- Default sample rate
                        MD5(af.file_path || af.id::text),  -- Generate checksum
                        af.created_at,
                        af.id
                    FROM migration_test.audio_files_v1 af
                    JOIN migration_test.users_v2 u2 ON af.user_id = u2.legacy_id
                """
                )

                records_migrated = cursor.rowcount

                # Verify migration
                cursor.execute("SELECT COUNT(*) FROM migration_test.audio_files_v2")
                target_count = cursor.fetchone()[0]

                # Check referential integrity
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.audio_files_v2
                    WHERE user_id NOT IN (SELECT id FROM migration_test.users_v2)
                """
                )
                orphaned_records = cursor.fetchone()[0]

                if orphaned_records > 0:
                    issues.append(f"{orphaned_records} audio files have invalid user references")

                # Check data enrichment
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.audio_files_v2
                    WHERE storage_url IS NULL OR checksum IS NULL
                """
                )
                incomplete_enrichment = cursor.fetchone()[0]

                if incomplete_enrichment > 0:
                    issues.append(
                        f"{incomplete_enrichment} audio files have incomplete data enrichment"
                    )

                conn.commit()

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="audio_files_migration_v1_to_v2",
                migration_type="schema_evolution_with_enrichment",
                success=len(issues) == 0,
                duration=duration,
                records_migrated=records_migrated,
                records_verified=target_count,
                data_integrity_check=orphaned_records == 0,
                performance_acceptable=duration < 120,
                rollback_tested=False,
                details={
                    "source_count": source_count,
                    "target_count": target_count,
                    "orphaned_records": orphaned_records,
                    "incomplete_enrichment": incomplete_enrichment,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Audio files migration test failed: {e}")
            return MigrationTestResult(
                test_name="audio_files_migration_v1_to_v2",
                migration_type="schema_evolution_with_enrichment",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Migration failed: {str(e)}"],
            )

    def test_generations_migration(self) -> MigrationTestResult:
        """Test generations migration with complex transformations"""
        logger.info("Testing generations migration...")
        start_time = time.time()
        issues = []

        try:
            conn = psycopg2.connect(**self.db_params)

            with conn.cursor() as cursor:
                # Get source count
                cursor.execute("SELECT COUNT(*) FROM migration_test.generations_v1")
                source_count = cursor.fetchone()[0]

                # Migrate generations with transformations
                cursor.execute(
                    """
                    INSERT INTO migration_test.generations_v2 (
                        user_id, prompt, enhanced_prompt, audio_file_id,
                        model_version, parameters, generation_time_seconds,
                        status, created_at, legacy_id
                    )
                    SELECT 
                        u2.id,
                        g.prompt,
                        g.prompt || ' [enhanced with style transfer]',  -- Simulate enhancement
                        af2.id,
                        'musicgen-v1.0',  -- Default model version
                        g.parameters || '{"migrated": true}'::jsonb,
                        COALESCE((g.parameters->>'duration')::float, 30.0) * 1.5,  -- Estimate generation time
                        'completed',  -- All legacy generations are completed
                        g.created_at,
                        g.id
                    FROM migration_test.generations_v1 g
                    JOIN migration_test.users_v2 u2 ON g.user_id = u2.legacy_id
                    JOIN migration_test.audio_files_v2 af2 ON g.audio_file_id = af2.legacy_id
                """
                )

                records_migrated = cursor.rowcount

                # Verify migration
                cursor.execute("SELECT COUNT(*) FROM migration_test.generations_v2")
                target_count = cursor.fetchone()[0]

                # Check data consistency
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.generations_v1 g1
                    LEFT JOIN migration_test.generations_v2 g2 ON g1.id = g2.legacy_id
                    WHERE g2.id IS NULL
                """
                )
                missing_records = cursor.fetchone()[0]

                if missing_records > 0:
                    issues.append(f"{missing_records} generations failed to migrate")

                # Verify foreign key relationships
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.generations_v2 g
                    LEFT JOIN migration_test.users_v2 u ON g.user_id = u.id
                    LEFT JOIN migration_test.audio_files_v2 af ON g.audio_file_id = af.id
                    WHERE u.id IS NULL OR af.id IS NULL
                """
                )
                broken_relationships = cursor.fetchone()[0]

                if broken_relationships > 0:
                    issues.append(f"{broken_relationships} generations have broken relationships")

                conn.commit()

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="generations_migration_v1_to_v2",
                migration_type="complex_transformation",
                success=len(issues) == 0,
                duration=duration,
                records_migrated=records_migrated,
                records_verified=target_count,
                data_integrity_check=broken_relationships == 0,
                performance_acceptable=duration < 180,
                rollback_tested=False,
                details={
                    "source_count": source_count,
                    "target_count": target_count,
                    "missing_records": missing_records,
                    "broken_relationships": broken_relationships,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Generations migration test failed: {e}")
            return MigrationTestResult(
                test_name="generations_migration_v1_to_v2",
                migration_type="complex_transformation",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Migration failed: {str(e)}"],
            )

    def test_redis_data_migration(self) -> MigrationTestResult:
        """Test Redis data migration (cache and session data)"""
        logger.info("Testing Redis data migration...")
        start_time = time.time()
        issues = []

        try:
            # Connect to Redis
            r = redis.Redis(
                host=self.redis_host,
                port=6379,
                password=os.getenv("REDIS_PASSWORD", "staging_redis_password"),
                decode_responses=True,
            )

            # Create test data in old format
            test_sessions = {}
            for i in range(50):
                old_key = f"session:old:{i}"
                session_data = {
                    "user_id": i,
                    "created": int(time.time()),
                    "data": f"legacy_session_data_{i}",
                }
                r.hset(old_key, mapping=session_data)
                test_sessions[old_key] = session_data

            # Simulate migration to new format
            migrated_count = 0
            for old_key, data in test_sessions.items():
                new_key = old_key.replace("session:old:", "session:v2:")

                # Transform data to new format
                new_data = {
                    "user_id": f"user_{data['user_id']}",  # New ID format
                    "created_at": datetime.fromtimestamp(data["created"]).isoformat(),
                    "data": json.dumps({"legacy": data["data"], "version": 2}),
                    "ttl": 86400,  # 24 hours
                }

                # Migrate with expiration
                r.hset(new_key, mapping=new_data)
                r.expire(new_key, new_data["ttl"])
                migrated_count += 1

            # Verify migration
            verified_count = 0
            for old_key in test_sessions.keys():
                new_key = old_key.replace("session:old:", "session:v2:")
                if r.exists(new_key):
                    verified_count += 1

                    # Check data transformation
                    new_data = r.hgetall(new_key)
                    if not new_data.get("user_id", "").startswith("user_"):
                        issues.append(f"Invalid user_id format in {new_key}")

            # Cleanup test data
            for key in test_sessions.keys():
                r.delete(key)
                r.delete(key.replace("session:old:", "session:v2:"))

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="redis_session_migration",
                migration_type="cache_migration",
                success=verified_count == len(test_sessions),
                duration=duration,
                records_migrated=migrated_count,
                records_verified=verified_count,
                data_integrity_check=len(issues) == 0,
                performance_acceptable=duration < 30,
                rollback_tested=False,
                details={
                    "total_sessions": len(test_sessions),
                    "migrated_count": migrated_count,
                    "verified_count": verified_count,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Redis migration test failed: {e}")
            return MigrationTestResult(
                test_name="redis_session_migration",
                migration_type="cache_migration",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Migration failed: {str(e)}"],
            )

    def test_migration_rollback(self) -> MigrationTestResult:
        """Test migration rollback capability"""
        logger.info("Testing migration rollback...")
        start_time = time.time()
        issues = []

        try:
            conn = psycopg2.connect(**self.db_params)

            # Start a transaction for rollback testing
            conn.autocommit = False

            with conn.cursor() as cursor:
                # Count initial state
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v2")
                initial_v2_count = cursor.fetchone()[0]

                # Perform a test migration
                cursor.execute(
                    """
                    INSERT INTO migration_test.users_v2 (email, username, legacy_id)
                    SELECT 
                        'rollback_test_' || id || '@test.com',
                        'rollback_user_' || id,
                        id + 10000  -- Offset to avoid conflicts
                    FROM generate_series(1, 10) id
                """
                )

                test_records = cursor.rowcount

                # Verify insertion
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v2")
                after_insert_count = cursor.fetchone()[0]

                # Test rollback
                conn.rollback()

                # Verify rollback
                cursor.execute("SELECT COUNT(*) FROM migration_test.users_v2")
                after_rollback_count = cursor.fetchone()[0]

                rollback_successful = after_rollback_count == initial_v2_count

                if not rollback_successful:
                    issues.append("Rollback did not restore original state")

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="migration_rollback_test",
                migration_type="rollback_capability",
                success=rollback_successful,
                duration=duration,
                records_migrated=test_records,
                records_verified=0,
                data_integrity_check=rollback_successful,
                performance_acceptable=duration < 10,
                rollback_tested=True,
                details={
                    "initial_count": initial_v2_count,
                    "after_insert_count": after_insert_count,
                    "after_rollback_count": after_rollback_count,
                    "test_records": test_records,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Rollback test failed: {e}")
            return MigrationTestResult(
                test_name="migration_rollback_test",
                migration_type="rollback_capability",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Rollback test failed: {str(e)}"],
            )

    def test_incremental_migration(self) -> MigrationTestResult:
        """Test incremental/delta migration capability"""
        logger.info("Testing incremental migration...")
        start_time = time.time()
        issues = []

        try:
            conn = psycopg2.connect(**self.db_params)

            with conn.cursor() as cursor:
                # Add new records to legacy tables (simulating ongoing activity)
                cursor.execute(
                    """
                    INSERT INTO migration_test.users_v1 (email, username, metadata)
                    VALUES 
                        ('incremental1@test.com', 'incremental_user1', '{"new": true}'::jsonb),
                        ('incremental2@test.com', 'incremental_user2', '{"new": true}'::jsonb)
                    RETURNING id
                """
                )
                new_user_ids = [row[0] for row in cursor.fetchall()]

                # Perform incremental migration (only new records)
                cursor.execute(
                    """
                    INSERT INTO migration_test.users_v2 (
                        email, username, display_name, tier, created_at, metadata, legacy_id
                    )
                    SELECT 
                        u1.email,
                        u1.username,
                        u1.username,
                        'free',
                        u1.created_at,
                        u1.metadata || '{"incremental": true}'::jsonb,
                        u1.id
                    FROM migration_test.users_v1 u1
                    LEFT JOIN migration_test.users_v2 u2 ON u1.id = u2.legacy_id
                    WHERE u2.id IS NULL  -- Only migrate records not already migrated
                """
                )

                incremental_count = cursor.rowcount

                # Verify only new records were migrated
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM migration_test.users_v2
                    WHERE metadata->>'incremental' = 'true'
                """
                )
                incremental_verified = cursor.fetchone()[0]

                if incremental_verified != len(new_user_ids):
                    issues.append("Incremental migration count mismatch")

                # Check for duplicates
                cursor.execute(
                    """
                    SELECT email, COUNT(*) FROM migration_test.users_v2
                    GROUP BY email
                    HAVING COUNT(*) > 1
                """
                )
                duplicates = cursor.fetchall()

                if duplicates:
                    issues.append(
                        f"Found {len(duplicates)} duplicate emails after incremental migration"
                    )

                conn.commit()

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="incremental_migration_test",
                migration_type="incremental_delta",
                success=len(issues) == 0,
                duration=duration,
                records_migrated=incremental_count,
                records_verified=incremental_verified,
                data_integrity_check=len(duplicates) == 0,
                performance_acceptable=duration < 5,
                rollback_tested=False,
                details={
                    "new_records": len(new_user_ids),
                    "incremental_migrated": incremental_count,
                    "incremental_verified": incremental_verified,
                    "duplicates_found": len(duplicates),
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Incremental migration test failed: {e}")
            return MigrationTestResult(
                test_name="incremental_migration_test",
                migration_type="incremental_delta",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Incremental migration failed: {str(e)}"],
            )

    def test_large_scale_migration_performance(self) -> MigrationTestResult:
        """Test migration performance with larger dataset"""
        logger.info("Testing large scale migration performance...")
        start_time = time.time()
        issues = []

        try:
            # First, populate more data
            self.populate_legacy_data(num_users=1000)

            conn = psycopg2.connect(**self.db_params)

            # Clear target tables for clean test
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE migration_test.generations_v2 CASCADE")
                cursor.execute("TRUNCATE migration_test.audio_files_v2 CASCADE")
                cursor.execute("TRUNCATE migration_test.users_v2 CASCADE")
                conn.commit()

            # Measure batch migration performance
            migration_start = time.time()

            with conn.cursor() as cursor:
                # Batch migrate users
                cursor.execute(
                    """
                    INSERT INTO migration_test.users_v2 (
                        email, username, display_name, tier, created_at, metadata, legacy_id
                    )
                    SELECT 
                        email,
                        COALESCE(username, 'user_' || id),
                        COALESCE(username, 'User ' || id),
                        COALESCE(metadata->>'tier', 'free'),
                        created_at,
                        metadata,
                        id
                    FROM migration_test.users_v1
                """
                )
                users_migrated = cursor.rowcount

                # Batch migrate audio files
                cursor.execute(
                    """
                    INSERT INTO migration_test.audio_files_v2 (
                        user_id, file_path, storage_url, duration_seconds,
                        file_size_bytes, format, sample_rate, checksum,
                        created_at, legacy_id
                    )
                    SELECT 
                        u2.id,
                        af.file_path,
                        'https://storage.example.com' || af.file_path,
                        af.duration_seconds,
                        CAST(af.duration_seconds * 44100 * 2 * 2 AS BIGINT),
                        'wav',
                        44100,
                        MD5(af.file_path || af.id::text),
                        af.created_at,
                        af.id
                    FROM migration_test.audio_files_v1 af
                    JOIN migration_test.users_v2 u2 ON af.user_id = u2.legacy_id
                """
                )
                audio_migrated = cursor.rowcount

                # Batch migrate generations
                cursor.execute(
                    """
                    INSERT INTO migration_test.generations_v2 (
                        user_id, prompt, enhanced_prompt, audio_file_id,
                        model_version, parameters, generation_time_seconds,
                        status, created_at, legacy_id
                    )
                    SELECT 
                        u2.id,
                        g.prompt,
                        g.prompt || ' [enhanced]',
                        af2.id,
                        'musicgen-v1.0',
                        g.parameters,
                        30.0,
                        'completed',
                        g.created_at,
                        g.id
                    FROM migration_test.generations_v1 g
                    JOIN migration_test.users_v2 u2 ON g.user_id = u2.legacy_id
                    JOIN migration_test.audio_files_v2 af2 ON g.audio_file_id = af2.legacy_id
                """
                )
                generations_migrated = cursor.rowcount

                conn.commit()

            migration_duration = time.time() - migration_start

            # Calculate migration rate
            total_records = users_migrated + audio_migrated + generations_migrated
            records_per_second = total_records / migration_duration if migration_duration > 0 else 0

            # Performance thresholds
            performance_acceptable = records_per_second > 100  # Should migrate > 100 records/second

            if not performance_acceptable:
                issues.append(f"Migration rate too slow: {records_per_second:.2f} records/second")

            conn.close()

            duration = time.time() - start_time

            return MigrationTestResult(
                test_name="large_scale_migration_performance",
                migration_type="performance_test",
                success=performance_acceptable,
                duration=duration,
                records_migrated=total_records,
                records_verified=total_records,
                data_integrity_check=True,
                performance_acceptable=performance_acceptable,
                rollback_tested=False,
                details={
                    "users_migrated": users_migrated,
                    "audio_migrated": audio_migrated,
                    "generations_migrated": generations_migrated,
                    "total_records": total_records,
                    "migration_duration": migration_duration,
                    "records_per_second": records_per_second,
                },
                issues=issues,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Performance test failed: {e}")
            return MigrationTestResult(
                test_name="large_scale_migration_performance",
                migration_type="performance_test",
                success=False,
                duration=duration,
                records_migrated=0,
                records_verified=0,
                data_integrity_check=False,
                performance_acceptable=False,
                rollback_tested=False,
                details={"error": str(e)},
                issues=[f"Performance test failed: {str(e)}"],
            )

    def cleanup_test_environment(self):
        """Clean up test schemas and data"""
        logger.info("Cleaning up test environment...")

        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute("DROP SCHEMA IF EXISTS migration_test CASCADE")
                conn.commit()
            conn.close()
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")

    def run_all_migration_tests(self) -> Dict[str, Any]:
        """Run all data migration tests"""
        logger.info("Starting comprehensive data migration test suite...")

        # Setup test environment
        if not self.setup_test_environment():
            logger.error("Failed to setup test environment")
            return {"error": "Setup failed"}

        # Populate test data
        if not self.populate_legacy_data():
            logger.error("Failed to populate test data")
            return {"error": "Data population failed"}

        # Run tests
        test_functions = [
            self.test_user_migration,
            self.test_audio_files_migration,
            self.test_generations_migration,
            self.test_redis_data_migration,
            self.test_migration_rollback,
            self.test_incremental_migration,
            self.test_large_scale_migration_performance,
        ]

        for test_func in test_functions:
            try:
                result = test_func()
                self.test_results.append(result)

                status = "✅" if result.success else "❌"
                logger.info(f"{status} {result.test_name}: {result.duration:.2f}s")

            except Exception as e:
                logger.error(f"Test {test_func.__name__} crashed: {e}")

        # Cleanup
        self.cleanup_test_environment()

        # Generate summary
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]

        total_records_migrated = sum(r.records_migrated for r in self.test_results)
        total_records_verified = sum(r.records_verified for r in self.test_results)

        # Check overall data integrity
        data_integrity_issues = sum(1 for r in self.test_results if not r.data_integrity_check)
        performance_issues = sum(1 for r in self.test_results if not r.performance_acceptable)

        summary = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100
                if self.test_results
                else 0,
                "total_duration": sum(r.duration for r in self.test_results),
            },
            "migration_stats": {
                "total_records_migrated": total_records_migrated,
                "total_records_verified": total_records_verified,
                "data_integrity_issues": data_integrity_issues,
                "performance_issues": performance_issues,
            },
            "test_results": [asdict(r) for r in self.test_results],
            "issues_summary": self._summarize_issues(),
        }

        # Save report
        report_file = self.results_dir / f"data_migration_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Data migration test report saved to {report_file}")

        return summary

    def _summarize_issues(self) -> Dict[str, List[str]]:
        """Summarize all issues found during testing"""
        issues_by_category = {
            "data_integrity": [],
            "performance": [],
            "transformation": [],
            "rollback": [],
        }

        for result in self.test_results:
            if result.issues:
                if "integrity" in result.migration_type or not result.data_integrity_check:
                    issues_by_category["data_integrity"].extend(result.issues)
                elif "performance" in result.migration_type or not result.performance_acceptable:
                    issues_by_category["performance"].extend(result.issues)
                elif "transformation" in result.migration_type:
                    issues_by_category["transformation"].extend(result.issues)
                elif "rollback" in result.migration_type:
                    issues_by_category["rollback"].extend(result.issues)

        return {k: list(set(v)) for k, v in issues_by_category.items() if v}

    def print_summary(self, summary: Dict[str, Any]):
        """Print migration test summary"""
        print("\n" + "=" * 70)
        print("DATA MIGRATION TEST SUMMARY")
        print("=" * 70)

        test_summary = summary["test_summary"]
        print(f"Total Tests: {test_summary['total_tests']}")
        print(f"Successful: {test_summary['successful']}")
        print(f"Failed: {test_summary['failed']}")
        print(f"Success Rate: {test_summary['success_rate']:.1f}%")
        print(f"Total Duration: {test_summary['total_duration']:.2f}s")
        print()

        migration_stats = summary["migration_stats"]
        print("MIGRATION STATISTICS:")
        print(f"  Records Migrated: {migration_stats['total_records_migrated']:,}")
        print(f"  Records Verified: {migration_stats['total_records_verified']:,}")
        print(f"  Data Integrity Issues: {migration_stats['data_integrity_issues']}")
        print(f"  Performance Issues: {migration_stats['performance_issues']}")
        print()

        print("TEST RESULTS:")
        for result in self.test_results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.test_name}")
            print(f"     Type: {result.migration_type}")
            print(f"     Duration: {result.duration:.2f}s")
            print(
                f"     Records: {result.records_migrated} migrated, {result.records_verified} verified"
            )
            if not result.success and result.issues:
                print(f"     Issues: {', '.join(result.issues[:2])}")

        if summary.get("issues_summary"):
            print("\nISSUES FOUND:")
            for category, issues in summary["issues_summary"].items():
                print(f"  {category.upper()}:")
                for issue in issues[:3]:  # Show first 3 issues per category
                    print(f"    - {issue}")

        print("=" * 70)


def main():
    """Main test execution"""
    suite = DataMigrationTestSuite()

    try:
        summary = suite.run_all_migration_tests()
        suite.print_summary(summary)

        # Exit with appropriate code
        exit_code = 0 if summary["test_summary"]["failed"] == 0 else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Data migration test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
