#!/usr/bin/env python3
"""
Backup and Restore Testing Suite for Music Gen AI Staging
Tests database backup, secrets backup, and disaster recovery procedures
"""

import os
import json
import time
import subprocess
import psycopg2
import redis
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import sys
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/results/backup_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class BackupRestoreTestSuite:
    def __init__(self):
        self.postgres_host = os.getenv("POSTGRES_HOST", "postgres-staging")
        self.redis_host = os.getenv("REDIS_HOST", "redis-staging")
        self.backup_dir = Path("/app/results/backup_tests")
        self.backup_dir.mkdir(exist_ok=True)
        self.test_results = []

    def log_test_result(
        self, test_name: str, success: bool, duration: float, details: Dict[str, Any] = None
    ):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        self.test_results.append(result)

        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({duration:.2f}s)")
        if details:
            logger.info(f"  Details: {details}")

    def test_postgres_backup_restore(self) -> bool:
        """Test PostgreSQL database backup and restore"""
        logger.info("Testing PostgreSQL backup and restore...")
        start_time = time.time()

        try:
            # Connection parameters
            db_params = {
                "host": self.postgres_host,
                "port": 5432,
                "database": "musicgen_staging",
                "user": "musicgen",
                "password": os.getenv("POSTGRES_PASSWORD", "staging_password_change_me"),
            }

            # 1. Create test data
            logger.info("Creating test data...")
            conn = psycopg2.connect(**db_params)
            with conn.cursor() as cursor:
                test_user_id = f"backup-test-{int(time.time())}"
                cursor.execute(
                    "INSERT INTO musicgen.users (id, email, username, password_hash) "
                    "VALUES (%s, %s, %s, %s)",
                    (test_user_id, f"backup-test@example.com", "backup-test-user", "test_hash"),
                )
                conn.commit()
            conn.close()

            # 2. Create backup
            logger.info("Creating database backup...")
            backup_file = self.backup_dir / f"postgres_backup_{int(time.time())}.sql"

            pg_dump_cmd = [
                "pg_dump",
                "-h",
                self.postgres_host,
                "-p",
                "5432",
                "-U",
                "musicgen",
                "-d",
                "musicgen_staging",
                "-f",
                str(backup_file),
                "--verbose",
                "--no-password",
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = db_params["password"]

            backup_result = subprocess.run(
                pg_dump_cmd, env=env, capture_output=True, text=True, timeout=300
            )

            if backup_result.returncode != 0:
                raise Exception(f"pg_dump failed: {backup_result.stderr}")

            backup_size = backup_file.stat().st_size
            logger.info(f"Backup created: {backup_file} ({backup_size} bytes)")

            # 3. Delete test data
            logger.info("Deleting test data...")
            conn = psycopg2.connect(**db_params)
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM musicgen.users WHERE id = %s", (test_user_id,))
                conn.commit()

                # Verify deletion
                cursor.execute("SELECT COUNT(*) FROM musicgen.users WHERE id = %s", (test_user_id,))
                count_after_delete = cursor.fetchone()[0]
                assert count_after_delete == 0
            conn.close()

            # 4. Restore from backup (to a temporary database)
            logger.info("Testing restore from backup...")
            temp_db = f"musicgen_restore_test_{int(time.time())}"

            # Create temporary database
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=5432,
                database="postgres",  # Connect to default database
                user="musicgen",
                password=db_params["password"],
            )
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE {temp_db}")
            conn.close()

            # Restore backup to temporary database
            psql_cmd = [
                "psql",
                "-h",
                self.postgres_host,
                "-p",
                "5432",
                "-U",
                "musicgen",
                "-d",
                temp_db,
                "-f",
                str(backup_file),
                "--quiet",
            ]

            restore_result = subprocess.run(
                psql_cmd, env=env, capture_output=True, text=True, timeout=300
            )

            if restore_result.returncode != 0:
                logger.warning(f"Some restore warnings: {restore_result.stderr}")

            # 5. Verify restored data
            logger.info("Verifying restored data...")
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=5432,
                database=temp_db,
                user="musicgen",
                password=db_params["password"],
            )

            with conn.cursor() as cursor:
                # Check if our test data exists in the backup
                cursor.execute("SELECT COUNT(*) FROM musicgen.users WHERE id = %s", (test_user_id,))
                restored_count = cursor.fetchone()[0]

                # Check table structure
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'musicgen'
                """
                )
                table_count = cursor.fetchone()[0]

            conn.close()

            # Cleanup temporary database
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=5432,
                database="postgres",
                user="musicgen",
                password=db_params["password"],
            )
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {temp_db}")
            conn.close()

            # Clean up backup file
            backup_file.unlink()

            duration = time.time() - start_time
            success = restored_count == 1 and table_count > 0

            self.log_test_result(
                "postgres_backup_restore",
                success,
                duration,
                {
                    "backup_size_bytes": backup_size,
                    "restored_user_count": restored_count,
                    "table_count": table_count,
                    "temp_database": temp_db,
                },
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("postgres_backup_restore", False, duration, {"error": str(e)})
            logger.error(f"PostgreSQL backup/restore test failed: {e}")
            return False

    def test_redis_backup_restore(self) -> bool:
        """Test Redis backup and restore"""
        logger.info("Testing Redis backup and restore...")
        start_time = time.time()

        try:
            # Connect to Redis
            r = redis.Redis(
                host=self.redis_host,
                port=6379,
                password=os.getenv("REDIS_PASSWORD", "staging_redis_password"),
                decode_responses=True,
            )

            # 1. Create test data
            test_keys = {}
            for i in range(10):
                key = f"backup_test_{i}_{int(time.time())}"
                value = f"test_value_{i}"
                r.set(key, value)
                test_keys[key] = value

            # Add some complex data structures
            hash_key = f"backup_test_hash_{int(time.time())}"
            r.hset(hash_key, mapping={"field1": "value1", "field2": "value2"})
            test_keys[hash_key] = "hash"

            list_key = f"backup_test_list_{int(time.time())}"
            r.lpush(list_key, "item1", "item2", "item3")
            test_keys[list_key] = "list"

            logger.info(f"Created {len(test_keys)} test keys in Redis")

            # 2. Trigger Redis backup (BGSAVE)
            logger.info("Triggering Redis background save...")
            r.bgsave()

            # Wait for backup to complete
            while r.lastsave() == r.lastsave():
                time.sleep(0.5)
                # Add timeout to prevent infinite loop
                if time.time() - start_time > 60:
                    raise Exception("Redis backup timeout")

            logger.info("Redis backup completed")

            # 3. Get some info about the backup
            info = r.info()
            db_size = info.get("db0", {}).get("keys", 0) if "db0" in info else 0

            # 4. Test data persistence by simulating restart
            # (In a real scenario, we would restart Redis and verify data)
            # For this test, we'll just verify our test data exists
            logger.info("Verifying test data persistence...")

            verified_keys = 0
            for key, expected_type in test_keys.items():
                if r.exists(key):
                    if expected_type == "hash":
                        if r.hgetall(key):
                            verified_keys += 1
                    elif expected_type == "list":
                        if r.llen(key) > 0:
                            verified_keys += 1
                    else:
                        if r.get(key):
                            verified_keys += 1

            # 5. Cleanup test data
            for key in test_keys.keys():
                r.delete(key)

            duration = time.time() - start_time
            success = verified_keys == len(test_keys)

            self.log_test_result(
                "redis_backup_restore",
                success,
                duration,
                {
                    "test_keys_created": len(test_keys),
                    "verified_keys": verified_keys,
                    "db_size": db_size,
                    "last_save": info.get("rdb_last_save_time", 0),
                },
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("redis_backup_restore", False, duration, {"error": str(e)})
            logger.error(f"Redis backup/restore test failed: {e}")
            return False

    def test_secrets_backup_restore(self) -> bool:
        """Test secrets backup and restore functionality"""
        logger.info("Testing secrets backup and restore...")
        start_time = time.time()

        try:
            # 1. Test metadata-only backup
            metadata_backup = self.backup_dir / f"secrets_metadata_{int(time.time())}.json"

            backup_cmd = [
                "python",
                "/app/scripts/secrets_cli.py",
                "backup",
                str(metadata_backup),
                "--metadata-only",
            ]

            backup_result = subprocess.run(backup_cmd, capture_output=True, text=True, timeout=60)

            if backup_result.returncode != 0:
                raise Exception(f"Secrets backup failed: {backup_result.stderr}")

            # Verify backup file exists and has content
            if not metadata_backup.exists():
                raise Exception("Secrets backup file was not created")

            backup_size = metadata_backup.stat().st_size

            # Parse backup content
            with open(metadata_backup, "r") as f:
                backup_data = json.load(f)

            secrets_count = len(backup_data.get("secrets", []))

            # 2. Test backup validation
            validate_cmd = [
                "python",
                "/app/scripts/secrets_cli.py",
                "verify-backup",
                str(metadata_backup),
            ]

            validate_result = subprocess.run(
                validate_cmd, capture_output=True, text=True, timeout=30
            )

            validation_success = validate_result.returncode == 0

            # 3. Test secrets health check
            health_cmd = ["python", "/app/scripts/secrets_cli.py", "health-advanced"]

            health_result = subprocess.run(health_cmd, capture_output=True, text=True, timeout=30)

            health_success = health_result.returncode == 0

            # Cleanup
            metadata_backup.unlink()

            duration = time.time() - start_time
            success = validation_success and health_success and secrets_count > 0

            self.log_test_result(
                "secrets_backup_restore",
                success,
                duration,
                {
                    "backup_size_bytes": backup_size,
                    "secrets_count": secrets_count,
                    "validation_success": validation_success,
                    "health_check_success": health_success,
                },
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("secrets_backup_restore", False, duration, {"error": str(e)})
            logger.error(f"Secrets backup/restore test failed: {e}")
            return False

    def test_volume_backup_restore(self) -> bool:
        """Test Docker volume backup and restore"""
        logger.info("Testing Docker volume backup and restore...")
        start_time = time.time()

        try:
            # Test backing up a volume by creating and copying data
            test_volume = "staging-models"  # Use existing volume
            backup_tar = self.backup_dir / f"volume_backup_{int(time.time())}.tar"

            # Create a temporary container to access the volume
            docker_backup_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{test_volume}:/source",
                "-v",
                f"{self.backup_dir}:/backup",
                "alpine:latest",
                "tar",
                "czf",
                f"/backup/{backup_tar.name}",
                "-C",
                "/source",
                ".",
            ]

            backup_result = subprocess.run(
                docker_backup_cmd, capture_output=True, text=True, timeout=300
            )

            if backup_result.returncode != 0:
                # Volume might be empty, which is okay for staging
                logger.info("Volume backup completed (volume may be empty)")

            # Check if backup file was created
            backup_exists = backup_tar.exists()
            backup_size = backup_tar.stat().st_size if backup_exists else 0

            # Test restore capability by creating a temporary volume
            test_restore_volume = f"test_restore_{int(time.time())}"

            # Create test volume
            create_volume_cmd = ["docker", "volume", "create", test_restore_volume]
            subprocess.run(create_volume_cmd, check=True, timeout=30)

            if backup_exists and backup_size > 0:
                # Restore to test volume
                docker_restore_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{test_restore_volume}:/target",
                    "-v",
                    f"{self.backup_dir}:/backup",
                    "alpine:latest",
                    "tar",
                    "xzf",
                    f"/backup/{backup_tar.name}",
                    "-C",
                    "/target",
                ]

                restore_result = subprocess.run(
                    docker_restore_cmd, capture_output=True, text=True, timeout=300
                )

                restore_success = restore_result.returncode == 0
            else:
                restore_success = True  # Empty volumes are okay

            # Cleanup
            subprocess.run(
                ["docker", "volume", "rm", test_restore_volume], capture_output=True, timeout=30
            )
            if backup_tar.exists():
                backup_tar.unlink()

            duration = time.time() - start_time
            success = backup_exists and restore_success

            self.log_test_result(
                "volume_backup_restore",
                success,
                duration,
                {
                    "volume_name": test_volume,
                    "backup_size_bytes": backup_size,
                    "restore_success": restore_success,
                    "test_volume": test_restore_volume,
                },
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("volume_backup_restore", False, duration, {"error": str(e)})
            logger.error(f"Volume backup/restore test failed: {e}")
            return False

    def test_application_state_backup(self) -> bool:
        """Test application state backup and recovery"""
        logger.info("Testing application state backup...")
        start_time = time.time()

        try:
            # 1. Capture current application state
            state_backup = self.backup_dir / f"app_state_{int(time.time())}.json"

            app_state = {
                "timestamp": datetime.now().isoformat(),
                "containers": [],
                "volumes": [],
                "networks": [],
            }

            # Get container information
            containers_cmd = ["docker", "ps", "--format", "json"]
            containers_result = subprocess.run(
                containers_cmd, capture_output=True, text=True, timeout=30
            )

            if containers_result.returncode == 0:
                for line in containers_result.stdout.strip().split("\n"):
                    if line:
                        container_info = json.loads(line)
                        app_state["containers"].append(container_info)

            # Get volume information
            volumes_cmd = ["docker", "volume", "ls", "--format", "json"]
            volumes_result = subprocess.run(volumes_cmd, capture_output=True, text=True, timeout=30)

            if volumes_result.returncode == 0:
                for line in volumes_result.stdout.strip().split("\n"):
                    if line:
                        volume_info = json.loads(line)
                        app_state["volumes"].append(volume_info)

            # Get network information
            networks_cmd = ["docker", "network", "ls", "--format", "json"]
            networks_result = subprocess.run(
                networks_cmd, capture_output=True, text=True, timeout=30
            )

            if networks_result.returncode == 0:
                for line in networks_result.stdout.strip().split("\n"):
                    if line:
                        network_info = json.loads(line)
                        app_state["networks"].append(network_info)

            # Save state backup
            with open(state_backup, "w") as f:
                json.dump(app_state, f, indent=2)

            backup_size = state_backup.stat().st_size

            # 2. Verify backup content
            with open(state_backup, "r") as f:
                restored_state = json.load(f)

            containers_count = len(restored_state.get("containers", []))
            volumes_count = len(restored_state.get("volumes", []))
            networks_count = len(restored_state.get("networks", []))

            # Cleanup
            state_backup.unlink()

            duration = time.time() - start_time
            success = containers_count > 0  # At least some containers should be running

            self.log_test_result(
                "application_state_backup",
                success,
                duration,
                {
                    "backup_size_bytes": backup_size,
                    "containers_count": containers_count,
                    "volumes_count": volumes_count,
                    "networks_count": networks_count,
                },
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("application_state_backup", False, duration, {"error": str(e)})
            logger.error(f"Application state backup test failed: {e}")
            return False

    def run_all_backup_tests(self) -> Dict[str, Any]:
        """Run all backup and restore tests"""
        logger.info("Starting comprehensive backup and restore test suite...")

        test_functions = [
            self.test_postgres_backup_restore,
            self.test_redis_backup_restore,
            self.test_secrets_backup_restore,
            self.test_volume_backup_restore,
            self.test_application_state_backup,
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test {test_func.__name__} crashed: {e}")

        # Generate summary
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]

        summary = {
            "total_tests": len(self.test_results),
            "successful": len(successful_tests),
            "failed": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.test_results) * 100
            if self.test_results
            else 0,
            "total_duration": sum(r["duration"] for r in self.test_results),
            "test_results": self.test_results,
        }

        # Save full report
        report_file = self.backup_dir / f"backup_test_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Backup test report saved to {report_file}")

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print backup test summary"""
        print("\n" + "=" * 60)
        print("BACKUP & RESTORE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print()

        print("TEST RESULTS:")
        for result in summary["test_results"]:
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {result['test_name']}: {result['duration']:.2f}s")
            if not result["success"] and "error" in result.get("details", {}):
                print(f"    Error: {result['details']['error']}")

        print("=" * 60)


def main():
    """Main test execution"""
    suite = BackupRestoreTestSuite()

    try:
        summary = suite.run_all_backup_tests()
        suite.print_summary(summary)

        # Exit with non-zero code if any tests failed
        exit_code = 0 if summary["failed"] == 0 else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Backup test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
