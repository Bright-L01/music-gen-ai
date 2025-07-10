#!/usr/bin/env python3
"""
Migration script to migrate tasks from in-memory or old Redis storage to new Redis Streams implementation.

Usage:
    python migrate_tasks_to_redis.py --source memory --target redis://localhost:6379
    python migrate_tasks_to_redis.py --source redis://old-redis:6379 --target redis://new-redis:6379
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from music_gen.infrastructure.repositories.task_repository import (
    InMemoryTaskRepository,
    RedisTaskRepository,
)
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    RedisTaskRepositoryAdvanced,
    TaskPriority,
    TaskStatus,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskMigrator:
    """Migrates tasks between different repository implementations."""

    def __init__(self, source_repo, target_repo):
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.stats = {
            "total": 0,
            "migrated": 0,
            "failed": 0,
            "skipped": 0,
        }

    async def migrate(self, batch_size: int = 100, dry_run: bool = False):
        """
        Migrate tasks from source to target repository.

        Args:
            batch_size: Number of tasks to process at once
            dry_run: If True, don't actually migrate, just report what would be done
        """
        logger.info(f"Starting task migration (dry_run={dry_run})")

        # Get all tasks from source
        offset = 0

        while True:
            tasks = await self.source_repo.list_tasks(limit=batch_size, offset=offset)
            if not tasks:
                break

            for task in tasks:
                self.stats["total"] += 1

                try:
                    await self._migrate_task(task, dry_run)
                except Exception as e:
                    logger.error(f"Failed to migrate task {task.get('id')}: {e}")
                    self.stats["failed"] += 1

            offset += batch_size

            # Progress update
            logger.info(f"Processed {self.stats['total']} tasks...")

        # Final report
        logger.info("Migration completed!")
        logger.info(f"Total tasks: {self.stats['total']}")
        logger.info(f"Migrated: {self.stats['migrated']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")

    async def _migrate_task(self, task: Dict[str, Any], dry_run: bool):
        """Migrate a single task."""
        task_id = task.get("id") or task.get("task_id")
        if not task_id:
            logger.warning("Task missing ID, skipping")
            self.stats["skipped"] += 1
            return

        # Check if task already exists in target
        existing = await self.target_repo.get_task(task_id)
        if existing:
            logger.debug(f"Task {task_id} already exists in target, skipping")
            self.stats["skipped"] += 1
            return

        # Prepare task data for new format
        migrated_task = self._transform_task(task)

        if dry_run:
            logger.info(f"Would migrate task {task_id}: {migrated_task}")
        else:
            # Create task in target repository
            await self.target_repo.create_task(task_id, migrated_task)

            # If task was in a specific status, update it
            status = migrated_task.get("status")
            if status and status != TaskStatus.PENDING.value:
                await self.target_repo.update_task(task_id, {"status": status})

            logger.debug(f"Migrated task {task_id}")

        self.stats["migrated"] += 1

    def _transform_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task data to new format."""
        # Determine priority from old data
        priority = task.get("priority", TaskPriority.NORMAL.value)
        if isinstance(priority, str):
            priority_map = {
                "low": TaskPriority.LOW.value,
                "normal": TaskPriority.NORMAL.value,
                "high": TaskPriority.HIGH.value,
                "critical": TaskPriority.CRITICAL.value,
            }
            priority = priority_map.get(priority.lower(), TaskPriority.NORMAL.value)

        # Build new task format
        migrated_task = {
            "request": task.get("request", {}),
            "priority": priority,
            "ttl": task.get("ttl", 86400),  # Default 24 hours
        }

        # Preserve important fields
        preserve_fields = [
            "status",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
            "error",
            "result",
            "metadata",
            "batch_id",
            "batch_index",
            "retry_count",
        ]

        for field in preserve_fields:
            if field in task:
                migrated_task[field] = task[field]

        return migrated_task


async def create_repository(repo_type: str, url: str = None):
    """Create a repository instance based on type."""
    if repo_type == "memory":
        return InMemoryTaskRepository()
    elif repo_type.startswith("redis://"):
        # Old Redis repository
        return RedisTaskRepository(repo_type)
    elif repo_type.startswith("redis-advanced://"):
        # New Redis repository
        url = repo_type.replace("redis-advanced://", "redis://")
        repo = RedisTaskRepositoryAdvanced(url)
        await repo.initialize()
        return repo
    else:
        raise ValueError(f"Unknown repository type: {repo_type}")


async def verify_migration(source_repo, target_repo, sample_size: int = 10):
    """Verify migration by comparing sample tasks."""
    logger.info(f"Verifying migration with {sample_size} sample tasks...")

    source_tasks = await source_repo.list_tasks(limit=sample_size)

    verified = 0
    failed = 0

    for task in source_tasks:
        task_id = task.get("id") or task.get("task_id")
        if not task_id:
            continue

        target_task = await target_repo.get_task(task_id)
        if target_task:
            # Basic verification - check key fields
            if task.get("request") == target_task.get("request") and task.get(
                "status"
            ) == target_task.get("status"):
                verified += 1
            else:
                logger.warning(f"Task {task_id} mismatch")
                failed += 1
        else:
            logger.warning(f"Task {task_id} not found in target")
            failed += 1

    logger.info(f"Verification complete: {verified} verified, {failed} failed")
    return failed == 0


async def main():
    parser = argparse.ArgumentParser(description="Migrate tasks between repositories")
    parser.add_argument(
        "--source", required=True, help="Source repository (memory, redis://host:port, etc.)"
    )
    parser.add_argument(
        "--target", required=True, help="Target repository (redis-advanced://host:port)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of tasks to process at once"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without actual migration"
    )
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    parser.add_argument(
        "--cleanup-source",
        action="store_true",
        help="Delete tasks from source after successful migration",
    )

    args = parser.parse_args()

    # Create repositories
    logger.info(f"Creating source repository: {args.source}")
    source_repo = await create_repository(args.source)

    logger.info(f"Creating target repository: {args.target}")
    target_repo = await create_repository(args.target)

    # Create migrator
    migrator = TaskMigrator(source_repo, target_repo)

    # Run migration
    await migrator.migrate(batch_size=args.batch_size, dry_run=args.dry_run)

    # Verify if requested
    if args.verify and not args.dry_run:
        success = await verify_migration(source_repo, target_repo)
        if not success:
            logger.error("Verification failed!")
            return 1

    # Cleanup source if requested
    if args.cleanup_source and not args.dry_run:
        logger.info("Cleaning up source repository...")
        tasks = await source_repo.list_tasks(limit=10000)
        for task in tasks:
            task_id = task.get("id") or task.get("task_id")
            if task_id:
                await source_repo.delete_task(task_id)
        logger.info(f"Deleted {len(tasks)} tasks from source")

    # Shutdown
    if hasattr(target_repo, "shutdown"):
        await target_repo.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
