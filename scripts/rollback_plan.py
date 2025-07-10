#!/usr/bin/env python3
"""
Production Rollback Plan for Music Gen AI
Emergency rollback procedures for production deployment
"""

import os
import json
import time
import subprocess
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"rollback_plan_{int(time.time())}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class RollbackStep:
    """Individual rollback step"""

    id: str
    name: str
    description: str
    command: str
    timeout: int
    critical: bool
    success: bool = False
    error: str = ""
    duration: float = 0.0


class ProductionRollbackPlan:
    def __init__(self):
        self.rollback_id = f"rollback_{int(time.time())}"
        self.rollback_start_time = None
        self.rollback_steps: List[RollbackStep] = []
        self.rollback_results = []

        # Configuration
        self.api_url = os.getenv("API_URL", "https://api.musicgen.ai")
        self.db_params = {
            "host": os.getenv("DB_HOST", "postgres"),
            "port": 5432,
            "database": "musicgen_prod",
            "user": "musicgen",
            "password": os.getenv("DB_PASSWORD"),
        }

        # Alert channels
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.pagerduty_token = os.getenv("PAGERDUTY_TOKEN")

        # Initialize rollback steps
        self._initialize_rollback_steps()

    def _initialize_rollback_steps(self):
        """Initialize all rollback steps"""
        self.rollback_steps = [
            RollbackStep(
                id="notification_start",
                name="Send Rollback Notification",
                description="Alert all teams that rollback is starting",
                command="self.send_rollback_notification('started')",
                timeout=30,
                critical=False,
            ),
            RollbackStep(
                id="enable_maintenance",
                name="Enable Maintenance Mode",
                description="Enable maintenance page to prevent new requests",
                command="kubectl set image deployment/nginx-ingress nginx-ingress=nginx-maintenance:latest --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="stop_workers",
                name="Stop Worker Services",
                description="Stop all Celery worker processes",
                command="kubectl scale deployment/musicgen-worker --replicas=0 --namespace=production",
                timeout=180,
                critical=True,
            ),
            RollbackStep(
                id="stop_scheduler",
                name="Stop Scheduler Services",
                description="Stop all scheduled tasks",
                command="kubectl scale deployment/musicgen-scheduler --replicas=0 --namespace=production",
                timeout=120,
                critical=True,
            ),
            RollbackStep(
                id="scale_down_api",
                name="Scale Down API Services",
                description="Scale down API services to 0 replicas",
                command="kubectl scale deployment/musicgen-api --replicas=0 --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="restore_database",
                name="Restore Database",
                description="Restore database from pre-deployment backup",
                command="self.restore_database_from_backup()",
                timeout=1800,
                critical=True,
            ),
            RollbackStep(
                id="restore_redis",
                name="Restore Redis Data",
                description="Restore Redis data from backup",
                command="self.restore_redis_from_backup()",
                timeout=600,
                critical=True,
            ),
            RollbackStep(
                id="rollback_api",
                name="Rollback API Services",
                description="Deploy previous version of API services",
                command="kubectl rollout undo deployment/musicgen-api --namespace=production",
                timeout=600,
                critical=True,
            ),
            RollbackStep(
                id="rollback_workers",
                name="Rollback Worker Services",
                description="Deploy previous version of worker services",
                command="kubectl rollout undo deployment/musicgen-worker --namespace=production",
                timeout=600,
                critical=True,
            ),
            RollbackStep(
                id="rollback_scheduler",
                name="Rollback Scheduler Services",
                description="Deploy previous version of scheduler services",
                command="kubectl rollout undo deployment/musicgen-scheduler --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="scale_up_api",
                name="Scale Up API Services",
                description="Scale API services to production levels",
                command="kubectl scale deployment/musicgen-api --replicas=3 --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="scale_up_workers",
                name="Scale Up Worker Services",
                description="Scale worker services to production levels",
                command="kubectl scale deployment/musicgen-worker --replicas=5 --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="scale_up_scheduler",
                name="Scale Up Scheduler Services",
                description="Scale scheduler services to production levels",
                command="kubectl scale deployment/musicgen-scheduler --replicas=1 --namespace=production",
                timeout=180,
                critical=True,
            ),
            RollbackStep(
                id="wait_for_rollout",
                name="Wait for Rollout",
                description="Wait for all services to be ready",
                command="self.wait_for_rollout_completion()",
                timeout=900,
                critical=True,
            ),
            RollbackStep(
                id="run_health_checks",
                name="Run Health Checks",
                description="Verify all services are healthy",
                command="self.run_health_checks()",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="run_smoke_tests",
                name="Run Smoke Tests",
                description="Run basic functionality tests",
                command="self.run_smoke_tests()",
                timeout=600,
                critical=True,
            ),
            RollbackStep(
                id="disable_maintenance",
                name="Disable Maintenance Mode",
                description="Restore normal load balancer configuration",
                command="kubectl set image deployment/nginx-ingress nginx-ingress=nginx:stable --namespace=production",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="verify_traffic",
                name="Verify Traffic Flow",
                description="Verify that traffic is flowing normally",
                command="self.verify_traffic_flow()",
                timeout=300,
                critical=True,
            ),
            RollbackStep(
                id="notification_complete",
                name="Send Completion Notification",
                description="Alert all teams that rollback is complete",
                command="self.send_rollback_notification('completed')",
                timeout=30,
                critical=False,
            ),
        ]

    def send_rollback_notification(self, status: str):
        """Send rollback notification to teams"""
        message = {
            "started": {
                "text": "🚨 *PRODUCTION ROLLBACK INITIATED*",
                "color": "danger",
                "fields": [
                    {"title": "Rollback ID", "value": self.rollback_id, "short": True},
                    {
                        "title": "Start Time",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "short": True,
                    },
                    {"title": "Status", "value": "IN PROGRESS", "short": True},
                    {"title": "Impact", "value": "Service temporarily unavailable", "short": True},
                ],
            },
            "completed": {
                "text": "✅ *PRODUCTION ROLLBACK COMPLETED*",
                "color": "good",
                "fields": [
                    {"title": "Rollback ID", "value": self.rollback_id, "short": True},
                    {
                        "title": "Duration",
                        "value": f"{self.get_rollback_duration():.1f} minutes",
                        "short": True,
                    },
                    {"title": "Status", "value": "COMPLETED", "short": True},
                    {"title": "Service", "value": "Restored to previous version", "short": True},
                ],
            },
            "failed": {
                "text": "❌ *PRODUCTION ROLLBACK FAILED*",
                "color": "danger",
                "fields": [
                    {"title": "Rollback ID", "value": self.rollback_id, "short": True},
                    {
                        "title": "Status",
                        "value": "FAILED - MANUAL INTERVENTION REQUIRED",
                        "short": True,
                    },
                    {
                        "title": "Action",
                        "value": "Escalate to Senior Engineering Team",
                        "short": True,
                    },
                ],
            },
        }

        if self.slack_webhook:
            try:
                slack_message = {
                    "text": message[status]["text"],
                    "attachments": [
                        {"color": message[status]["color"], "fields": message[status]["fields"]}
                    ],
                }
                requests.post(self.slack_webhook, json=slack_message, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")

        # PagerDuty for critical rollback events
        if status in ["started", "failed"] and self.pagerduty_token:
            try:
                pagerduty_event = {
                    "routing_key": self.pagerduty_token,
                    "event_action": "trigger",
                    "payload": {
                        "summary": f"Production Rollback {status.upper()}",
                        "severity": "critical",
                        "source": "rollback-plan",
                        "custom_details": {
                            "rollback_id": self.rollback_id,
                            "status": status,
                            "timestamp": datetime.now().isoformat(),
                        },
                    },
                }
                requests.post(
                    "https://events.pagerduty.com/v2/enqueue", json=pagerduty_event, timeout=5
                )
            except Exception as e:
                logger.error(f"Failed to send PagerDuty alert: {e}")

    def restore_database_from_backup(self):
        """Restore database from pre-deployment backup"""
        try:
            logger.info("Starting database restoration...")

            # Find most recent backup
            backup_file = self.find_latest_backup()
            if not backup_file:
                raise Exception("No backup file found")

            logger.info(f"Restoring from backup: {backup_file}")

            # Stop all connections to the database
            conn = psycopg2.connect(**self.db_params)
            conn.autocommit = True

            with conn.cursor() as cursor:
                # Terminate all connections
                cursor.execute(
                    """
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = 'musicgen_prod'
                    AND pid <> pg_backend_pid()
                """
                )

                # Drop and recreate database
                cursor.execute("DROP DATABASE IF EXISTS musicgen_prod")
                cursor.execute("CREATE DATABASE musicgen_prod")

            conn.close()

            # Restore from backup
            restore_cmd = [
                "psql",
                "-h",
                self.db_params["host"],
                "-p",
                str(self.db_params["port"]),
                "-U",
                self.db_params["user"],
                "-d",
                "musicgen_prod",
                "-f",
                backup_file,
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_params["password"]

            result = subprocess.run(
                restore_cmd, env=env, capture_output=True, text=True, timeout=1800
            )

            if result.returncode != 0:
                raise Exception(f"Database restore failed: {result.stderr}")

            logger.info("Database restoration completed successfully")

        except Exception as e:
            logger.error(f"Database restoration failed: {e}")
            raise

    def restore_redis_from_backup(self):
        """Restore Redis data from backup"""
        try:
            logger.info("Starting Redis data restoration...")

            # Find Redis backup file
            redis_backup = self.find_latest_redis_backup()
            if not redis_backup:
                logger.warning("No Redis backup found, skipping Redis restoration")
                return

            # Stop Redis temporarily
            subprocess.run(
                [
                    "kubectl",
                    "scale",
                    "deployment/redis-master",
                    "--replicas=0",
                    "--namespace=production",
                ],
                check=True,
                timeout=300,
            )

            # Wait for Redis to stop
            time.sleep(30)

            # Restore Redis data
            restore_cmd = [
                "kubectl",
                "exec",
                "redis-master-0",
                "--namespace=production",
                "--",
                "redis-cli",
                "--rdb",
                redis_backup,
            ]

            subprocess.run(restore_cmd, check=True, timeout=600)

            # Restart Redis
            subprocess.run(
                [
                    "kubectl",
                    "scale",
                    "deployment/redis-master",
                    "--replicas=1",
                    "--namespace=production",
                ],
                check=True,
                timeout=300,
            )

            logger.info("Redis restoration completed successfully")

        except Exception as e:
            logger.error(f"Redis restoration failed: {e}")
            raise

    def find_latest_backup(self) -> Optional[str]:
        """Find the most recent database backup"""
        try:
            # Look for backup files
            backup_patterns = [
                "/backups/pre_deployment_backup_*.sql",
                "/backups/postgres_backup_*.sql",
                "/app/results/backup_tests/pre_migration_backup_*.sql.gz",
            ]

            latest_backup = None
            latest_time = 0

            for pattern in backup_patterns:
                result = subprocess.run(
                    ["find", os.path.dirname(pattern), "-name", os.path.basename(pattern)],
                    capture_output=True,
                    text=True,
                )

                for file_path in result.stdout.strip().split("\n"):
                    if file_path and os.path.exists(file_path):
                        mtime = os.path.getmtime(file_path)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_backup = file_path

            return latest_backup

        except Exception as e:
            logger.error(f"Error finding backup: {e}")
            return None

    def find_latest_redis_backup(self) -> Optional[str]:
        """Find the most recent Redis backup"""
        try:
            result = subprocess.run(
                ["find", "/backups", "-name", "redis_backup_*.rdb"], capture_output=True, text=True
            )

            backups = result.stdout.strip().split("\n")
            if backups and backups[0]:
                return max(backups, key=os.path.getmtime)

            return None

        except Exception as e:
            logger.error(f"Error finding Redis backup: {e}")
            return None

    def wait_for_rollout_completion(self):
        """Wait for all Kubernetes rollouts to complete"""
        deployments = ["musicgen-api", "musicgen-worker", "musicgen-scheduler"]

        for deployment in deployments:
            logger.info(f"Waiting for {deployment} rollout to complete...")

            cmd = [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--namespace=production",
                "--timeout=600s",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Rollout failed for {deployment}: {result.stderr}")

            logger.info(f"{deployment} rollout completed successfully")

    def run_health_checks(self):
        """Run comprehensive health checks"""
        health_checks = [
            {
                "name": "API Health",
                "url": f"{self.api_url}/health",
                "expected_status": 200,
                "timeout": 30,
            },
            {
                "name": "Database Health",
                "url": f"{self.api_url}/health/database",
                "expected_status": 200,
                "timeout": 30,
            },
            {
                "name": "Redis Health",
                "url": f"{self.api_url}/health/redis",
                "expected_status": 200,
                "timeout": 30,
            },
            {
                "name": "Workers Health",
                "url": f"{self.api_url}/health/workers",
                "expected_status": 200,
                "timeout": 30,
            },
        ]

        failed_checks = []

        for check in health_checks:
            try:
                response = requests.get(check["url"], timeout=check["timeout"])

                if response.status_code != check["expected_status"]:
                    failed_checks.append(f"{check['name']}: {response.status_code}")
                else:
                    logger.info(f"Health check passed: {check['name']}")

            except Exception as e:
                failed_checks.append(f"{check['name']}: {str(e)}")

        if failed_checks:
            raise Exception(f"Health checks failed: {', '.join(failed_checks)}")

        logger.info("All health checks passed")

    def run_smoke_tests(self):
        """Run basic smoke tests"""
        test_token = os.getenv("TEST_API_TOKEN")
        if not test_token:
            logger.warning("No test API token configured, skipping smoke tests")
            return

        smoke_tests = [
            {
                "name": "Authentication Test",
                "method": "POST",
                "url": f"{self.api_url}/api/v1/auth/login",
                "data": {"email": "test@musicgen.ai", "password": "TestPassword123!"},
                "expected_status": [200, 401],
            },
            {
                "name": "Models List Test",
                "method": "GET",
                "url": f"{self.api_url}/api/v1/models",
                "headers": {"Authorization": f"Bearer {test_token}"},
                "expected_status": [200],
            },
            {
                "name": "User Profile Test",
                "method": "GET",
                "url": f"{self.api_url}/api/v1/user/profile",
                "headers": {"Authorization": f"Bearer {test_token}"},
                "expected_status": [200, 401],
            },
        ]

        failed_tests = []

        for test in smoke_tests:
            try:
                if test["method"] == "GET":
                    response = requests.get(
                        test["url"], headers=test.get("headers", {}), timeout=30
                    )
                else:
                    response = requests.post(
                        test["url"],
                        json=test.get("data", {}),
                        headers=test.get("headers", {}),
                        timeout=30,
                    )

                if response.status_code not in test["expected_status"]:
                    failed_tests.append(f"{test['name']}: {response.status_code}")
                else:
                    logger.info(f"Smoke test passed: {test['name']}")

            except Exception as e:
                failed_tests.append(f"{test['name']}: {str(e)}")

        if failed_tests:
            raise Exception(f"Smoke tests failed: {', '.join(failed_tests)}")

        logger.info("All smoke tests passed")

    def verify_traffic_flow(self):
        """Verify that traffic is flowing normally"""
        try:
            # Check that we can reach the main API
            response = requests.get(f"{self.api_url}/health", timeout=30)

            if response.status_code != 200:
                raise Exception(f"API not responding: {response.status_code}")

            # Check response time
            if response.elapsed.total_seconds() > 5:
                logger.warning(f"Slow response time: {response.elapsed.total_seconds():.2f}s")

            # Verify we're not in maintenance mode
            if "maintenance" in response.text.lower():
                raise Exception("Service still in maintenance mode")

            logger.info("Traffic flow verification passed")

        except Exception as e:
            logger.error(f"Traffic flow verification failed: {e}")
            raise

    def execute_step(self, step: RollbackStep) -> bool:
        """Execute a single rollback step"""
        logger.info(f"Executing step: {step.name}")

        start_time = time.time()

        try:
            if step.command.startswith("self."):
                # Execute method
                method_name = step.command.split("(")[0].replace("self.", "")
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    method()
                else:
                    raise Exception(f"Method {method_name} not found")
            else:
                # Execute shell command
                result = subprocess.run(
                    step.command.split(), capture_output=True, text=True, timeout=step.timeout
                )

                if result.returncode != 0:
                    raise Exception(f"Command failed: {result.stderr}")

            step.success = True
            step.duration = time.time() - start_time
            logger.info(f"Step completed successfully: {step.name} ({step.duration:.2f}s)")

        except Exception as e:
            step.success = False
            step.error = str(e)
            step.duration = time.time() - start_time
            logger.error(f"Step failed: {step.name} - {e}")

            if step.critical:
                logger.critical(f"Critical step failed: {step.name}")
                return False

        return True

    def execute_rollback(self, start_step: str = None) -> bool:
        """Execute the complete rollback plan"""
        self.rollback_start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("STARTING PRODUCTION ROLLBACK PROCEDURE")
        logger.info("=" * 80)
        logger.info(f"Rollback ID: {self.rollback_id}")
        logger.info(f"Start time: {self.rollback_start_time}")

        # Find starting step
        start_index = 0
        if start_step:
            for i, step in enumerate(self.rollback_steps):
                if step.id == start_step:
                    start_index = i
                    break

        # Execute steps
        for i in range(start_index, len(self.rollback_steps)):
            step = self.rollback_steps[i]

            if not self.execute_step(step):
                logger.critical(f"Rollback failed at step: {step.name}")
                self.send_rollback_notification("failed")
                return False

            # Brief pause between steps
            if i < len(self.rollback_steps) - 1:
                time.sleep(5)

        # Generate final report
        self.generate_rollback_report()

        logger.info("=" * 80)
        logger.info("PRODUCTION ROLLBACK COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return True

    def get_rollback_duration(self) -> float:
        """Get rollback duration in minutes"""
        if self.rollback_start_time:
            return (datetime.now() - self.rollback_start_time).total_seconds() / 60
        return 0

    def generate_rollback_report(self):
        """Generate rollback execution report"""
        report = {
            "rollback_id": self.rollback_id,
            "start_time": self.rollback_start_time.isoformat()
            if self.rollback_start_time
            else None,
            "end_time": datetime.now().isoformat(),
            "duration_minutes": self.get_rollback_duration(),
            "total_steps": len(self.rollback_steps),
            "successful_steps": len([s for s in self.rollback_steps if s.success]),
            "failed_steps": len([s for s in self.rollback_steps if not s.success and s.error]),
            "critical_failures": len(
                [s for s in self.rollback_steps if not s.success and s.critical]
            ),
            "step_details": [asdict(step) for step in self.rollback_steps],
            "overall_success": all(s.success or not s.critical for s in self.rollback_steps),
        }

        # Save report
        report_file = f"rollback_report_{self.rollback_id}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Rollback report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("ROLLBACK EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Rollback ID: {self.rollback_id}")
        print(f"Duration: {self.get_rollback_duration():.1f} minutes")
        print(f"Steps: {report['successful_steps']}/{report['total_steps']} successful")
        print(f"Critical Failures: {report['critical_failures']}")
        print(f"Overall Success: {'YES' if report['overall_success'] else 'NO'}")

        print("\nSTEP DETAILS:")
        for step in self.rollback_steps:
            status = "✅" if step.success else "❌"
            critical = " (CRITICAL)" if step.critical else ""
            print(f"  {status} {step.name}{critical}: {step.duration:.2f}s")
            if step.error:
                print(f"    Error: {step.error}")

        print("=" * 60)

        return report


def main():
    """Main rollback execution"""
    rollback_plan = ProductionRollbackPlan()

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dry-run":
            print("DRY RUN MODE - No actual changes will be made")
            print("\nRollback Steps:")
            for i, step in enumerate(rollback_plan.rollback_steps, 1):
                print(f"{i:2d}. {step.name}")
                print(f"    {step.description}")
                print(f"    Command: {step.command}")
                print(f"    Timeout: {step.timeout}s")
                print(f"    Critical: {step.critical}")
                print()
            return

        elif sys.argv[1] == "--from-step":
            if len(sys.argv) < 3:
                print("Error: --from-step requires a step ID")
                sys.exit(1)
            start_step = sys.argv[2]
        else:
            start_step = None
    else:
        start_step = None

    # Confirm rollback
    if not os.getenv("ROLLBACK_CONFIRMED"):
        response = input("Are you sure you want to execute the production rollback? (yes/no): ")
        if response.lower() != "yes":
            print("Rollback cancelled")
            sys.exit(0)

    # Execute rollback
    try:
        success = rollback_plan.execute_rollback(start_step)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Rollback execution failed: {e}")
        rollback_plan.send_rollback_notification("failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
