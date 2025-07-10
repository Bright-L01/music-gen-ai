#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Music Gen AI Staging
Tests all system components and their interactions
"""

import asyncio
import json
import time
import requests
import psycopg2
import redis
import websockets
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/results/integration_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    component: str
    test_name: str
    success: bool
    duration: float
    error_message: str = ""
    response_data: Dict[str, Any] = None


class IntegrationTestSuite:
    def __init__(self):
        self.base_url = os.getenv("TARGET_HOST", "http://nginx-staging")
        self.api_key = os.getenv("STAGING_API_KEY", "staging_api_key_change_me")
        self.postgres_host = os.getenv("POSTGRES_HOST", "postgres-staging")
        self.redis_host = os.getenv("REDIS_HOST", "redis-staging")
        self.results: List[TestResult] = []

        # HTTP session for API tests
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "IntegrationTestSuite/1.0",
            }
        )

    def test_database_connectivity(self) -> TestResult:
        """Test PostgreSQL database connection and basic queries"""
        start_time = time.time()
        try:
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=5432,
                database="musicgen_staging",
                user="musicgen",
                password=os.getenv("POSTGRES_PASSWORD", "staging_password_change_me"),
                connect_timeout=10,
            )

            with conn.cursor() as cursor:
                # Test basic query
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

                # Test schema exists
                cursor.execute(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'musicgen'"
                )
                schema = cursor.fetchone()
                assert schema is not None

                # Test user table exists and has data
                cursor.execute("SELECT COUNT(*) FROM musicgen.users")
                user_count = cursor.fetchone()[0]
                assert user_count > 0

            conn.close()

            duration = time.time() - start_time
            return TestResult(
                component="database",
                test_name="connectivity_and_schema",
                success=True,
                duration=duration,
                response_data={"user_count": user_count},
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="database",
                test_name="connectivity_and_schema",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_redis_connectivity(self) -> TestResult:
        """Test Redis connection and basic operations"""
        start_time = time.time()
        try:
            r = redis.Redis(
                host=self.redis_host,
                port=6379,
                password=os.getenv("REDIS_PASSWORD", "staging_redis_password"),
                decode_responses=True,
                socket_timeout=10,
            )

            # Test ping
            ping_result = r.ping()
            assert ping_result is True

            # Test set/get
            test_key = f"integration_test_{int(time.time())}"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            assert value == "test_value"

            # Clean up
            r.delete(test_key)

            # Test queue operations (simulate Celery)
            queue_name = "test_queue"
            r.lpush(queue_name, json.dumps({"task": "test_task"}))
            queue_length = r.llen(queue_name)
            assert queue_length > 0

            # Clean up queue
            r.delete(queue_name)

            duration = time.time() - start_time
            return TestResult(
                component="redis",
                test_name="connectivity_and_operations",
                success=True,
                duration=duration,
                response_data={"ping": ping_result, "queue_length": queue_length},
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="redis",
                test_name="connectivity_and_operations",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_api_health_endpoints(self) -> TestResult:
        """Test API health check endpoints"""
        start_time = time.time()
        try:
            # Basic health check
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            assert response.status_code == 200

            # Detailed health check
            response = self.session.get(f"{self.base_url}/health/detailed", timeout=10)
            health_data = response.json()

            assert response.status_code == 200
            assert health_data.get("status") == "healthy"
            assert "database" in health_data.get("checks", {})
            assert "redis" in health_data.get("checks", {})

            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="health_endpoints",
                success=True,
                duration=duration,
                response_data=health_data,
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="health_endpoints",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_api_authentication(self) -> TestResult:
        """Test API authentication mechanisms"""
        start_time = time.time()
        try:
            # Test with valid API key
            response = self.session.get(f"{self.base_url}/api/v1/user/profile", timeout=10)
            assert response.status_code in [200, 404]  # 404 is ok if no profile exists

            # Test with invalid API key
            invalid_session = requests.Session()
            invalid_session.headers.update(
                {"Authorization": "Bearer invalid_key", "Content-Type": "application/json"}
            )

            response = invalid_session.get(f"{self.base_url}/api/v1/user/profile", timeout=10)
            assert response.status_code == 401

            # Test rate limiting
            rate_limit_responses = []
            for i in range(15):  # Try to exceed rate limit
                response = self.session.get(f"{self.base_url}/api/v1/models", timeout=5)
                rate_limit_responses.append(response.status_code)
                if response.status_code == 429:
                    break
                time.sleep(0.1)

            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="authentication_and_rate_limiting",
                success=True,
                duration=duration,
                response_data={
                    "rate_limit_triggered": 429 in rate_limit_responses,
                    "response_codes": rate_limit_responses,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="authentication_and_rate_limiting",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_music_generation_api(self) -> TestResult:
        """Test music generation API endpoint"""
        start_time = time.time()
        try:
            payload = {
                "prompt": "upbeat electronic music for testing",
                "duration": 5.0,
                "temperature": 0.8,
                "format": "wav",
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/generate", json=payload, timeout=120
            )

            assert response.status_code in [200, 202]  # 202 for async processing

            response_data = response.json()

            if response.status_code == 202:
                # If async, check task status
                task_id = response_data.get("task_id")
                if task_id:
                    # Poll for completion (with timeout)
                    for _ in range(30):  # 30 attempts, 2 second intervals = 60 seconds max
                        status_response = self.session.get(
                            f"{self.base_url}/api/v1/tasks/{task_id}", timeout=10
                        )
                        if status_response.status_code == 200:
                            task_data = status_response.json()
                            if task_data.get("status") in ["completed", "failed"]:
                                response_data.update(task_data)
                                break
                        time.sleep(2)

            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="music_generation",
                success=True,
                duration=duration,
                response_data=response_data,
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="api",
                test_name="music_generation",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    async def test_websocket_streaming(self) -> TestResult:
        """Test WebSocket streaming functionality"""
        start_time = time.time()
        try:
            ws_url = self.base_url.replace("http", "ws") + "/ws/generate"

            async with websockets.connect(
                ws_url, extra_headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30
            ) as websocket:
                # Send generation request
                request = {"prompt": "ambient background music", "duration": 3.0, "stream": True}
                await websocket.send(json.dumps(request))

                # Receive responses
                responses = []
                timeout_count = 0

                while timeout_count < 10:  # Max 10 receives
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        responses.append(data)

                        if data.get("status") in ["completed", "error"]:
                            break

                    except asyncio.TimeoutError:
                        timeout_count += 1
                        if timeout_count >= 3:  # 3 consecutive timeouts
                            break

                duration = time.time() - start_time
                return TestResult(
                    component="websocket",
                    test_name="streaming_generation",
                    success=len(responses) > 0,
                    duration=duration,
                    response_data={
                        "response_count": len(responses),
                        "final_status": responses[-1].get("status") if responses else None,
                    },
                )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="websocket",
                test_name="streaming_generation",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_load_balancer_functionality(self) -> TestResult:
        """Test Nginx load balancer distribution"""
        start_time = time.time()
        try:
            server_counts = {}

            # Make multiple requests to see distribution
            for i in range(20):
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                assert response.status_code == 200

                # Try to identify which backend server responded
                server_header = response.headers.get("X-Backend-Server", "unknown")
                server_counts[server_header] = server_counts.get(server_header, 0) + 1

            # Check nginx status endpoint
            status_response = requests.get(f"{self.base_url}/nginx_status", timeout=5)
            nginx_stats = {}
            if status_response.status_code == 200:
                # Parse nginx stub_status format
                lines = status_response.text.strip().split("\n")
                for line in lines:
                    if "Active connections" in line:
                        nginx_stats["active_connections"] = int(line.split(":")[1].strip())

            duration = time.time() - start_time
            return TestResult(
                component="load_balancer",
                test_name="distribution_and_status",
                success=True,
                duration=duration,
                response_data={
                    "server_distribution": server_counts,
                    "nginx_stats": nginx_stats,
                    "total_requests": 20,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="load_balancer",
                test_name="distribution_and_status",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_monitoring_endpoints(self) -> TestResult:
        """Test monitoring and metrics endpoints"""
        start_time = time.time()
        try:
            # Test Prometheus metrics
            prometheus_response = requests.get(
                "http://prometheus-staging:9090/api/v1/query?query=up", timeout=10
            )
            assert prometheus_response.status_code == 200

            # Test application metrics endpoint
            metrics_response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            assert metrics_response.status_code == 200

            # Test Grafana health
            grafana_response = requests.get("http://grafana-staging:3000/api/health", timeout=10)

            # Test Flower (Celery monitoring)
            flower_response = requests.get(
                "http://flower-staging:5555/api/workers",
                auth=("admin", os.getenv("FLOWER_PASSWORD", "staging_flower_password")),
                timeout=10,
            )

            duration = time.time() - start_time
            return TestResult(
                component="monitoring",
                test_name="endpoints_availability",
                success=True,
                duration=duration,
                response_data={
                    "prometheus_status": prometheus_response.status_code,
                    "metrics_status": metrics_response.status_code,
                    "grafana_status": grafana_response.status_code if grafana_response else None,
                    "flower_status": flower_response.status_code if flower_response else None,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="monitoring",
                test_name="endpoints_availability",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    def test_celery_worker_functionality(self) -> TestResult:
        """Test Celery worker task processing"""
        start_time = time.time()
        try:
            # Submit a task through the API that should use Celery
            payload = {"prompt": "test celery processing", "duration": 2.0, "async": True}

            response = self.session.post(
                f"{self.base_url}/api/v1/generate", json=payload, timeout=30
            )

            task_submitted = response.status_code in [200, 202]
            task_id = None

            if task_submitted:
                task_data = response.json()
                task_id = task_data.get("task_id")

            # Check Flower for worker status
            worker_info = {}
            try:
                flower_response = requests.get(
                    "http://flower-staging:5555/api/workers",
                    auth=("admin", os.getenv("FLOWER_PASSWORD", "staging_flower_password")),
                    timeout=10,
                )
                if flower_response.status_code == 200:
                    worker_info = flower_response.json()
            except:
                pass

            duration = time.time() - start_time
            return TestResult(
                component="celery",
                test_name="worker_task_processing",
                success=task_submitted,
                duration=duration,
                response_data={
                    "task_submitted": task_submitted,
                    "task_id": task_id,
                    "worker_count": len(worker_info) if worker_info else 0,
                    "workers": list(worker_info.keys()) if worker_info else [],
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                component="celery",
                test_name="worker_task_processing",
                success=False,
                duration=duration,
                error_message=str(e),
            )

    async def run_all_tests(self) -> List[TestResult]:
        """Run all integration tests"""
        logger.info("Starting comprehensive integration test suite...")

        # Database and infrastructure tests
        logger.info("Testing database connectivity...")
        self.results.append(self.test_database_connectivity())

        logger.info("Testing Redis connectivity...")
        self.results.append(self.test_redis_connectivity())

        # API tests
        logger.info("Testing API health endpoints...")
        self.results.append(self.test_api_health_endpoints())

        logger.info("Testing API authentication...")
        self.results.append(self.test_authentication())

        logger.info("Testing music generation API...")
        self.results.append(self.test_music_generation_api())

        # WebSocket tests
        logger.info("Testing WebSocket streaming...")
        ws_result = await self.test_websocket_streaming()
        self.results.append(ws_result)

        # Infrastructure tests
        logger.info("Testing load balancer...")
        self.results.append(self.test_load_balancer_functionality())

        logger.info("Testing monitoring endpoints...")
        self.results.append(self.test_monitoring_endpoints())

        logger.info("Testing Celery workers...")
        self.results.append(self.test_celery_worker_functionality())

        return self.results

    def generate_report(self, output_file: str = "/app/results/integration_test_report.json"):
        """Generate comprehensive integration test report"""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        # Group by component
        component_results = {}
        for result in self.results:
            if result.component not in component_results:
                component_results[result.component] = {"passed": 0, "failed": 0, "tests": []}

            if result.success:
                component_results[result.component]["passed"] += 1
            else:
                component_results[result.component]["failed"] += 1

            component_results[result.component]["tests"].append(asdict(result))

        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100,
                "total_duration": sum(r.duration for r in self.results),
            },
            "components": component_results,
            "failed_tests": [asdict(r) for r in failed_tests],
            "timestamp": time.time(),
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Integration test report saved to {output_file}")
        self.print_summary(report)

        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Duration: {report['summary']['total_duration']:.2f}s")
        print()

        print("COMPONENT BREAKDOWN:")
        for component, results in report["components"].items():
            total = results["passed"] + results["failed"]
            rate = results["passed"] / total * 100 if total > 0 else 0
            print(f"  {component.upper()}: {results['passed']}/{total} ({rate:.1f}%)")

        if report["failed_tests"]:
            print()
            print("FAILED TESTS:")
            for test in report["failed_tests"]:
                print(f"  ❌ {test['component']}.{test['test_name']}: {test['error_message']}")

        print("=" * 60)


async def main():
    """Main test execution"""
    suite = IntegrationTestSuite()

    try:
        await suite.run_all_tests()
        report = suite.generate_report()

        # Exit with non-zero code if any tests failed
        exit_code = 0 if report["summary"]["failed"] == 0 else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
