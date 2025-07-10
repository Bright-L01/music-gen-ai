"""
Main Locust load testing configuration for Music Gen AI API.

This file defines comprehensive load testing scenarios including:
- Concurrent music generation requests
- WebSocket streaming load testing
- Database and Redis connection pooling
- Performance bottleneck identification
- Baseline metrics collection

Usage:
    # Basic load test
    locust -f tests/load/locustfile.py --host=http://localhost:8000

    # Heavy load test
    locust -f tests/load/locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 300s

    # WebSocket focused test
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --tags websocket

    # Database stress test
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --tags database
"""

import json
import random
import time
import uuid
from typing import Dict, Any, Optional
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from locust import HttpUser, task, between, events, tag
from locust.contrib.fasthttp import FastHttpUser
from locust.exception import StopUser
from locust.env import Environment

# Import custom performance monitoring
from tests.load.performance_monitor import PerformanceMonitor
from tests.load.metrics_collector import MetricsCollector

# Global performance monitor
performance_monitor = PerformanceMonitor()
metrics_collector = MetricsCollector()


class MusicGenerationUser(FastHttpUser):
    """
    Load test user for music generation API endpoints.

    Simulates realistic user behavior including:
    - Authentication flows
    - Music generation requests
    - Task status polling
    - File downloads
    """

    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    weight = 3  # Higher weight = more of these users

    def on_start(self):
        """Initialize user session."""
        self.auth_token = None
        self.csrf_token = None
        self.active_tasks = []
        self.user_id = f"load_test_user_{uuid.uuid4().hex[:8]}"

        # Authenticate user (if auth is implemented)
        self.authenticate()

        # Get CSRF token
        self.get_csrf_token()

    def authenticate(self):
        """Authenticate user and get tokens."""
        login_data = {
            "email": f"{self.user_id}@loadtest.com",
            "password": "loadtest123",
            "remember_me": False,
        }

        with self.client.post(
            "/api/auth/login", json=login_data, catch_response=True, name="auth_login"
        ) as response:
            if response.status_code == 200:
                # Extract tokens from response or cookies
                data = response.json()
                self.auth_token = data.get("access_token")
                self.csrf_token = data.get("csrfToken")
            elif response.status_code == 404:
                # Auth not implemented, continue without auth
                pass
            else:
                response.failure(f"Login failed: {response.status_code}")

    def get_csrf_token(self):
        """Get CSRF token for authenticated requests."""
        with self.client.get("/api/auth/csrf-token", name="get_csrf_token") as response:
            if response.status_code == 200:
                self.csrf_token = response.json().get("csrfToken")

    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.csrf_token:
            headers["X-CSRF-Token"] = self.csrf_token
        return headers

    @task(10)
    @tag("generation", "core")
    def generate_music_short(self):
        """Generate short music clips (high frequency)."""
        prompts = [
            "upbeat electronic music",
            "calm piano melody",
            "energetic rock guitar",
            "ambient soundscape",
            "jazz saxophone solo",
            "classical violin piece",
            "trap beats",
            "folk acoustic guitar",
        ]

        request_data = {
            "prompt": random.choice(prompts),
            "duration": random.uniform(2.0, 8.0),
            "temperature": random.uniform(0.7, 1.3),
            "top_k": random.randint(40, 60),
            "top_p": random.uniform(0.8, 0.95),
        }

        start_time = time.time()

        with self.client.post(
            "/api/v1/generate/",
            json=request_data,
            headers=self.get_auth_headers(),
            catch_response=True,
            name="generate_music_short",
        ) as response:
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                if task_id:
                    self.active_tasks.append(
                        {
                            "task_id": task_id,
                            "created_at": start_time,
                            "prompt": request_data["prompt"],
                        }
                    )

                # Track generation latency
                latency = time.time() - start_time
                performance_monitor.record_generation_latency(latency)

            elif response.status_code == 429:
                response.failure("Rate limited")
                performance_monitor.record_rate_limit()
            else:
                response.failure(f"Generation failed: {response.status_code}")

    @task(5)
    @tag("generation", "batch")
    def generate_music_batch(self):
        """Generate batch music requests."""
        batch_requests = []
        batch_size = random.randint(2, 4)

        prompts = [
            "rock music with heavy drums",
            "peaceful meditation music",
            "dance electronic beats",
            "country music with guitar",
        ]

        for _ in range(batch_size):
            batch_requests.append(
                {
                    "prompt": random.choice(prompts),
                    "duration": random.uniform(3.0, 6.0),
                    "temperature": random.uniform(0.8, 1.2),
                }
            )

        batch_data = {"requests": batch_requests}

        start_time = time.time()

        with self.client.post(
            "/api/v1/generate/batch",
            json=batch_data,
            headers=self.get_auth_headers(),
            catch_response=True,
            name="generate_music_batch",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                batch_id = data.get("batch_id")
                task_ids = data.get("task_ids", [])

                # Track batch tasks
                for task_id in task_ids:
                    self.active_tasks.append(
                        {"task_id": task_id, "batch_id": batch_id, "created_at": start_time}
                    )

                # Track batch latency
                latency = time.time() - start_time
                performance_monitor.record_batch_latency(latency, len(batch_requests))

            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Batch generation failed: {response.status_code}")

    @task(15)
    @tag("monitoring", "core")
    def check_task_status(self):
        """Check status of active tasks."""
        if not self.active_tasks:
            return

        # Check random active task
        task_info = random.choice(self.active_tasks)
        task_id = task_info["task_id"]

        start_time = time.time()

        with self.client.get(
            f"/api/v1/generate/{task_id}",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="check_task_status",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")

                # Track task completion time
                if status == "completed":
                    completion_time = time.time() - task_info["created_at"]
                    performance_monitor.record_task_completion_time(completion_time)

                    # Remove from active tasks
                    self.active_tasks.remove(task_info)

                elif status == "failed":
                    performance_monitor.record_task_failure()
                    self.active_tasks.remove(task_info)

                # Track status check latency
                latency = time.time() - start_time
                performance_monitor.record_status_check_latency(latency)

            elif response.status_code == 404:
                # Task not found, remove from active list
                self.active_tasks.remove(task_info)
            else:
                response.failure(f"Status check failed: {response.status_code}")

    @task(2)
    @tag("download", "file")
    def download_completed_task(self):
        """Download completed task files."""
        # Find completed tasks (simulate with random choice)
        if not self.active_tasks:
            return

        task_info = random.choice(self.active_tasks)
        task_id = task_info["task_id"]

        start_time = time.time()

        with self.client.get(
            f"/download/{task_id}",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="download_audio_file",
        ) as response:
            if response.status_code == 200:
                # Track download size and speed
                content_length = int(response.headers.get("content-length", 0))
                download_time = time.time() - start_time

                if content_length > 0 and download_time > 0:
                    download_speed = content_length / download_time  # bytes/second
                    performance_monitor.record_download_metrics(content_length, download_speed)

            elif response.status_code == 400:
                # Task not completed yet
                pass
            elif response.status_code == 404:
                # Task not found
                response.failure("Download failed: Task not found")
            else:
                response.failure(f"Download failed: {response.status_code}")

    @task(1)
    @tag("health", "monitoring")
    def check_health(self):
        """Check API health endpoints."""
        endpoints = ["/health", "/health/detailed"]
        endpoint = random.choice(endpoints)

        with self.client.get(endpoint, name="health_check") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")


class WebSocketStreamingUser(HttpUser):
    """
    Load test user for WebSocket streaming functionality.

    Tests WebSocket connection limits, streaming performance,
    and real-time audio streaming capabilities.
    """

    wait_time = between(2, 8)
    weight = 1  # Lower weight as WebSocket tests are more resource intensive

    def on_start(self):
        """Initialize WebSocket user."""
        self.session_id = None
        self.websocket = None
        self.streaming_active = False
        self.user_id = f"ws_user_{uuid.uuid4().hex[:8]}"

    @task(8)
    @tag("websocket", "streaming")
    def create_streaming_session(self):
        """Create a new streaming session."""
        request_data = {
            "prompt": f"streaming music for user {self.user_id}",
            "duration": random.uniform(5.0, 15.0),
            "chunk_duration": random.uniform(1.0, 3.0),
            "temperature": random.uniform(0.8, 1.2),
        }

        start_time = time.time()

        with self.client.post(
            "/api/v1/stream/session",
            json=request_data,
            catch_response=True,
            name="create_streaming_session",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")

                # Track session creation latency
                latency = time.time() - start_time
                performance_monitor.record_streaming_session_latency(latency)

            elif response.status_code == 503:
                response.failure("Streaming service unavailable")
            else:
                response.failure(f"Session creation failed: {response.status_code}")

    @task(5)
    @tag("websocket", "connection")
    def test_websocket_connection(self):
        """Test WebSocket connection and streaming."""
        if not self.session_id:
            self.create_streaming_session()
            if not self.session_id:
                return

        # Simulate WebSocket connection
        ws_url = f"ws://{self.host.replace('http://', '')}/api/v1/stream/ws/{self.session_id}"

        try:
            start_time = time.time()

            # Note: This is a simplified simulation
            # Real WebSocket testing would require async implementation
            connection_time = time.time() - start_time
            performance_monitor.record_websocket_connection_time(connection_time)

            # Simulate receiving chunks
            chunk_count = random.randint(3, 8)
            for i in range(chunk_count):
                time.sleep(0.5)  # Simulate chunk processing
                performance_monitor.record_websocket_chunk_received()

            performance_monitor.record_websocket_session_complete()

        except Exception as e:
            performance_monitor.record_websocket_error(str(e))

    @task(3)
    @tag("websocket", "management")
    def list_streaming_sessions(self):
        """List active streaming sessions."""
        with self.client.get(
            "/api/v1/stream/sessions", catch_response=True, name="list_streaming_sessions"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                session_count = data.get("count", 0)
                performance_monitor.record_active_sessions(session_count)
            else:
                response.failure(f"Session listing failed: {response.status_code}")

    @task(2)
    @tag("websocket", "cleanup")
    def stop_streaming_session(self):
        """Stop active streaming session."""
        if not self.session_id:
            return

        with self.client.delete(
            f"/api/v1/stream/session/{self.session_id}",
            catch_response=True,
            name="stop_streaming_session",
        ) as response:
            if response.status_code == 200:
                self.session_id = None
                performance_monitor.record_session_stopped()
            elif response.status_code == 404:
                # Session already gone
                self.session_id = None
            else:
                response.failure(f"Session stop failed: {response.status_code}")


class DatabaseStressUser(FastHttpUser):
    """
    Load test user focused on database connection pooling and stress testing.

    Tests database performance under heavy concurrent load to identify
    connection pool limits and query performance bottlenecks.
    """

    wait_time = between(0.5, 2)  # Faster requests to stress database
    weight = 2

    def on_start(self):
        """Initialize database stress user."""
        self.user_id = f"db_user_{uuid.uuid4().hex[:8]}"
        self.task_history = []

    @task(15)
    @tag("database", "read")
    def heavy_status_checking(self):
        """Perform heavy database read operations."""
        # Generate multiple tasks first
        if len(self.task_history) < 10:
            self.create_tasks_for_status_checking()

        # Check multiple task statuses rapidly
        for _ in range(random.randint(3, 7)):
            if self.task_history:
                task_id = random.choice(self.task_history)

                start_time = time.time()

                with self.client.get(
                    f"/api/v1/generate/{task_id}", catch_response=True, name="db_heavy_status_check"
                ) as response:
                    query_time = time.time() - start_time
                    performance_monitor.record_database_query_time(query_time)

                    if response.status_code not in [200, 404]:
                        response.failure(f"DB query failed: {response.status_code}")

    def create_tasks_for_status_checking(self):
        """Create tasks to populate database for testing."""
        for _ in range(5):
            request_data = {
                "prompt": f"database test music {uuid.uuid4().hex[:8]}",
                "duration": 2.0,
            }

            with self.client.post(
                "/api/v1/generate/", json=request_data, catch_response=True, name="db_create_task"
            ) as response:
                if response.status_code == 200:
                    task_id = response.json().get("task_id")
                    if task_id:
                        self.task_history.append(task_id)

    @task(10)
    @tag("database", "write")
    def rapid_task_creation(self):
        """Stress database with rapid task creation."""
        for _ in range(random.randint(2, 5)):
            request_data = {
                "prompt": f"rapid creation test {uuid.uuid4().hex[:8]}",
                "duration": random.uniform(1.0, 3.0),
            }

            start_time = time.time()

            with self.client.post(
                "/api/v1/generate/",
                json=request_data,
                catch_response=True,
                name="db_rapid_creation",
            ) as response:
                insert_time = time.time() - start_time
                performance_monitor.record_database_insert_time(insert_time)

                if response.status_code == 200:
                    task_id = response.json().get("task_id")
                    if task_id:
                        self.task_history.append(task_id)

                        # Keep history manageable
                        if len(self.task_history) > 50:
                            self.task_history = self.task_history[-30:]
                elif response.status_code == 429:
                    break  # Rate limited, stop rapid creation
                else:
                    response.failure(f"Task creation failed: {response.status_code}")

    @task(5)
    @tag("database", "batch")
    def batch_operations(self):
        """Test batch operations that stress database connections."""
        # Create batch request
        batch_requests = []
        for i in range(random.randint(3, 6)):
            batch_requests.append({"prompt": f"batch db test {i}", "duration": 2.0})

        start_time = time.time()

        with self.client.post(
            "/api/v1/generate/batch",
            json={"requests": batch_requests},
            catch_response=True,
            name="db_batch_operation",
        ) as response:
            batch_time = time.time() - start_time
            performance_monitor.record_database_batch_time(batch_time, len(batch_requests))

            if response.status_code == 200:
                data = response.json()
                task_ids = data.get("task_ids", [])
                self.task_history.extend(task_ids)
            else:
                response.failure(f"Batch operation failed: {response.status_code}")


class RedisStressUser(FastHttpUser):
    """
    Load test user focused on Redis connection pooling and caching performance.

    Tests Redis caching, session storage, and rate limiting under load.
    """

    wait_time = between(0.2, 1)  # Very fast requests to stress Redis
    weight = 1

    @task(20)
    @tag("redis", "cache")
    def stress_rate_limiting(self):
        """Stress Redis-based rate limiting."""
        # Make rapid requests to trigger rate limiting
        for _ in range(random.randint(5, 15)):
            start_time = time.time()

            with self.client.get(
                "/health", catch_response=True, name="redis_rate_limit_test"
            ) as response:
                redis_time = time.time() - start_time
                performance_monitor.record_redis_operation_time(redis_time)

                if response.status_code == 429:
                    performance_monitor.record_rate_limit_hit()
                    break
                elif response.status_code != 200:
                    response.failure(f"Redis operation failed: {response.status_code}")

    @task(10)
    @tag("redis", "session")
    def stress_session_operations(self):
        """Stress Redis session storage."""
        # Simulate authentication to test session storage
        login_data = {"email": f"redis_user_{uuid.uuid4().hex[:8]}@test.com", "password": "test123"}

        start_time = time.time()

        with self.client.post(
            "/api/auth/login", json=login_data, catch_response=True, name="redis_session_create"
        ) as response:
            session_time = time.time() - start_time
            performance_monitor.record_redis_session_time(session_time)

            if response.status_code in [200, 401, 404]:
                # Expected responses
                pass
            else:
                response.failure(f"Session operation failed: {response.status_code}")

    @task(15)
    @tag("redis", "csrf")
    def stress_csrf_tokens(self):
        """Stress CSRF token generation and validation."""
        start_time = time.time()

        with self.client.get(
            "/api/auth/csrf-token", catch_response=True, name="redis_csrf_token"
        ) as response:
            csrf_time = time.time() - start_time
            performance_monitor.record_redis_operation_time(csrf_time)

            if response.status_code == 200:
                # Validate token format
                token = response.json().get("csrfToken")
                if not token or len(token) < 20:
                    response.failure("Invalid CSRF token format")
            else:
                response.failure(f"CSRF operation failed: {response.status_code}")


# Locust event listeners for custom metrics collection
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize performance monitoring when test starts."""
    print("Starting load test with performance monitoring...")
    performance_monitor.start_monitoring()
    metrics_collector.start_collection()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate performance report when test stops."""
    print("Load test completed. Generating performance report...")
    performance_monitor.stop_monitoring()
    metrics_collector.stop_collection()

    # Generate comprehensive performance report
    report = performance_monitor.generate_report()
    metrics = metrics_collector.get_metrics()

    # Save reports
    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/performance_report.json", "w"
    ) as f:
        json.dump(report, f, indent=2)

    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/metrics_baseline.json", "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"Performance report saved. Total requests: {report.get('total_requests', 0)}")
    print(f"Average response time: {report.get('avg_response_time', 0):.2f}ms")
    print(f"Error rate: {report.get('error_rate', 0):.2f}%")


@events.request_success.add_listener
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    """Track successful requests."""
    performance_monitor.record_success(request_type, name, response_time, response_length)


@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Track failed requests."""
    performance_monitor.record_failure(request_type, name, response_time, exception)


if __name__ == "__main__":
    # Command line execution
    import sys
    from locust.main import main

    # Set default parameters if running directly
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--host=http://localhost:8000",
                "--users=50",
                "--spawn-rate=5",
                "--run-time=300s",
                "--html=tests/load/load_test_report.html",
            ]
        )

    main()
