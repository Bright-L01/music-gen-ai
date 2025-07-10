"""
Advanced WebSocket load testing for Music Gen AI streaming functionality.

Tests WebSocket connection limits, streaming performance, and concurrent
session handling under various load conditions.
"""

import asyncio
import websockets
import json
import time
import uuid
import random
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import requests
import aiohttp


@dataclass
class WebSocketTestResult:
    """Result of a WebSocket test session."""

    session_id: str
    connection_time: float
    total_duration: float
    messages_received: int
    bytes_received: int
    errors: List[str]
    success: bool
    disconnection_reason: Optional[str] = None


class WebSocketLoadTester:
    """
    Comprehensive WebSocket load testing for streaming functionality.

    Tests connection limits, message throughput, concurrent sessions,
    and streaming performance under various load conditions.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.results = []
        self.active_connections = []
        self.max_concurrent_connections = 0
        self.total_messages_received = 0
        self.total_bytes_received = 0
        self.connection_errors = []

        # Performance metrics
        self.connection_times = []
        self.message_latencies = []
        self.throughput_measurements = []

        # Thread safety
        self.lock = threading.Lock()

    async def create_streaming_session(self) -> Optional[str]:
        """Create a streaming session via REST API."""
        try:
            request_data = {
                "prompt": f"WebSocket load test music {uuid.uuid4().hex[:8]}",
                "duration": random.uniform(10.0, 30.0),
                "chunk_duration": random.uniform(1.0, 3.0),
                "temperature": random.uniform(0.8, 1.2),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/stream/session",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("session_id")
                    else:
                        print(f"Failed to create session: {response.status}")
                        return None

        except Exception as e:
            print(f"Error creating streaming session: {e}")
            return None

    async def test_single_websocket_connection(
        self, session_id: str, duration: float = 30.0
    ) -> WebSocketTestResult:
        """Test a single WebSocket connection."""
        connection_start = time.time()
        messages_received = 0
        bytes_received = 0
        errors = []
        success = False
        disconnection_reason = None

        ws_url = f"{self.ws_base_url}/api/v1/stream/ws/{session_id}"

        try:
            async with websockets.connect(
                ws_url,
                timeout=10,
                max_size=10**7,  # 10MB max message size
                ping_interval=20,
                ping_timeout=10,
            ) as websocket:
                connection_time = time.time() - connection_start

                with self.lock:
                    self.active_connections.append(websocket)
                    self.connection_times.append(connection_time)
                    self.max_concurrent_connections = max(
                        self.max_concurrent_connections, len(self.active_connections)
                    )

                start_time = time.time()

                try:
                    while time.time() - start_time < duration:
                        try:
                            # Wait for message with timeout
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)

                            message_received_time = time.time()
                            messages_received += 1

                            # Handle different message types
                            if isinstance(message, bytes):
                                # Audio chunk
                                bytes_received += len(message)
                                with self.lock:
                                    self.total_bytes_received += len(message)
                            else:
                                # JSON metadata
                                try:
                                    data = json.loads(message)
                                    message_type = data.get("type")

                                    if message_type == "chunk":
                                        # Record chunk latency
                                        chunk_index = data.get("chunk_index", 0)
                                        latency = message_received_time - start_time
                                        with self.lock:
                                            self.message_latencies.append(latency)

                                    elif message_type == "complete":
                                        # Streaming completed
                                        success = True
                                        break

                                    elif message_type == "error":
                                        error_msg = data.get("error", "Unknown error")
                                        errors.append(error_msg)
                                        disconnection_reason = "server_error"
                                        break

                                except json.JSONDecodeError:
                                    errors.append("Invalid JSON received")

                            with self.lock:
                                self.total_messages_received += 1

                        except asyncio.TimeoutError:
                            # No message received within timeout
                            continue

                        except websockets.exceptions.ConnectionClosed as e:
                            disconnection_reason = f"connection_closed_{e.code}"
                            break

                except Exception as e:
                    errors.append(f"Connection error: {str(e)}")
                    disconnection_reason = "connection_error"

                finally:
                    with self.lock:
                        if websocket in self.active_connections:
                            self.active_connections.remove(websocket)

                total_duration = time.time() - connection_start

                if not success and not errors:
                    success = (
                        messages_received > 0
                    )  # Consider successful if we received any messages

                return WebSocketTestResult(
                    session_id=session_id,
                    connection_time=connection_time,
                    total_duration=total_duration,
                    messages_received=messages_received,
                    bytes_received=bytes_received,
                    errors=errors,
                    success=success,
                    disconnection_reason=disconnection_reason,
                )

        except Exception as e:
            error_msg = f"WebSocket connection failed: {str(e)}"
            errors.append(error_msg)

            with self.lock:
                self.connection_errors.append(error_msg)

            return WebSocketTestResult(
                session_id=session_id,
                connection_time=0,
                total_duration=time.time() - connection_start,
                messages_received=0,
                bytes_received=0,
                errors=errors,
                success=False,
                disconnection_reason="connection_failed",
            )

    async def test_concurrent_connections(
        self, num_connections: int, duration: float = 60.0
    ) -> List[WebSocketTestResult]:
        """Test multiple concurrent WebSocket connections."""
        print(f"Testing {num_connections} concurrent WebSocket connections for {duration}s...")

        # Create sessions first
        session_creation_tasks = []
        for _ in range(num_connections):
            session_creation_tasks.append(self.create_streaming_session())

        session_ids = await asyncio.gather(*session_creation_tasks, return_exceptions=True)
        valid_session_ids = [sid for sid in session_ids if isinstance(sid, str)]

        print(f"Created {len(valid_session_ids)} valid sessions out of {num_connections} requested")

        if not valid_session_ids:
            print("No valid sessions created, skipping WebSocket test")
            return []

        # Test WebSocket connections
        connection_tasks = []
        for session_id in valid_session_ids:
            task = self.test_single_websocket_connection(session_id, duration)
            connection_tasks.append(task)

        results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [result for result in results if isinstance(result, WebSocketTestResult)]

        with self.lock:
            self.results.extend(valid_results)

        return valid_results

    async def test_connection_limits(
        self, max_connections: int = 100, step_size: int = 10
    ) -> Dict[str, Any]:
        """Test WebSocket connection limits by gradually increasing load."""
        print(f"Testing WebSocket connection limits up to {max_connections} connections...")

        limit_results = {}
        current_connections = step_size

        while current_connections <= max_connections:
            print(f"Testing {current_connections} concurrent connections...")

            start_time = time.time()
            results = await self.test_concurrent_connections(current_connections, duration=30.0)
            test_duration = time.time() - start_time

            # Analyze results
            successful_connections = len([r for r in results if r.success])
            failed_connections = len(results) - successful_connections
            avg_connection_time = (
                statistics.mean([r.connection_time for r in results if r.connection_time > 0])
                if results
                else 0
            )

            limit_results[current_connections] = {
                "successful_connections": successful_connections,
                "failed_connections": failed_connections,
                "success_rate": (successful_connections / len(results)) * 100 if results else 0,
                "avg_connection_time_ms": avg_connection_time * 1000,
                "test_duration_seconds": test_duration,
                "max_concurrent_achieved": self.max_concurrent_connections,
            }

            print(f"  Success rate: {limit_results[current_connections]['success_rate']:.1f}%")
            print(f"  Avg connection time: {avg_connection_time * 1000:.1f}ms")

            # Stop if success rate drops below 80%
            if limit_results[current_connections]["success_rate"] < 80:
                print(f"Success rate dropped below 80% at {current_connections} connections")
                break

            current_connections += step_size

            # Cool down between tests
            await asyncio.sleep(5)

        return limit_results

    async def test_message_throughput(
        self, num_connections: int = 20, duration: float = 120.0
    ) -> Dict[str, Any]:
        """Test message throughput under sustained load."""
        print(f"Testing message throughput with {num_connections} connections for {duration}s...")

        start_time = time.time()
        results = await self.test_concurrent_connections(num_connections, duration)
        total_test_time = time.time() - start_time

        # Calculate throughput metrics
        total_messages = sum(r.messages_received for r in results)
        total_bytes = sum(r.bytes_received for r in results)
        successful_connections = len([r for r in results if r.success])

        messages_per_second = total_messages / total_test_time if total_test_time > 0 else 0
        bytes_per_second = total_bytes / total_test_time if total_test_time > 0 else 0
        avg_messages_per_connection = total_messages / len(results) if results else 0

        return {
            "test_duration_seconds": total_test_time,
            "total_connections": len(results),
            "successful_connections": successful_connections,
            "total_messages_received": total_messages,
            "total_bytes_received": total_bytes,
            "messages_per_second": messages_per_second,
            "bytes_per_second": bytes_per_second,
            "megabytes_per_second": bytes_per_second / (1024 * 1024),
            "avg_messages_per_connection": avg_messages_per_connection,
            "avg_connection_duration": statistics.mean([r.total_duration for r in results])
            if results
            else 0,
        }

    async def test_reconnection_behavior(self, num_tests: int = 10) -> Dict[str, Any]:
        """Test WebSocket reconnection behavior after failures."""
        print(f"Testing WebSocket reconnection behavior with {num_tests} tests...")

        reconnection_results = []

        for i in range(num_tests):
            session_id = await self.create_streaming_session()
            if not session_id:
                continue

            # Test initial connection
            initial_result = await self.test_single_websocket_connection(session_id, duration=10.0)

            # Simulate reconnection after short delay
            await asyncio.sleep(1.0)

            reconnect_start = time.time()
            reconnect_result = await self.test_single_websocket_connection(
                session_id, duration=10.0
            )
            reconnect_time = time.time() - reconnect_start

            reconnection_results.append(
                {
                    "test_id": i + 1,
                    "initial_success": initial_result.success,
                    "reconnect_success": reconnect_result.success,
                    "reconnect_time_seconds": reconnect_time,
                    "initial_messages": initial_result.messages_received,
                    "reconnect_messages": reconnect_result.messages_received,
                }
            )

        # Analyze reconnection results
        successful_reconnections = len([r for r in reconnection_results if r["reconnect_success"]])
        avg_reconnect_time = statistics.mean(
            [r["reconnect_time_seconds"] for r in reconnection_results]
        )

        return {
            "total_tests": len(reconnection_results),
            "successful_reconnections": successful_reconnections,
            "reconnection_success_rate": (successful_reconnections / len(reconnection_results))
            * 100
            if reconnection_results
            else 0,
            "avg_reconnect_time_seconds": avg_reconnect_time,
            "reconnection_tests": reconnection_results,
        }

    def generate_websocket_report(self) -> Dict[str, Any]:
        """Generate comprehensive WebSocket performance report."""
        if not self.results:
            return {"error": "No test results available"}

        # Calculate overall statistics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        success_rate = (successful_tests / total_tests) * 100

        # Connection time statistics
        connection_times_ms = [
            r.connection_time * 1000 for r in self.results if r.connection_time > 0
        ]
        avg_connection_time = statistics.mean(connection_times_ms) if connection_times_ms else 0

        # Message statistics
        total_messages = sum(r.messages_received for r in self.results)
        total_bytes = sum(r.bytes_received for r in self.results)
        avg_messages_per_connection = total_messages / total_tests if total_tests > 0 else 0

        # Duration statistics
        durations = [r.total_duration for r in self.results]
        avg_duration = statistics.mean(durations) if durations else 0

        # Error analysis
        all_errors = []
        disconnection_reasons = {}
        for result in self.results:
            all_errors.extend(result.errors)
            if result.disconnection_reason:
                disconnection_reasons[result.disconnection_reason] = (
                    disconnection_reasons.get(result.disconnection_reason, 0) + 1
                )

        error_frequency = {}
        for error in all_errors:
            error_frequency[error] = error_frequency.get(error, 0) + 1

        # Performance percentiles
        percentiles = {}
        if connection_times_ms:
            sorted_times = sorted(connection_times_ms)
            percentiles = {
                "p50": sorted_times[int(0.5 * len(sorted_times))],
                "p90": sorted_times[int(0.9 * len(sorted_times))],
                "p95": sorted_times[int(0.95 * len(sorted_times))],
                "p99": sorted_times[int(0.99 * len(sorted_times))],
            }

        return {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate_percent": success_rate,
                "max_concurrent_connections": self.max_concurrent_connections,
                "total_messages_received": self.total_messages_received,
                "total_bytes_received": self.total_bytes_received,
            },
            "connection_performance": {
                "avg_connection_time_ms": avg_connection_time,
                "connection_time_percentiles": percentiles,
                "connection_errors": len(self.connection_errors),
                "unique_connection_errors": len(set(self.connection_errors)),
            },
            "message_performance": {
                "total_messages": total_messages,
                "total_bytes": total_bytes,
                "avg_messages_per_connection": avg_messages_per_connection,
                "avg_bytes_per_connection": total_bytes / total_tests if total_tests > 0 else 0,
                "message_latencies": {
                    "count": len(self.message_latencies),
                    "avg_ms": statistics.mean(self.message_latencies) * 1000
                    if self.message_latencies
                    else 0,
                    "max_ms": max(self.message_latencies) * 1000 if self.message_latencies else 0,
                },
            },
            "session_performance": {
                "avg_session_duration_seconds": avg_duration,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
            },
            "error_analysis": {
                "total_errors": len(all_errors),
                "unique_errors": len(error_frequency),
                "error_frequency": error_frequency,
                "disconnection_reasons": disconnection_reasons,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


async def run_websocket_load_tests():
    """Run comprehensive WebSocket load tests."""
    tester = WebSocketLoadTester()

    print("Starting WebSocket Load Testing...")
    print("=" * 50)

    # Test 1: Basic concurrent connections
    print("\n1. Testing basic concurrent connections...")
    await tester.test_concurrent_connections(num_connections=10, duration=30.0)

    # Test 2: Connection limits
    print("\n2. Testing connection limits...")
    limit_results = await tester.test_connection_limits(max_connections=50, step_size=10)

    # Test 3: Message throughput
    print("\n3. Testing message throughput...")
    throughput_results = await tester.test_message_throughput(num_connections=15, duration=60.0)

    # Test 4: Reconnection behavior
    print("\n4. Testing reconnection behavior...")
    reconnection_results = await tester.test_reconnection_behavior(num_tests=5)

    # Generate comprehensive report
    print("\n5. Generating report...")
    report = tester.generate_websocket_report()

    # Combine all results
    full_report = {
        "websocket_load_test_report": report,
        "connection_limits_test": limit_results,
        "throughput_test": throughput_results,
        "reconnection_test": reconnection_results,
    }

    # Save report
    import json

    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/websocket_load_report.json", "w"
    ) as f:
        json.dump(full_report, f, indent=2)

    print(f"\nWebSocket Load Testing Complete!")
    print(f"Total tests: {report['test_summary']['total_tests']}")
    print(f"Success rate: {report['test_summary']['success_rate_percent']:.1f}%")
    print(f"Max concurrent: {report['test_summary']['max_concurrent_connections']}")
    print(f"Report saved to: websocket_load_report.json")

    return full_report


if __name__ == "__main__":
    # Run WebSocket load tests
    asyncio.run(run_websocket_load_tests())
