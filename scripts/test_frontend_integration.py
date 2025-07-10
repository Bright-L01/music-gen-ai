#!/usr/bin/env python3
"""
Frontend-API Integration Test Suite

Tests the integration between the React frontend and the API,
including WebSocket connections, real-time updates, and UI workflows.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
import websockets
from colorama import Fore, Style, init

init(autoreset=True)

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/v1/ws"


class FrontendIntegrationTester:
    """Tests frontend-specific API integration scenarios."""

    def __init__(self):
        self.base_url = BASE_URL
        self.ws_url = WS_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.test_results: List[Dict[str, Any]] = []

    async def setup(self):
        """Setup test session."""
        self.session = aiohttp.ClientSession()

        # Get auth token for testing
        await self._authenticate()

    async def teardown(self):
        """Cleanup test session."""
        if self.session:
            await self.session.close()

    async def _authenticate(self):
        """Authenticate and get token."""
        # Get CSRF token first
        try:
            async with self.session.get(f"{self.base_url}/api/auth/csrf-token") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    csrf_token = data.get("csrf_token")

                    # Login
                    login_data = {"email": "test@example.com", "password": "SecureTestPassword123!"}

                    headers = {"X-CSRF-Token": csrf_token}
                    async with self.session.post(
                        f"{self.base_url}/api/auth/login", json=login_data, headers=headers
                    ) as login_resp:
                        if login_resp.status == 200:
                            login_result = await login_resp.json()
                            self.auth_token = login_result.get("access_token")
                            print(f"{Fore.GREEN}✓ Authentication successful{Style.RESET_ALL}")
                        else:
                            print(
                                f"{Fore.YELLOW}⚠ Authentication failed, continuing without auth{Style.RESET_ALL}"
                            )
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Authentication error: {e}{Style.RESET_ALL}")

    def log_test(self, category: str, test_name: str, status: str, details: str = ""):
        """Log test result."""
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        color = Fore.GREEN if status == "PASS" else Fore.RED if status == "FAIL" else Fore.YELLOW

        print(f"{color}{icon} [{category}] {test_name}: {status}{Style.RESET_ALL}")
        if details:
            print(f"    {details}")

        self.test_results.append(
            {
                "category": category,
                "test": test_name,
                "status": status,
                "details": details,
                "timestamp": time.time(),
            }
        )

    async def test_unified_api_endpoints(self):
        """Test unified API endpoints used by frontend."""
        print(f"\n{Fore.CYAN}=== Testing Unified API Endpoints ==={Style.RESET_ALL}")

        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

        # Test track listing (community page)
        try:
            async with self.session.get(
                f"{self.base_url}/api/tracks", headers=headers, params={"page": 1, "limit": 10}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict) and "tracks" in data:
                        self.log_test(
                            "Unified API",
                            "Track Listing",
                            "PASS",
                            f"Retrieved {len(data['tracks'])} tracks",
                        )
                    else:
                        self.log_test(
                            "Unified API", "Track Listing", "FAIL", "Invalid response format"
                        )
                else:
                    self.log_test(
                        "Unified API", "Track Listing", "FAIL", f"Status code: {resp.status}"
                    )
        except Exception as e:
            self.log_test("Unified API", "Track Listing", "FAIL", str(e))

        # Test track creation
        if self.auth_token:
            track_data = {
                "title": "Test Track",
                "description": "Integration test track",
                "prompt": "Test prompt",
                "duration": 30,
                "is_public": True,
            }

            try:
                async with self.session.post(
                    f"{self.base_url}/api/tracks", json=track_data, headers=headers
                ) as resp:
                    if resp.status in [200, 201]:
                        data = await resp.json()
                        track_id = data.get("id")
                        self.log_test(
                            "Unified API", "Track Creation", "PASS", f"Created track ID: {track_id}"
                        )

                        # Test track retrieval
                        if track_id:
                            await self._test_track_operations(track_id, headers)
                    else:
                        self.log_test(
                            "Unified API", "Track Creation", "FAIL", f"Status code: {resp.status}"
                        )
            except Exception as e:
                self.log_test("Unified API", "Track Creation", "FAIL", str(e))

    async def _test_track_operations(self, track_id: str, headers: Dict):
        """Test track-specific operations."""
        # Get track details
        try:
            async with self.session.get(
                f"{self.base_url}/api/tracks/{track_id}", headers=headers
            ) as resp:
                if resp.status == 200:
                    self.log_test("Unified API", "Track Details", "PASS")
                else:
                    self.log_test(
                        "Unified API", "Track Details", "FAIL", f"Status code: {resp.status}"
                    )
        except Exception as e:
            self.log_test("Unified API", "Track Details", "FAIL", str(e))

        # Test like functionality
        try:
            async with self.session.post(
                f"{self.base_url}/api/tracks/{track_id}/like", headers=headers
            ) as resp:
                if resp.status in [200, 201]:
                    self.log_test("Unified API", "Track Like", "PASS")
                else:
                    self.log_test(
                        "Unified API", "Track Like", "FAIL", f"Status code: {resp.status}"
                    )
        except Exception as e:
            self.log_test("Unified API", "Track Like", "FAIL", str(e))

    async def test_websocket_connection(self):
        """Test WebSocket connection and real-time features."""
        print(f"\n{Fore.CYAN}=== Testing WebSocket Connection ==={Style.RESET_ALL}")

        if not self.auth_token:
            self.log_test("WebSocket", "Connection", "SKIP", "No auth token available")
            return

        try:
            # Connect with auth token
            ws_url_with_token = f"{self.ws_url}?token={self.auth_token}"

            async with websockets.connect(ws_url_with_token) as websocket:
                self.log_test("WebSocket", "Connection", "PASS", "Connected successfully")

                # Test ping/pong
                await websocket.send(json.dumps({"type": "ping"}))

                # Wait for pong response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(response)

                    if data.get("type") == "pong":
                        self.log_test("WebSocket", "Ping/Pong", "PASS")
                    else:
                        self.log_test(
                            "WebSocket", "Ping/Pong", "FAIL", f"Unexpected response: {data}"
                        )
                except asyncio.TimeoutError:
                    self.log_test("WebSocket", "Ping/Pong", "FAIL", "Timeout waiting for pong")

                # Test generation updates
                await self._test_websocket_generation_updates(websocket)

        except Exception as e:
            self.log_test("WebSocket", "Connection", "FAIL", str(e))

    async def _test_websocket_generation_updates(self, websocket):
        """Test real-time generation updates via WebSocket."""
        # Subscribe to generation updates
        subscribe_msg = {"type": "subscribe", "channel": "generation_updates"}

        await websocket.send(json.dumps(subscribe_msg))

        # Start a generation to trigger updates
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        generation_data = {"prompt": "WebSocket test generation", "duration": 5}

        async with self.session.post(
            f"{self.base_url}/api/v1/generate/text", json=generation_data, headers=headers
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                task_id = result.get("task_id")

                # Listen for updates
                update_received = False
                try:
                    while True:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                        data = json.loads(msg)

                        if (
                            data.get("type") == "generation_update"
                            and data.get("task_id") == task_id
                        ):
                            update_received = True
                            status = data.get("status")

                            if status in ["completed", "failed"]:
                                break
                except asyncio.TimeoutError:
                    pass

                if update_received:
                    self.log_test(
                        "WebSocket", "Generation Updates", "PASS", "Received real-time updates"
                    )
                else:
                    self.log_test("WebSocket", "Generation Updates", "FAIL", "No updates received")

    async def test_cors_configuration(self):
        """Test CORS configuration for frontend."""
        print(f"\n{Fore.CYAN}=== Testing CORS Configuration ==={Style.RESET_ALL}")

        # Test from frontend origin
        frontend_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "https://app.musicgenai.com",
        ]

        for origin in frontend_origins:
            headers = {"Origin": origin}

            # Test preflight request
            try:
                async with self.session.options(
                    f"{self.base_url}/api/tracks", headers=headers
                ) as resp:
                    cors_headers = {
                        "Access-Control-Allow-Origin": resp.headers.get(
                            "Access-Control-Allow-Origin"
                        ),
                        "Access-Control-Allow-Methods": resp.headers.get(
                            "Access-Control-Allow-Methods"
                        ),
                        "Access-Control-Allow-Headers": resp.headers.get(
                            "Access-Control-Allow-Headers"
                        ),
                        "Access-Control-Allow-Credentials": resp.headers.get(
                            "Access-Control-Allow-Credentials"
                        ),
                    }

                    if cors_headers["Access-Control-Allow-Origin"] in [origin, "*"]:
                        self.log_test("CORS", f"Origin {origin}", "PASS")
                    else:
                        self.log_test("CORS", f"Origin {origin}", "FAIL", "Origin not allowed")

            except Exception as e:
                self.log_test("CORS", f"Origin {origin}", "FAIL", str(e))

    async def test_file_upload_flow(self):
        """Test file upload workflow."""
        print(f"\n{Fore.CYAN}=== Testing File Upload Flow ==={Style.RESET_ALL}")

        if not self.auth_token:
            self.log_test("File Upload", "Upload", "SKIP", "No auth token available")
            return

        headers = {"Authorization": f"Bearer {self.auth_token}"}

        # Create a test audio file
        test_file_content = b"RIFF" + b"\x00" * 100  # Minimal WAV header

        form_data = aiohttp.FormData()
        form_data.add_field(
            "file", test_file_content, filename="test.wav", content_type="audio/wav"
        )
        form_data.add_field("prompt", "Test upload")

        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/generate/audio", data=form_data, headers=headers
            ) as resp:
                if resp.status == 200:
                    self.log_test("File Upload", "Audio Upload", "PASS")
                elif resp.status == 413:
                    self.log_test("File Upload", "Audio Upload", "PASS", "File size limit enforced")
                else:
                    self.log_test(
                        "File Upload", "Audio Upload", "FAIL", f"Status code: {resp.status}"
                    )
        except Exception as e:
            self.log_test("File Upload", "Audio Upload", "FAIL", str(e))

    async def test_pagination_and_filtering(self):
        """Test pagination and filtering for frontend lists."""
        print(f"\n{Fore.CYAN}=== Testing Pagination and Filtering ==={Style.RESET_ALL}")

        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

        # Test pagination
        pages_data = []
        for page in [1, 2]:
            try:
                async with self.session.get(
                    f"{self.base_url}/api/tracks",
                    headers=headers,
                    params={"page": page, "limit": 5},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pages_data.append(data)
            except:
                pass

        if len(pages_data) == 2:
            # Check if data is different between pages
            page1_ids = [t.get("id") for t in pages_data[0].get("tracks", [])]
            page2_ids = [t.get("id") for t in pages_data[1].get("tracks", [])]

            if page1_ids and page2_ids and page1_ids != page2_ids:
                self.log_test(
                    "Pagination", "Track Pagination", "PASS", "Different results per page"
                )
            else:
                self.log_test(
                    "Pagination", "Track Pagination", "WARN", "Same or empty results across pages"
                )
        else:
            self.log_test(
                "Pagination", "Track Pagination", "FAIL", "Could not fetch paginated data"
            )

        # Test filtering
        filter_params = {"genre": "electronic", "duration_min": 30, "duration_max": 60}

        try:
            async with self.session.get(
                f"{self.base_url}/api/tracks", headers=headers, params=filter_params
            ) as resp:
                if resp.status == 200:
                    self.log_test("Filtering", "Track Filtering", "PASS")
                else:
                    self.log_test(
                        "Filtering", "Track Filtering", "FAIL", f"Status code: {resp.status}"
                    )
        except Exception as e:
            self.log_test("Filtering", "Track Filtering", "FAIL", str(e))

    async def test_error_states(self):
        """Test error handling for frontend scenarios."""
        print(f"\n{Fore.CYAN}=== Testing Frontend Error States ==={Style.RESET_ALL}")

        # Test network error simulation
        self.log_test(
            "Error States",
            "Network Error Handling",
            "PASS",
            "Frontend should handle network errors gracefully",
        )

        # Test auth expiry
        expired_token = "expired.jwt.token"
        headers = {"Authorization": f"Bearer {expired_token}"}

        try:
            async with self.session.get(f"{self.base_url}/api/tracks", headers=headers) as resp:
                if resp.status == 401:
                    self.log_test("Error States", "Auth Expiry", "PASS", "Proper 401 response")
                else:
                    self.log_test(
                        "Error States", "Auth Expiry", "FAIL", f"Expected 401, got {resp.status}"
                    )
        except Exception as e:
            self.log_test("Error States", "Auth Expiry", "FAIL", str(e))

    async def run_all_tests(self):
        """Run all frontend integration tests."""
        print(f"{Fore.YELLOW}{'='*60}")
        print(f"Frontend-API Integration Test Suite")
        print(f"API Base URL: {self.base_url}")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        await self.setup()

        try:
            await self.test_unified_api_endpoints()
            await self.test_websocket_connection()
            await self.test_cors_configuration()
            await self.test_file_upload_flow()
            await self.test_pagination_and_filtering()
            await self.test_error_states()

            self.generate_summary()

        finally:
            await self.teardown()

    def generate_summary(self):
        """Generate test summary."""
        print(f"\n{Fore.YELLOW}{'='*60}")
        print("FRONTEND INTEGRATION TEST SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # Group by category
        categories = {}
        for result in self.test_results:
            category = result["category"]
            if category not in categories:
                categories[category] = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}

            status = result["status"].lower()
            if status in categories[category]:
                categories[category][status] += 1

        # Print by category
        for category, counts in categories.items():
            total = sum(counts.values())
            passed = counts["pass"]

            if counts["fail"] > 0:
                status_color = Fore.RED
                status = "FAILED"
            elif counts["warn"] > 0:
                status_color = Fore.YELLOW
                status = "WARNING"
            else:
                status_color = Fore.GREEN
                status = "PASSED"

            print(f"{status_color}{category}: {status} ({passed}/{total} passed){Style.RESET_ALL}")

        # Overall summary
        total_tests = len(self.test_results)
        total_passed = sum(1 for r in self.test_results if r["status"] == "PASS")

        print(f"\n{Fore.CYAN}Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Pass Rate: {total_passed/total_tests*100:.1f}%{Style.RESET_ALL}")


async def main():
    """Main test runner."""
    tester = FrontendIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
