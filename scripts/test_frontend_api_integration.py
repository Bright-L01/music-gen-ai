#!/usr/bin/env python3
"""
Test frontend API integration to ensure all endpoints work correctly.

This script verifies:
- All frontend API calls match backend endpoints
- Proper error responses
- Request/response formats
- Authentication flow
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import aiohttp
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrontendAPITester:
    """Test harness for frontend API integration."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.csrf_token = None
        self.auth_token = None
        self.test_results = {}

    async def setup(self):
        """Initialize test session."""
        self.session = aiohttp.ClientSession()
        logger.info(f"Testing API at {self.base_url}")

    async def teardown(self):
        """Clean up test session."""
        if self.session:
            await self.session.close()

    async def test_endpoint(
        self,
        method: str,
        path: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        expected_status: int = 200,
        test_name: str = None,
    ) -> Dict[str, Any]:
        """Test a single API endpoint."""
        if not test_name:
            test_name = f"{method} {path}"

        url = f"{self.base_url}{path}"
        headers = headers or {}

        # Add auth header if available
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Add CSRF token for state-changing requests
        if method in ["POST", "PUT", "PATCH", "DELETE"] and self.csrf_token:
            headers["X-CSRF-Token"] = self.csrf_token

        # Add content type
        if data and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        try:
            start_time = time.time()

            async with self.session.request(method, url, json=data, headers=headers) as response:
                duration = (time.time() - start_time) * 1000

                # Get response data
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()

                # Check response format
                is_api_response = (
                    isinstance(response_data, dict)
                    and "success" in response_data
                    and "data" in response_data
                )

                # Log result
                success = response.status == expected_status
                result = {
                    "test": test_name,
                    "success": success,
                    "status": response.status,
                    "expected_status": expected_status,
                    "duration_ms": duration,
                    "is_api_response": is_api_response,
                    "response": response_data if success else None,
                    "error": response_data if not success else None,
                }

                if success:
                    logger.info(f"✅ {test_name}: {response.status} ({duration:.2f}ms)")
                else:
                    logger.error(f"❌ {test_name}: {response.status} (expected {expected_status})")
                    if not success:
                        logger.error(f"   Response: {json.dumps(response_data, indent=2)}")

                self.test_results[test_name] = result
                return result

        except Exception as e:
            logger.error(f"❌ {test_name}: {str(e)}")
            self.test_results[test_name] = {
                "test": test_name,
                "success": False,
                "error": str(e),
            }
            return {"success": False, "error": str(e)}

    async def test_auth_flow(self):
        """Test authentication flow."""
        logger.info("\n=== Testing Authentication Flow ===")

        # Test CSRF token endpoint
        await self.test_endpoint("GET", "/api/auth/csrf-token", test_name="Get CSRF Token")

        # Test registration
        register_data = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "SecurePass123!",
            "name": "Test User",
            "username": f"testuser_{int(time.time())}",
        }

        result = await self.test_endpoint(
            "POST", "/api/auth/register", data=register_data, test_name="User Registration"
        )

        if result["success"] and result.get("response", {}).get("data", {}).get("token"):
            self.auth_token = result["response"]["data"]["token"]

        # Test login
        login_data = {"email": register_data["email"], "password": register_data["password"]}

        result = await self.test_endpoint(
            "POST", "/api/auth/login", data=login_data, test_name="User Login"
        )

        if result["success"] and result.get("response", {}).get("data", {}).get("token"):
            self.auth_token = result["response"]["data"]["token"]

        # Test session
        await self.test_endpoint("GET", "/api/auth/session", test_name="Get Session")

        # Test refresh
        await self.test_endpoint("POST", "/api/auth/refresh", test_name="Refresh Session")

        # Test logout
        await self.test_endpoint("POST", "/api/auth/logout", test_name="User Logout")

    async def test_generation_endpoints(self):
        """Test music generation endpoints."""
        logger.info("\n=== Testing Generation Endpoints ===")

        # Test generation start
        generation_data = {
            "prompt": "Relaxing lofi hip hop beat",
            "genre": "lofi",
            "mood": "relaxing",
            "duration": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
        }

        result = await self.test_endpoint(
            "POST", "/api/v1/generate", data=generation_data, test_name="Start Generation"
        )

        task_id = None
        if result["success"] and result.get("response", {}).get("data", {}).get("id"):
            task_id = result["response"]["data"]["id"]

        # Test generation status
        if task_id:
            await self.test_endpoint(
                "GET", f"/api/v1/generate/{task_id}", test_name="Get Generation Status"
            )

            # Test save generation
            await self.test_endpoint(
                "POST",
                f"/api/v1/generate/{task_id}/save",
                data={"metadata": {"test": True}},
                test_name="Save Generation",
            )

            # Test cancel generation
            await self.test_endpoint(
                "DELETE", f"/api/v1/generate/{task_id}", test_name="Cancel Generation"
            )

        # Test generation history
        await self.test_endpoint(
            "GET", "/api/v1/generate/history?page=1&limit=10", test_name="Get Generation History"
        )

    async def test_user_endpoints(self):
        """Test user profile endpoints."""
        logger.info("\n=== Testing User Endpoints ===")

        # Test get profile
        await self.test_endpoint("GET", "/api/v1/user/profile", test_name="Get User Profile")

        # Test update profile
        profile_updates = {"name": "Updated Test User", "bio": "Test bio"}

        await self.test_endpoint(
            "PUT", "/api/v1/user/profile", data=profile_updates, test_name="Update User Profile"
        )

    async def test_track_endpoints(self):
        """Test track management endpoints."""
        logger.info("\n=== Testing Track Endpoints ===")

        # Test trending tracks
        await self.test_endpoint(
            "GET", "/api/v1/tracks/trending?limit=5", test_name="Get Trending Tracks"
        )

        # Test recent tracks
        await self.test_endpoint(
            "GET", "/api/v1/tracks/recent?page=1&limit=10", test_name="Get Recent Tracks"
        )

        # Test search tracks
        await self.test_endpoint(
            "GET", "/api/v1/tracks/search?q=test&genre=electronic", test_name="Search Tracks"
        )

    async def test_community_endpoints(self):
        """Test community endpoints."""
        logger.info("\n=== Testing Community Endpoints ===")

        # Test community stats
        await self.test_endpoint("GET", "/api/v1/community/stats", test_name="Get Community Stats")

        # Test featured users
        await self.test_endpoint(
            "GET", "/api/v1/community/featured-users?limit=5", test_name="Get Featured Users"
        )

        # Test trending topics
        await self.test_endpoint(
            "GET", "/api/v1/community/trending-topics", test_name="Get Trending Topics"
        )

    async def test_analytics_endpoints(self):
        """Test analytics endpoints."""
        logger.info("\n=== Testing Analytics Endpoints ===")

        # Test track event
        event_data = {
            "event": "test_event",
            "properties": {"test": True},
            "timestamp": int(time.time() * 1000),
        }

        await self.test_endpoint(
            "POST", "/api/v1/analytics/track", data=event_data, test_name="Track Analytics Event"
        )

    async def test_error_handling(self):
        """Test error handling and response formats."""
        logger.info("\n=== Testing Error Handling ===")

        # Test 404 error
        await self.test_endpoint(
            "GET",
            "/api/v1/generate/non-existent-id",
            expected_status=404,
            test_name="404 Not Found",
        )

        # Test validation error
        invalid_data = {"prompt": "", "duration": 1000}  # Empty prompt should fail  # Too long

        await self.test_endpoint(
            "POST",
            "/api/v1/generate",
            data=invalid_data,
            expected_status=422,
            test_name="Validation Error",
        )

        # Test unauthorized
        saved_token = self.auth_token
        self.auth_token = None

        await self.test_endpoint(
            "GET", "/api/v1/user/profile", expected_status=401, test_name="401 Unauthorized"
        )

        self.auth_token = saved_token

    async def test_openapi_endpoints(self):
        """Test OpenAPI documentation endpoints."""
        logger.info("\n=== Testing OpenAPI Endpoints ===")

        # Test OpenAPI schema
        await self.test_endpoint(
            "GET", "/openapi.json", expected_status=200, test_name="OpenAPI Schema"
        )

        # Test docs
        await self.test_endpoint("GET", "/docs", expected_status=200, test_name="Swagger UI")

        # Test redoc
        await self.test_endpoint("GET", "/redoc", expected_status=200, test_name="ReDoc UI")

    async def run_all_tests(self):
        """Run all API tests."""
        test_groups = [
            self.test_openapi_endpoints,
            self.test_auth_flow,
            self.test_generation_endpoints,
            self.test_user_endpoints,
            self.test_track_endpoints,
            self.test_community_endpoints,
            self.test_analytics_endpoints,
            self.test_error_handling,
        ]

        for test_group in test_groups:
            try:
                await test_group()
            except Exception as e:
                logger.error(f"Test group {test_group.__name__} failed: {e}")

    def generate_report(self):
        """Generate test report."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r["success"])
        failed = total - passed

        # Check API response format compliance
        api_compliant = sum(
            1 for r in self.test_results.values() if r.get("is_api_response", False)
        )

        logger.info("\n" + "=" * 60)
        logger.info("API INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
        logger.info(f"API Format Compliant: {api_compliant}/{total}")
        logger.info("")

        # List failed tests
        if failed > 0:
            logger.info("Failed Tests:")
            for name, result in self.test_results.items():
                if not result["success"]:
                    logger.info(f"  - {name}: {result.get('error', 'Unknown error')}")

        # Performance stats
        durations = [r["duration_ms"] for r in self.test_results.values() if "duration_ms" in r]

        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)

            logger.info("\nPerformance Stats:")
            logger.info(f"  Average Response Time: {avg_duration:.2f}ms")
            logger.info(f"  Max Response Time: {max_duration:.2f}ms")
            logger.info(f"  Min Response Time: {min_duration:.2f}ms")

        # Save detailed report
        report_path = Path("frontend_api_test_results.json")
        with open(report_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": failed,
                        "success_rate": passed / total if total > 0 else 0,
                        "api_compliant": api_compliant,
                    },
                    "results": self.test_results,
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"\nDetailed report saved to {report_path}")

        return passed == total


async def main():
    """Run frontend API integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Frontend API Integration")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", help="Run specific test group")

    args = parser.parse_args()

    tester = FrontendAPITester(args.base_url)

    try:
        await tester.setup()

        if args.test:
            # Run specific test
            test_method = getattr(tester, f"test_{args.test}", None)
            if test_method:
                await test_method()
            else:
                logger.error(f"Unknown test: {args.test}")
                return 1
        else:
            # Run all tests
            await tester.run_all_tests()

        # Generate report
        success = tester.generate_report()
        return 0 if success else 1

    finally:
        await tester.teardown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
