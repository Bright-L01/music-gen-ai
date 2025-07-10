#!/usr/bin/env python3
"""
Full End-to-End Integration Test Suite for Music Gen AI

This script performs comprehensive testing of all user workflows,
error handling, performance, and security aspects.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pytest
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "SecureTestPassword123!"


class IntegrationTestRunner:
    """Runs comprehensive integration tests for the Music Gen AI system."""

    def __init__(self):
        self.base_url = BASE_URL.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.csrf_token: Optional[str] = None
        self.test_results: List[Dict[str, Any]] = []
        self.issues_found: List[Dict[str, Any]] = []

    async def setup(self):
        """Setup test session."""
        self.session = aiohttp.ClientSession()

    async def teardown(self):
        """Cleanup test session."""
        if self.session:
            await self.session.close()

    def log_test(self, test_name: str, status: str, details: str = "", issue: Optional[str] = None):
        """Log test result."""
        icon = "✓" if status == "PASS" else "✗"
        color = Fore.GREEN if status == "PASS" else Fore.RED

        print(f"{color}{icon} {test_name}: {status}{Style.RESET_ALL}")
        if details:
            print(f"  {details}")

        result = {"test": test_name, "status": status, "details": details, "timestamp": time.time()}

        self.test_results.append(result)

        if issue:
            self.issues_found.append(
                {
                    "test": test_name,
                    "issue": issue,
                    "severity": "high" if status == "FAIL" else "medium",
                }
            )

    async def test_health_endpoints(self):
        """Test health check endpoints."""
        print(f"\n{Fore.CYAN}=== Testing Health Endpoints ==={Style.RESET_ALL}")

        # Test basic health
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.log_test("Basic Health Check", "PASS", f"Status: {data.get('status')}")
                else:
                    self.log_test("Basic Health Check", "FAIL", f"Status code: {resp.status}")
        except Exception as e:
            self.log_test("Basic Health Check", "FAIL", str(e), "API not accessible")

        # Test detailed health status
        try:
            async with self.session.get(f"{self.base_url}/health/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    services_ok = all(
                        s.get("status") == "healthy" for s in data.get("services", [])
                    )
                    if services_ok:
                        self.log_test("Detailed Health Status", "PASS", "All services healthy")
                    else:
                        self.log_test("Detailed Health Status", "WARN", "Some services unhealthy")
                else:
                    self.log_test("Detailed Health Status", "FAIL", f"Status code: {resp.status}")
        except Exception as e:
            self.log_test("Detailed Health Status", "FAIL", str(e))

    async def test_authentication_flow(self):
        """Test complete authentication workflow."""
        print(f"\n{Fore.CYAN}=== Testing Authentication Flow ==={Style.RESET_ALL}")

        # Get CSRF token
        try:
            async with self.session.get(f"{self.base_url}/api/auth/csrf-token") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.csrf_token = data.get("csrf_token")
                    self.log_test("CSRF Token Retrieval", "PASS")
                else:
                    self.log_test("CSRF Token Retrieval", "FAIL", f"Status code: {resp.status}")
                    return
        except Exception as e:
            self.log_test("CSRF Token Retrieval", "FAIL", str(e))
            return

        # Test registration
        register_data = {"email": TEST_EMAIL, "password": TEST_PASSWORD, "name": "Test User"}

        try:
            headers = {"X-CSRF-Token": self.csrf_token}
            async with self.session.post(
                f"{self.base_url}/api/auth/register", json=register_data, headers=headers
            ) as resp:
                if resp.status in [200, 201]:
                    self.log_test("User Registration", "PASS")
                elif resp.status == 409:
                    self.log_test("User Registration", "SKIP", "User already exists")
                else:
                    data = await resp.text()
                    self.log_test(
                        "User Registration", "FAIL", f"Status: {resp.status}, Body: {data}"
                    )
        except Exception as e:
            self.log_test("User Registration", "FAIL", str(e))

        # Test login
        login_data = {"email": TEST_EMAIL, "password": TEST_PASSWORD}

        try:
            async with self.session.post(
                f"{self.base_url}/api/auth/login", json=login_data, headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.auth_token = data.get("access_token")
                    self.log_test("User Login", "PASS", "Token received")
                else:
                    data = await resp.text()
                    self.log_test("User Login", "FAIL", f"Status: {resp.status}, Body: {data}")
        except Exception as e:
            self.log_test("User Login", "FAIL", str(e))

        # Test token validation
        if self.auth_token:
            try:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                async with self.session.get(
                    f"{self.base_url}/api/auth/me", headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Token Validation", "PASS", f"User: {data.get('email')}")
                    else:
                        self.log_test("Token Validation", "FAIL", f"Status: {resp.status}")
            except Exception as e:
                self.log_test("Token Validation", "FAIL", str(e))

    async def test_generation_workflow(self):
        """Test music generation workflow."""
        print(f"\n{Fore.CYAN}=== Testing Generation Workflow ==={Style.RESET_ALL}")

        if not self.auth_token:
            self.log_test("Generation Workflow", "SKIP", "No auth token available")
            return

        headers = {"Authorization": f"Bearer {self.auth_token}"}

        # Test text-to-music generation
        generation_data = {
            "prompt": "A peaceful piano melody with strings",
            "duration": 5,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
        }

        task_id = None

        # Start generation
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/generate/text", json=generation_data, headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    task_id = data.get("task_id")
                    self.log_test("Start Generation", "PASS", f"Task ID: {task_id}")
                else:
                    data = await resp.text()
                    self.log_test(
                        "Start Generation", "FAIL", f"Status: {resp.status}, Body: {data}"
                    )
                    return
        except Exception as e:
            self.log_test("Start Generation", "FAIL", str(e))
            return

        # Poll for completion
        if task_id:
            max_polls = 30  # 30 seconds max
            poll_count = 0

            while poll_count < max_polls:
                try:
                    async with self.session.get(
                        f"{self.base_url}/api/v1/generate/status/{task_id}", headers=headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get("status")

                            if status == "completed":
                                self.log_test("Generation Status", "PASS", "Generation completed")

                                # Test download
                                audio_url = data.get("result", {}).get("audio_path")
                                if audio_url:
                                    await self.test_audio_download(task_id)
                                break
                            elif status == "failed":
                                error = data.get("error", "Unknown error")
                                self.log_test(
                                    "Generation Status", "FAIL", f"Generation failed: {error}"
                                )
                                break
                            elif status == "processing":
                                poll_count += 1
                                await asyncio.sleep(1)
                            else:
                                self.log_test(
                                    "Generation Status", "FAIL", f"Unknown status: {status}"
                                )
                                break
                        else:
                            self.log_test(
                                "Generation Status", "FAIL", f"Status code: {resp.status}"
                            )
                            break
                except Exception as e:
                    self.log_test("Generation Status", "FAIL", str(e))
                    break

            if poll_count >= max_polls:
                self.log_test("Generation Status", "FAIL", "Timeout waiting for completion")

    async def test_audio_download(self, task_id: str):
        """Test audio file download."""
        try:
            async with self.session.get(f"{self.base_url}/download/{task_id}") as resp:
                if resp.status == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    content_length = int(resp.headers.get("Content-Length", 0))

                    if "audio" in content_type and content_length > 0:
                        self.log_test(
                            "Audio Download", "PASS", f"Size: {content_length/1024:.1f}KB"
                        )
                    else:
                        self.log_test("Audio Download", "FAIL", "Invalid audio response")
                else:
                    self.log_test("Audio Download", "FAIL", f"Status code: {resp.status}")
        except Exception as e:
            self.log_test("Audio Download", "FAIL", str(e))

    async def test_error_handling(self):
        """Test error handling scenarios."""
        print(f"\n{Fore.CYAN}=== Testing Error Handling ==={Style.RESET_ALL}")

        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

        # Test invalid endpoint
        try:
            async with self.session.get(f"{self.base_url}/api/v1/invalid") as resp:
                if resp.status == 404:
                    self.log_test("404 Error Handling", "PASS", "Proper 404 response")
                else:
                    self.log_test("404 Error Handling", "FAIL", f"Expected 404, got {resp.status}")
        except Exception as e:
            self.log_test("404 Error Handling", "FAIL", str(e))

        # Test invalid generation parameters
        invalid_data = {
            "prompt": "",  # Empty prompt
            "duration": -1,  # Invalid duration
            "temperature": 10.0,  # Invalid temperature
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/generate/text", json=invalid_data, headers=headers
            ) as resp:
                if resp.status in [400, 422]:
                    data = await resp.json()
                    if "detail" in data or "message" in data:
                        self.log_test(
                            "Validation Error Handling", "PASS", "Proper validation errors"
                        )
                    else:
                        self.log_test(
                            "Validation Error Handling", "FAIL", "No error details provided"
                        )
                else:
                    self.log_test(
                        "Validation Error Handling", "FAIL", f"Expected 4xx, got {resp.status}"
                    )
        except Exception as e:
            self.log_test("Validation Error Handling", "FAIL", str(e))

        # Test rate limiting
        rate_limit_hit = False
        for i in range(100):  # Try to hit rate limit
            try:
                async with self.session.get(f"{self.base_url}/health") as resp:
                    if resp.status == 429:
                        rate_limit_hit = True
                        break
            except:
                pass

        if rate_limit_hit:
            self.log_test("Rate Limiting", "PASS", "Rate limit enforced")
        else:
            self.log_test("Rate Limiting", "WARN", "Rate limit not hit (may be disabled)")

    async def test_performance(self):
        """Test performance characteristics."""
        print(f"\n{Fore.CYAN}=== Testing Performance ==={Style.RESET_ALL}")

        # Test response times
        endpoints = [
            ("/health", "GET", None),
            ("/api/v1/models", "GET", None),
            ("/api/v1/resources", "GET", None),
        ]

        for endpoint, method, data in endpoints:
            times = []

            for _ in range(5):
                start = time.time()
                try:
                    if method == "GET":
                        async with self.session.get(f"{self.base_url}{endpoint}") as resp:
                            await resp.read()
                            elapsed = time.time() - start
                            times.append(elapsed)
                except:
                    pass

            if times:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms

                if avg_time < 100:
                    self.log_test(f"Response Time {endpoint}", "PASS", f"Avg: {avg_time:.1f}ms")
                elif avg_time < 500:
                    self.log_test(
                        f"Response Time {endpoint}", "WARN", f"Avg: {avg_time:.1f}ms (slow)"
                    )
                else:
                    self.log_test(
                        f"Response Time {endpoint}", "FAIL", f"Avg: {avg_time:.1f}ms (too slow)"
                    )
                    self.issues_found.append(
                        {
                            "test": f"Response Time {endpoint}",
                            "issue": f"Slow response time: {avg_time:.1f}ms",
                            "severity": "high",
                        }
                    )

    async def test_security_headers(self):
        """Test security headers."""
        print(f"\n{Fore.CYAN}=== Testing Security Headers ==={Style.RESET_ALL}")

        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                headers = resp.headers

                # Check for security headers
                security_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=",
                    "Content-Security-Policy": None,  # Just check existence
                }

                for header, expected in security_headers.items():
                    value = headers.get(header)

                    if value:
                        if expected is None:
                            self.log_test(f"Security Header: {header}", "PASS", "Present")
                        elif isinstance(expected, list):
                            if any(exp in value for exp in expected):
                                self.log_test(
                                    f"Security Header: {header}", "PASS", f"Value: {value}"
                                )
                            else:
                                self.log_test(
                                    f"Security Header: {header}", "FAIL", f"Invalid value: {value}"
                                )
                        elif expected in value:
                            self.log_test(f"Security Header: {header}", "PASS", f"Value: {value}")
                        else:
                            self.log_test(
                                f"Security Header: {header}", "FAIL", f"Invalid value: {value}"
                            )
                    else:
                        self.log_test(f"Security Header: {header}", "FAIL", "Missing")
                        self.issues_found.append(
                            {
                                "test": "Security Headers",
                                "issue": f"Missing security header: {header}",
                                "severity": "medium",
                            }
                        )

                # Check for sensitive headers that shouldn't be present
                sensitive_headers = ["Server", "X-Powered-By"]
                for header in sensitive_headers:
                    if header in headers:
                        self.log_test(
                            f"Sensitive Header: {header}", "FAIL", "Should not be exposed"
                        )
                        self.issues_found.append(
                            {
                                "test": "Security Headers",
                                "issue": f"Sensitive header exposed: {header}",
                                "severity": "low",
                            }
                        )
                    else:
                        self.log_test(f"Sensitive Header: {header}", "PASS", "Not exposed")

        except Exception as e:
            self.log_test("Security Headers", "FAIL", str(e))

    async def test_mobile_simulation(self):
        """Simulate mobile client requests."""
        print(f"\n{Fore.CYAN}=== Testing Mobile Client Simulation ==={Style.RESET_ALL}")

        # Mobile user agent
        mobile_headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }

        try:
            async with self.session.get(f"{self.base_url}/", headers=mobile_headers) as resp:
                if resp.status == 200:
                    content = await resp.text()

                    # Check for viewport meta tag
                    if "viewport" in content.lower():
                        self.log_test("Mobile Viewport", "PASS", "Viewport meta tag present")
                    else:
                        self.log_test("Mobile Viewport", "WARN", "No viewport meta tag found")

                    # Check response size
                    size_kb = len(content.encode()) / 1024
                    if size_kb < 100:
                        self.log_test("Mobile Page Size", "PASS", f"Size: {size_kb:.1f}KB")
                    else:
                        self.log_test(
                            "Mobile Page Size", "WARN", f"Size: {size_kb:.1f}KB (large for mobile)"
                        )
                else:
                    self.log_test("Mobile Access", "FAIL", f"Status code: {resp.status}")
        except Exception as e:
            self.log_test("Mobile Access", "FAIL", str(e))

    async def test_slow_connection(self):
        """Simulate slow connection scenarios."""
        print(f"\n{Fore.CYAN}=== Testing Slow Connection Handling ==={Style.RESET_ALL}")

        # Test with timeout
        timeout = aiohttp.ClientTimeout(total=2)  # 2 second timeout

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(f"{self.base_url}/api/v1/models") as resp:
                        if resp.status == 200:
                            self.log_test(
                                "Slow Connection - Fast Endpoint",
                                "PASS",
                                "Responded within timeout",
                            )
                except asyncio.TimeoutError:
                    self.log_test(
                        "Slow Connection - Fast Endpoint", "FAIL", "Timeout on fast endpoint"
                    )
                    self.issues_found.append(
                        {
                            "test": "Slow Connection",
                            "issue": "Fast endpoints timeout on slow connections",
                            "severity": "high",
                        }
                    )
        except Exception as e:
            self.log_test("Slow Connection Test", "FAIL", str(e))

    async def run_all_tests(self):
        """Run all integration tests."""
        print(f"{Fore.YELLOW}{'='*60}")
        print(f"Music Gen AI - Full Integration Test Suite")
        print(f"API Base URL: {self.base_url}")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        await self.setup()

        try:
            # Run all test suites
            await self.test_health_endpoints()
            await self.test_authentication_flow()
            await self.test_generation_workflow()
            await self.test_error_handling()
            await self.test_performance()
            await self.test_security_headers()
            await self.test_mobile_simulation()
            await self.test_slow_connection()

            # Generate summary
            self.generate_summary()

        finally:
            await self.teardown()

    def generate_summary(self):
        """Generate test summary and punch list."""
        print(f"\n{Fore.YELLOW}{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # Count results
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
        warned = sum(1 for r in self.test_results if r["status"] == "WARN")
        skipped = sum(1 for r in self.test_results if r["status"] == "SKIP")

        total = len(self.test_results)

        print(f"Total Tests: {total}")
        print(f"{Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {failed}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Warnings: {warned}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Skipped: {skipped}{Style.RESET_ALL}")

        if failed > 0:
            print(f"\n{Fore.RED}OVERALL: FAILED{Style.RESET_ALL}")
        elif warned > 0:
            print(f"\n{Fore.YELLOW}OVERALL: PASSED WITH WARNINGS{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}OVERALL: PASSED{Style.RESET_ALL}")

        # Generate punch list
        if self.issues_found:
            print(f"\n{Fore.YELLOW}{'='*60}")
            print("PUNCH LIST - Issues to Address")
            print(f"{'='*60}{Style.RESET_ALL}\n")

            # Group by severity
            high_priority = [i for i in self.issues_found if i["severity"] == "high"]
            medium_priority = [i for i in self.issues_found if i["severity"] == "medium"]
            low_priority = [i for i in self.issues_found if i["severity"] == "low"]

            if high_priority:
                print(f"{Fore.RED}HIGH PRIORITY:{Style.RESET_ALL}")
                for issue in high_priority:
                    print(f"  - [{issue['test']}] {issue['issue']}")

            if medium_priority:
                print(f"\n{Fore.YELLOW}MEDIUM PRIORITY:{Style.RESET_ALL}")
                for issue in medium_priority:
                    print(f"  - [{issue['test']}] {issue['issue']}")

            if low_priority:
                print(f"\n{Fore.CYAN}LOW PRIORITY:{Style.RESET_ALL}")
                for issue in low_priority:
                    print(f"  - [{issue['test']}] {issue['issue']}")

        # Save detailed report
        report = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warned": warned,
                "skipped": skipped,
            },
            "results": self.test_results,
            "issues": self.issues_found,
        }

        report_path = Path("integration_test_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{Fore.CYAN}Detailed report saved to: {report_path}{Style.RESET_ALL}")


async def main():
    """Main test runner."""
    runner = IntegrationTestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
