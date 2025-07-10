#!/usr/bin/env python3
"""
Validation script for all critical area implementations.

This script validates that all security and performance implementations
are working correctly and meet the specified requirements.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

BASE_URL = "http://localhost:8000"


class CriticalAreaValidator:
    """Validates all critical area implementations."""

    def __init__(self):
        self.base_url = BASE_URL
        self.session: aiohttp.ClientSession = None
        self.results: List[Dict[str, Any]] = []

    async def setup(self):
        """Setup validation session."""
        self.session = aiohttp.ClientSession()

    async def teardown(self):
        """Cleanup validation session."""
        if self.session:
            await self.session.close()

    def log_result(self, area: str, test: str, status: str, details: str = ""):
        """Log validation result."""
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        color = Fore.GREEN if status == "PASS" else Fore.RED if status == "FAIL" else Fore.YELLOW

        print(f"{icon} {color}[{area}] {test}: {status}{Style.RESET_ALL}")
        if details:
            print(f"    {details}")

        self.results.append(
            {
                "area": area,
                "test": test,
                "status": status,
                "details": details,
                "timestamp": time.time(),
            }
        )

    async def validate_database_pool(self):
        """Validate database connection pooling implementation."""
        print(f"\n{Fore.CYAN}=== Validating Database Connection Pool ==={Style.RESET_ALL}")

        # Test health endpoint includes database status
        try:
            async with self.session.get(f"{self.base_url}/health/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    services = data.get("services", [])

                    # Look for database service
                    db_service = next((s for s in services if s.get("name") == "database"), None)

                    if db_service:
                        if db_service.get("status") == "healthy":
                            pool_stats = db_service.get("details", {})
                            self.log_result(
                                "Database Pool",
                                "Health Check",
                                "PASS",
                                f"Pool size: {pool_stats.get('pool_size', 'N/A')}, "
                                f"Free: {pool_stats.get('free_connections', 'N/A')}",
                            )
                        else:
                            self.log_result(
                                "Database Pool", "Health Check", "FAIL", "Database unhealthy"
                            )
                    else:
                        self.log_result(
                            "Database Pool", "Health Check", "FAIL", "No database service found"
                        )
                else:
                    self.log_result(
                        "Database Pool", "Health Check", "FAIL", f"Status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Database Pool", "Health Check", "FAIL", str(e))

        # Test connection pool statistics
        try:
            async with self.session.get(f"{self.base_url}/api/v1/resources") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    db_stats = data.get("database", {})

                    if db_stats:
                        pool_size = db_stats.get("pool_size", 0)
                        if pool_size >= 10:
                            self.log_result(
                                "Database Pool",
                                "Pool Configuration",
                                "PASS",
                                f"Pool size: {pool_size}",
                            )
                        else:
                            self.log_result(
                                "Database Pool",
                                "Pool Configuration",
                                "FAIL",
                                f"Pool too small: {pool_size}",
                            )
                    else:
                        self.log_result(
                            "Database Pool", "Pool Configuration", "FAIL", "No database stats"
                        )
                else:
                    self.log_result(
                        "Database Pool", "Pool Configuration", "FAIL", f"Status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Database Pool", "Pool Configuration", "FAIL", str(e))

    async def validate_security_headers(self):
        """Validate security headers implementation."""
        print(f"\n{Fore.CYAN}=== Validating Security Headers ==={Style.RESET_ALL}")

        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                headers = resp.headers

                # Required security headers
                security_headers = {
                    "Content-Security-Policy": "CSP",
                    "X-Content-Type-Options": "Content Type Options",
                    "Referrer-Policy": "Referrer Policy",
                    "Permissions-Policy": "Permissions Policy",
                }

                for header, name in security_headers.items():
                    if header in headers:
                        value = headers[header]
                        self.log_result("Security Headers", name, "PASS", f"Value: {value[:50]}...")
                    else:
                        self.log_result("Security Headers", name, "FAIL", "Header missing")

                # Check CSP quality
                csp = headers.get("Content-Security-Policy", "")
                if csp:
                    if "default-src 'none'" in csp:
                        self.log_result(
                            "Security Headers", "CSP Quality", "PASS", "Restrictive default-src"
                        )
                    else:
                        self.log_result(
                            "Security Headers", "CSP Quality", "WARN", "CSP not restrictive enough"
                        )

                # Check for unwanted headers
                unwanted_headers = ["Server", "X-Powered-By"]
                for header in unwanted_headers:
                    if header not in headers:
                        self.log_result(
                            "Security Headers", f"No {header}", "PASS", "Sensitive header hidden"
                        )
                    else:
                        self.log_result(
                            "Security Headers",
                            f"No {header}",
                            "FAIL",
                            f"Exposes: {headers[header]}",
                        )

        except Exception as e:
            self.log_result("Security Headers", "Validation", "FAIL", str(e))

    async def validate_request_size_limiting(self):
        """Validate request size limiting implementation."""
        print(f"\n{Fore.CYAN}=== Validating Request Size Limiting ==={Style.RESET_ALL}")

        # Test small request (should pass)
        try:
            small_data = {"test": "small request"}
            async with self.session.post(
                f"{self.base_url}/api/auth/login", json=small_data
            ) as resp:
                # We expect this to fail due to validation, not size
                if resp.status in [400, 401, 422]:  # Validation errors, not size errors
                    self.log_result(
                        "Request Size", "Small Request", "PASS", "Size limit not triggered"
                    )
                elif resp.status == 413:
                    self.log_result(
                        "Request Size", "Small Request", "FAIL", "Small request rejected"
                    )
                else:
                    self.log_result(
                        "Request Size", "Small Request", "WARN", f"Unexpected status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Request Size", "Small Request", "FAIL", str(e))

        # Test large request (should be rejected)
        try:
            large_data = {"test": "x" * (2 * 1024)}  # 2KB, should exceed 1KB limit for auth
            async with self.session.post(
                f"{self.base_url}/api/auth/login", json=large_data
            ) as resp:
                if resp.status == 413:
                    self.log_result(
                        "Request Size", "Large Request", "PASS", "Request properly rejected"
                    )
                else:
                    self.log_result(
                        "Request Size", "Large Request", "FAIL", f"Status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Request Size", "Large Request", "FAIL", str(e))

        # Test Content-Length requirement
        try:
            # Try to send request without Content-Length
            async with self.session.post(
                f"{self.base_url}/api/auth/login",
                data="test",
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 411:  # Length Required
                    self.log_result(
                        "Request Size", "Content-Length Required", "PASS", "Missing header rejected"
                    )
                else:
                    self.log_result(
                        "Request Size", "Content-Length Required", "WARN", f"Status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Request Size", "Content-Length Required", "FAIL", str(e))

    async def validate_websocket_security(self):
        """Validate WebSocket authentication hardening."""
        print(f"\n{Fore.CYAN}=== Validating WebSocket Security ==={Style.RESET_ALL}")

        # Test WebSocket connection without authentication
        try:
            import websockets

            # Try to connect without proper authentication
            ws_url = f"ws://localhost:8000/api/v1/ws"

            try:
                async with websockets.connect(ws_url) as websocket:
                    # Wait for authentication timeout (should be ~30 seconds)
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=35)
                        data = json.loads(message)

                        if (
                            data.get("type") == "disconnect"
                            and "timeout" in data.get("reason", "").lower()
                        ):
                            self.log_result(
                                "WebSocket Security", "Auth Timeout", "PASS", "Connection timed out"
                            )
                        else:
                            self.log_result(
                                "WebSocket Security", "Auth Timeout", "FAIL", "No timeout enforced"
                            )
                    except asyncio.TimeoutError:
                        self.log_result(
                            "WebSocket Security",
                            "Auth Timeout",
                            "FAIL",
                            "No timeout message received",
                        )

            except Exception as e:
                if "403" in str(e) or "unauthorized" in str(e).lower():
                    self.log_result(
                        "WebSocket Security",
                        "Connection Rejection",
                        "PASS",
                        "Unauthorized rejected",
                    )
                else:
                    self.log_result("WebSocket Security", "Connection Rejection", "WARN", str(e))

        except ImportError:
            self.log_result(
                "WebSocket Security", "Validation", "SKIP", "websockets library not available"
            )
        except Exception as e:
            self.log_result("WebSocket Security", "Validation", "FAIL", str(e))

    async def validate_performance_optimizations(self):
        """Validate performance optimization implementations."""
        print(f"\n{Fore.CYAN}=== Validating Performance Optimizations ==={Style.RESET_ALL}")

        # Test response compression
        try:
            headers = {"Accept-Encoding": "gzip"}
            async with self.session.get(f"{self.base_url}/api/v1/models", headers=headers) as resp:
                if resp.headers.get("Content-Encoding") == "gzip":
                    self.log_result(
                        "Performance", "Response Compression", "PASS", "GZIP compression enabled"
                    )
                else:
                    self.log_result(
                        "Performance", "Response Compression", "WARN", "No compression detected"
                    )
        except Exception as e:
            self.log_result("Performance", "Response Compression", "FAIL", str(e))

        # Test cache headers
        try:
            async with self.session.get(f"{self.base_url}/api/v1/models") as resp:
                etag = resp.headers.get("ETag")
                cache_control = resp.headers.get("Cache-Control")

                if etag:
                    self.log_result(
                        "Performance", "ETag Generation", "PASS", f"ETag: {etag[:20]}..."
                    )
                else:
                    self.log_result("Performance", "ETag Generation", "WARN", "No ETag header")

                if cache_control:
                    self.log_result(
                        "Performance", "Cache Headers", "PASS", f"Cache-Control: {cache_control}"
                    )
                else:
                    self.log_result(
                        "Performance", "Cache Headers", "WARN", "No Cache-Control header"
                    )
        except Exception as e:
            self.log_result("Performance", "Cache Headers", "FAIL", str(e))

        # Test response times
        start_time = time.time()
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                response_time = time.time() - start_time

                if response_time < 0.1:  # 100ms
                    self.log_result(
                        "Performance", "Response Time", "PASS", f"{response_time*1000:.1f}ms"
                    )
                elif response_time < 0.5:  # 500ms
                    self.log_result(
                        "Performance", "Response Time", "WARN", f"{response_time*1000:.1f}ms (slow)"
                    )
                else:
                    self.log_result(
                        "Performance",
                        "Response Time",
                        "FAIL",
                        f"{response_time*1000:.1f}ms (too slow)",
                    )
        except Exception as e:
            self.log_result("Performance", "Response Time", "FAIL", str(e))

    async def validate_mobile_optimizations(self):
        """Validate mobile performance optimizations."""
        print(f"\n{Fore.CYAN}=== Validating Mobile Optimizations ==={Style.RESET_ALL}")

        # Check if mobile optimization files exist
        mobile_files = [
            Path("frontend/src/hooks/usePerformanceOptimization.ts"),
            Path("frontend/webpack.mobile.config.js"),
        ]

        for file_path in mobile_files:
            if file_path.exists():
                self.log_result(
                    "Mobile Optimization", f"File: {file_path.name}", "PASS", "File exists"
                )
            else:
                self.log_result(
                    "Mobile Optimization", f"File: {file_path.name}", "FAIL", "File missing"
                )

        # Test mobile-friendly headers
        try:
            mobile_headers = {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
            }
            async with self.session.get(f"{self.base_url}/", headers=mobile_headers) as resp:
                if resp.status == 200:
                    content = await resp.text()

                    # Check for viewport meta tag
                    if "viewport" in content.lower():
                        self.log_result(
                            "Mobile Optimization",
                            "Viewport Meta Tag",
                            "PASS",
                            "Mobile viewport configured",
                        )
                    else:
                        self.log_result(
                            "Mobile Optimization",
                            "Viewport Meta Tag",
                            "WARN",
                            "No viewport meta tag",
                        )

                    # Check content size
                    size_kb = len(content.encode()) / 1024
                    if size_kb < 100:
                        self.log_result(
                            "Mobile Optimization", "Page Size", "PASS", f"{size_kb:.1f}KB"
                        )
                    else:
                        self.log_result(
                            "Mobile Optimization", "Page Size", "WARN", f"{size_kb:.1f}KB (large)"
                        )
                else:
                    self.log_result(
                        "Mobile Optimization", "Mobile Response", "FAIL", f"Status: {resp.status}"
                    )
        except Exception as e:
            self.log_result("Mobile Optimization", "Mobile Response", "FAIL", str(e))

    async def run_all_validations(self):
        """Run all validation tests."""
        print(f"{Fore.YELLOW}{'='*60}")
        print("Critical Areas Implementation Validation")
        print(f"{'='*60}{Style.RESET_ALL}")

        await self.setup()

        try:
            await self.validate_database_pool()
            await self.validate_security_headers()
            await self.validate_request_size_limiting()
            await self.validate_websocket_security()
            await self.validate_performance_optimizations()
            await self.validate_mobile_optimizations()

            self.generate_summary()

        finally:
            await self.teardown()

    def generate_summary(self):
        """Generate validation summary."""
        print(f"\n{Fore.YELLOW}{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")

        # Count results by status
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        warnings = sum(1 for r in self.results if r["status"] == "WARN")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")

        print(f"Total Tests: {total}")
        print(f"{Fore.GREEN}Passed: {passed}")
        print(f"{Fore.RED}Failed: {failed}")
        print(f"{Fore.YELLOW}Warnings: {warnings}")
        print(f"{Fore.CYAN}Skipped: {skipped}{Style.RESET_ALL}")

        # Overall status
        if failed == 0:
            if warnings == 0:
                print(f"\n{Fore.GREEN}✅ ALL VALIDATIONS PASSED{Style.RESET_ALL}")
                print("🚀 System is ready for production deployment!")
            else:
                print(f"\n{Fore.YELLOW}⚠️ PASSED WITH WARNINGS{Style.RESET_ALL}")
                print("📋 Review warnings before production deployment.")
        else:
            print(f"\n{Fore.RED}❌ VALIDATION FAILED{Style.RESET_ALL}")
            print("🔧 Fix failed tests before proceeding to production.")

        # Group results by area
        areas = {}
        for result in self.results:
            area = result["area"]
            if area not in areas:
                areas[area] = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
            areas[area][result["status"].lower()] += 1

        print(f"\n{Fore.CYAN}Results by Area:{Style.RESET_ALL}")
        for area, counts in areas.items():
            total_area = sum(counts.values())
            passed_area = counts["pass"]
            status_icon = "✅" if counts["fail"] == 0 else "❌"

            print(f"{status_icon} {area}: {passed_area}/{total_area} passed")

        # Save detailed results
        report_path = Path("validation_report.json")
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": failed,
                        "warnings": warnings,
                        "skipped": skipped,
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\n📄 Detailed report saved to: {report_path}")


async def main():
    """Main validation function."""
    validator = CriticalAreaValidator()
    await validator.run_all_validations()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Validation interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Validation failed: {e}{Style.RESET_ALL}")
        sys.exit(1)
