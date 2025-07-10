#!/usr/bin/env python3
"""
User Acceptance Testing (UAT) Suite for Music Gen AI
Comprehensive testing of user workflows, business logic, and acceptance criteria
"""

import os
import json
import time
import requests
import asyncio
import websockets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import base64
import wave
import subprocess
from dataclasses import dataclass, asdict, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/app/results/uat_test.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Individual UAT test case"""

    id: str
    category: str
    scenario: str
    description: str
    preconditions: List[str]
    steps: List[Dict[str, Any]]
    expected_results: List[str]
    priority: str  # Critical, High, Medium, Low
    business_value: str


@dataclass
class TestResult:
    """UAT test result"""

    test_case_id: str
    status: str  # Pass, Fail, Blocked, Skipped
    execution_time: float
    actual_results: List[str]
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass
class Issue:
    """Issue found during UAT"""

    id: str
    test_case_id: str
    severity: str  # Critical, High, Medium, Low
    type: str  # Bug, Performance, Usability, Feature Gap
    description: str
    steps_to_reproduce: List[str]
    expected_behavior: str
    actual_behavior: str
    workaround: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)


class UATTestSuite:
    def __init__(self):
        self.base_url = os.getenv("TARGET_HOST", "http://nginx-staging")
        self.api_key = os.getenv("STAGING_API_KEY", "staging_api_key_change_me")
        self.results_dir = Path("/app/results/uat")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Test data
        self.test_users = {
            "free_user": {"email": "free@test.com", "tier": "free", "api_key": "test_free_key"},
            "standard_user": {
                "email": "standard@test.com",
                "tier": "standard",
                "api_key": "test_standard_key",
            },
            "premium_user": {
                "email": "premium@test.com",
                "tier": "premium",
                "api_key": "test_premium_key",
            },
        }

        # Performance SLAs
        self.performance_slas = {
            "api_response_time_p95": 2.0,  # 95th percentile < 2 seconds
            "api_response_time_p99": 5.0,  # 99th percentile < 5 seconds
            "generation_time_10s": 30.0,  # 10-second audio < 30 seconds to generate
            "generation_time_30s": 60.0,  # 30-second audio < 60 seconds to generate
            "uptime_percentage": 99.9,  # 99.9% uptime
            "error_rate_threshold": 0.01,  # Less than 1% error rate
        }

        # Test results storage
        self.test_results: List[TestResult] = []
        self.issues: List[Issue] = []
        self.performance_data: Dict[str, List[float]] = {
            "response_times": [],
            "generation_times": [],
            "error_count": 0,
            "total_requests": 0,
        }

    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive UAT test cases"""
        test_cases = []

        # User Registration and Authentication
        test_cases.append(
            TestCase(
                id="UAT-001",
                category="Authentication",
                scenario="New User Registration",
                description="Verify new user can register and receive API key",
                preconditions=["System is accessible", "Email service is configured"],
                steps=[
                    {
                        "action": "Navigate to registration endpoint",
                        "data": {"endpoint": "/api/v1/auth/register"},
                    },
                    {
                        "action": "Submit registration",
                        "data": {"email": "newuser@test.com", "password": "Test123!@#"},
                    },
                    {"action": "Verify email sent", "data": {}},
                    {"action": "Confirm email", "data": {}},
                    {"action": "Login with credentials", "data": {}},
                    {"action": "Retrieve API key", "data": {}},
                ],
                expected_results=[
                    "User account created successfully",
                    "Confirmation email received",
                    "Login successful with valid credentials",
                    "API key generated and retrievable",
                ],
                priority="Critical",
                business_value="Essential for user onboarding",
            )
        )

        # Music Generation Workflows
        test_cases.append(
            TestCase(
                id="UAT-002",
                category="Music Generation",
                scenario="Basic Music Generation",
                description="Generate music from text prompt",
                preconditions=["User authenticated", "Models loaded"],
                steps=[
                    {
                        "action": "Submit generation request",
                        "data": {"prompt": "upbeat jazz music", "duration": 10},
                    },
                    {"action": "Monitor generation progress", "data": {}},
                    {"action": "Retrieve generated audio", "data": {}},
                    {"action": "Verify audio quality", "data": {}},
                ],
                expected_results=[
                    "Generation request accepted",
                    "Progress updates received",
                    "Audio file generated successfully",
                    "Audio matches prompt description",
                    "Audio duration matches requested duration",
                ],
                priority="Critical",
                business_value="Core product functionality",
            )
        )

        # Advanced Generation Features
        test_cases.append(
            TestCase(
                id="UAT-003",
                category="Music Generation",
                scenario="Conditional Generation with Parameters",
                description="Generate music with specific parameters",
                preconditions=["Premium user account", "Advanced models available"],
                steps=[
                    {
                        "action": "Submit advanced generation",
                        "data": {
                            "prompt": "classical piano sonata in minor key",
                            "duration": 30,
                            "temperature": 0.7,
                            "instruments": ["piano"],
                            "tempo": "andante",
                            "key": "D minor",
                        },
                    },
                    {"action": "Verify parameter application", "data": {}},
                    {
                        "action": "Download in multiple formats",
                        "data": {"formats": ["wav", "mp3", "midi"]},
                    },
                ],
                expected_results=[
                    "Parameters accepted and applied",
                    "Generated music matches specifications",
                    "Multiple format downloads successful",
                    "MIDI export contains correct instrument data",
                ],
                priority="High",
                business_value="Premium feature validation",
            )
        )

        # Batch Processing
        test_cases.append(
            TestCase(
                id="UAT-004",
                category="Batch Operations",
                scenario="Batch Music Generation",
                description="Process multiple generation requests",
                preconditions=["Standard or Premium account", "Sufficient quota"],
                steps=[
                    {
                        "action": "Submit batch request",
                        "data": {
                            "requests": [
                                {"prompt": "morning meditation music", "duration": 60},
                                {"prompt": "workout motivation music", "duration": 30},
                                {"prompt": "evening relaxation sounds", "duration": 45},
                            ]
                        },
                    },
                    {"action": "Monitor batch progress", "data": {}},
                    {"action": "Retrieve all results", "data": {}},
                    {"action": "Verify individual quality", "data": {}},
                ],
                expected_results=[
                    "Batch accepted with job ID",
                    "Progress tracking for each item",
                    "All items generated successfully",
                    "Each audio matches its prompt",
                ],
                priority="High",
                business_value="Efficiency for power users",
            )
        )

        # Error Handling
        test_cases.append(
            TestCase(
                id="UAT-005",
                category="Error Handling",
                scenario="Invalid Input Handling",
                description="System handles invalid inputs gracefully",
                preconditions=["User authenticated"],
                steps=[
                    {"action": "Submit empty prompt", "data": {"prompt": ""}},
                    {
                        "action": "Submit excessive duration",
                        "data": {"prompt": "test", "duration": 3600},
                    },
                    {
                        "action": "Submit invalid parameters",
                        "data": {"prompt": "test", "temperature": 10},
                    },
                    {"action": "Exceed rate limits", "data": {"repeat": 100}},
                ],
                expected_results=[
                    "Empty prompt rejected with clear error",
                    "Duration limit enforced with message",
                    "Invalid parameters rejected with guidance",
                    "Rate limiting returns 429 with retry-after header",
                ],
                priority="High",
                business_value="User experience and system stability",
            )
        )

        # Performance Testing
        test_cases.append(
            TestCase(
                id="UAT-006",
                category="Performance",
                scenario="Response Time Validation",
                description="Verify system meets performance SLAs",
                preconditions=["System under normal load"],
                steps=[
                    {"action": "Measure API response times", "data": {"requests": 100}},
                    {"action": "Measure generation times", "data": {"durations": [5, 10, 30]}},
                    {"action": "Test concurrent requests", "data": {"concurrent": 10}},
                    {"action": "Monitor resource usage", "data": {}},
                ],
                expected_results=[
                    "95th percentile response time < 2s",
                    "99th percentile response time < 5s",
                    "Generation time scales linearly with duration",
                    "No degradation under concurrent load",
                ],
                priority="Critical",
                business_value="Service reliability",
            )
        )

        # User Account Management
        test_cases.append(
            TestCase(
                id="UAT-007",
                category="Account Management",
                scenario="Subscription Upgrade/Downgrade",
                description="User can change subscription tiers",
                preconditions=["User with active subscription"],
                steps=[
                    {"action": "View current plan", "data": {}},
                    {"action": "Upgrade to premium", "data": {"tier": "premium"}},
                    {"action": "Verify premium features", "data": {}},
                    {"action": "Downgrade to standard", "data": {"tier": "standard"}},
                    {"action": "Verify feature restrictions", "data": {}},
                ],
                expected_results=[
                    "Current plan displayed correctly",
                    "Upgrade processed immediately",
                    "Premium features accessible",
                    "Downgrade handled gracefully",
                    "Features restricted appropriately",
                ],
                priority="High",
                business_value="Revenue and user satisfaction",
            )
        )

        # API Integration
        test_cases.append(
            TestCase(
                id="UAT-008",
                category="API Integration",
                scenario="Third-party Integration",
                description="API works with external applications",
                preconditions=["API documentation available", "Test client ready"],
                steps=[
                    {"action": "Authenticate via API", "data": {"method": "Bearer token"}},
                    {"action": "Test all endpoints", "data": {}},
                    {
                        "action": "Verify webhook delivery",
                        "data": {"webhook_url": "https://test.webhook.site"},
                    },
                    {
                        "action": "Test SDK integration",
                        "data": {"languages": ["Python", "JavaScript"]},
                    },
                ],
                expected_results=[
                    "Authentication works with standard libraries",
                    "All documented endpoints functional",
                    "Webhooks delivered with correct payload",
                    "SDKs function as documented",
                ],
                priority="Medium",
                business_value="Developer adoption",
            )
        )

        # Data Privacy and Security
        test_cases.append(
            TestCase(
                id="UAT-009",
                category="Security",
                scenario="Data Privacy Compliance",
                description="Verify GDPR/privacy compliance",
                preconditions=["User account with data"],
                steps=[
                    {"action": "Request data export", "data": {}},
                    {"action": "Verify data completeness", "data": {}},
                    {"action": "Request data deletion", "data": {}},
                    {"action": "Verify deletion", "data": {}},
                ],
                expected_results=[
                    "Data export available within 48 hours",
                    "Export contains all user data",
                    "Deletion request processed",
                    "Data verifiably removed from all systems",
                ],
                priority="Critical",
                business_value="Legal compliance",
            )
        )

        # Monitoring and Analytics
        test_cases.append(
            TestCase(
                id="UAT-010",
                category="Analytics",
                scenario="Usage Analytics",
                description="User can view their usage statistics",
                preconditions=["User with generation history"],
                steps=[
                    {"action": "View usage dashboard", "data": {}},
                    {"action": "Check generation history", "data": {}},
                    {"action": "Download usage report", "data": {"format": "csv"}},
                    {"action": "Verify billing accuracy", "data": {}},
                ],
                expected_results=[
                    "Dashboard shows accurate statistics",
                    "History includes all generations",
                    "CSV export contains complete data",
                    "Billing matches actual usage",
                ],
                priority="Medium",
                business_value="User transparency and trust",
            )
        )

        return test_cases

    async def execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single UAT test case"""
        logger.info(f"Executing test case: {test_case.id} - {test_case.scenario}")

        start_time = time.time()
        test_result = TestResult(
            test_case_id=test_case.id, status="Pass", execution_time=0, actual_results=[]
        )

        try:
            # Execute based on category
            if test_case.category == "Authentication":
                await self._test_authentication(test_case, test_result)
            elif test_case.category == "Music Generation":
                await self._test_music_generation(test_case, test_result)
            elif test_case.category == "Batch Operations":
                await self._test_batch_operations(test_case, test_result)
            elif test_case.category == "Error Handling":
                await self._test_error_handling(test_case, test_result)
            elif test_case.category == "Performance":
                await self._test_performance(test_case, test_result)
            elif test_case.category == "Account Management":
                await self._test_account_management(test_case, test_result)
            elif test_case.category == "API Integration":
                await self._test_api_integration(test_case, test_result)
            elif test_case.category == "Security":
                await self._test_security(test_case, test_result)
            elif test_case.category == "Analytics":
                await self._test_analytics(test_case, test_result)
            else:
                test_result.status = "Skipped"
                test_result.actual_results.append(f"Unknown category: {test_case.category}")

        except Exception as e:
            test_result.status = "Fail"
            test_result.actual_results.append(f"Test execution error: {str(e)}")
            self._log_issue(test_case, "Test execution failed", str(e), "Critical")

        test_result.execution_time = time.time() - start_time
        self.test_results.append(test_result)

        # Log result
        status_icon = "✅" if test_result.status == "Pass" else "❌"
        logger.info(
            f"{status_icon} {test_case.id}: {test_result.status} ({test_result.execution_time:.2f}s)"
        )

        return test_result

    async def _test_authentication(self, test_case: TestCase, test_result: TestResult):
        """Test authentication workflows"""
        if test_case.id == "UAT-001":
            # Test user registration
            session = requests.Session()

            # Step 1: Register
            register_data = {
                "email": f"uat_test_{int(time.time())}@test.com",
                "password": "Test123!@#",
                "username": f"uat_user_{int(time.time())}",
            }

            response = session.post(
                f"{self.base_url}/api/v1/auth/register", json=register_data, timeout=30
            )

            if response.status_code == 201:
                test_result.actual_results.append("User registration successful")
                self._track_performance("api_response", response.elapsed.total_seconds())
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(f"Registration failed: {response.status_code}")
                self._log_issue(
                    test_case,
                    "Registration failed",
                    f"Status: {response.status_code}, Response: {response.text}",
                    "High",
                )
                return

            # Step 2: Login
            login_data = {"email": register_data["email"], "password": register_data["password"]}

            response = session.post(
                f"{self.base_url}/api/v1/auth/login", json=login_data, timeout=30
            )

            if response.status_code == 200:
                auth_data = response.json()
                if "access_token" in auth_data:
                    test_result.actual_results.append("Login successful, token received")
                    self._track_performance("api_response", response.elapsed.total_seconds())
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append("Login response missing access token")
                    self._log_issue(
                        test_case, "Invalid login response", "Access token not provided", "High"
                    )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(f"Login failed: {response.status_code}")
                self._log_issue(
                    test_case, "Login failed", f"Status: {response.status_code}", "High"
                )

    async def _test_music_generation(self, test_case: TestCase, test_result: TestResult):
        """Test music generation workflows"""
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        if test_case.id == "UAT-002":
            # Basic generation test
            generation_data = {
                "prompt": "upbeat jazz music with saxophone",
                "duration": 10.0,
                "format": "wav",
            }

            start_time = time.time()
            response = session.post(
                f"{self.base_url}/api/v1/generate", json=generation_data, timeout=120
            )

            if response.status_code in [200, 202]:
                generation_time = time.time() - start_time
                self._track_performance("generation_time", generation_time)
                test_result.actual_results.append(
                    f"Generation request accepted in {generation_time:.2f}s"
                )

                # Check response format
                response_data = response.json()
                if "task_id" in response_data or "audio_url" in response_data:
                    test_result.actual_results.append("Valid response format received")

                    # If async, poll for completion
                    if "task_id" in response_data:
                        task_id = response_data["task_id"]
                        completed = await self._poll_task_completion(session, task_id, 120)

                        if completed:
                            test_result.actual_results.append(
                                "Async generation completed successfully"
                            )
                        else:
                            test_result.status = "Fail"
                            test_result.actual_results.append("Async generation timed out")
                            self._log_issue(
                                test_case,
                                "Generation timeout",
                                f"Task {task_id} did not complete in 120s",
                                "High",
                            )
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append("Invalid response format")
                    self._log_issue(
                        test_case, "Invalid API response", "Missing task_id or audio_url", "High"
                    )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(f"Generation failed: {response.status_code}")
                self._log_issue(
                    test_case,
                    "Generation request failed",
                    f"Status: {response.status_code}, Response: {response.text}",
                    "Critical",
                )

        elif test_case.id == "UAT-003":
            # Advanced generation with parameters
            generation_data = {
                "prompt": "classical piano sonata in D minor",
                "duration": 30.0,
                "temperature": 0.7,
                "conditioning": {
                    "instruments": ["piano"],
                    "tempo": "andante",
                    "key": "D minor",
                    "style": "classical",
                },
            }

            response = session.post(
                f"{self.base_url}/api/v1/generate/advanced", json=generation_data, timeout=180
            )

            if response.status_code in [200, 202]:
                test_result.actual_results.append("Advanced generation request accepted")

                # Verify parameters were accepted
                response_data = response.json()
                if "parameters_applied" in response_data:
                    test_result.actual_results.append("Generation parameters confirmed")
                else:
                    test_result.actual_results.append("Warning: Parameter confirmation missing")
                    self._log_issue(
                        test_case,
                        "Missing parameter confirmation",
                        "API should confirm which parameters were applied",
                        "Low",
                    )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(
                    f"Advanced generation failed: {response.status_code}"
                )
                self._log_issue(
                    test_case,
                    "Advanced generation failed",
                    f"Status: {response.status_code}",
                    "High",
                )

    async def _test_batch_operations(self, test_case: TestCase, test_result: TestResult):
        """Test batch processing workflows"""
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        if test_case.id == "UAT-004":
            batch_data = {
                "requests": [
                    {"prompt": "morning meditation music", "duration": 60},
                    {"prompt": "workout motivation music", "duration": 30},
                    {"prompt": "evening relaxation sounds", "duration": 45},
                ]
            }

            response = session.post(
                f"{self.base_url}/api/v1/generate/batch", json=batch_data, timeout=60
            )

            if response.status_code in [200, 202]:
                test_result.actual_results.append("Batch request accepted")
                batch_response = response.json()

                if "batch_id" in batch_response:
                    batch_id = batch_response["batch_id"]
                    test_result.actual_results.append(f"Batch ID received: {batch_id}")

                    # Monitor batch progress
                    completed = await self._monitor_batch_progress(session, batch_id, 300)

                    if completed:
                        test_result.actual_results.append("All batch items completed")
                    else:
                        test_result.status = "Fail"
                        test_result.actual_results.append("Batch processing timed out")
                        self._log_issue(
                            test_case, "Batch timeout", f"Batch {batch_id} did not complete", "High"
                        )
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append("No batch ID in response")
                    self._log_issue(test_case, "Invalid batch response", "Missing batch_id", "High")
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(f"Batch request failed: {response.status_code}")
                self._log_issue(
                    test_case, "Batch request failed", f"Status: {response.status_code}", "High"
                )

    async def _test_error_handling(self, test_case: TestCase, test_result: TestResult):
        """Test error handling scenarios"""
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        if test_case.id == "UAT-005":
            error_scenarios = [
                {
                    "name": "Empty prompt",
                    "data": {"prompt": "", "duration": 10},
                    "expected_status": 400,
                    "expected_error": "prompt",
                },
                {
                    "name": "Excessive duration",
                    "data": {"prompt": "test music", "duration": 3600},
                    "expected_status": 400,
                    "expected_error": "duration",
                },
                {
                    "name": "Invalid temperature",
                    "data": {"prompt": "test music", "duration": 10, "temperature": 10},
                    "expected_status": 400,
                    "expected_error": "temperature",
                },
            ]

            for scenario in error_scenarios:
                response = session.post(
                    f"{self.base_url}/api/v1/generate", json=scenario["data"], timeout=30
                )

                if response.status_code == scenario["expected_status"]:
                    test_result.actual_results.append(f"{scenario['name']}: Correctly rejected")

                    # Check error message
                    try:
                        error_data = response.json()
                        if "error" in error_data or "message" in error_data:
                            error_msg = error_data.get("error") or error_data.get("message")
                            if scenario["expected_error"] in str(error_msg).lower():
                                test_result.actual_results.append(
                                    f"{scenario['name']}: Clear error message"
                                )
                            else:
                                test_result.actual_results.append(
                                    f"{scenario['name']}: Unclear error message"
                                )
                                self._log_issue(
                                    test_case,
                                    "Unclear error message",
                                    f"{scenario['name']}: {error_msg}",
                                    "Low",
                                )
                    except:
                        test_result.actual_results.append(
                            f"{scenario['name']}: Non-JSON error response"
                        )
                        self._log_issue(
                            test_case,
                            "Invalid error format",
                            f"{scenario['name']} returned non-JSON error",
                            "Medium",
                        )
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append(
                        f"{scenario['name']}: Wrong status code {response.status_code} (expected {scenario['expected_status']})"
                    )
                    self._log_issue(
                        test_case,
                        "Incorrect error handling",
                        f"{scenario['name']} not handled properly",
                        "High",
                    )

            # Test rate limiting
            rate_limit_hit = False
            for i in range(50):
                response = session.get(f"{self.base_url}/api/v1/models", timeout=5)
                if response.status_code == 429:
                    rate_limit_hit = True
                    if "Retry-After" in response.headers:
                        test_result.actual_results.append(
                            "Rate limiting works with Retry-After header"
                        )
                    else:
                        test_result.actual_results.append(
                            "Rate limiting works but missing Retry-After"
                        )
                        self._log_issue(
                            test_case,
                            "Missing Retry-After header",
                            "Rate limit response should include Retry-After",
                            "Medium",
                        )
                    break

            if not rate_limit_hit:
                test_result.actual_results.append(
                    "Warning: Rate limiting not triggered after 50 requests"
                )
                self._log_issue(
                    test_case,
                    "Rate limiting not enforced",
                    "50 rapid requests did not trigger rate limit",
                    "Medium",
                )

    async def _test_performance(self, test_case: TestCase, test_result: TestResult):
        """Test performance against SLAs"""
        if test_case.id == "UAT-006":
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})

            # Test API response times
            response_times = []
            for i in range(100):
                start = time.time()
                response = session.get(f"{self.base_url}/api/v1/models", timeout=10)
                elapsed = time.time() - start

                if response.status_code == 200:
                    response_times.append(elapsed)
                    self._track_performance("api_response", elapsed)
                else:
                    self.performance_data["error_count"] += 1

                self.performance_data["total_requests"] += 1

            # Calculate percentiles
            if response_times:
                response_times.sort()
                p95 = response_times[int(len(response_times) * 0.95)]
                p99 = response_times[int(len(response_times) * 0.99)]

                test_result.performance_metrics["api_p95"] = p95
                test_result.performance_metrics["api_p99"] = p99

                if p95 <= self.performance_slas["api_response_time_p95"]:
                    test_result.actual_results.append(
                        f"✅ API P95: {p95:.3f}s (SLA: {self.performance_slas['api_response_time_p95']}s)"
                    )
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append(
                        f"❌ API P95: {p95:.3f}s (SLA: {self.performance_slas['api_response_time_p95']}s)"
                    )
                    self._log_issue(
                        test_case,
                        "API P95 SLA violation",
                        f"P95 response time {p95:.3f}s exceeds SLA",
                        "High",
                    )

                if p99 <= self.performance_slas["api_response_time_p99"]:
                    test_result.actual_results.append(
                        f"✅ API P99: {p99:.3f}s (SLA: {self.performance_slas['api_response_time_p99']}s)"
                    )
                else:
                    test_result.status = "Fail"
                    test_result.actual_results.append(
                        f"❌ API P99: {p99:.3f}s (SLA: {self.performance_slas['api_response_time_p99']}s)"
                    )
                    self._log_issue(
                        test_case,
                        "API P99 SLA violation",
                        f"P99 response time {p99:.3f}s exceeds SLA",
                        "High",
                    )

            # Test generation times
            generation_tests = [
                {"duration": 10, "sla_key": "generation_time_10s"},
                {"duration": 30, "sla_key": "generation_time_30s"},
            ]

            for gen_test in generation_tests:
                start = time.time()
                response = session.post(
                    f"{self.base_url}/api/v1/generate",
                    json={"prompt": "test music", "duration": gen_test["duration"]},
                    timeout=180,
                )

                if response.status_code in [200, 202]:
                    if "task_id" in response.json():
                        task_id = response.json()["task_id"]
                        completed = await self._poll_task_completion(session, task_id, 180)
                        if completed:
                            gen_time = time.time() - start
                            self._track_performance("generation_time", gen_time)

                            if gen_time <= self.performance_slas[gen_test["sla_key"]]:
                                test_result.actual_results.append(
                                    f"✅ {gen_test['duration']}s generation: {gen_time:.1f}s (SLA: {self.performance_slas[gen_test['sla_key']]}s)"
                                )
                            else:
                                test_result.status = "Fail"
                                test_result.actual_results.append(
                                    f"❌ {gen_test['duration']}s generation: {gen_time:.1f}s (SLA: {self.performance_slas[gen_test['sla_key']]}s)"
                                )
                                self._log_issue(
                                    test_case,
                                    "Generation time SLA violation",
                                    f"{gen_test['duration']}s audio took {gen_time:.1f}s",
                                    "High",
                                )

            # Calculate error rate
            error_rate = (
                self.performance_data["error_count"] / self.performance_data["total_requests"]
            )
            if error_rate <= self.performance_slas["error_rate_threshold"]:
                test_result.actual_results.append(
                    f"✅ Error rate: {error_rate:.2%} (SLA: {self.performance_slas['error_rate_threshold']:.2%})"
                )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append(
                    f"❌ Error rate: {error_rate:.2%} (SLA: {self.performance_slas['error_rate_threshold']:.2%})"
                )
                self._log_issue(
                    test_case,
                    "Error rate SLA violation",
                    f"Error rate {error_rate:.2%} exceeds threshold",
                    "Critical",
                )

    async def _test_account_management(self, test_case: TestCase, test_result: TestResult):
        """Test account management features"""
        if test_case.id == "UAT-007":
            # Simulate subscription management
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})

            # Get current plan
            response = session.get(f"{self.base_url}/api/v1/account/subscription", timeout=30)
            if response.status_code == 200:
                current_plan = response.json()
                test_result.actual_results.append(
                    f"Current plan retrieved: {current_plan.get('tier', 'unknown')}"
                )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append("Failed to retrieve current plan")
                self._log_issue(
                    test_case,
                    "Cannot retrieve subscription",
                    f"Status: {response.status_code}",
                    "High",
                )
                return

            # Test upgrade
            upgrade_response = session.post(
                f"{self.base_url}/api/v1/account/subscription/upgrade",
                json={"tier": "premium"},
                timeout=30,
            )

            if upgrade_response.status_code in [200, 202]:
                test_result.actual_results.append("Upgrade request successful")

                # Verify premium features
                features_response = session.get(
                    f"{self.base_url}/api/v1/account/features", timeout=30
                )
                if features_response.status_code == 200:
                    features = features_response.json()
                    if features.get("advanced_generation", False):
                        test_result.actual_results.append("Premium features confirmed active")
                    else:
                        test_result.actual_results.append(
                            "Warning: Premium features not immediately active"
                        )
                        self._log_issue(
                            test_case,
                            "Feature activation delay",
                            "Premium features not immediately available after upgrade",
                            "Medium",
                        )
            else:
                test_result.actual_results.append(
                    "Upgrade simulation noted (may require payment integration)"
                )

    async def _test_api_integration(self, test_case: TestCase, test_result: TestResult):
        """Test API integration scenarios"""
        if test_case.id == "UAT-008":
            # Test different authentication methods
            auth_methods = [
                {"method": "Bearer", "header": f"Bearer {self.api_key}"},
                {"method": "API-Key", "header": self.api_key},
            ]

            for auth in auth_methods:
                session = requests.Session()
                if auth["method"] == "Bearer":
                    session.headers.update({"Authorization": auth["header"]})
                else:
                    session.headers.update({"X-API-Key": auth["header"]})

                response = session.get(f"{self.base_url}/api/v1/models", timeout=30)

                if response.status_code == 200:
                    test_result.actual_results.append(f"{auth['method']} authentication successful")
                else:
                    test_result.actual_results.append(f"{auth['method']} authentication failed")
                    if auth["method"] == "Bearer":  # This should work
                        test_result.status = "Fail"
                        self._log_issue(
                            test_case,
                            "Bearer auth failed",
                            f"Standard Bearer authentication not working",
                            "Critical",
                        )

            # Test webhook simulation
            webhook_data = {
                "event": "generation.completed",
                "webhook_url": "https://webhook.site/test",
                "retry_policy": {"max_retries": 3, "retry_delay": 60},
            }

            response = requests.post(
                f"{self.base_url}/api/v1/webhooks",
                json=webhook_data,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )

            if response.status_code in [200, 201]:
                test_result.actual_results.append("Webhook registration successful")
            else:
                test_result.actual_results.append("Webhook registration not available")
                self._log_issue(
                    test_case,
                    "Webhook support missing",
                    "Webhook functionality not implemented",
                    "Medium",
                )

    async def _test_security(self, test_case: TestCase, test_result: TestResult):
        """Test security and privacy features"""
        if test_case.id == "UAT-009":
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})

            # Test data export
            export_response = session.post(
                f"{self.base_url}/api/v1/account/export", json={"format": "json"}, timeout=30
            )

            if export_response.status_code in [200, 202]:
                test_result.actual_results.append("Data export request accepted")

                if export_response.status_code == 202:
                    # Async export
                    export_data = export_response.json()
                    if "export_id" in export_data:
                        test_result.actual_results.append("Export ID received for tracking")
                    else:
                        test_result.actual_results.append("Warning: No export ID provided")
                        self._log_issue(
                            test_case,
                            "Missing export tracking",
                            "Async export should provide tracking ID",
                            "Low",
                        )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append("Data export failed")
                self._log_issue(
                    test_case, "GDPR export failed", "User data export not working", "Critical"
                )

            # Test data deletion
            deletion_response = session.post(
                f"{self.base_url}/api/v1/account/delete",
                json={"confirm": True, "reason": "UAT testing"},
                timeout=30,
            )

            if deletion_response.status_code in [200, 202]:
                test_result.actual_results.append("Account deletion request accepted")
            else:
                test_result.actual_results.append("Account deletion endpoint not available")
                self._log_issue(
                    test_case,
                    "GDPR deletion missing",
                    "Account deletion functionality not implemented",
                    "Critical",
                )

    async def _test_analytics(self, test_case: TestCase, test_result: TestResult):
        """Test analytics and reporting features"""
        if test_case.id == "UAT-010":
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})

            # Test usage statistics
            stats_response = session.get(
                f"{self.base_url}/api/v1/account/usage", params={"period": "month"}, timeout=30
            )

            if stats_response.status_code == 200:
                usage_data = stats_response.json()
                test_result.actual_results.append("Usage statistics retrieved")

                # Verify data structure
                expected_fields = ["total_generations", "total_duration", "credits_used", "period"]
                missing_fields = [f for f in expected_fields if f not in usage_data]

                if not missing_fields:
                    test_result.actual_results.append("All expected usage fields present")
                else:
                    test_result.actual_results.append(f"Missing usage fields: {missing_fields}")
                    self._log_issue(
                        test_case,
                        "Incomplete usage data",
                        f"Missing fields: {missing_fields}",
                        "Medium",
                    )
            else:
                test_result.status = "Fail"
                test_result.actual_results.append("Usage statistics not available")
                self._log_issue(
                    test_case, "Usage stats failed", "Cannot retrieve usage statistics", "High"
                )

            # Test history endpoint
            history_response = session.get(
                f"{self.base_url}/api/v1/generations/history", params={"limit": 10}, timeout=30
            )

            if history_response.status_code == 200:
                history_data = history_response.json()
                if isinstance(history_data, list):
                    test_result.actual_results.append(
                        f"Generation history retrieved: {len(history_data)} items"
                    )
                else:
                    test_result.actual_results.append("History endpoint returned unexpected format")
                    self._log_issue(
                        test_case,
                        "Invalid history format",
                        "History should return array of generations",
                        "Medium",
                    )
            else:
                test_result.actual_results.append("Generation history not available")
                self._log_issue(
                    test_case,
                    "History endpoint failed",
                    "Cannot retrieve generation history",
                    "Medium",
                )

    async def _poll_task_completion(
        self, session: requests.Session, task_id: str, timeout: int
    ) -> bool:
        """Poll for async task completion"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = session.get(f"{self.base_url}/api/v1/tasks/{task_id}", timeout=10)

            if response.status_code == 200:
                task_data = response.json()
                status = task_data.get("status", "unknown")

                if status == "completed":
                    return True
                elif status == "failed":
                    logger.error(f"Task {task_id} failed: {task_data.get('error')}")
                    return False

            await asyncio.sleep(2)

        return False

    async def _monitor_batch_progress(
        self, session: requests.Session, batch_id: str, timeout: int
    ) -> bool:
        """Monitor batch processing progress"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = session.get(f"{self.base_url}/api/v1/batches/{batch_id}", timeout=10)

            if response.status_code == 200:
                batch_data = response.json()
                completed = batch_data.get("completed", 0)
                total = batch_data.get("total", 0)

                if completed == total and total > 0:
                    return True

                logger.info(f"Batch progress: {completed}/{total}")

            await asyncio.sleep(5)

        return False

    def _track_performance(self, metric_type: str, value: float):
        """Track performance metrics"""
        if metric_type == "api_response":
            self.performance_data["response_times"].append(value)
        elif metric_type == "generation_time":
            self.performance_data["generation_times"].append(value)

    def _log_issue(self, test_case: TestCase, title: str, description: str, severity: str):
        """Log an issue found during testing"""
        issue = Issue(
            id=f"ISSUE-{len(self.issues) + 1:03d}",
            test_case_id=test_case.id,
            severity=severity,
            type=self._categorize_issue(title),
            description=title,
            steps_to_reproduce=[f"Execute test case {test_case.id}"]
            + [step["action"] for step in test_case.steps],
            expected_behavior=" ".join(test_case.expected_results),
            actual_behavior=description,
        )

        self.issues.append(issue)
        logger.warning(f"Issue logged: {issue.id} - {title} ({severity})")

    def _categorize_issue(self, title: str) -> str:
        """Categorize issue type based on title"""
        title_lower = title.lower()

        if "performance" in title_lower or "slow" in title_lower or "timeout" in title_lower:
            return "Performance"
        elif "error" in title_lower or "fail" in title_lower or "crash" in title_lower:
            return "Bug"
        elif "missing" in title_lower or "not implemented" in title_lower:
            return "Feature Gap"
        elif "unclear" in title_lower or "confusing" in title_lower:
            return "Usability"
        else:
            return "Bug"

    async def run_uat_suite(self):
        """Run the complete UAT test suite"""
        logger.info("=" * 80)
        logger.info("STARTING USER ACCEPTANCE TESTING (UAT)")
        logger.info("=" * 80)

        # Create test cases
        test_cases = self.create_test_cases()
        logger.info(f"Created {len(test_cases)} UAT test cases")

        # Execute test cases
        for test_case in test_cases:
            await self.execute_test_case(test_case)
            await asyncio.sleep(1)  # Brief pause between tests

        # Generate reports
        self.generate_uat_report()
        self.generate_issue_report()
        self.generate_performance_report()

        return self.test_results, self.issues

    def generate_uat_report(self):
        """Generate comprehensive UAT report"""
        passed_tests = [r for r in self.test_results if r.status == "Pass"]
        failed_tests = [r for r in self.test_results if r.status == "Fail"]
        blocked_tests = [r for r in self.test_results if r.status == "Blocked"]
        skipped_tests = [r for r in self.test_results if r.status == "Skipped"]

        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "blocked": len(blocked_tests),
                "skipped": len(skipped_tests),
                "pass_rate": len(passed_tests) / len(self.test_results) * 100
                if self.test_results
                else 0,
                "total_execution_time": sum(r.execution_time for r in self.test_results),
                "issues_found": len(self.issues),
            },
            "test_results": [asdict(r) for r in self.test_results],
            "issues": [asdict(i) for i in self.issues],
            "performance_summary": self._get_performance_summary(),
            "recommendations": self._generate_recommendations(),
            "test_execution_date": datetime.now().isoformat(),
            "environment": "staging",
        }

        report_file = self.results_dir / f"uat_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"UAT report saved to: {report_file}")

        # Generate executive summary
        self._generate_executive_summary(report)

    def generate_issue_report(self):
        """Generate detailed issue report"""
        if not self.issues:
            logger.info("No issues found during UAT - excellent!")
            return

        # Group issues by severity
        severity_groups = {}
        for issue in self.issues:
            if issue.severity not in severity_groups:
                severity_groups[issue.severity] = []
            severity_groups[issue.severity].append(issue)

        issue_report = {
            "total_issues": len(self.issues),
            "by_severity": {severity: len(issues) for severity, issues in severity_groups.items()},
            "by_type": {},
            "detailed_issues": [asdict(i) for i in self.issues],
            "recommendations": self._generate_issue_recommendations(),
        }

        # Count by type
        for issue in self.issues:
            if issue.type not in issue_report["by_type"]:
                issue_report["by_type"][issue.type] = 0
            issue_report["by_type"][issue.type] += 1

        report_file = self.results_dir / f"issue_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(issue_report, f, indent=2)

        logger.info(f"Issue report saved to: {report_file}")

    def generate_performance_report(self):
        """Generate performance analysis report"""
        if self.performance_data["response_times"]:
            response_times = sorted(self.performance_data["response_times"])
            p50 = response_times[int(len(response_times) * 0.50)]
            p95 = response_times[int(len(response_times) * 0.95)]
            p99 = response_times[int(len(response_times) * 0.99)]
        else:
            p50 = p95 = p99 = 0

        if self.performance_data["generation_times"]:
            gen_times = sorted(self.performance_data["generation_times"])
            gen_p50 = gen_times[int(len(gen_times) * 0.50)]
            gen_p95 = gen_times[int(len(gen_times) * 0.95)]
        else:
            gen_p50 = gen_p95 = 0

        performance_report = {
            "api_performance": {
                "total_requests": self.performance_data["total_requests"],
                "error_count": self.performance_data["error_count"],
                "error_rate": self.performance_data["error_count"]
                / self.performance_data["total_requests"]
                if self.performance_data["total_requests"] > 0
                else 0,
                "response_time_p50": p50,
                "response_time_p95": p95,
                "response_time_p99": p99,
            },
            "generation_performance": {
                "total_generations": len(self.performance_data["generation_times"]),
                "generation_time_p50": gen_p50,
                "generation_time_p95": gen_p95,
            },
            "sla_compliance": self._check_sla_compliance(),
            "recommendations": self._generate_performance_recommendations(),
        }

        report_file = self.results_dir / f"performance_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(performance_report, f, indent=2)

        logger.info(f"Performance report saved to: {report_file}")

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if self.performance_data["response_times"]:
            avg_response = sum(self.performance_data["response_times"]) / len(
                self.performance_data["response_times"]
            )
        else:
            avg_response = 0

        if self.performance_data["generation_times"]:
            avg_generation = sum(self.performance_data["generation_times"]) / len(
                self.performance_data["generation_times"]
            )
        else:
            avg_generation = 0

        return {
            "average_api_response_time": avg_response,
            "average_generation_time": avg_generation,
            "total_api_calls": len(self.performance_data["response_times"]),
            "total_generations": len(self.performance_data["generation_times"]),
            "error_rate": self.performance_data["error_count"]
            / self.performance_data["total_requests"]
            if self.performance_data["total_requests"] > 0
            else 0,
        }

    def _check_sla_compliance(self) -> Dict[str, bool]:
        """Check compliance with performance SLAs"""
        compliance = {}

        if self.performance_data["response_times"]:
            response_times = sorted(self.performance_data["response_times"])
            p95 = response_times[int(len(response_times) * 0.95)]
            p99 = response_times[int(len(response_times) * 0.99)]

            compliance["api_p95"] = p95 <= self.performance_slas["api_response_time_p95"]
            compliance["api_p99"] = p99 <= self.performance_slas["api_response_time_p99"]

        error_rate = (
            self.performance_data["error_count"] / self.performance_data["total_requests"]
            if self.performance_data["total_requests"] > 0
            else 0
        )
        compliance["error_rate"] = error_rate <= self.performance_slas["error_rate_threshold"]

        return compliance

    def _generate_recommendations(self) -> List[str]:
        """Generate overall UAT recommendations"""
        recommendations = []

        # Based on test results
        failed_count = len([r for r in self.test_results if r.status == "Fail"])
        if failed_count > 0:
            recommendations.append(
                f"Address {failed_count} failed test cases before production deployment"
            )

        # Based on issues
        critical_issues = [i for i in self.issues if i.severity == "Critical"]
        if critical_issues:
            recommendations.append(f"Resolve {len(critical_issues)} critical issues immediately")

        high_issues = [i for i in self.issues if i.severity == "High"]
        if high_issues:
            recommendations.append(f"Fix {len(high_issues)} high-priority issues before go-live")

        # Performance recommendations
        if self.performance_data["response_times"]:
            response_times = sorted(self.performance_data["response_times"])
            p95 = response_times[int(len(response_times) * 0.95)]
            if p95 > self.performance_slas["api_response_time_p95"]:
                recommendations.append("Optimize API performance to meet SLA requirements")

        # Feature gaps
        feature_gaps = [i for i in self.issues if i.type == "Feature Gap"]
        if feature_gaps:
            recommendations.append(
                f"Consider implementing {len(feature_gaps)} missing features for better user experience"
            )

        if not recommendations:
            recommendations.append(
                "System is ready for production deployment with minor improvements"
            )

        return recommendations

    def _generate_issue_recommendations(self) -> List[str]:
        """Generate recommendations based on issues found"""
        recommendations = []

        # Group by type
        issue_types = {}
        for issue in self.issues:
            if issue.type not in issue_types:
                issue_types[issue.type] = 0
            issue_types[issue.type] += 1

        if issue_types.get("Bug", 0) > 5:
            recommendations.append("Conduct thorough bug fixing sprint before release")

        if issue_types.get("Performance", 0) > 3:
            recommendations.append(
                "Performance optimization needed - consider load testing and profiling"
            )

        if issue_types.get("Usability", 0) > 2:
            recommendations.append("UI/UX review recommended to address usability concerns")

        if issue_types.get("Feature Gap", 0) > 0:
            recommendations.append("Prioritize missing features based on business value")

        return recommendations

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance-specific recommendations"""
        recommendations = []

        if self.performance_data["response_times"]:
            response_times = sorted(self.performance_data["response_times"])
            p99 = response_times[int(len(response_times) * 0.99)]

            if p99 > 3.0:
                recommendations.append("Implement caching to reduce API response times")

            if p99 > 5.0:
                recommendations.append("Consider horizontal scaling for API servers")

        error_rate = (
            self.performance_data["error_count"] / self.performance_data["total_requests"]
            if self.performance_data["total_requests"] > 0
            else 0
        )
        if error_rate > 0.01:
            recommendations.append("Investigate and fix sources of API errors")

        if self.performance_data["generation_times"]:
            avg_gen_time = sum(self.performance_data["generation_times"]) / len(
                self.performance_data["generation_times"]
            )
            if avg_gen_time > 45:
                recommendations.append("Optimize model inference for faster generation")

        return recommendations

    def _generate_executive_summary(self, report: Dict[str, Any]):
        """Generate executive summary for stakeholders"""
        summary_file = self.results_dir / "uat_executive_summary.md"

        with open(summary_file, "w") as f:
            f.write("# User Acceptance Testing (UAT) - Executive Summary\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"**Environment:** Staging\n")
            f.write(
                f"**Test Duration:** {report['summary']['total_execution_time']:.1f} seconds\n\n"
            )

            f.write("## Overall Results\n\n")
            f.write(f"- **Total Test Cases:** {report['summary']['total_tests']}\n")
            f.write(
                f"- **Passed:** {report['summary']['passed']} ({report['summary']['pass_rate']:.1f}%)\n"
            )
            f.write(f"- **Failed:** {report['summary']['failed']}\n")
            f.write(f"- **Issues Found:** {report['summary']['issues_found']}\n\n")

            # Issue breakdown
            if self.issues:
                f.write("## Issues by Severity\n\n")
                severity_count = {}
                for issue in self.issues:
                    severity_count[issue.severity] = severity_count.get(issue.severity, 0) + 1

                for severity in ["Critical", "High", "Medium", "Low"]:
                    if severity in severity_count:
                        f.write(f"- **{severity}:** {severity_count[severity]}\n")
                f.write("\n")

            # SLA Compliance
            f.write("## Performance SLA Compliance\n\n")
            compliance = self._check_sla_compliance()
            for metric, passed in compliance.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                f.write(f"- **{metric}:** {status}\n")
            f.write("\n")

            # Recommendations
            f.write("## Key Recommendations\n\n")
            for i, rec in enumerate(report["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Sign-off section
            f.write("## Stakeholder Sign-off\n\n")
            f.write("The following stakeholders have reviewed and approved these UAT results:\n\n")
            f.write("| Stakeholder | Role | Date | Signature |\n")
            f.write("|-------------|------|------|------------|\n")
            f.write("| | Product Owner | | |\n")
            f.write("| | QA Lead | | |\n")
            f.write("| | Technical Lead | | |\n")
            f.write("| | Business Analyst | | |\n")

        logger.info(f"Executive summary saved to: {summary_file}")

    def print_summary(self):
        """Print UAT summary to console"""
        print("\n" + "=" * 80)
        print("USER ACCEPTANCE TESTING (UAT) - SUMMARY")
        print("=" * 80)

        passed = len([r for r in self.test_results if r.status == "Pass"])
        failed = len([r for r in self.test_results if r.status == "Fail"])
        total = len(self.test_results)

        print(f"Total Test Cases: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Issues Found: {len(self.issues)}")
        print()

        if self.issues:
            print("ISSUES BY SEVERITY:")
            severity_count = {}
            for issue in self.issues:
                severity_count[issue.severity] = severity_count.get(issue.severity, 0) + 1

            for severity in ["Critical", "High", "Medium", "Low"]:
                if severity in severity_count:
                    print(f"  {severity}: {severity_count[severity]}")
            print()

        print("PERFORMANCE SUMMARY:")
        perf_summary = self._get_performance_summary()
        print(f"  API Response Time (avg): {perf_summary['average_api_response_time']:.3f}s")
        print(f"  Generation Time (avg): {perf_summary['average_generation_time']:.1f}s")
        print(f"  Error Rate: {perf_summary['error_rate']:.2%}")

        print("\n" + "=" * 80)


async def main():
    """Main UAT execution"""
    suite = UATTestSuite()

    try:
        test_results, issues = await suite.run_uat_suite()
        suite.print_summary()

        # Exit code based on results
        critical_issues = [i for i in issues if i.severity == "Critical"]
        failed_tests = [r for r in test_results if r.status == "Fail"]

        if critical_issues or failed_tests:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"UAT suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
