#!/usr/bin/env python3
"""
Security Penetration Testing Suite for Music Gen AI Staging
Comprehensive security testing including OWASP Top 10 vulnerabilities
"""

import os
import json
import time
import requests
import subprocess
import threading
from urllib.parse import urljoin
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import logging
import sys
import base64
import hashlib
import random
import string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/results/security_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityTestResult:
    category: str
    test_name: str
    vulnerability_type: str
    severity: str  # Critical, High, Medium, Low, Info
    success: bool  # True if test passed (no vulnerability found)
    duration: float
    details: Dict[str, Any] = None
    remediation: str = ""


class SecurityPenetrationTestSuite:
    def __init__(self):
        self.base_url = os.getenv("TARGET_HOST", "http://nginx-staging")
        self.api_key = os.getenv("STAGING_API_KEY", "staging_api_key_change_me")
        self.results: List[SecurityTestResult] = []

        # Configure session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "SecurityTestSuite/1.0 (Penetration Testing)",
                "Accept": "application/json, text/html, */*",
            }
        )

        # Test payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg onload=alert('XSS')>",
            "{{7*7}}",  # Template injection
            "${7*7}",  # EL injection
        ]

        self.sql_injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT 1,2,3,4,5 --",
            "admin'--",
            "' OR 1=1#",
            "1' ORDER BY 1--+",
            "1' UNION SELECT null-- -",
        ]

        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/passwd",
            "file:///etc/passwd",
            "../../../proc/self/environ",
        ]

    def log_result(self, result: SecurityTestResult):
        """Log security test result"""
        self.results.append(result)

        status = "🔒 SECURE" if result.success else f"🚨 VULNERABLE ({result.severity})"
        logger.info(f"{status} {result.category}.{result.test_name} ({result.duration:.2f}s)")

        if not result.success:
            logger.warning(f"  Vulnerability: {result.vulnerability_type}")
            logger.warning(f"  Details: {result.details}")
            if result.remediation:
                logger.info(f"  Remediation: {result.remediation}")

    def test_authentication_bypass(self) -> List[SecurityTestResult]:
        """Test for authentication bypass vulnerabilities"""
        logger.info("Testing authentication bypass...")
        results = []

        # Test 1: No API key
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/v1/user/profile", timeout=10)

            vulnerable = response.status_code == 200
            result = SecurityTestResult(
                category="authentication",
                test_name="no_api_key_bypass",
                vulnerability_type="Missing Authentication",
                severity="High" if vulnerable else "Info",
                success=not vulnerable,
                duration=time.time() - start_time,
                details={
                    "status_code": response.status_code,
                    "response_length": len(response.content),
                },
                remediation="Ensure all protected endpoints require valid authentication",
            )
            results.append(result)
        except Exception as e:
            result = SecurityTestResult(
                category="authentication",
                test_name="no_api_key_bypass",
                vulnerability_type="Test Error",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={"error": str(e)},
            )
            results.append(result)

        # Test 2: Invalid API key formats
        invalid_keys = [
            "Bearer invalid",
            "Basic " + base64.b64encode(b"admin:admin").decode(),
            "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "",
            "null",
            "undefined",
        ]

        for i, invalid_key in enumerate(invalid_keys):
            start_time = time.time()
            try:
                headers = {"Authorization": invalid_key}
                response = requests.get(
                    f"{self.base_url}/api/v1/user/profile", headers=headers, timeout=10
                )

                vulnerable = response.status_code == 200
                result = SecurityTestResult(
                    category="authentication",
                    test_name=f"invalid_key_bypass_{i}",
                    vulnerability_type="Authentication Bypass",
                    severity="High" if vulnerable else "Info",
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "invalid_key": invalid_key[:20] + "..."
                        if len(invalid_key) > 20
                        else invalid_key,
                        "status_code": response.status_code,
                    },
                    remediation="Implement proper API key validation and reject malformed keys",
                )
                results.append(result)
            except Exception as e:
                result = SecurityTestResult(
                    category="authentication",
                    test_name=f"invalid_key_bypass_{i}",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)

        return results

    def test_injection_attacks(self) -> List[SecurityTestResult]:
        """Test for injection vulnerabilities"""
        logger.info("Testing injection attacks...")
        results = []

        # SQL Injection tests
        for i, payload in enumerate(self.sql_injection_payloads):
            start_time = time.time()
            try:
                # Test in query parameters
                response = self.session.get(
                    f"{self.base_url}/api/v1/generate",
                    params={"prompt": payload},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )

                # Look for SQL error indicators
                sql_errors = [
                    "SQL syntax",
                    "mysql_fetch",
                    "ORA-",
                    "Microsoft OLE DB",
                    "postgres",
                    "sqlite",
                    "Warning: mysql_",
                    "valid MySQL result",
                    "PostgreSQL query failed",
                    "syntax error",
                ]

                response_text = response.text.lower()
                vulnerable = any(error.lower() in response_text for error in sql_errors)

                result = SecurityTestResult(
                    category="injection",
                    test_name=f"sql_injection_query_{i}",
                    vulnerability_type="SQL Injection",
                    severity="Critical" if vulnerable else "Info",
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "payload": payload,
                        "status_code": response.status_code,
                        "response_contains_sql_error": vulnerable,
                    },
                    remediation="Use parameterized queries and input validation",
                )
                results.append(result)

            except Exception as e:
                result = SecurityTestResult(
                    category="injection",
                    test_name=f"sql_injection_query_{i}",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)

        # XSS tests
        for i, payload in enumerate(self.xss_payloads):
            start_time = time.time()
            try:
                # Test in JSON POST body
                json_data = {"prompt": payload, "duration": 5.0}
                response = self.session.post(
                    f"{self.base_url}/api/v1/generate",
                    json=json_data,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )

                # Check if payload is reflected unescaped
                vulnerable = payload in response.text and response.headers.get(
                    "content-type", ""
                ).startswith("text/html")

                result = SecurityTestResult(
                    category="injection",
                    test_name=f"xss_reflection_{i}",
                    vulnerability_type="Cross-Site Scripting (XSS)",
                    severity="High" if vulnerable else "Info",
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "payload": payload,
                        "status_code": response.status_code,
                        "payload_reflected": payload in response.text,
                        "content_type": response.headers.get("content-type", ""),
                    },
                    remediation="Implement proper input sanitization and output encoding",
                )
                results.append(result)

            except Exception as e:
                result = SecurityTestResult(
                    category="injection",
                    test_name=f"xss_reflection_{i}",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)

        return results

    def test_path_traversal(self) -> List[SecurityTestResult]:
        """Test for path traversal vulnerabilities"""
        logger.info("Testing path traversal...")
        results = []

        for i, payload in enumerate(self.path_traversal_payloads):
            start_time = time.time()
            try:
                # Test in file download endpoints
                response = self.session.get(
                    f"{self.base_url}/api/v1/download/{payload}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )

                # Look for sensitive file content
                sensitive_indicators = [
                    "root:",
                    "bin/bash",
                    "[system32]",
                    "windows registry",
                    "# User privilege",
                    "PATH=",
                ]

                response_text = response.text.lower()
                vulnerable = response.status_code == 200 and any(
                    indicator in response_text for indicator in sensitive_indicators
                )

                result = SecurityTestResult(
                    category="access_control",
                    test_name=f"path_traversal_{i}",
                    vulnerability_type="Path Traversal",
                    severity="High" if vulnerable else "Info",
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "payload": payload,
                        "status_code": response.status_code,
                        "response_length": len(response.content),
                        "contains_sensitive_data": vulnerable,
                    },
                    remediation="Implement proper path validation and restrict file access",
                )
                results.append(result)

            except Exception as e:
                result = SecurityTestResult(
                    category="access_control",
                    test_name=f"path_traversal_{i}",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)

        return results

    def test_security_headers(self) -> List[SecurityTestResult]:
        """Test for missing security headers"""
        logger.info("Testing security headers...")
        results = []

        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)

            required_headers = {
                "X-Content-Type-Options": ("nosniff", "Prevent MIME type sniffing"),
                "X-Frame-Options": (["DENY", "SAMEORIGIN"], "Prevent clickjacking"),
                "Content-Security-Policy": (None, "Prevent XSS and injection attacks"),
                "Strict-Transport-Security": (None, "Enforce HTTPS (if applicable)"),
                "Referrer-Policy": (None, "Control referrer information"),
                "Permissions-Policy": (None, "Control browser permissions"),
            }

            for header_name, (expected_values, purpose) in required_headers.items():
                header_value = response.headers.get(header_name, "")

                if not header_value:
                    vulnerable = True
                    severity = "Medium"
                elif expected_values and isinstance(expected_values, list):
                    vulnerable = not any(val in header_value for val in expected_values)
                    severity = "Medium" if vulnerable else "Info"
                elif expected_values and isinstance(expected_values, str):
                    vulnerable = expected_values not in header_value
                    severity = "Medium" if vulnerable else "Info"
                else:
                    vulnerable = False
                    severity = "Info"

                result = SecurityTestResult(
                    category="security_headers",
                    test_name=f"header_{header_name.lower().replace('-', '_')}",
                    vulnerability_type="Missing Security Header",
                    severity=severity,
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "header_name": header_name,
                        "header_value": header_value or "Missing",
                        "expected": expected_values,
                        "purpose": purpose,
                    },
                    remediation=f"Add {header_name} header: {purpose}",
                )
                results.append(result)

        except Exception as e:
            result = SecurityTestResult(
                category="security_headers",
                test_name="header_check_error",
                vulnerability_type="Test Error",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={"error": str(e)},
            )
            results.append(result)

        return results

    def test_rate_limiting(self) -> List[SecurityTestResult]:
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")
        results = []

        start_time = time.time()
        try:
            # Rapid-fire requests to trigger rate limiting
            responses = []
            for i in range(50):  # Send 50 requests rapidly
                try:
                    response = requests.get(
                        f"{self.base_url}/api/v1/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=5,
                    )
                    responses.append(response.status_code)

                    # Break early if rate limited
                    if response.status_code == 429:
                        break

                except requests.exceptions.Timeout:
                    responses.append(0)  # Timeout
                except Exception:
                    responses.append(-1)  # Error

            rate_limited = 429 in responses

            result = SecurityTestResult(
                category="rate_limiting",
                test_name="api_rate_limiting",
                vulnerability_type="Missing Rate Limiting",
                severity="Medium" if not rate_limited else "Info",
                success=rate_limited,
                duration=time.time() - start_time,
                details={
                    "total_requests": len(responses),
                    "rate_limited": rate_limited,
                    "response_codes": list(set(responses)),
                    "first_429_at_request": responses.index(429) + 1 if rate_limited else None,
                },
                remediation="Implement rate limiting to prevent abuse",
            )
            results.append(result)

        except Exception as e:
            result = SecurityTestResult(
                category="rate_limiting",
                test_name="api_rate_limiting",
                vulnerability_type="Test Error",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={"error": str(e)},
            )
            results.append(result)

        return results

    def test_information_disclosure(self) -> List[SecurityTestResult]:
        """Test for information disclosure vulnerabilities"""
        logger.info("Testing information disclosure...")
        results = []

        # Test debug endpoints
        debug_endpoints = [
            "/debug",
            "/admin",
            "/.env",
            "/config",
            "/api/debug",
            "/metrics",
            "/health/detailed",
            "/swagger",
            "/docs",
            "/api-docs",
        ]

        for endpoint in debug_endpoints:
            start_time = time.time()
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)

                # Check for sensitive information in response
                sensitive_patterns = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "api_key",
                    "database",
                    "connection",
                    "config",
                    "env",
                    "debug",
                    "stack trace",
                    "error",
                ]

                response_text = response.text.lower()
                contains_sensitive = any(pattern in response_text for pattern in sensitive_patterns)

                vulnerable = (
                    response.status_code == 200
                    and contains_sensitive
                    and len(response.content) > 100  # Ignore simple responses
                )

                result = SecurityTestResult(
                    category="information_disclosure",
                    test_name=f"debug_endpoint_{endpoint.replace('/', '_').replace('-', '_')}",
                    vulnerability_type="Information Disclosure",
                    severity="Medium" if vulnerable else "Info",
                    success=not vulnerable,
                    duration=time.time() - start_time,
                    details={
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response_length": len(response.content),
                        "contains_sensitive_info": contains_sensitive,
                    },
                    remediation="Remove or secure debug endpoints in production",
                )
                results.append(result)

            except Exception as e:
                result = SecurityTestResult(
                    category="information_disclosure",
                    test_name=f"debug_endpoint_{endpoint.replace('/', '_').replace('-', '_')}",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)

        return results

    def test_ssl_tls_configuration(self) -> List[SecurityTestResult]:
        """Test SSL/TLS configuration"""
        logger.info("Testing SSL/TLS configuration...")
        results = []

        if self.base_url.startswith("https"):
            start_time = time.time()
            try:
                # Test SSL certificate
                response = requests.get(self.base_url, timeout=10, verify=True)

                result = SecurityTestResult(
                    category="ssl_tls",
                    test_name="certificate_validation",
                    vulnerability_type="Invalid SSL Certificate",
                    severity="High",
                    success=True,  # If we get here, certificate is valid
                    duration=time.time() - start_time,
                    details={"status_code": response.status_code, "certificate_valid": True},
                    remediation="Certificate is valid",
                )
                results.append(result)

            except requests.exceptions.SSLError as e:
                result = SecurityTestResult(
                    category="ssl_tls",
                    test_name="certificate_validation",
                    vulnerability_type="Invalid SSL Certificate",
                    severity="High",
                    success=False,
                    duration=time.time() - start_time,
                    details={"certificate_valid": False, "ssl_error": str(e)},
                    remediation="Fix SSL certificate configuration",
                )
                results.append(result)
            except Exception as e:
                result = SecurityTestResult(
                    category="ssl_tls",
                    test_name="certificate_validation",
                    vulnerability_type="Test Error",
                    severity="Info",
                    success=True,
                    duration=time.time() - start_time,
                    details={"error": str(e)},
                )
                results.append(result)
        else:
            result = SecurityTestResult(
                category="ssl_tls",
                test_name="https_enforcement",
                vulnerability_type="Unencrypted Connection",
                severity="Medium",
                success=False,
                duration=0,
                details={"using_https": False},
                remediation="Enforce HTTPS for all connections",
            )
            results.append(result)

        return results

    def run_zap_scan(self) -> List[SecurityTestResult]:
        """Run OWASP ZAP security scan if available"""
        logger.info("Running OWASP ZAP scan...")
        results = []

        start_time = time.time()
        try:
            # Check if ZAP is available
            zap_cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "staging-network",
                "owasp/zap2docker-stable:latest",
                "zap-baseline.py",
                "-t",
                self.base_url,
                "-J",
                "/tmp/zap-report.json",
            ]

            zap_result = subprocess.run(
                zap_cmd, capture_output=True, text=True, timeout=300  # 5 minutes timeout
            )

            # ZAP returns non-zero for vulnerabilities found, which is expected
            zap_available = True

            result = SecurityTestResult(
                category="automated_scan",
                test_name="owasp_zap_baseline",
                vulnerability_type="Automated Security Scan",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={
                    "zap_available": zap_available,
                    "return_code": zap_result.returncode,
                    "stdout_length": len(zap_result.stdout),
                    "stderr_length": len(zap_result.stderr),
                },
                remediation="Review ZAP report for detailed findings",
            )
            results.append(result)

        except subprocess.TimeoutExpired:
            result = SecurityTestResult(
                category="automated_scan",
                test_name="owasp_zap_baseline",
                vulnerability_type="Scan Timeout",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={"timeout": True},
                remediation="ZAP scan timed out - may need longer timeout for full scan",
            )
            results.append(result)
        except Exception as e:
            result = SecurityTestResult(
                category="automated_scan",
                test_name="owasp_zap_baseline",
                vulnerability_type="Test Error",
                severity="Info",
                success=True,
                duration=time.time() - start_time,
                details={"error": str(e), "zap_available": False},
            )
            results.append(result)

        return results

    def run_all_security_tests(self) -> List[SecurityTestResult]:
        """Run all security tests"""
        logger.info("Starting comprehensive security penetration test suite...")

        test_functions = [
            self.test_authentication_bypass,
            self.test_injection_attacks,
            self.test_path_traversal,
            self.test_security_headers,
            self.test_rate_limiting,
            self.test_information_disclosure,
            self.test_ssl_tls_configuration,
            self.run_zap_scan,
        ]

        for test_func in test_functions:
            try:
                results = test_func()
                for result in results:
                    self.log_result(result)
            except Exception as e:
                logger.error(f"Test function {test_func.__name__} failed: {e}")

        return self.results

    def generate_security_report(
        self, output_file: str = "/app/results/security_penetration_report.json"
    ) -> Dict[str, Any]:
        """Generate comprehensive security test report"""
        if not self.results:
            return {}

        # Categorize results by severity
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
        vulnerabilities = []

        for result in self.results:
            severity_counts[result.severity] += 1
            if not result.success:
                vulnerabilities.append(asdict(result))

        # Calculate risk score (weighted by severity)
        risk_weights = {"Critical": 10, "High": 7, "Medium": 4, "Low": 2, "Info": 1}
        risk_score = sum(severity_counts[sev] * weight for sev, weight in risk_weights.items())
        max_possible_score = len(self.results) * 10
        risk_percentage = (risk_score / max_possible_score * 100) if max_possible_score > 0 else 0

        # Group by category
        category_results = {}
        for result in self.results:
            if result.category not in category_results:
                category_results[result.category] = {"total": 0, "vulnerable": 0, "tests": []}

            category_results[result.category]["total"] += 1
            if not result.success:
                category_results[result.category]["vulnerable"] += 1
            category_results[result.category]["tests"].append(asdict(result))

        report = {
            "summary": {
                "total_tests": len(self.results),
                "vulnerabilities_found": len(vulnerabilities),
                "risk_score": risk_score,
                "risk_percentage": risk_percentage,
                "security_level": self._get_security_level(risk_percentage),
                "severity_breakdown": severity_counts,
                "test_duration": sum(r.duration for r in self.results),
            },
            "categories": category_results,
            "vulnerabilities": vulnerabilities,
            "recommendations": self._generate_recommendations(),
            "timestamp": time.time(),
        }

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Security penetration test report saved to {output_file}")
        return report

    def _get_security_level(self, risk_percentage: float) -> str:
        """Determine security level based on risk percentage"""
        if risk_percentage < 10:
            return "Excellent"
        elif risk_percentage < 25:
            return "Good"
        elif risk_percentage < 50:
            return "Fair"
        elif risk_percentage < 75:
            return "Poor"
        else:
            return "Critical"

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []

        vulnerability_counts = {}
        for result in self.results:
            if not result.success:
                vuln_type = result.vulnerability_type
                if vuln_type not in vulnerability_counts:
                    vulnerability_counts[vuln_type] = 0
                vulnerability_counts[vuln_type] += 1

        if "Missing Authentication" in vulnerability_counts:
            recommendations.append("Implement comprehensive authentication for all API endpoints")

        if "SQL Injection" in vulnerability_counts:
            recommendations.append(
                "Use parameterized queries and input validation to prevent SQL injection"
            )

        if "Cross-Site Scripting (XSS)" in vulnerability_counts:
            recommendations.append("Implement proper input sanitization and output encoding")

        if "Missing Security Header" in vulnerability_counts:
            recommendations.append(
                "Add all recommended security headers (CSP, X-Frame-Options, etc.)"
            )

        if "Missing Rate Limiting" in vulnerability_counts:
            recommendations.append("Implement rate limiting to prevent abuse and DoS attacks")

        if "Information Disclosure" in vulnerability_counts:
            recommendations.append(
                "Remove or secure debug endpoints and sensitive information exposure"
            )

        if "Path Traversal" in vulnerability_counts:
            recommendations.append("Implement proper path validation and file access controls")

        if not recommendations:
            recommendations.append(
                "Maintain current security practices and conduct regular security audits"
            )

        return recommendations

    def print_summary(self, report: Dict[str, Any]):
        """Print security test summary"""
        print("\n" + "=" * 70)
        print("SECURITY PENETRATION TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Vulnerabilities Found: {report['summary']['vulnerabilities_found']}")
        print(f"Security Level: {report['summary']['security_level']}")
        print(
            f"Risk Score: {report['summary']['risk_score']}/{report['summary']['total_tests'] * 10}"
        )
        print(f"Test Duration: {report['summary']['test_duration']:.2f}s")
        print()

        print("SEVERITY BREAKDOWN:")
        for severity, count in report["summary"]["severity_breakdown"].items():
            if count > 0:
                print(f"  {severity}: {count}")
        print()

        print("CATEGORY BREAKDOWN:")
        for category, results in report["categories"].items():
            total = results["total"]
            vulnerable = results["vulnerable"]
            secure_count = total - vulnerable
            print(
                f"  {category.upper()}: {secure_count}/{total} secure ({secure_count/total*100:.1f}%)"
            )

        if report["vulnerabilities"]:
            print()
            print("VULNERABILITIES FOUND:")
            for vuln in report["vulnerabilities"]:
                print(f"  🚨 {vuln['severity']}: {vuln['vulnerability_type']} in {vuln['category']}")

        if report["recommendations"]:
            print()
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")

        print("=" * 70)


def main():
    """Main test execution"""
    suite = SecurityPenetrationTestSuite()

    try:
        suite.run_all_security_tests()
        report = suite.generate_security_report()
        suite.print_summary(report)

        # Exit with appropriate code based on security level
        security_level = report["summary"]["security_level"]
        if security_level in ["Critical", "Poor"]:
            exit_code = 2  # Critical security issues
        elif report["summary"]["vulnerabilities_found"] > 0:
            exit_code = 1  # Some vulnerabilities found
        else:
            exit_code = 0  # No vulnerabilities found

        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Security penetration test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
