#!/usr/bin/env python3
"""
Comprehensive OWASP Security Audit for Music Gen AI
Performs automated security scanning and vulnerability assessment
"""
import subprocess
import json
import os
import sys
import requests
import time
from typing import Dict, List, Any
from pathlib import Path
import hashlib
import re
import ast


class SecurityAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        self.fixed_issues = []

    def run_comprehensive_audit(self):
        """Run complete OWASP security audit"""
        print("🔍 Starting Comprehensive OWASP Security Audit")
        print("=" * 60)

        # 1. OWASP Dependency Check
        self.owasp_dependency_check()

        # 2. Static Code Analysis
        self.static_code_analysis()

        # 3. Authentication Path Review
        self.review_authentication_paths()

        # 4. Information Disclosure Check
        self.check_information_disclosure()

        # 5. Input Validation Review
        self.verify_input_validation()

        # 6. Rate Limiting Test
        self.test_rate_limiting()

        # 7. Security Headers Review
        self.review_security_headers()

        # 8. Additional Security Checks
        self.additional_security_checks()

        # 9. Generate Report
        self.generate_audit_report()

        return self.findings

    def owasp_dependency_check(self):
        """Run OWASP Dependency Check for known vulnerabilities"""
        print("🔎 Running OWASP Dependency Check...")

        try:
            # Check Python dependencies with safety
            result = subprocess.run(
                ["python", "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            packages = json.loads(result.stdout)

            # Check for known vulnerable packages
            vulnerable_packages = self.check_vulnerable_packages(packages)

            for vuln in vulnerable_packages:
                self.findings["high"].append(
                    {
                        "type": "vulnerable_dependency",
                        "package": vuln["name"],
                        "version": vuln["version"],
                        "vulnerability": vuln["vulnerability"],
                        "severity": "HIGH",
                        "fix": f"Update {vuln['name']} to version {vuln['safe_version']} or later",
                    }
                )

            # Check requirements files for pinned versions
            self.check_requirements_security()

        except Exception as e:
            self.findings["medium"].append(
                {
                    "type": "dependency_check_error",
                    "error": str(e),
                    "recommendation": "Install safety: pip install safety",
                }
            )

    def check_vulnerable_packages(self, packages: List[Dict]) -> List[Dict]:
        """Check packages against known vulnerability database"""
        vulnerable = []

        # Known vulnerability patterns (simplified - in production use safety/PyUp)
        known_vulns = {
            "requests": {"vulnerable_versions": ["< 2.20.0"], "cve": "CVE-2018-18074"},
            "urllib3": {"vulnerable_versions": ["< 1.24.2"], "cve": "CVE-2019-11324"},
            "jinja2": {"vulnerable_versions": ["< 2.11.3"], "cve": "CVE-2020-28493"},
            "flask": {"vulnerable_versions": ["< 1.0"], "cve": "CVE-2018-1000656"},
            "django": {"vulnerable_versions": ["< 2.2.13"], "cve": "CVE-2020-13254"},
            "pillow": {"vulnerable_versions": ["< 8.3.2"], "cve": "CVE-2021-34552"},
            "pycryptodome": {"vulnerable_versions": ["< 3.9.8"], "cve": "CVE-2018-15560"},
        }

        for package in packages:
            name = package["name"].lower()
            version = package["version"]

            if name in known_vulns:
                # Simplified version check - in production use proper version parsing
                if any(
                    self.version_matches_pattern(version, pattern)
                    for pattern in known_vulns[name]["vulnerable_versions"]
                ):
                    vulnerable.append(
                        {
                            "name": name,
                            "version": version,
                            "vulnerability": known_vulns[name]["cve"],
                            "safe_version": "latest",
                        }
                    )

        return vulnerable

    def version_matches_pattern(self, version: str, pattern: str) -> bool:
        """Simple version pattern matching"""
        # Simplified - in production use packaging.version
        if pattern.startswith("< "):
            threshold = pattern[2:]
            return version < threshold
        return False

    def check_requirements_security(self):
        """Check requirements files for security issues"""
        req_files = ["requirements.txt", "requirements-prod.txt", "requirements-dev.txt"]

        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                with open(req_path, "r") as f:
                    content = f.read()

                # Check for unpinned versions
                unpinned = re.findall(
                    r"^([a-zA-Z0-9_-]+)(?!.*==)(?!.*>=)(?!.*<=)(?!.*>)(?!.*<)(?!.*~=)",
                    content,
                    re.MULTILINE,
                )

                for package in unpinned:
                    if package and not package.startswith("#"):
                        self.findings["medium"].append(
                            {
                                "type": "unpinned_dependency",
                                "file": str(req_file),
                                "package": package,
                                "recommendation": f"Pin {package} to specific version for security",
                            }
                        )

    def static_code_analysis(self):
        """Run static code analysis with bandit for Python security issues"""
        print("🔍 Running Static Code Analysis...")

        try:
            # Run bandit security scanner
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "bandit",
                    "-r",
                    str(self.project_root / "music_gen"),
                    "-f",
                    "json",
                    "--skip",
                    "B101",  # Skip assert statements
                ],
                capture_output=True,
                text=True,
            )

            if result.stdout:
                bandit_results = json.loads(result.stdout)
                self.process_bandit_results(bandit_results)

        except FileNotFoundError:
            self.findings["info"].append(
                {
                    "type": "missing_tool",
                    "tool": "bandit",
                    "recommendation": "Install bandit: pip install bandit",
                }
            )
        except Exception as e:
            self.findings["medium"].append({"type": "static_analysis_error", "error": str(e)})

        # Manual code pattern analysis
        self.analyze_code_patterns()

    def process_bandit_results(self, results: Dict):
        """Process bandit security scan results"""
        if "results" not in results:
            return

        for issue in results["results"]:
            severity_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}

            severity = severity_map.get(issue.get("issue_severity", "LOW"), "low")

            self.findings[severity].append(
                {
                    "type": "static_analysis",
                    "test_name": issue.get("test_name", ""),
                    "test_id": issue.get("test_id", ""),
                    "file": issue.get("filename", ""),
                    "line": issue.get("line_number", 0),
                    "issue": issue.get("issue_text", ""),
                    "confidence": issue.get("issue_confidence", ""),
                    "severity": issue.get("issue_severity", ""),
                    "code": issue.get("code", ""),
                }
            )

    def analyze_code_patterns(self):
        """Analyze code patterns for security issues"""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for hardcoded secrets
                self.check_hardcoded_secrets(file_path, content)

                # Check for SQL injection patterns
                self.check_sql_injection(file_path, content)

                # Check for XSS vulnerabilities
                self.check_xss_vulnerabilities(file_path, content)

                # Check for insecure random usage
                self.check_insecure_random(file_path, content)

                # Check for debug mode in production
                self.check_debug_mode(file_path, content)

            except Exception as e:
                continue  # Skip files that can't be read

    def check_hardcoded_secrets(self, file_path: Path, content: str):
        """Check for hardcoded secrets and credentials"""
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "hardcoded_password"),
            (r'secret\s*=\s*["\'][^"\']{16,}["\']', "hardcoded_secret"),
            (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', "hardcoded_api_key"),
            (r'token\s*=\s*["\'][^"\']{16,}["\']', "hardcoded_token"),
            (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', "base64_secret"),
            (r"sk_live_[a-zA-Z0-9]{48,}", "stripe_secret_key"),
            (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
            (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "uuid_secret"),
        ]

        for pattern, secret_type in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                # Skip test files and documentation
                if any(x in str(file_path).lower() for x in ["test", "example", "demo", "doc"]):
                    continue

                self.findings["high"].append(
                    {
                        "type": "hardcoded_secret",
                        "secret_type": secret_type,
                        "file": str(file_path),
                        "line": line_num,
                        "matched_text": match.group()[:50] + "...",
                        "recommendation": "Move secret to environment variable or secure vault",
                    }
                )

    def check_sql_injection(self, file_path: Path, content: str):
        """Check for SQL injection vulnerabilities"""
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*\s*\+\s*[^"\']*["\']',  # String concatenation in SQL
            r'query\s*\(\s*["\'][^"\']*%s[^"\']*["\']',  # String formatting in SQL
            r'\.format\s*\([^)]*\)\s*["\'][^"\']*SELECT',  # .format() with SQL
            r'f["\'][^"\']*SELECT[^"\']*\{[^}]*\}',  # f-strings with SQL
        ]

        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                self.findings["high"].append(
                    {
                        "type": "sql_injection_risk",
                        "file": str(file_path),
                        "line": line_num,
                        "pattern": match.group(),
                        "recommendation": "Use parameterized queries or ORM methods",
                    }
                )

    def check_xss_vulnerabilities(self, file_path: Path, content: str):
        """Check for XSS vulnerabilities"""
        xss_patterns = [
            r"render_template_string\s*\([^)]*\+",  # Template injection
            r"Markup\s*\([^)]*\+",  # Direct markup concatenation
            r"\.innerHTML\s*=\s*[^;]*\+",  # innerHTML concatenation
            r"document\.write\s*\([^)]*\+",  # document.write concatenation
        ]

        for pattern in xss_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                self.findings["medium"].append(
                    {
                        "type": "xss_risk",
                        "file": str(file_path),
                        "line": line_num,
                        "pattern": match.group(),
                        "recommendation": "Use proper escaping or templating engine",
                    }
                )

    def check_insecure_random(self, file_path: Path, content: str):
        """Check for insecure random number generation"""
        if "import random" in content or "from random import" in content:
            # Check if it's used for security purposes
            security_contexts = ["password", "token", "secret", "key", "salt", "nonce", "csrf"]

            for context in security_contexts:
                if context in content.lower() and "random." in content:
                    self.findings["medium"].append(
                        {
                            "type": "insecure_random",
                            "file": str(file_path),
                            "context": context,
                            "recommendation": "Use secrets module for cryptographic randomness",
                        }
                    )

    def check_debug_mode(self, file_path: Path, content: str):
        """Check for debug mode enabled in production"""
        debug_patterns = [
            r"debug\s*=\s*True",
            r"DEBUG\s*=\s*True",
            r"app\.run\([^)]*debug\s*=\s*True",
        ]

        for pattern in debug_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                self.findings["medium"].append(
                    {
                        "type": "debug_mode_enabled",
                        "file": str(file_path),
                        "line": line_num,
                        "recommendation": "Disable debug mode in production",
                    }
                )

    def review_authentication_paths(self):
        """Review all authentication and authorization paths"""
        print("🔐 Reviewing Authentication Paths...")

        auth_files = [
            "music_gen/api/endpoints/auth.py",
            "music_gen/api/middleware/auth.py",
            "music_gen/core/config.py",
        ]

        for auth_file in auth_files:
            file_path = self.project_root / auth_file
            if file_path.exists():
                self.analyze_auth_file(file_path)

    def analyze_auth_file(self, file_path: Path):
        """Analyze authentication file for security issues"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check for weak authentication
            weak_auth_patterns = [
                (r'password\s*==\s*["\']password["\']', "weak_default_password"),
                (r'if.*user.*==.*["\']admin["\']', "hardcoded_admin_check"),
                (r'secret.*=.*["\'][^"\']{1,10}["\']', "weak_secret"),
                (r'jwt.*secret.*=.*["\']your-secret', "default_jwt_secret"),
            ]

            for pattern, issue_type in weak_auth_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1

                    severity = "critical" if "default" in issue_type else "high"

                    self.findings[severity].append(
                        {
                            "type": "weak_authentication",
                            "issue_type": issue_type,
                            "file": str(file_path),
                            "line": line_num,
                            "match": match.group(),
                            "recommendation": "Use strong, randomized secrets and proper authentication",
                        }
                    )

            # Check for missing authentication
            if "auth.py" in str(file_path):
                self.check_auth_bypass_risks(file_path, content)

        except Exception as e:
            self.findings["medium"].append(
                {"type": "auth_analysis_error", "file": str(file_path), "error": str(e)}
            )

    def check_auth_bypass_risks(self, file_path: Path, content: str):
        """Check for authentication bypass risks"""
        bypass_patterns = [
            r"if.*debug.*:.*return.*True",
            r"if.*test.*:.*return.*True",
            r"if.*skip.*auth.*:.*return",
            r"return.*True.*#.*bypass",
        ]

        for pattern in bypass_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                self.findings["critical"].append(
                    {
                        "type": "auth_bypass_risk",
                        "file": str(file_path),
                        "line": line_num,
                        "pattern": match.group(),
                        "recommendation": "Remove authentication bypass code from production",
                    }
                )

    def check_information_disclosure(self):
        """Check for information disclosure vulnerabilities"""
        print("🔍 Checking Information Disclosure...")

        # Check configuration files
        config_files = (
            list(self.project_root.rglob("*.yaml"))
            + list(self.project_root.rglob("*.yml"))
            + list(self.project_root.rglob("*.json"))
            + list(self.project_root.rglob("*.env*"))
        )

        for config_file in config_files:
            self.analyze_config_file(config_file)

        # Check Python files for information disclosure
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            self.check_python_info_disclosure(py_file)

    def analyze_config_file(self, file_path: Path):
        """Analyze configuration files for sensitive information"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Sensitive patterns in config files
            sensitive_patterns = [
                (r"password:\s*[^\s]+", "password_in_config"),
                (r"secret:\s*[^\s]+", "secret_in_config"),
                (r"key:\s*[^\s]+", "key_in_config"),
                (r"token:\s*[^\s]+", "token_in_config"),
                (r"database.*://.*:.*@", "database_credentials"),
                (r"[A-Za-z0-9+/]{20,}={0,2}", "base64_encoded_secret"),
            ]

            for pattern, disclosure_type in sensitive_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1

                    # Skip obvious placeholders
                    if any(
                        placeholder in match.group().lower()
                        for placeholder in ["example", "placeholder", "your-", "change-me"]
                    ):
                        continue

                    self.findings["high"].append(
                        {
                            "type": "information_disclosure",
                            "disclosure_type": disclosure_type,
                            "file": str(file_path),
                            "line": line_num,
                            "recommendation": "Move sensitive data to environment variables or secure vault",
                        }
                    )

        except Exception as e:
            pass  # Skip files that can't be read

    def check_python_info_disclosure(self, file_path: Path):
        """Check Python files for information disclosure"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check for error messages that might leak information
            disclosure_patterns = [
                (r"print\s*\([^)]*password[^)]*\)", "password_in_logs"),
                (r"logger.*info.*password", "password_in_logs"),
                (r'raise.*Exception.*["\'][^"\']*{[^}]*}', "data_in_exception"),
                (r"traceback\.print_exc\(\)", "stack_trace_disclosure"),
            ]

            for pattern, disclosure_type in disclosure_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1

                    self.findings["medium"].append(
                        {
                            "type": "information_disclosure",
                            "disclosure_type": disclosure_type,
                            "file": str(file_path),
                            "line": line_num,
                            "recommendation": "Sanitize error messages and avoid logging sensitive data",
                        }
                    )

        except Exception as e:
            pass

    def verify_input_validation(self):
        """Verify input validation across all endpoints"""
        print("✅ Verifying Input Validation...")

        # Find API endpoint files
        api_files = list((self.project_root / "music_gen" / "api").rglob("*.py"))

        for api_file in api_files:
            self.analyze_input_validation(api_file)

    def analyze_input_validation(self, file_path: Path):
        """Analyze input validation in API files"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check for missing validation patterns
            validation_issues = []

            # Look for FastAPI endpoints
            endpoint_pattern = r"@app\.(get|post|put|delete)\s*\([^)]*\)"
            endpoints = re.finditer(endpoint_pattern, content, re.IGNORECASE)

            for endpoint in endpoints:
                line_num = content[: endpoint.start()].count("\n") + 1

                # Get the function definition following the decorator
                func_start = endpoint.end()
                func_pattern = r"def\s+(\w+)\s*\([^)]*\):"
                func_match = re.search(func_pattern, content[func_start : func_start + 200])

                if func_match:
                    func_name = func_match.group(1)

                    # Check if validation is present
                    if not self.has_input_validation(content, func_start, func_name):
                        self.findings["medium"].append(
                            {
                                "type": "missing_input_validation",
                                "file": str(file_path),
                                "line": line_num,
                                "function": func_name,
                                "recommendation": "Add proper input validation using Pydantic models",
                            }
                        )

        except Exception as e:
            pass

    def has_input_validation(self, content: str, start_pos: int, func_name: str) -> bool:
        """Check if function has proper input validation"""
        # Look for Pydantic model usage
        pydantic_patterns = [
            r":\s*BaseModel",
            r":\s*\w+Request",
            r":\s*\w+Model",
            r"Field\(",
            r"validator\(",
        ]

        # Check next 500 characters for validation patterns
        func_content = content[start_pos : start_pos + 500]

        return any(re.search(pattern, func_content, re.IGNORECASE) for pattern in pydantic_patterns)

    def test_rate_limiting(self):
        """Test rate limiting effectiveness"""
        print("🚦 Testing Rate Limiting...")

        # Analyze rate limiting configuration
        rate_limit_files = ["music_gen/api/middleware/rate_limiting.py", "music_gen/core/config.py"]

        for file_path in rate_limit_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.analyze_rate_limiting(full_path)

    def analyze_rate_limiting(self, file_path: Path):
        """Analyze rate limiting implementation"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check for rate limiting configuration
            rate_limit_checks = [
                ("redis", "Uses Redis for distributed rate limiting"),
                ("time.time()", "Has time-based rate limiting"),
                ("requests_per_minute", "Configures requests per minute"),
                ("rate_limit_exceeded", "Handles rate limit exceeded cases"),
            ]

            has_rate_limiting = False
            for check, description in rate_limit_checks:
                if check in content:
                    has_rate_limiting = True
                    self.findings["info"].append(
                        {
                            "type": "rate_limiting_feature",
                            "file": str(file_path),
                            "feature": description,
                        }
                    )

            if not has_rate_limiting:
                self.findings["medium"].append(
                    {
                        "type": "missing_rate_limiting",
                        "file": str(file_path),
                        "recommendation": "Implement rate limiting to prevent abuse",
                    }
                )

            # Check for bypass conditions
            bypass_patterns = [
                r"if.*admin.*:.*continue",
                r"if.*whitelist.*:.*return",
                r"if.*debug.*:.*skip",
            ]

            for pattern in bypass_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1

                    self.findings["medium"].append(
                        {
                            "type": "rate_limit_bypass",
                            "file": str(file_path),
                            "line": line_num,
                            "pattern": match.group(),
                            "recommendation": "Review rate limiting bypass conditions",
                        }
                    )

        except Exception as e:
            pass

    def review_security_headers(self):
        """Review security headers implementation"""
        print("🛡️ Reviewing Security Headers...")

        security_header_files = [
            "music_gen/api/middleware/security_headers.py",
            "music_gen/api/cors_config.py",
        ]

        for file_path in security_header_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.analyze_security_headers(full_path)

    def analyze_security_headers(self, file_path: Path):
        """Analyze security headers configuration"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Required security headers
            required_headers = [
                ("X-Content-Type-Options", "nosniff"),
                ("X-Frame-Options", "DENY"),
                ("X-XSS-Protection", "1; mode=block"),
                ("Strict-Transport-Security", "max-age"),
                ("Content-Security-Policy", "default-src"),
                ("Referrer-Policy", "strict-origin"),
            ]

            missing_headers = []
            weak_headers = []

            for header_name, expected_value in required_headers:
                if header_name not in content:
                    missing_headers.append(header_name)
                elif expected_value not in content:
                    weak_headers.append((header_name, expected_value))

            for header in missing_headers:
                self.findings["medium"].append(
                    {
                        "type": "missing_security_header",
                        "file": str(file_path),
                        "header": header,
                        "recommendation": f"Add {header} security header",
                    }
                )

            for header, expected in weak_headers:
                self.findings["low"].append(
                    {
                        "type": "weak_security_header",
                        "file": str(file_path),
                        "header": header,
                        "expected": expected,
                        "recommendation": f"Strengthen {header} header configuration",
                    }
                )

            # Check for unsafe CSP directives
            if "'unsafe-inline'" in content:
                self.findings["medium"].append(
                    {
                        "type": "unsafe_csp_directive",
                        "file": str(file_path),
                        "directive": "unsafe-inline",
                        "recommendation": "Use nonces or hashes instead of unsafe-inline",
                    }
                )

            if "'unsafe-eval'" in content:
                self.findings["high"].append(
                    {
                        "type": "unsafe_csp_directive",
                        "file": str(file_path),
                        "directive": "unsafe-eval",
                        "recommendation": "Remove unsafe-eval from CSP",
                    }
                )

        except Exception as e:
            pass

    def additional_security_checks(self):
        """Additional security checks"""
        print("🔒 Running Additional Security Checks...")

        # Check for common security misconfigurations
        self.check_file_permissions()
        self.check_environment_files()
        self.check_dockerfile_security()
        self.check_kubernetes_security()

    def check_file_permissions(self):
        """Check file permissions for security issues"""
        sensitive_files = [
            ".env",
            ".env.local",
            ".env.production",
            "config.yaml",
            "secrets.yaml",
            "private.key",
            "*.pem",
        ]

        for pattern in sensitive_files:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]

                    # Check if file is readable by others
                    if mode[-1] in ["4", "5", "6", "7"]:
                        self.findings["medium"].append(
                            {
                                "type": "insecure_file_permissions",
                                "file": str(file_path),
                                "permissions": mode,
                                "recommendation": "Restrict file permissions (chmod 600)",
                            }
                        )
                except Exception:
                    pass

    def check_environment_files(self):
        """Check environment files for security issues"""
        env_files = list(self.project_root.rglob(".env*"))

        for env_file in env_files:
            if env_file.name == ".env.example":
                continue

            try:
                with open(env_file, "r") as f:
                    content = f.read()

                # Check for actual secrets in .env files
                if any(
                    pattern in content for pattern in ["password=", "secret=", "key=", "token="]
                ):
                    self.findings["high"].append(
                        {
                            "type": "secrets_in_env_file",
                            "file": str(env_file),
                            "recommendation": "Do not commit .env files with real secrets",
                        }
                    )

            except Exception:
                pass

    def check_dockerfile_security(self):
        """Check Dockerfile for security issues"""
        dockerfiles = list(self.project_root.rglob("Dockerfile*"))

        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, "r") as f:
                    content = f.read()

                # Check for security issues in Dockerfile
                if (
                    "USER root" in content
                    and "USER " not in content[content.index("USER root") + 9 :]
                ):
                    self.findings["medium"].append(
                        {
                            "type": "dockerfile_root_user",
                            "file": str(dockerfile),
                            "recommendation": "Do not run containers as root user",
                        }
                    )

                if "COPY . ." in content:
                    self.findings["low"].append(
                        {
                            "type": "dockerfile_copy_all",
                            "file": str(dockerfile),
                            "recommendation": "Use .dockerignore to exclude sensitive files",
                        }
                    )

            except Exception:
                pass

    def check_kubernetes_security(self):
        """Check Kubernetes manifests for security issues"""
        k8s_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))

        for k8s_file in k8s_files:
            try:
                with open(k8s_file, "r") as f:
                    content = f.read()

                # Skip non-k8s files
                if "apiVersion" not in content:
                    continue

                # Check for security issues
                if "privileged: true" in content:
                    self.findings["high"].append(
                        {
                            "type": "k8s_privileged_container",
                            "file": str(k8s_file),
                            "recommendation": "Avoid privileged containers",
                        }
                    )

                if "runAsRoot: true" in content:
                    self.findings["medium"].append(
                        {
                            "type": "k8s_run_as_root",
                            "file": str(k8s_file),
                            "recommendation": "Do not run containers as root",
                        }
                    )

            except Exception:
                pass

    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "=" * 60)
        print("🔒 SECURITY AUDIT REPORT")
        print("=" * 60)

        total_findings = sum(len(findings) for findings in self.findings.values())

        print(f"\n📊 SUMMARY:")
        print(f"Critical: {len(self.findings['critical'])}")
        print(f"High:     {len(self.findings['high'])}")
        print(f"Medium:   {len(self.findings['medium'])}")
        print(f"Low:      {len(self.findings['low'])}")
        print(f"Info:     {len(self.findings['info'])}")
        print(f"Total:    {total_findings}")

        # Critical findings
        if self.findings["critical"]:
            print(f"\n🚨 CRITICAL FINDINGS ({len(self.findings['critical'])}):")
            for i, finding in enumerate(self.findings["critical"], 1):
                print(f"\n{i}. {finding.get('type', 'Unknown').replace('_', ' ').title()}")
                print(f"   File: {finding.get('file', 'Unknown')}")
                if "line" in finding:
                    print(f"   Line: {finding['line']}")
                if "recommendation" in finding:
                    print(f"   Fix: {finding['recommendation']}")

        # High findings
        if self.findings["high"]:
            print(f"\n⚠️  HIGH FINDINGS ({len(self.findings['high'])}):")
            for i, finding in enumerate(self.findings["high"], 1):
                print(f"\n{i}. {finding.get('type', 'Unknown').replace('_', ' ').title()}")
                print(f"   File: {finding.get('file', 'Unknown')}")
                if "line" in finding:
                    print(f"   Line: {finding['line']}")
                if "recommendation" in finding:
                    print(f"   Fix: {finding['recommendation']}")

        # Save detailed report
        self.save_detailed_report()

        return total_findings

    def save_detailed_report(self):
        """Save detailed security audit report"""
        report = {
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.project_root),
            "summary": {
                "critical": len(self.findings["critical"]),
                "high": len(self.findings["high"]),
                "medium": len(self.findings["medium"]),
                "low": len(self.findings["low"]),
                "info": len(self.findings["info"]),
                "total": sum(len(findings) for findings in self.findings.values()),
            },
            "findings": self.findings,
            "fixed_issues": self.fixed_issues,
        }

        report_file = self.project_root / "security_audit_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Run security audit"""
    project_root = os.getcwd()
    auditor = SecurityAuditor(project_root)

    findings_count = auditor.run_comprehensive_audit()

    # Exit with error code if critical or high findings
    critical_high = len(auditor.findings["critical"]) + len(auditor.findings["high"])
    if critical_high > 0:
        print(f"\n❌ Security audit failed: {critical_high} critical/high severity issues found")
        sys.exit(1)
    else:
        print(f"\n✅ Security audit passed: No critical or high severity issues found")
        sys.exit(0)


if __name__ == "__main__":
    main()
