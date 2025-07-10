"""
Audit logging middleware for security-sensitive operations.

Logs authentication, authorization, data access, and security events
following compliance requirements for audit trails.
"""

import json
import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.logging_config import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs security-sensitive operations for audit trails."""

    def __init__(
        self,
        app: ASGIApp,
        log_all_requests: bool = False,
        sensitive_paths: Optional[list] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ):
        super().__init__(app)
        self.log_all_requests = log_all_requests
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

        # Default sensitive paths that should always be audited
        self.sensitive_paths = {
            "/api/auth/login",
            "/api/auth/logout",
            "/api/auth/register",
            "/api/auth/refresh",
            "/api/auth/reset-password",
            "/api/v1/admin/",
            "/api/v1/users/",
            "/api/v1/generate/",
            "/api/v1/models/upload",
            "/api/v1/models/delete",
        }

        # Add custom sensitive paths
        if sensitive_paths:
            self.sensitive_paths.update(sensitive_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log audit events for sensitive operations."""

        # Check if this request should be audited
        should_audit = self._should_audit_request(request)

        if not should_audit and not self.log_all_requests:
            return await call_next(request)

        # Get correlation ID and user context
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        user_id = self._extract_user_id(request)
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Capture request details
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": self._filter_sensitive_headers(dict(request.headers)),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "user_id": user_id,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }

        # Optionally capture request body for certain operations
        if self.log_request_body and should_audit:
            try:
                if request.headers.get("content-type", "").startswith("application/json"):
                    body = await request.body()
                    if body:
                        request_data["request_body"] = self._filter_sensitive_data(
                            json.loads(body.decode())
                        )
            except Exception as e:
                logger.warning("Failed to capture request body for audit", error=str(e))

        # Log request start for sensitive operations
        if should_audit:
            audit_logger.log_data_access(
                user_id=user_id or "anonymous",
                data_type="api_request",
                action="access_attempt",
                resource_id=request.url.path,
                ip_address=ip_address,
                correlation_id=correlation_id,
                method=request.method,
                user_agent=user_agent,
            )

        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms

            # Log successful operations
            if should_audit:
                self._log_successful_operation(request_data, response, processing_time)

            # Log authentication events
            if self._is_auth_endpoint(request.url.path):
                self._log_authentication_event(
                    request, response, user_id, ip_address, user_agent, correlation_id
                )

            # Log data access events
            if self._is_data_access_endpoint(request.url.path):
                self._log_data_access_event(request, response, user_id, ip_address, correlation_id)

            # Log admin actions
            if self._is_admin_endpoint(request.url.path):
                self._log_admin_action(request, response, user_id, ip_address, correlation_id)

            return response

        except Exception as e:
            # Log failed operations
            processing_time = (time.time() - start_time) * 1000  # ms

            if should_audit:
                audit_logger.log_security_event(
                    event_type="operation_failure",
                    severity="medium",
                    description=f"Operation failed: {str(e)}",
                    ip_address=ip_address,
                    user_id=user_id,
                    correlation_id=correlation_id,
                    path=request.url.path,
                    method=request.method,
                    processing_time_ms=processing_time,
                    error_type=type(e).__name__,
                )

            raise

    def _should_audit_request(self, request: Request) -> bool:
        """Determine if a request should be audited."""
        path = request.url.path

        # Always audit sensitive paths
        if any(sensitive_path in path for sensitive_path in self.sensitive_paths):
            return True

        # Audit admin endpoints
        if path.startswith("/api/v1/admin/"):
            return True

        # Audit write operations
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            return True

        return False

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request context."""
        # Try to get from JWT token in request state
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "id"):
                return str(user.id)
            elif isinstance(user, dict):
                return user.get("user_id") or user.get("id")

        # Try to get from JWT token directly
        if hasattr(request.state, "jwt_payload"):
            return request.state.jwt_payload.get("sub")

        return None

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (common with reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to client IP
        return request.client.host if request.client else "unknown"

    def _filter_sensitive_headers(self, headers: dict) -> dict:
        """Remove sensitive information from headers."""
        sensitive_headers = {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
            "x-csrf-token",
            "x-session-id",
        }

        filtered = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in sensitive_headers:
                if len(value) > 8:
                    filtered[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value

        return filtered

    def _filter_sensitive_data(self, data: dict) -> dict:
        """Remove sensitive information from request/response data."""
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "key",
            "authorization",
            "credit_card",
            "ssn",
            "social_security",
            "api_key",
            "jwt",
            "session",
            "csrf_token",
            "refresh_token",
        }

        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 8:
                    filtered[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value

        return filtered

    def _is_auth_endpoint(self, path: str) -> bool:
        """Check if path is an authentication endpoint."""
        auth_paths = ["/api/auth/", "/api/v1/auth/"]
        return any(auth_path in path for auth_path in auth_paths)

    def _is_data_access_endpoint(self, path: str) -> bool:
        """Check if path involves data access."""
        data_paths = ["/api/v1/generate/", "/api/v1/models/", "/api/v1/users/"]
        return any(data_path in path for data_path in data_paths)

    def _is_admin_endpoint(self, path: str) -> bool:
        """Check if path is an admin endpoint."""
        return "/api/v1/admin/" in path

    def _log_successful_operation(
        self, request_data: dict, response: Response, processing_time: float
    ):
        """Log successful operation details."""
        audit_logger.log_data_access(
            user_id=request_data["user_id"] or "anonymous",
            data_type="api_operation",
            action="completed",
            resource_id=request_data["path"],
            ip_address=request_data["ip_address"],
            correlation_id=request_data["correlation_id"],
            status_code=response.status_code,
            processing_time_ms=processing_time,
            method=request_data["method"],
        )

    def _log_authentication_event(
        self,
        request: Request,
        response: Response,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        correlation_id: str,
    ):
        """Log authentication-related events."""
        path = request.url.path

        # Determine event type
        if "/login" in path:
            event_type = "login"
        elif "/logout" in path:
            event_type = "logout"
        elif "/register" in path:
            event_type = "register"
        elif "/refresh" in path:
            event_type = "token_refresh"
        elif "/reset-password" in path:
            event_type = "password_reset"
        else:
            event_type = "authentication"

        # Determine success based on status code
        success = 200 <= response.status_code < 300
        failure_reason = None

        if not success:
            if response.status_code == 401:
                failure_reason = "invalid_credentials"
            elif response.status_code == 403:
                failure_reason = "access_forbidden"
            elif response.status_code == 429:
                failure_reason = "rate_limited"
            else:
                failure_reason = f"http_{response.status_code}"

        audit_logger.log_authentication(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            correlation_id=correlation_id,
            failure_reason=failure_reason,
        )

    def _log_data_access_event(
        self,
        request: Request,
        response: Response,
        user_id: Optional[str],
        ip_address: str,
        correlation_id: str,
    ):
        """Log data access events."""
        path = request.url.path
        method = request.method

        # Determine action type
        if method == "GET":
            action = "read"
        elif method == "POST":
            action = "create"
        elif method == "PUT" or method == "PATCH":
            action = "update"
        elif method == "DELETE":
            action = "delete"
        else:
            action = method.lower()

        # Determine data type
        if "/models/" in path:
            data_type = "model"
        elif "/generate/" in path:
            data_type = "generation"
        elif "/users/" in path:
            data_type = "user_data"
        else:
            data_type = "api_resource"

        # Extract resource ID from path
        resource_id = path
        if response.status_code in [200, 201]:
            # Try to extract resource ID from path segments
            path_segments = path.strip("/").split("/")
            if len(path_segments) > 3:
                resource_id = path_segments[-1]

        audit_logger.log_data_access(
            user_id=user_id or "anonymous",
            data_type=data_type,
            action=action,
            resource_id=resource_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            status_code=response.status_code,
            success=200 <= response.status_code < 300,
        )

    def _log_admin_action(
        self,
        request: Request,
        response: Response,
        user_id: Optional[str],
        ip_address: str,
        correlation_id: str,
    ):
        """Log administrative actions."""
        audit_logger.log_security_event(
            event_type="admin_action",
            severity="high",
            description=f"Admin action: {request.method} {request.url.path}",
            ip_address=ip_address,
            user_id=user_id,
            correlation_id=correlation_id,
            status_code=response.status_code,
            success=200 <= response.status_code < 300,
        )
