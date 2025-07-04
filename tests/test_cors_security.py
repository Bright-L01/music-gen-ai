"""
CORS Security Tests for Music Gen AI API.
Tests various CORS scenarios to ensure proper security implementation.
"""

import pytest
import os
from unittest.mock import patch, Mock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from music_gen.api.app import create_app
from music_gen.api.cors_config import CORSConfig, cors_config


# Test fixtures
@pytest.fixture
def mock_env_development():
    """Mock development environment."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "development",
        "ALLOWED_ORIGINS": "https://custom-dev.com"
    }, clear=True):
        yield


@pytest.fixture
def mock_env_staging():
    """Mock staging environment."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "staging",
        "STAGING_DEV_ORIGINS": "http://localhost:3000"
    }, clear=True):
        yield


@pytest.fixture
def mock_env_production():
    """Mock production environment."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "ALLOWED_ORIGINS": "https://app.musicgen.ai,https://admin.musicgen.ai"
    }, clear=True):
        yield


@pytest.fixture
def test_app():
    """Create a test FastAPI app with CORS and test endpoints."""
    from music_gen.api.cors_config import get_cors_config, cors_config
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI()
    
    # Add CORS middleware
    cors_options = get_cors_config()
    app.add_middleware(CORSMiddleware, **cors_options)
    
    # Add custom CORS validation middleware (same as in main app)
    @app.middleware("http")
    async def validate_cors_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        
        if request.method == "OPTIONS":
            if origin and not cors_config.validate_origin_header(origin):
                return JSONResponse(
                    content={"error": "CORS origin not allowed"},
                    status_code=403,
                    headers={"Vary": "Origin"}
                )
        
        response = await call_next(request)
        
        if origin:
            cors_headers = cors_config.get_response_headers(origin)
            for header, value in cors_headers.items():
                response.headers[header] = value
        
        return response
    
    # Test endpoints
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    @app.post("/test")
    async def test_post():
        return {"message": "posted"}
    
    @app.put("/test")
    async def test_put():
        return {"message": "updated"}
    
    @app.delete("/test")
    async def test_delete():
        return {"message": "deleted"}
    
    @app.get("/test/credentials")
    async def test_credentials(request: Request):
        # Check if credentials were sent
        auth_header = request.headers.get("authorization")
        cookie_header = request.headers.get("cookie")
        return {
            "has_auth": bool(auth_header),
            "has_cookie": bool(cookie_header)
        }
    
    return app


class TestCORSAllowedOrigins:
    """Test that allowed origins can access the API."""
    
    def test_development_localhost_allowed(self, mock_env_development, test_app):
        """Test localhost origins are allowed in development."""
        client = TestClient(test_app)
        
        # Test various localhost origins
        localhost_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://[::1]:3000"
        ]
        
        for origin in localhost_origins:
            response = client.get("/test", headers={"Origin": origin})
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == origin
            assert response.headers.get("access-control-allow-credentials") == "true"
            assert "access-control-expose-headers" in response.headers
    
    def test_staging_origins_allowed(self, mock_env_staging, test_app):
        """Test staging origins are allowed in staging environment."""
        client = TestClient(test_app)
        
        staging_origins = [
            "https://staging.musicgen.ai",
            "https://preview.musicgen.ai",
            "https://beta.musicgen.ai"
        ]
        
        for origin in staging_origins:
            response = client.get("/test", headers={"Origin": origin})
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == origin
    
    def test_production_origins_allowed(self, mock_env_production, test_app):
        """Test production origins are allowed in production environment."""
        client = TestClient(test_app)
        
        prod_origins = [
            "https://app.musicgen.ai",
            "https://admin.musicgen.ai"
        ]
        
        for origin in prod_origins:
            response = client.get("/test", headers={"Origin": origin})
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == origin
    
    def test_custom_allowed_origins(self, mock_env_development, test_app):
        """Test custom origins from environment variable are allowed."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "https://custom-dev.com"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://custom-dev.com"
    
    def test_staging_dev_origins(self, mock_env_staging, test_app):
        """Test development origins allowed in staging via STAGING_DEV_ORIGINS."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestCORSBlockedOrigins:
    """Test that non-allowed origins are blocked."""
    
    def test_unknown_origin_blocked(self, mock_env_production, test_app):
        """Test unknown origins don't get CORS headers."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "https://evil.com"})
        
        # Request should succeed but without CORS headers
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
        assert "access-control-allow-credentials" not in response.headers
    
    def test_http_origin_blocked_in_production(self, mock_env_production, test_app):
        """Test HTTP origins are blocked in production."""
        client = TestClient(test_app)
        
        # Even if explicitly added, HTTP should be rejected in production
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "http://insecure.com"}):
            response = client.get("/test", headers={"Origin": "http://insecure.com"})
            
            assert response.status_code == 200
            assert "access-control-allow-origin" not in response.headers
    
    def test_localhost_blocked_in_production(self, mock_env_production, test_app):
        """Test localhost is blocked in production by default."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
    
    def test_malformed_origin_blocked(self, mock_env_development, test_app):
        """Test malformed origins are blocked."""
        client = TestClient(test_app)
        
        malformed_origins = [
            "not-a-valid-origin",
            "//example.com",
            "https://",
            "https://example.com/path",
            "https://example.com?query=1",
            "ftp://example.com"
        ]
        
        for origin in malformed_origins:
            response = client.get("/test", headers={"Origin": origin})
            
            assert response.status_code == 200
            assert "access-control-allow-origin" not in response.headers


class TestCORSPreflightRequests:
    """Test CORS preflight (OPTIONS) requests."""
    
    def test_preflight_allowed_origin(self, mock_env_development, test_app):
        """Test preflight request from allowed origin."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, Authorization"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"
        assert "POST" in response.headers.get("access-control-allow-methods", "")
        assert "Content-Type" in response.headers.get("access-control-allow-headers", "")
        assert "Authorization" in response.headers.get("access-control-allow-headers", "")
        assert response.headers.get("access-control-max-age") == "86400"
    
    def test_preflight_blocked_origin(self, mock_env_production, test_app):
        """Test preflight request from blocked origin."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # Should be blocked by custom middleware
        assert response.status_code == 403
        assert response.json() == {"error": "CORS origin not allowed"}
    
    def test_preflight_disallowed_method(self, mock_env_development, test_app):
        """Test preflight with disallowed method."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "TRACE"  # Not allowed
            }
        )
        
        # FastAPI's CORS middleware handles this differently - it allows the preflight
        # but won't include TRACE in allowed methods
        assert response.status_code == 200
        assert "TRACE" not in response.headers.get("access-control-allow-methods", "")
    
    def test_preflight_disallowed_header(self, mock_env_development, test_app):
        """Test preflight with custom disallowed header."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Custom-Evil-Header"
            }
        )
        
        # FastAPI's CORS middleware is permissive with headers
        # Our custom validation would need to be stricter
        assert response.status_code == 200
    
    def test_preflight_cache_header(self, mock_env_development, test_app):
        """Test preflight cache duration header."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-max-age") == "86400"  # 24 hours


class TestCORSCredentials:
    """Test CORS credentials handling."""
    
    def test_credentials_allowed_origin(self, mock_env_development, test_app):
        """Test credentials are allowed for whitelisted origins."""
        client = TestClient(test_app)
        
        response = client.get(
            "/test/credentials",
            headers={
                "Origin": "http://localhost:3000",
                "Authorization": "Bearer token123",
                "Cookie": "session=abc123"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-credentials") == "true"
        assert response.json()["has_auth"] is True
        assert response.json()["has_cookie"] is True
    
    def test_credentials_blocked_origin(self, mock_env_production, test_app):
        """Test credentials handling for non-whitelisted origins."""
        client = TestClient(test_app)
        
        response = client.get(
            "/test/credentials",
            headers={
                "Origin": "https://evil.com",
                "Authorization": "Bearer token123",
                "Cookie": "session=abc123"
            }
        )
        
        # Request succeeds but without CORS headers
        assert response.status_code == 200
        assert "access-control-allow-credentials" not in response.headers
        # Credentials are still sent (this is browser's job to block)
        assert response.json()["has_auth"] is True
        assert response.json()["has_cookie"] is True
    
    def test_credentials_in_preflight(self, mock_env_development, test_app):
        """Test credentials header in preflight response."""
        client = TestClient(test_app)
        
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-credentials") == "true"


class TestCORSHttpMethods:
    """Test various HTTP methods with CORS."""
    
    def test_get_method(self, mock_env_development, test_app):
        """Test GET method with CORS."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    
    def test_post_method(self, mock_env_development, test_app):
        """Test POST method with CORS."""
        client = TestClient(test_app)
        
        response = client.post("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    
    def test_put_method(self, mock_env_development, test_app):
        """Test PUT method with CORS."""
        client = TestClient(test_app)
        
        response = client.put("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    
    def test_delete_method(self, mock_env_development, test_app):
        """Test DELETE method with CORS."""
        client = TestClient(test_app)
        
        response = client.delete("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    
    def test_patch_method(self, mock_env_development, test_app):
        """Test PATCH method with CORS."""
        client = TestClient(test_app)
        
        # First do preflight
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PATCH"
            }
        )
        
        assert response.status_code == 200
        assert "PATCH" in response.headers.get("access-control-allow-methods", "")


class TestCORSHeaders:
    """Test CORS header handling."""
    
    def test_vary_header(self, mock_env_development, test_app):
        """Test Vary header is set for caching."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert "Origin" in response.headers.get("vary", "")
    
    def test_expose_headers(self, mock_env_development, test_app):
        """Test exposed headers configuration."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        exposed = response.headers.get("access-control-expose-headers", "")
        assert "X-Request-ID" in exposed
        assert "X-RateLimit-Limit" in exposed
    
    def test_no_origin_header(self, mock_env_development, test_app):
        """Test request without Origin header."""
        client = TestClient(test_app)
        
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestCORSIntegration:
    """Integration tests with the full application."""
    
    @pytest.mark.asyncio
    async def test_real_app_cors(self, mock_env_development):
        """Test CORS with the real application."""
        app = create_app()
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"
    
    @pytest.mark.asyncio
    async def test_api_endpoints_cors(self, mock_env_production):
        """Test CORS on actual API endpoints."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://app.musicgen.ai"
        }):
            app = create_app()
            client = TestClient(app)
            
            # Test generation endpoint preflight
            response = client.options(
                "/api/v1/generate",
                headers={
                    "Origin": "https://app.musicgen.ai",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type, Authorization"
                }
            )
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == "https://app.musicgen.ai"
            
            # Test blocked origin
            response = client.options(
                "/api/v1/generate",
                headers={
                    "Origin": "https://hacker.com",
                    "Access-Control-Request-Method": "POST"
                }
            )
            
            assert response.status_code == 403


class TestCORSEdgeCases:
    """Test edge cases and security scenarios."""
    
    def test_case_sensitive_origin(self, mock_env_development, test_app):
        """Test that origin matching is case-sensitive."""
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://Example.COM"}):
            client = TestClient(test_app)
            
            # Exact case match should work
            response = client.get("/test", headers={"Origin": "https://Example.COM"})
            assert response.headers.get("access-control-allow-origin") == "https://Example.COM"
            
            # Different case should not work
            response = client.get("/test", headers={"Origin": "https://example.com"})
            assert "access-control-allow-origin" not in response.headers
    
    def test_origin_with_port(self, mock_env_development, test_app):
        """Test origins with explicit ports."""
        client = TestClient(test_app)
        
        # Standard ports
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        
        # Non-standard port not in whitelist
        response = client.get("/test", headers={"Origin": "http://localhost:9999"})
        assert "access-control-allow-origin" not in response.headers
    
    def test_subdomain_without_wildcard(self, mock_env_production, test_app):
        """Test subdomains are not automatically allowed."""
        with patch.dict(os.environ, {
            "ALLOWED_ORIGINS": "https://musicgen.ai"
        }):
            client = TestClient(test_app)
            
            # Exact domain works
            response = client.get("/test", headers={"Origin": "https://musicgen.ai"})
            assert response.headers.get("access-control-allow-origin") == "https://musicgen.ai"
            
            # Subdomain should not work without wildcard
            response = client.get("/test", headers={"Origin": "https://app.musicgen.ai"})
            assert "access-control-allow-origin" not in response.headers
    
    def test_empty_origin(self, mock_env_development, test_app):
        """Test empty origin header."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": ""})
        
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
    
    def test_null_origin(self, mock_env_development, test_app):
        """Test null origin (from data: URIs, file://, etc)."""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"Origin": "null"})
        
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestCORSWildcardSubdomains:
    """Test wildcard subdomain functionality."""
    
    def test_wildcard_enabled(self):
        """Test wildcard subdomain matching when enabled."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://*.musicgen.ai",
            "ALLOW_SUBDOMAIN_WILDCARDS": "true"
        }, clear=True):
            config = CORSConfig()
            
            # Should match subdomains
            assert config.is_origin_allowed("https://app.musicgen.ai")
            assert config.is_origin_allowed("https://api.musicgen.ai")
            assert config.is_origin_allowed("https://staging.musicgen.ai")
            assert config.is_origin_allowed("https://musicgen.ai")
            
            # Should not match different domains
            assert not config.is_origin_allowed("https://musicgen.com")
            assert not config.is_origin_allowed("https://fake-musicgen.ai")
    
    def test_wildcard_disabled(self):
        """Test wildcard subdomain matching when disabled."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://*.musicgen.ai",
            "ALLOW_SUBDOMAIN_WILDCARDS": "false"
        }, clear=True):
            config = CORSConfig()
            
            # Wildcard should be treated as literal
            assert "https://*.musicgen.ai" in config.allowed_origins
            
            # Should not match subdomains
            assert not config.is_origin_allowed("https://app.musicgen.ai")


# Performance test
class TestCORSPerformance:
    """Test CORS doesn't significantly impact performance."""
    
    def test_cors_overhead(self, mock_env_development, test_app):
        """Test CORS adds minimal overhead to requests."""
        import time
        client = TestClient(test_app)
        
        # Warm up
        client.get("/test")
        
        # Time without origin
        start = time.time()
        for _ in range(100):
            client.get("/test")
        time_without_cors = time.time() - start
        
        # Time with origin
        start = time.time()
        for _ in range(100):
            client.get("/test", headers={"Origin": "http://localhost:3000"})
        time_with_cors = time.time() - start
        
        # CORS should add less than 50% overhead
        overhead = (time_with_cors - time_without_cors) / time_without_cors
        assert overhead < 0.5, f"CORS overhead too high: {overhead:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])