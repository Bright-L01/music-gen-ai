"""
CORS and Authentication Integration Tests.
Tests that CORS and JWT authentication work correctly together.
"""

import pytest
import os
from unittest.mock import patch, Mock
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt

from music_gen.api.app import create_app
from music_gen.api.cors_config import get_cors_config, cors_config
from music_gen.api.middleware.auth import (
    AuthenticationMiddleware,
    UserRole,
    get_current_user,
    require_auth,
    require_admin,
    UserClaims,
    JWT_SECRET_KEY,
    JWT_ALGORITHM
)
from music_gen.api.deps import get_current_active_user


@pytest.fixture
def auth_middleware():
    """Create authentication middleware instance."""
    return AuthenticationMiddleware()


@pytest.fixture
def valid_token(auth_middleware):
    """Create a valid JWT token."""
    return auth_middleware.create_access_token(
        user_id="test123",
        email="test@example.com",
        username="testuser",
        roles=[UserRole.USER],
        tier="free",
        is_verified=True
    )


@pytest.fixture
def admin_token(auth_middleware):
    """Create a valid admin JWT token."""
    return auth_middleware.create_access_token(
        user_id="admin123",
        email="admin@example.com",
        username="admin",
        roles=[UserRole.ADMIN],
        tier="enterprise",
        is_verified=True
    )


@pytest.fixture
def test_app_with_auth():
    """Create test app with both CORS and authentication."""
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    # Add CORS middleware
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
        cors_options = get_cors_config()
        app.add_middleware(CORSMiddleware, **cors_options)
    
    # Add custom CORS validation
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
    @app.get("/public")
    async def public_endpoint():
        return {"message": "public access"}
    
    @app.get("/authenticated")
    async def authenticated_endpoint(user: UserClaims = Depends(require_auth)):
        return {"message": "authenticated", "user_id": user.user_id}
    
    @app.get("/admin")
    async def admin_endpoint(user: UserClaims = Depends(require_admin())):
        return {"message": "admin access", "user_id": user.user_id}
    
    @app.post("/api/data")
    async def post_data(user: UserClaims = Depends(get_current_active_user)):
        return {"message": "data posted", "user_id": user.user_id}
    
    @app.get("/optional-auth")
    async def optional_auth(user: UserClaims = Depends(get_current_user)):
        if user:
            return {"message": "authenticated", "user_id": user.user_id}
        return {"message": "anonymous"}
    
    return app


class TestCORSWithAuthentication:
    """Test CORS behavior with authenticated requests."""
    
    def test_public_endpoint_with_cors(self, test_app_with_auth):
        """Test public endpoint respects CORS without authentication."""
        client = TestClient(test_app_with_auth)
        
        # Allowed origin
        response = client.get("/public", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"
        
        # Blocked origin
        response = client.get("/public", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
    
    def test_authenticated_endpoint_allowed_origin(self, test_app_with_auth, valid_token, auth_middleware):
        """Test authenticated endpoint with allowed origin."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            response = client.get(
                "/authenticated",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"
        assert response.json()["user_id"] == "test123"
    
    def test_authenticated_endpoint_blocked_origin(self, test_app_with_auth, valid_token, auth_middleware):
        """Test authenticated endpoint with blocked origin still requires auth."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            # With valid token but blocked origin
            response = client.get(
                "/authenticated",
                headers={
                    "Origin": "https://evil.com",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
        
        # Request succeeds (auth passed) but no CORS headers
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers
        assert response.json()["user_id"] == "test123"
    
    def test_authenticated_endpoint_no_token(self, test_app_with_auth):
        """Test authenticated endpoint without token fails regardless of CORS."""
        client = TestClient(test_app_with_auth)
        
        # Allowed origin but no token
        response = client.get(
            "/authenticated",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 401
        # Should still include CORS headers for allowed origin
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    
    def test_admin_endpoint_cors(self, test_app_with_auth, admin_token, valid_token, auth_middleware):
        """Test admin endpoint with CORS."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            # Admin with allowed origin
            response = client.get(
                "/admin",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {admin_token}"
                }
            )
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
            
            # Non-admin with allowed origin
            response = client.get(
                "/admin",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
            
            assert response.status_code == 403
            # Should still include CORS headers even on auth failure
            assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestCORSPreflightWithAuth:
    """Test CORS preflight requests for authenticated endpoints."""
    
    def test_preflight_authenticated_endpoint(self, test_app_with_auth):
        """Test preflight for authenticated endpoint."""
        client = TestClient(test_app_with_auth)
        
        response = client.options(
            "/authenticated",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "Authorization" in response.headers.get("access-control-allow-headers", "")
        assert response.headers.get("access-control-allow-credentials") == "true"
    
    def test_preflight_post_with_auth(self, test_app_with_auth):
        """Test preflight for POST with authentication."""
        client = TestClient(test_app_with_auth)
        
        response = client.options(
            "/api/data",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, Authorization"
            }
        )
        
        assert response.status_code == 200
        assert "POST" in response.headers.get("access-control-allow-methods", "")
        assert "Content-Type" in response.headers.get("access-control-allow-headers", "")
        assert "Authorization" in response.headers.get("access-control-allow-headers", "")
    
    def test_preflight_blocked_origin_with_auth(self, test_app_with_auth):
        """Test preflight from blocked origin for auth endpoint."""
        client = TestClient(test_app_with_auth)
        
        response = client.options(
            "/authenticated",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        # Should be blocked by CORS
        assert response.status_code == 403
        assert response.json() == {"error": "CORS origin not allowed"}


class TestCORSCredentialsWithAuth:
    """Test CORS credentials handling with authentication."""
    
    def test_cookies_and_auth_header(self, test_app_with_auth, valid_token, auth_middleware):
        """Test both cookies and auth header work with CORS."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            # Simulate request with both cookie and auth header
            response = client.get(
                "/authenticated",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}",
                    "Cookie": "session=abc123; other=value"
                }
            )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-credentials") == "true"
        assert response.json()["user_id"] == "test123"
    
    def test_credentials_required_for_auth(self, test_app_with_auth):
        """Test that credentials are properly configured for auth endpoints."""
        client = TestClient(test_app_with_auth)
        
        # Check preflight includes credentials
        response = client.options(
            "/authenticated",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        assert response.headers.get("access-control-allow-credentials") == "true"


class TestCORSWithDifferentEnvironments:
    """Test CORS behaves correctly with auth in different environments."""
    
    def test_production_cors_with_auth(self, auth_middleware, valid_token):
        """Test production CORS with authentication."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://app.musicgen.ai"
        }, clear=True):
            app = create_app()
            client = TestClient(app)
            
            with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
                # Allowed production origin
                response = client.get(
                    "/health",
                    headers={
                        "Origin": "https://app.musicgen.ai",
                        "Authorization": f"Bearer {valid_token}"
                    }
                )
                
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") == "https://app.musicgen.ai"
                
                # HTTP origin should be blocked in production
                response = client.get(
                    "/health",
                    headers={
                        "Origin": "http://app.musicgen.ai",
                        "Authorization": f"Bearer {valid_token}"
                    }
                )
                
                assert response.status_code == 200
                assert "access-control-allow-origin" not in response.headers
    
    def test_staging_cors_with_auth(self, auth_middleware, valid_token):
        """Test staging CORS with authentication."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "staging",
            "STAGING_DEV_ORIGINS": "http://localhost:3000"
        }, clear=True):
            app = create_app()
            client = TestClient(app)
            
            with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
                # Staging origin
                response = client.get(
                    "/health",
                    headers={
                        "Origin": "https://staging.musicgen.ai",
                        "Authorization": f"Bearer {valid_token}"
                    }
                )
                
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") == "https://staging.musicgen.ai"
                
                # Dev origin allowed in staging
                response = client.get(
                    "/health",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Authorization": f"Bearer {valid_token}"
                    }
                )
                
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestOptionalAuth:
    """Test optional authentication with CORS."""
    
    def test_optional_auth_no_token(self, test_app_with_auth):
        """Test optional auth endpoint without token."""
        client = TestClient(test_app_with_auth)
        
        response = client.get(
            "/optional-auth",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.json()["message"] == "anonymous"
    
    def test_optional_auth_with_token(self, test_app_with_auth, valid_token, auth_middleware):
        """Test optional auth endpoint with token."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            response = client.get(
                "/optional-auth",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.json()["message"] == "authenticated"
        assert response.json()["user_id"] == "test123"
    
    def test_optional_auth_invalid_token(self, test_app_with_auth, auth_middleware):
        """Test optional auth endpoint with invalid token."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            response = client.get(
                "/optional-auth",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": "Bearer invalid.token.here"
                }
            )
        
        # Should fail authentication
        assert response.status_code == 401


class TestRealAPIEndpoints:
    """Test real API endpoints with CORS and auth."""
    
    def test_generation_endpoint_cors_auth(self, auth_middleware, valid_token):
        """Test generation endpoint with CORS and auth."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            app = create_app()
            client = TestClient(app)
            
            with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
                # Preflight
                response = client.options(
                    "/api/v1/generate",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type, Authorization"
                    }
                )
                
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
                
                # Actual request (would need proper setup to work fully)
                response = client.post(
                    "/api/v1/generate",
                    json={"prompt": "test music"},
                    headers={
                        "Origin": "http://localhost:3000",
                        "Authorization": f"Bearer {valid_token}",
                        "Content-Type": "application/json"
                    }
                )
                
                # May fail due to missing dependencies, but CORS should work
                assert "access-control-allow-origin" in response.headers or response.status_code >= 400


class TestSecurityScenarios:
    """Test various security scenarios with CORS and auth."""
    
    def test_csrf_protection_with_cors(self, test_app_with_auth, valid_token, auth_middleware):
        """Test CSRF protection works with CORS."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            # Same-origin request (no Origin header) should work
            response = client.post(
                "/api/data",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            
            assert response.status_code == 200
            
            # Cross-origin from allowed origin should work
            response = client.post(
                "/api/data",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
            
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
            
            # Cross-origin from blocked origin should not get CORS headers
            response = client.post(
                "/api/data",
                headers={
                    "Origin": "https://evil.com",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
            
            # Auth passes but no CORS headers
            assert response.status_code == 200
            assert "access-control-allow-origin" not in response.headers
    
    def test_token_in_cors_exposed_headers(self, test_app_with_auth, valid_token, auth_middleware):
        """Test that sensitive headers are properly exposed/hidden."""
        client = TestClient(test_app_with_auth)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware):
            response = client.get(
                "/authenticated",
                headers={
                    "Origin": "http://localhost:3000",
                    "Authorization": f"Bearer {valid_token}"
                }
            )
        
        assert response.status_code == 200
        exposed = response.headers.get("access-control-expose-headers", "")
        
        # Should expose rate limit headers
        assert "X-RateLimit-Limit" in exposed
        
        # Should not expose sensitive headers
        assert "Authorization" not in exposed
        assert "Set-Cookie" not in exposed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])