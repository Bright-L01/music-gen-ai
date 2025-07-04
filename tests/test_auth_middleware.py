"""
Comprehensive tests for JWT authentication middleware.
Ensures 100% code coverage for all authentication functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, JWTError
import redis
from pydantic import ValidationError

from music_gen.api.middleware.auth import (
    AuthenticationMiddleware,
    UserRole,
    TokenType,
    UserClaims,
    RoleChecker,
    TierChecker,
    auth_middleware,
    get_current_user,
    require_auth,
    require_admin,
    require_user,
    require_premium,
    require_moderator,
    require_developer,
    require_pro_tier,
    require_enterprise_tier,
    logout_user,
    refresh_token,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
)
from music_gen.core.exceptions import AuthenticationError, AuthorizationError


# Test fixtures
@pytest.fixture
def auth_middleware_instance():
    """Create AuthenticationMiddleware instance."""
    return AuthenticationMiddleware()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.ping.return_value = True
    client.exists.return_value = 0
    client.setex.return_value = True
    return client


@pytest.fixture
def valid_user_claims():
    """Create valid user claims."""
    now = datetime.now(timezone.utc)
    return UserClaims(
        user_id="user123",
        email="test@example.com",
        username="testuser",
        roles=[UserRole.USER],
        tier="free",
        is_verified=True,
        token_type=TokenType.ACCESS,
        issued_at=now,
        expires_at=now + timedelta(minutes=15),
        jti="test_jwt_id"
    )


@pytest.fixture
def test_app():
    """Create test FastAPI app with auth endpoints."""
    app = FastAPI()
    
    @app.get("/public")
    async def public_endpoint():
        return {"message": "public"}
    
    @app.get("/optional-auth")
    async def optional_auth(user: Optional[UserClaims] = Depends(get_current_user)):
        if user:
            return {"message": "authenticated", "user_id": user.user_id}
        return {"message": "anonymous"}
    
    @app.get("/protected")
    async def protected_endpoint(user: UserClaims = Depends(require_auth)):
        return {"message": "protected", "user_id": user.user_id}
    
    @app.get("/admin")
    async def admin_endpoint(user: UserClaims = Depends(require_admin())):
        return {"message": "admin", "user_id": user.user_id}
    
    @app.get("/premium")
    async def premium_endpoint(user: UserClaims = Depends(require_premium())):
        return {"message": "premium", "user_id": user.user_id}
    
    @app.get("/pro-tier")
    async def pro_tier_endpoint(user: UserClaims = Depends(require_pro_tier())):
        return {"message": "pro", "user_id": user.user_id}
    
    @app.post("/logout")
    async def logout_endpoint(result = Depends(logout_user)):
        return result
    
    @app.post("/refresh")
    async def refresh_endpoint(token: str):
        return await refresh_token(token)
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestUserClaims:
    """Test UserClaims model."""
    
    def test_parse_timestamp_int(self):
        """Test timestamp parsing from int."""
        timestamp = int(datetime.now(timezone.utc).timestamp())
        claims = UserClaims(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=timestamp,
            expires_at=timestamp + 900,
        )
        assert isinstance(claims.issued_at, datetime)
        assert isinstance(claims.expires_at, datetime)
    
    def test_parse_timestamp_datetime(self):
        """Test timestamp parsing from datetime."""
        now = datetime.now(timezone.utc)
        claims = UserClaims(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        assert claims.issued_at == now
        assert claims.expires_at == now + timedelta(minutes=15)
    
    def test_parse_roles_string(self):
        """Test role parsing from string."""
        claims = UserClaims(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles="admin",  # Single string
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        assert claims.roles == [UserRole.ADMIN]
    
    def test_parse_roles_list_strings(self):
        """Test role parsing from list of strings."""
        claims = UserClaims(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=["admin", "user"],  # List of strings
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        assert claims.roles == [UserRole.ADMIN, UserRole.USER]
    
    def test_parse_roles_mixed(self):
        """Test role parsing from mixed list."""
        claims = UserClaims(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=["admin", UserRole.USER],  # Mixed types
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        assert claims.roles == [UserRole.ADMIN, UserRole.USER]


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware class."""
    
    def test_initialization(self, auth_middleware_instance):
        """Test middleware initialization."""
        assert auth_middleware_instance.secret_key == JWT_SECRET_KEY
        assert auth_middleware_instance.algorithm == JWT_ALGORITHM
    
    def test_create_access_token(self, auth_middleware_instance):
        """Test access token creation."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER, UserRole.PREMIUM_USER],
            tier="pro",
            is_verified=True,
            expires_delta=30  # 30 minutes
        )
        
        # Decode and verify token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"
        assert payload["username"] == "testuser"
        assert payload["roles"] == ["user", "premium_user"]
        assert payload["tier"] == "pro"
        assert payload["is_verified"] is True
        assert payload["token_type"] == "access"
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload
    
    def test_create_access_token_default_expiry(self, auth_middleware_instance):
        """Test access token with default expiry."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        iat = payload["iat"]
        exp = payload["exp"]
        # Default expiry should be 15 minutes
        assert (exp - iat) == JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    def test_create_access_token_error(self, auth_middleware_instance, monkeypatch):
        """Test access token creation error."""
        # Mock jwt.encode to raise exception
        def mock_encode(*args, **kwargs):
            raise Exception("Encoding error")
        
        monkeypatch.setattr(jwt, "encode", mock_encode)
        
        with pytest.raises(AuthenticationError, match="Failed to create access token"):
            auth_middleware_instance.create_access_token(
                user_id="user123",
                email="test@example.com",
                username="testuser",
                roles=[UserRole.USER]
            )
    
    def test_create_refresh_token(self, auth_middleware_instance):
        """Test refresh token creation."""
        token = auth_middleware_instance.create_refresh_token(
            user_id="user123",
            expires_delta=60  # 60 minutes
        )
        
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == "user123"
        assert payload["token_type"] == "refresh"
        assert "iat" in payload
        assert "exp" in payload
        assert payload["jti"].startswith("refresh_user123_")
    
    def test_create_refresh_token_default_expiry(self, auth_middleware_instance):
        """Test refresh token with default expiry."""
        token = auth_middleware_instance.create_refresh_token(user_id="user123")
        
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        iat = payload["iat"]
        exp = payload["exp"]
        # Default expiry should be 7 days
        expected_seconds = JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        assert abs((exp - iat) - expected_seconds) < 2  # Allow 2 second tolerance
    
    def test_create_refresh_token_error(self, auth_middleware_instance, monkeypatch):
        """Test refresh token creation error."""
        def mock_encode(*args, **kwargs):
            raise Exception("Encoding error")
        
        monkeypatch.setattr(jwt, "encode", mock_encode)
        
        with pytest.raises(AuthenticationError, match="Failed to create refresh token"):
            auth_middleware_instance.create_refresh_token(user_id="user123")
    
    def test_verify_token_valid(self, auth_middleware_instance):
        """Test valid token verification."""
        # Create a valid token
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            tier="free",
            is_verified=True
        )
        
        # Verify token
        claims = auth_middleware_instance.verify_token(token)
        assert claims.user_id == "user123"
        assert claims.email == "test@example.com"
        assert claims.username == "testuser"
        assert claims.roles == [UserRole.USER]
        assert claims.tier == "free"
        assert claims.is_verified is True
        assert claims.token_type == TokenType.ACCESS
    
    def test_verify_token_expired(self, auth_middleware_instance):
        """Test expired token verification."""
        # Create an expired token
        now = datetime.now(timezone.utc)
        expired_time = now - timedelta(hours=1)
        
        payload = {
            "sub": "user123",
            "email": "test@example.com",
            "username": "testuser",
            "roles": ["user"],
            "tier": "free",
            "is_verified": True,
            "token_type": "access",
            "iat": expired_time.timestamp(),
            "exp": (expired_time + timedelta(minutes=15)).timestamp(),
            "jti": "expired_token"
        }
        
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        with pytest.raises(AuthenticationError, match="Token has expired"):
            auth_middleware_instance.verify_token(expired_token)
    
    def test_verify_token_invalid(self, auth_middleware_instance):
        """Test invalid token verification."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            auth_middleware_instance.verify_token(invalid_token)
    
    def test_verify_token_wrong_secret(self, auth_middleware_instance):
        """Test token with wrong secret."""
        wrong_secret_token = jwt.encode(
            {"sub": "user123"}, 
            "wrong_secret", 
            algorithm=JWT_ALGORITHM
        )
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            auth_middleware_instance.verify_token(wrong_secret_token)
    
    def test_verify_token_blacklisted(self, auth_middleware_instance, mock_redis_client):
        """Test blacklisted token verification."""
        auth_middleware_instance.redis_client = mock_redis_client
        mock_redis_client.exists.return_value = 1  # Token is blacklisted
        
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            auth_middleware_instance.verify_token(token)
    
    def test_verify_token_missing_user_id(self, auth_middleware_instance):
        """Test token without user ID."""
        payload = {
            "email": "test@example.com",
            "roles": ["user"],
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": (datetime.now(timezone.utc) + timedelta(minutes=15)).timestamp(),
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        with pytest.raises(AuthenticationError, match="Invalid token: missing user ID"):
            auth_middleware_instance.verify_token(token)
    
    def test_verify_token_unverified_user(self, auth_middleware_instance):
        """Test unverified user token."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            is_verified=False  # Unverified user
        )
        
        with pytest.raises(AuthenticationError, match="Email verification required"):
            auth_middleware_instance.verify_token(token)
    
    def test_verify_token_refresh_unverified_allowed(self, auth_middleware_instance):
        """Test refresh token for unverified user is allowed."""
        payload = {
            "sub": "user123",
            "token_type": "refresh",
            "is_verified": False,
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": (datetime.now(timezone.utc) + timedelta(days=7)).timestamp(),
            "jti": "refresh_token"
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Should not raise exception for refresh tokens
        claims = auth_middleware_instance.verify_token(token)
        assert claims.user_id == "user123"
        assert claims.token_type == TokenType.REFRESH
        assert claims.is_verified is False
    
    def test_verify_token_generic_exception(self, auth_middleware_instance, monkeypatch):
        """Test generic exception during verification."""
        def mock_decode(*args, **kwargs):
            raise Exception("Unexpected error")
        
        monkeypatch.setattr(jwt, "decode", mock_decode)
        
        with pytest.raises(AuthenticationError, match="Token verification failed"):
            auth_middleware_instance.verify_token("some.token.here")
    
    def test_is_token_blacklisted_no_redis(self, auth_middleware_instance):
        """Test blacklist check without Redis."""
        auth_middleware_instance.redis_client = None
        assert auth_middleware_instance._is_token_blacklisted("token123") is False
    
    def test_is_token_blacklisted_no_jti(self, auth_middleware_instance, mock_redis_client):
        """Test blacklist check without JTI."""
        auth_middleware_instance.redis_client = mock_redis_client
        assert auth_middleware_instance._is_token_blacklisted(None) is False
    
    def test_is_token_blacklisted_redis_error(self, auth_middleware_instance, mock_redis_client):
        """Test blacklist check with Redis error."""
        auth_middleware_instance.redis_client = mock_redis_client
        mock_redis_client.exists.side_effect = Exception("Redis error")
        
        # Should return False on error
        assert auth_middleware_instance._is_token_blacklisted("token123") is False
    
    def test_blacklist_token_success(self, auth_middleware_instance, mock_redis_client):
        """Test successful token blacklisting."""
        auth_middleware_instance.redis_client = mock_redis_client
        
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        result = auth_middleware_instance.blacklist_token("token123", expires_at)
        
        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args[0]
        assert call_args[0] == "blacklist:token123"
        assert call_args[2] == "revoked"
        # TTL should be approximately 3600 seconds (1 hour)
        assert 3595 <= call_args[1] <= 3605
    
    def test_blacklist_token_no_redis(self, auth_middleware_instance):
        """Test blacklisting without Redis."""
        auth_middleware_instance.redis_client = None
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        result = auth_middleware_instance.blacklist_token("token123", expires_at)
        assert result is False
    
    def test_blacklist_token_no_jti(self, auth_middleware_instance, mock_redis_client):
        """Test blacklisting without JTI."""
        auth_middleware_instance.redis_client = mock_redis_client
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        result = auth_middleware_instance.blacklist_token(None, expires_at)
        assert result is False
    
    def test_blacklist_token_expired(self, auth_middleware_instance, mock_redis_client):
        """Test blacklisting already expired token."""
        auth_middleware_instance.redis_client = mock_redis_client
        expires_at = datetime.now(timezone.utc) - timedelta(hours=1)  # Already expired
        
        result = auth_middleware_instance.blacklist_token("token123", expires_at)
        assert result is False
        mock_redis_client.setex.assert_not_called()
    
    def test_blacklist_token_redis_error(self, auth_middleware_instance, mock_redis_client):
        """Test blacklisting with Redis error."""
        auth_middleware_instance.redis_client = mock_redis_client
        mock_redis_client.setex.side_effect = Exception("Redis error")
        
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        result = auth_middleware_instance.blacklist_token("token123", expires_at)
        assert result is False
    
    def test_refresh_access_token_success(self, auth_middleware_instance):
        """Test successful token refresh."""
        # Create a refresh token
        refresh_token = auth_middleware_instance.create_refresh_token("user123")
        
        # Mock the token to include all required fields
        payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        payload.update({
            "email": "test@example.com",
            "username": "testuser",
            "roles": ["user"],
            "tier": "pro",
            "is_verified": True
        })
        refresh_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Refresh tokens
        new_access, new_refresh = auth_middleware_instance.refresh_access_token(refresh_token)
        
        # Verify new tokens
        access_payload = jwt.decode(new_access, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert access_payload["sub"] == "user123"
        assert access_payload["token_type"] == "access"
        
        refresh_payload = jwt.decode(new_refresh, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert refresh_payload["sub"] == "user123"
        assert refresh_payload["token_type"] == "refresh"
    
    def test_refresh_access_token_invalid_type(self, auth_middleware_instance):
        """Test refresh with access token instead of refresh token."""
        # Create an access token
        access_token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        with pytest.raises(AuthenticationError, match="Invalid refresh token"):
            auth_middleware_instance.refresh_access_token(access_token)
    
    def test_refresh_access_token_with_blacklist(self, auth_middleware_instance, mock_redis_client):
        """Test token refresh with blacklisting."""
        auth_middleware_instance.redis_client = mock_redis_client
        
        # Create a refresh token with all fields
        payload = {
            "sub": "user123",
            "email": "test@example.com",
            "username": "testuser",
            "roles": ["user"],
            "tier": "pro",
            "is_verified": True,
            "token_type": "refresh",
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": (datetime.now(timezone.utc) + timedelta(days=7)).timestamp(),
            "jti": "refresh_123"
        }
        refresh_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Refresh tokens
        new_access, new_refresh = auth_middleware_instance.refresh_access_token(refresh_token)
        
        # Verify old token was blacklisted
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args[0]
        assert call_args[0] == "blacklist:refresh_123"
    
    def test_refresh_access_token_error(self, auth_middleware_instance):
        """Test token refresh error."""
        with pytest.raises(AuthenticationError, match="Token refresh failed"):
            auth_middleware_instance.refresh_access_token("invalid.token")


class TestRoleChecker:
    """Test RoleChecker class."""
    
    def test_require_all_roles_success(self, valid_user_claims):
        """Test requiring all roles - success case."""
        valid_user_claims.roles = [UserRole.USER, UserRole.PREMIUM_USER]
        checker = RoleChecker([UserRole.USER, UserRole.PREMIUM_USER], require_all=True)
        
        result = checker(valid_user_claims)
        assert result == valid_user_claims
    
    def test_require_all_roles_failure(self, valid_user_claims):
        """Test requiring all roles - failure case."""
        valid_user_claims.roles = [UserRole.USER]
        checker = RoleChecker([UserRole.USER, UserRole.ADMIN], require_all=True)
        
        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            checker(valid_user_claims)
    
    def test_require_any_role_success(self, valid_user_claims):
        """Test requiring any role - success case."""
        valid_user_claims.roles = [UserRole.USER]
        checker = RoleChecker([UserRole.USER, UserRole.ADMIN], require_all=False)
        
        result = checker(valid_user_claims)
        assert result == valid_user_claims
    
    def test_require_any_role_failure(self, valid_user_claims):
        """Test requiring any role - failure case."""
        valid_user_claims.roles = [UserRole.USER]
        checker = RoleChecker([UserRole.ADMIN, UserRole.MODERATOR], require_all=False)
        
        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            checker(valid_user_claims)
    
    def test_no_user_claims(self):
        """Test with no user claims."""
        checker = RoleChecker([UserRole.USER])
        
        with pytest.raises(AuthorizationError, match="Authentication required"):
            checker(None)


class TestTierChecker:
    """Test TierChecker class."""
    
    def test_tier_allowed(self, valid_user_claims):
        """Test allowed tier."""
        valid_user_claims.tier = "pro"
        checker = TierChecker(["pro", "enterprise"])
        
        result = checker(valid_user_claims)
        assert result == valid_user_claims
    
    def test_tier_not_allowed(self, valid_user_claims):
        """Test disallowed tier."""
        valid_user_claims.tier = "free"
        checker = TierChecker(["pro", "enterprise"])
        
        with pytest.raises(AuthorizationError, match="Subscription upgrade required"):
            checker(valid_user_claims)
    
    def test_no_user_claims(self):
        """Test with no user claims."""
        checker = TierChecker(["pro"])
        
        with pytest.raises(AuthorizationError, match="Authentication required"):
            checker(None)


class TestDependencyFunctions:
    """Test dependency functions."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_with_bearer_token(self, auth_middleware_instance):
        """Test get_current_user with bearer token."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        request = Mock(spec=Request)
        request.state = Mock()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            user = await get_current_user(request, credentials)
        
        assert user.user_id == "user123"
        assert request.state.user_id == "user123"
        assert request.state.user_roles == ["user"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_with_oauth2_token(self, auth_middleware_instance):
        """Test get_current_user with OAuth2 token."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        request = Mock(spec=Request)
        request.state = Mock()
        
        # Mock oauth2_scheme to return token
        with patch('music_gen.api.middleware.auth.oauth2_scheme', return_value=token):
            with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
                user = await get_current_user(request, None)
        
        assert user.user_id == "user123"
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Test get_current_user without token."""
        request = Mock(spec=Request)
        
        with patch('music_gen.api.middleware.auth.oauth2_scheme', return_value=None):
            user = await get_current_user(request, None)
        
        assert user is None
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, auth_middleware_instance):
        """Test get_current_user with invalid token."""
        request = Mock(spec=Request)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token")
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, credentials)
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_current_user_unexpected_error(self, auth_middleware_instance):
        """Test get_current_user with unexpected error."""
        request = Mock(spec=Request)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
        
        # Mock verify_token to raise unexpected exception
        auth_middleware_instance.verify_token = Mock(side_effect=Exception("Unexpected"))
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, credentials)
        
        assert exc_info.value.status_code == 500
        assert "Authentication service unavailable" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_require_auth_success(self, valid_user_claims):
        """Test require_auth with valid user."""
        user = await require_auth(valid_user_claims)
        assert user == valid_user_claims
    
    @pytest.mark.asyncio
    async def test_require_auth_no_user(self):
        """Test require_auth without user."""
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(None)
        
        assert exc_info.value.status_code == 401
        assert "Authentication required" in str(exc_info.value.detail)
    
    def test_require_admin(self):
        """Test require_admin function."""
        checker = require_admin()
        assert isinstance(checker, RoleChecker)
        assert UserRole.ADMIN in checker.required_roles
    
    def test_require_user(self):
        """Test require_user function."""
        checker = require_user()
        assert isinstance(checker, RoleChecker)
        assert UserRole.USER in checker.required_roles
        assert UserRole.PREMIUM_USER in checker.required_roles
        assert UserRole.ADMIN in checker.required_roles
    
    def test_require_premium(self):
        """Test require_premium function."""
        checker = require_premium()
        assert isinstance(checker, RoleChecker)
        assert UserRole.PREMIUM_USER in checker.required_roles
        assert UserRole.ADMIN in checker.required_roles
    
    def test_require_moderator(self):
        """Test require_moderator function."""
        checker = require_moderator()
        assert isinstance(checker, RoleChecker)
        assert UserRole.MODERATOR in checker.required_roles
        assert UserRole.ADMIN in checker.required_roles
    
    def test_require_developer(self):
        """Test require_developer function."""
        checker = require_developer()
        assert isinstance(checker, RoleChecker)
        assert UserRole.DEVELOPER in checker.required_roles
        assert UserRole.ADMIN in checker.required_roles
    
    def test_require_pro_tier(self):
        """Test require_pro_tier function."""
        checker = require_pro_tier()
        assert isinstance(checker, TierChecker)
        assert "pro" in checker.required_tiers
        assert "enterprise" in checker.required_tiers
    
    def test_require_enterprise_tier(self):
        """Test require_enterprise_tier function."""
        checker = require_enterprise_tier()
        assert isinstance(checker, TierChecker)
        assert "enterprise" in checker.required_tiers
    
    @pytest.mark.asyncio
    async def test_logout_user_success(self, valid_user_claims, auth_middleware_instance):
        """Test successful logout."""
        auth_middleware_instance.blacklist_token = Mock(return_value=True)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            result = await logout_user(valid_user_claims)
        
        assert result["message"] == "Logged out successfully"
        auth_middleware_instance.blacklist_token.assert_called_once_with(
            valid_user_claims.jti, valid_user_claims.expires_at
        )
    
    @pytest.mark.asyncio
    async def test_logout_user_blacklist_failed(self, valid_user_claims, auth_middleware_instance):
        """Test logout with blacklist failure."""
        auth_middleware_instance.blacklist_token = Mock(return_value=False)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            result = await logout_user(valid_user_claims)
        
        assert "token blacklist unavailable" in result["message"]
    
    @pytest.mark.asyncio
    async def test_logout_user_no_jti(self, valid_user_claims):
        """Test logout without JTI."""
        valid_user_claims.jti = None
        
        result = await logout_user(valid_user_claims)
        assert result["message"] == "Logout completed"
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, auth_middleware_instance):
        """Test successful token refresh."""
        new_access = "new.access.token"
        new_refresh = "new.refresh.token"
        auth_middleware_instance.refresh_access_token = Mock(
            return_value=(new_access, new_refresh)
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            result = await refresh_token("old.refresh.token")
        
        assert result["access_token"] == new_access
        assert result["refresh_token"] == new_refresh
        assert result["token_type"] == "bearer"
        assert result["expires_in"] == JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    @pytest.mark.asyncio
    async def test_refresh_token_error(self, auth_middleware_instance):
        """Test token refresh error."""
        auth_middleware_instance.refresh_access_token = Mock(
            side_effect=AuthenticationError("Invalid refresh token")
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            with pytest.raises(HTTPException) as exc_info:
                await refresh_token("invalid.token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid refresh token" in str(exc_info.value.detail)


class TestIntegration:
    """Integration tests with FastAPI."""
    
    def test_public_endpoint(self, client):
        """Test public endpoint access."""
        response = client.get("/public")
        assert response.status_code == 200
        assert response.json() == {"message": "public"}
    
    def test_optional_auth_anonymous(self, client):
        """Test optional auth endpoint without token."""
        response = client.get("/optional-auth")
        assert response.status_code == 200
        assert response.json() == {"message": "anonymous"}
    
    def test_optional_auth_authenticated(self, client, auth_middleware_instance):
        """Test optional auth endpoint with token."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/optional-auth",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "authenticated"
        assert data["user_id"] == "user123"
    
    def test_protected_endpoint_no_auth(self, client):
        """Test protected endpoint without authentication."""
        response = client.get("/protected")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_auth(self, client, auth_middleware_instance):
        """Test protected endpoint with authentication."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/protected",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "protected"
        assert data["user_id"] == "user123"
    
    def test_admin_endpoint_as_user(self, client, auth_middleware_instance):
        """Test admin endpoint as regular user."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]  # Not admin
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/admin",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 403
    
    def test_admin_endpoint_as_admin(self, client, auth_middleware_instance):
        """Test admin endpoint as admin."""
        token = auth_middleware_instance.create_access_token(
            user_id="admin123",
            email="admin@example.com",
            username="admin",
            roles=[UserRole.ADMIN]
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/admin",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "admin"
        assert data["user_id"] == "admin123"
    
    def test_premium_endpoint_as_free_user(self, client, auth_middleware_instance):
        """Test premium endpoint as free user."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]  # Not premium
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/premium",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 403
    
    def test_premium_endpoint_as_premium_user(self, client, auth_middleware_instance):
        """Test premium endpoint as premium user."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.PREMIUM_USER]
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/premium",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "premium"
    
    def test_pro_tier_endpoint_as_free_tier(self, client, auth_middleware_instance):
        """Test pro tier endpoint as free tier user."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            tier="free"  # Free tier
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/pro-tier",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 403
    
    def test_pro_tier_endpoint_as_pro_tier(self, client, auth_middleware_instance):
        """Test pro tier endpoint as pro tier user."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER],
            tier="pro"  # Pro tier
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.get(
                "/pro-tier",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pro"
    
    def test_logout_endpoint(self, client, auth_middleware_instance):
        """Test logout endpoint."""
        token = auth_middleware_instance.create_access_token(
            user_id="user123",
            email="test@example.com",
            username="testuser",
            roles=[UserRole.USER]
        )
        
        auth_middleware_instance.blacklist_token = Mock(return_value=True)
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.post(
                "/logout",
                headers={"Authorization": f"Bearer {token}"}
            )
        
        assert response.status_code == 200
        assert "Logged out successfully" in response.json()["message"]
    
    def test_refresh_endpoint(self, client, auth_middleware_instance):
        """Test refresh endpoint."""
        auth_middleware_instance.refresh_access_token = Mock(
            return_value=("new.access.token", "new.refresh.token")
        )
        
        with patch('music_gen.api.middleware.auth.auth_middleware', auth_middleware_instance):
            response = client.post(
                "/refresh",
                json={"token": "refresh.token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new.access.token"
        assert data["refresh_token"] == "new.refresh.token"
        assert data["token_type"] == "bearer"


class TestRedisIntegration:
    """Test Redis integration."""
    
    @patch('music_gen.api.middleware.auth.redis.Redis')
    def test_redis_connection_success(self, mock_redis_class):
        """Test successful Redis connection."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        # Re-import to trigger Redis connection
        from music_gen.api.middleware.auth import redis_client
        
        mock_redis_class.assert_called_once()
        mock_redis.ping.assert_called_once()
    
    @patch('music_gen.api.middleware.auth.redis.Redis')
    def test_redis_connection_failure(self, mock_redis_class):
        """Test Redis connection failure."""
        mock_redis_class.side_effect = redis.ConnectionError("Connection failed")
        
        # Re-import to trigger Redis connection
        import importlib
        import music_gen.api.middleware.auth
        importlib.reload(music_gen.api.middleware.auth)
        
        # Should handle error gracefully
        assert True  # If we get here, error was handled


class TestEnums:
    """Test enum values."""
    
    def test_user_roles(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"
        assert UserRole.PREMIUM_USER.value == "premium_user"
        assert UserRole.MODERATOR.value == "moderator"
        assert UserRole.DEVELOPER.value == "developer"
    
    def test_token_types(self):
        """Test TokenType enum values."""
        assert TokenType.ACCESS.value == "access"
        assert TokenType.REFRESH.value == "refresh"


# Run tests with coverage
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=music_gen.api.middleware.auth",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-fail-under=100"
    ])