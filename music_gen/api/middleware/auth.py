"""
Production-ready JWT Authentication Middleware for MusicGen AI
Implements secure JWT validation with RBAC support and proper error handling.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from collections import defaultdict
import time
import json
import secrets
import uuid

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt, ExpiredSignatureError
from pydantic import BaseModel, validator
import redis
from sqlalchemy.orm import Session

from music_gen.core.config import get_config
from music_gen.core.exceptions import AuthenticationError, AuthorizationError
from music_gen.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Configuration
config = get_config()

# JWT Configuration
JWT_SECRET_KEY = config.jwt_secret_key or "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

# OAuth2 scheme for FastAPI integration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    auto_error=False  # We'll handle errors manually for better control
)

# HTTP Bearer scheme for Authorization header
security = HTTPBearer(auto_error=False)

# Redis connection for token blacklist
try:
    redis_client = redis.Redis(
        host=config.redis_host or "localhost",
        port=config.redis_port or 6379,
        db=config.redis_db or 0,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
except (redis.ConnectionError, AttributeError):
    logger.warning("Redis not available. Token blacklist will be disabled.")
    redis_client = None


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM_USER = "premium_user"
    MODERATOR = "moderator"
    DEVELOPER = "developer"


class Permission(str, Enum):
    """Granular permissions for fine-grained access control"""
    # Music generation permissions
    GENERATE_MUSIC = "music:generate"
    GENERATE_PREMIUM = "music:generate:premium"
    GENERATE_BATCH = "music:generate:batch"
    
    # Model management permissions
    VIEW_MODELS = "models:view"
    MANAGE_MODELS = "models:manage"
    UPLOAD_MODELS = "models:upload"
    
    # User management permissions
    VIEW_USERS = "users:view"
    MANAGE_USERS = "users:manage"
    DELETE_USERS = "users:delete"
    
    # Analytics permissions
    VIEW_ANALYTICS = "analytics:view"
    EXPORT_ANALYTICS = "analytics:export"
    
    # System permissions
    VIEW_LOGS = "system:logs:view"
    MANAGE_SYSTEM = "system:manage"
    
    # Admin wildcard
    ADMIN_ALL = "admin:*"


class TokenType(str, Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"


# Role to permission mapping
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.USER: [
        Permission.GENERATE_MUSIC,
        Permission.VIEW_MODELS,
    ],
    UserRole.PREMIUM_USER: [
        Permission.GENERATE_MUSIC,
        Permission.GENERATE_PREMIUM,
        Permission.GENERATE_BATCH,
        Permission.VIEW_MODELS,
    ],
    UserRole.MODERATOR: [
        Permission.GENERATE_MUSIC,
        Permission.VIEW_MODELS,
        Permission.VIEW_USERS,
        Permission.VIEW_ANALYTICS,
    ],
    UserRole.DEVELOPER: [
        Permission.GENERATE_MUSIC,
        Permission.GENERATE_PREMIUM,
        Permission.VIEW_MODELS,
        Permission.MANAGE_MODELS,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_LOGS,
    ],
    UserRole.ADMIN: [
        Permission.ADMIN_ALL,
    ],
}


class UserClaims(BaseModel):
    """User claims extracted from JWT token"""
    user_id: str
    email: str
    username: str
    roles: List[UserRole]
    tier: str
    is_verified: bool
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    jti: Optional[str] = None  # JWT ID for token revocation

    @validator('issued_at', 'expires_at', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v

    @validator('roles', pre=True)
    def parse_roles(cls, v):
        if isinstance(v, str):
            return [UserRole(v)]
        if isinstance(v, list):
            return [UserRole(role) if isinstance(role, str) else role for role in v]
        return v


class AuthenticationMiddleware:
    """JWT Authentication Middleware with comprehensive security features"""

    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.redis_client = redis_client

    def create_access_token(
        self,
        user_id: str,
        email: str,
        username: str,
        roles: List[UserRole],
        tier: str = "free",
        is_verified: bool = True,
        expires_delta: Optional[int] = None
    ) -> str:
        """Create a new access token"""
        if expires_delta is None:
            expires_delta = JWT_ACCESS_TOKEN_EXPIRE_MINUTES

        now = datetime.now(timezone.utc)
        expire = now.timestamp() + (expires_delta * 60)

        # Use UUID for unpredictable JTI
        import uuid
        jti = str(uuid.uuid4())

        payload = {
            "sub": user_id,
            "email": email,
            "username": username,
            "roles": [role.value for role in roles],
            "tier": tier,
            "is_verified": is_verified,
            "token_type": TokenType.ACCESS.value,
            "iat": now.timestamp(),
            "exp": expire,
            "jti": jti,
            "iss": "music-gen-auth",  # Issuer claim
            "aud": "music-gen-api",    # Audience claim
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            # Don't log user_id in production to prevent correlation attacks
            logger.debug("Access token created successfully")
            return token
        except Exception as e:
            logger.error("Failed to create access token")
            raise AuthenticationError("Authentication service unavailable")

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[int] = None
    ) -> str:
        """Create a new refresh token"""
        if expires_delta is None:
            expires_delta = JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60

        now = datetime.now(timezone.utc)
        expire = now.timestamp() + (expires_delta * 60)

        # Use UUID for unpredictable JTI
        import uuid
        jti = str(uuid.uuid4())

        payload = {
            "sub": user_id,
            "token_type": TokenType.REFRESH.value,
            "iat": now.timestamp(),
            "exp": expire,
            "jti": jti,
            "iss": "music-gen-auth",
            "aud": "music-gen-api",
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug("Refresh token created successfully")
            return token
        except Exception as e:
            logger.error("Failed to create refresh token")
            raise AuthenticationError("Authentication service unavailable")

    def verify_token(self, token: str) -> UserClaims:
        """
        Verify and decode JWT token with comprehensive validation
        
        Args:
            token: JWT token string
            
        Returns:
            UserClaims: Validated user claims
            
        Raises:
            AuthenticationError: If token is invalid, expired, or revoked
        """
        try:
            # Decode token with enhanced security options
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # Only allow specified algorithm
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "require": ["exp", "iat", "sub", "jti", "iss", "aud"]
                },
                audience="music-gen-api",
                issuer="music-gen-auth"
            )

            # Validate issued-at time (prevent future tokens)
            iat = payload.get("iat", 0)
            current_time = datetime.now(timezone.utc).timestamp()
            if iat > current_time + 60:  # Allow 1 minute clock skew
                logger.warning("Token issued in the future detected")
                raise AuthenticationError("Authentication failed")

            # Validate JTI format (should be UUID)
            jti = payload.get("jti")
            if jti:
                import uuid
                try:
                    uuid.UUID(jti)
                except ValueError:
                    logger.warning("Invalid JTI format detected")
                    raise AuthenticationError("Authentication failed")

            # Check if token is blacklisted
            if self._is_token_blacklisted(jti):
                logger.warning("Blacklisted token access attempt")
                raise AuthenticationError("Authentication failed")

            # Extract and validate claims
            user_claims = UserClaims(
                user_id=payload.get("sub"),
                email=payload.get("email", ""),
                username=payload.get("username", ""),
                roles=payload.get("roles", [UserRole.USER]),
                tier=payload.get("tier", "free"),
                is_verified=payload.get("is_verified", False),
                token_type=TokenType(payload.get("token_type", TokenType.ACCESS)),
                issued_at=payload.get("iat"),
                expires_at=payload.get("exp"),
                jti=jti
            )

            # Additional validation
            if not user_claims.user_id:
                logger.warning("Token missing user ID")
                raise AuthenticationError("Authentication failed")

            if not user_claims.is_verified and user_claims.token_type == TokenType.ACCESS:
                logger.warning("Unverified user access attempt")
                raise AuthenticationError("Authentication failed")

            logger.debug("Token verified successfully")
            return user_claims

        except ExpiredSignatureError:
            logger.debug("Token has expired")
            raise AuthenticationError("Authentication failed")
        except JWTError as e:
            logger.debug("JWT validation failed")
            raise AuthenticationError("Authentication failed")
        except AuthenticationError:
            # Re-raise our own errors
            raise
        except Exception as e:
            logger.error("Unexpected error during token verification")
            raise AuthenticationError("Authentication failed")

    def _is_token_blacklisted(self, jti: Optional[str]) -> bool:
        """Check if token is blacklisted in Redis"""
        if not self.redis_client or not jti:
            return False

        try:
            return self.redis_client.exists(f"blacklist:{jti}") > 0
        except Exception as e:
            logger.error(f"Failed to check token blacklist: {e}")
            return False

    def blacklist_token(self, jti: str, expires_at: datetime) -> bool:
        """Add token to blacklist"""
        if not self.redis_client or not jti:
            return False

        try:
            # Calculate TTL based on token expiration
            now = datetime.now(timezone.utc)
            ttl = max(int((expires_at - now).total_seconds()), 0)

            if ttl > 0:
                self.redis_client.setex(f"blacklist:{jti}", ttl, "revoked")
                logger.info(f"Token blacklisted: {jti}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Create new access token from refresh token
        
        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        try:
            # Verify refresh token
            claims = self.verify_token(refresh_token)
            
            if claims.token_type != TokenType.REFRESH:
                raise AuthenticationError("Invalid refresh token")

            # TODO: In production, fetch current user data from database
            # For now, we'll use basic claims
            new_access_token = self.create_access_token(
                user_id=claims.user_id,
                email=claims.email,
                username=claims.username,
                roles=claims.roles,
                tier=claims.tier,
                is_verified=claims.is_verified
            )

            # Create new refresh token
            new_refresh_token = self.create_refresh_token(claims.user_id)

            # Blacklist old refresh token
            if claims.jti:
                self.blacklist_token(claims.jti, claims.expires_at)

            logger.info(f"Tokens refreshed for user {claims.user_id}")
            return new_access_token, new_refresh_token

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Token refresh failed")


class SecurityAuditLogger:
    """Log security-relevant events for compliance and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
        # Set up a separate handler for security logs
        handler = logging.FileHandler("logs/security_audit.log")
        handler.setFormatter(
            logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}'
            )
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_authentication(
        self, 
        event_type: str, 
        user_id: Optional[str], 
        client_ip: str, 
        success: bool, 
        details: Optional[Dict[str, Any]] = None
    ):
        """Log authentication events"""
        event = {
            "type": "authentication",
            "event": event_type,
            "user_id": user_id,
            "client_ip": client_ip,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        self.logger.info(json.dumps(event))
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
        reason: Optional[str] = None
    ):
        """Log authorization decisions"""
        event = {
            "type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.logger.info(json.dumps(event))


class AuthRateLimiter:
    """Enhanced rate limiting for authentication endpoints"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.lockout_duration = 900  # 15 minutes
        self.max_attempts = 5
        self.attempt_window = 300  # 5 minutes
    
    def check_auth_limit(self, client_ip: str) -> bool:
        """Check if IP is locked out or has exceeded attempts"""
        if self.redis_client:
            return self._check_redis_limit(client_ip)
        else:
            return self._check_memory_limit(client_ip)
    
    def _check_memory_limit(self, client_ip: str) -> bool:
        """Check rate limit using in-memory storage"""
        now = time.time()
        
        # Clean old attempts
        self.failed_attempts[client_ip] = [
            ts for ts in self.failed_attempts[client_ip]
            if now - ts < self.attempt_window
        ]
        
        # Check if locked out
        attempts = self.failed_attempts[client_ip]
        if len(attempts) >= self.max_attempts:
            oldest_attempt = min(attempts)
            if now - oldest_attempt < self.lockout_duration:
                logger.warning(f"IP {client_ip} locked out due to failed attempts")
                return False
        
        return True
    
    def _check_redis_limit(self, client_ip: str) -> bool:
        """Check rate limit using Redis"""
        key = f"auth_attempts:{client_ip}"
        try:
            attempts = self.redis_client.llen(key)
            if attempts >= self.max_attempts:
                # Check if still in lockout period
                oldest = self.redis_client.lindex(key, 0)
                if oldest:
                    oldest_time = float(oldest)
                    if time.time() - oldest_time < self.lockout_duration:
                        return False
            return True
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True  # Fail open on Redis errors
    
    def record_failed_attempt(self, client_ip: str):
        """Record a failed authentication attempt"""
        if self.redis_client:
            self._record_redis_attempt(client_ip)
        else:
            self._record_memory_attempt(client_ip)
    
    def _record_memory_attempt(self, client_ip: str):
        """Record attempt in memory"""
        self.failed_attempts[client_ip].append(time.time())
    
    def _record_redis_attempt(self, client_ip: str):
        """Record attempt in Redis"""
        key = f"auth_attempts:{client_ip}"
        try:
            pipe = self.redis_client.pipeline()
            pipe.rpush(key, str(time.time()))
            pipe.expire(key, self.lockout_duration)
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to record auth attempt: {e}")


# Global instances
auth_middleware = AuthenticationMiddleware()
security_audit = SecurityAuditLogger()
auth_rate_limiter = AuthRateLimiter(redis_client)


class PermissionChecker:
    """Permission-based access control with role hierarchy support"""
    
    def __init__(self, required_permissions: List[Permission]):
        """
        Args:
            required_permissions: List of permissions required for access
        """
        self.required_permissions = set(required_permissions)
    
    def __call__(self, user_claims: UserClaims = Depends(lambda: None)) -> UserClaims:
        """Check if user has required permissions based on their roles"""
        if not user_claims:
            raise AuthorizationError("Authentication required")
        
        user_permissions = set()
        
        # Collect all permissions from user's roles
        for role in user_claims.roles:
            perms = ROLE_PERMISSIONS.get(role, [])
            for perm in perms:
                if perm == Permission.ADMIN_ALL:
                    # Admin has all permissions
                    security_audit.log_authorization(
                        user_id=user_claims.user_id,
                        resource="system",
                        action="admin_access",
                        allowed=True,
                        reason="Admin role has all permissions"
                    )
                    return user_claims
                user_permissions.add(perm)
        
        # Check if user has required permissions
        if not self.required_permissions.issubset(user_permissions):
            missing = self.required_permissions - user_permissions
            security_audit.log_authorization(
                user_id=user_claims.user_id,
                resource=str(self.required_permissions),
                action="access",
                allowed=False,
                reason=f"Missing permissions: {missing}"
            )
            raise AuthorizationError("Authentication failed")
        
        security_audit.log_authorization(
            user_id=user_claims.user_id,
            resource=str(self.required_permissions),
            action="access",
            allowed=True
        )
        return user_claims


class RoleChecker:
    """Role-based access control checker"""
    
    def __init__(self, required_roles: List[UserRole], require_all: bool = False):
        """
        Args:
            required_roles: List of roles required for access
            require_all: If True, user must have ALL roles. If False, user needs ANY role.
        """
        self.required_roles = set(required_roles)
        self.require_all = require_all

    def __call__(self, user_claims: UserClaims = Depends(lambda: None)) -> UserClaims:
        """Check if user has required roles"""
        if not user_claims:
            raise AuthorizationError("Authentication required")

        user_roles = set(user_claims.roles)
        
        if self.require_all:
            if not self.required_roles.issubset(user_roles):
                missing_roles = self.required_roles - user_roles
                security_audit.log_authorization(
                    user_id=user_claims.user_id,
                    resource="role_check",
                    action=f"require_all:{self.required_roles}",
                    allowed=False,
                    reason=f"Missing roles: {missing_roles}"
                )
                raise AuthorizationError("Authentication failed")
        else:
            if not self.required_roles.intersection(user_roles):
                security_audit.log_authorization(
                    user_id=user_claims.user_id,
                    resource="role_check",
                    action=f"require_any:{self.required_roles}",
                    allowed=False,
                    reason="No matching roles"
                )
                raise AuthorizationError("Authentication failed")
        
        security_audit.log_authorization(
            user_id=user_claims.user_id,
            resource="role_check",
            action=f"roles:{self.required_roles}",
            allowed=True
        )
        return user_claims


class TierChecker:
    """Subscription tier access control"""
    
    def __init__(self, required_tiers: List[str]):
        """
        Args:
            required_tiers: List of tiers that have access (e.g., ['pro', 'enterprise'])
        """
        self.required_tiers = set(required_tiers)

    def __call__(self, user_claims: UserClaims = Depends(lambda: None)) -> UserClaims:
        """Check if user's tier has access"""
        if not user_claims:
            raise AuthorizationError("Authentication required")

        if user_claims.tier not in self.required_tiers:
            logger.warning(
                f"Tier access denied for user {user_claims.user_id}. "
                f"User tier: {user_claims.tier}, Required: {self.required_tiers}"
            )
            raise AuthorizationError(
                f"Subscription upgrade required. Required tier: {self.required_tiers}"
            )

        return user_claims


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request, handling proxies"""
    # Check X-Forwarded-For header (common proxy header)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        else:
            # Fall back to direct connection
            client_ip = request.client.host if request.client else "unknown"
    return client_ip


# Authentication dependency functions
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserClaims]:
    """
    Extract and validate user from JWT token in Authorization header
    
    Returns None if no token provided (for optional authentication)
    Raises HTTPException for invalid tokens
    """
    token = None
    client_ip = get_client_ip(request)
    
    # Check rate limiting
    if not auth_rate_limiter.check_auth_limit(client_ip):
        security_audit.log_authentication(
            event_type="rate_limit_exceeded",
            user_id=None,
            client_ip=client_ip,
            success=False,
            details={"reason": "Too many failed attempts"}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later.",
            headers={"Retry-After": "900"}  # 15 minutes
        )
    
    # Try to get token from Authorization header
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    
    # Fallback: try to get token from OAuth2 scheme (for compatibility)
    if not token:
        token = await oauth2_scheme(request)
    
    if not token:
        return None

    try:
        user_claims = auth_middleware.verify_token(token)
        
        # Add user context to request state for logging
        request.state.user_id = user_claims.user_id
        request.state.user_roles = [role.value for role in user_claims.roles]
        request.state.client_ip = client_ip
        
        # Log successful authentication
        security_audit.log_authentication(
            event_type="token_validation",
            user_id=user_claims.user_id,
            client_ip=client_ip,
            success=True,
            details={
                "token_type": user_claims.token_type.value,
                "user_agent": request.headers.get("User-Agent")
            }
        )
        
        return user_claims
        
    except AuthenticationError as e:
        # Record failed attempt
        auth_rate_limiter.record_failed_attempt(client_ip)
        
        # Log failed authentication
        security_audit.log_authentication(
            event_type="token_validation",
            user_id=None,
            client_ip=client_ip,
            success=False,
            details={
                "error": "authentication_failed",
                "user_agent": request.headers.get("User-Agent")
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected authentication error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )


async def require_auth(
    user_claims: Optional[UserClaims] = Depends(get_current_user)
) -> UserClaims:
    """Require authentication - raises 401 if not authenticated"""
    if not user_claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_claims


# Convenience functions for common role checks
def require_admin() -> RoleChecker:
    """Require admin role"""
    return RoleChecker([UserRole.ADMIN])


def require_user() -> RoleChecker:
    """Require user role (basic access)"""
    return RoleChecker([UserRole.USER, UserRole.PREMIUM_USER, UserRole.ADMIN])


def require_premium() -> RoleChecker:
    """Require premium user or admin role"""
    return RoleChecker([UserRole.PREMIUM_USER, UserRole.ADMIN])


def require_moderator() -> RoleChecker:
    """Require moderator or admin role"""
    return RoleChecker([UserRole.MODERATOR, UserRole.ADMIN])


def require_developer() -> RoleChecker:
    """Require developer access"""
    return RoleChecker([UserRole.DEVELOPER, UserRole.ADMIN])


# Convenience functions for tier checks
def require_pro_tier() -> TierChecker:
    """Require pro or enterprise tier"""
    return TierChecker(["pro", "enterprise"])


def require_enterprise_tier() -> TierChecker:
    """Require enterprise tier"""
    return TierChecker(["enterprise"])


# Convenience functions for permission checks
def require_permission(permissions: List[Permission]) -> PermissionChecker:
    """Require specific permissions"""
    return PermissionChecker(permissions)


def require_generate_permission() -> PermissionChecker:
    """Require basic music generation permission"""
    return PermissionChecker([Permission.GENERATE_MUSIC])


def require_premium_features() -> PermissionChecker:
    """Require premium generation features"""
    return PermissionChecker([Permission.GENERATE_PREMIUM])


def require_model_management() -> PermissionChecker:
    """Require model management permissions"""
    return PermissionChecker([Permission.MANAGE_MODELS])


def require_user_management() -> PermissionChecker:
    """Require user management permissions"""
    return PermissionChecker([Permission.MANAGE_USERS])


# Token management functions
async def logout_user(
    user_claims: UserClaims = Depends(require_auth)
) -> Dict[str, str]:
    """Logout user by blacklisting their current token"""
    if user_claims.jti:
        success = auth_middleware.blacklist_token(user_claims.jti, user_claims.expires_at)
        if success:
            logger.info(f"User {user_claims.user_id} logged out successfully")
            return {"message": "Logged out successfully"}
        else:
            logger.warning(f"Failed to blacklist token for user {user_claims.user_id}")
            return {"message": "Logout completed (token blacklist unavailable)"}
    
    return {"message": "Logout completed"}


async def refresh_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh access token using refresh token"""
    try:
        new_access_token, new_refresh_token = auth_middleware.refresh_access_token(refresh_token)
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


# Export public interface
__all__ = [
    # Core components
    "auth_middleware",
    "security_audit",
    "auth_rate_limiter",
    
    # Enums
    "UserRole",
    "Permission",
    "TokenType", 
    
    # Models
    "UserClaims",
    
    # Authentication functions
    "get_current_user",
    "require_auth",
    
    # Role-based access control
    "require_admin",
    "require_user", 
    "require_premium",
    "require_moderator",
    "require_developer",
    "RoleChecker",
    
    # Permission-based access control
    "require_permission",
    "require_generate_permission",
    "require_premium_features",
    "require_model_management",
    "require_user_management",
    "PermissionChecker",
    
    # Tier-based access control
    "require_pro_tier",
    "require_enterprise_tier",
    "TierChecker",
    
    # Token management
    "logout_user",
    "refresh_token",
    
    # Utilities
    "get_client_ip",
    "ROLE_PERMISSIONS",
]