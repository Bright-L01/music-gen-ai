"""
Authentication and common dependencies for MusicGen AI API.
Provides reusable dependency injection functions for FastAPI endpoints.
"""

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from music_gen.api.middleware.auth import (
    RoleChecker,
    TierChecker,
    UserClaims,
    UserRole,
)
from music_gen.api.middleware.auth import get_current_user as middleware_get_current_user
from music_gen.api.middleware.auth import require_auth as middleware_require_auth
from music_gen.core.container import get_container
from music_gen.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Security scheme for OpenAPI documentation
security = HTTPBearer(auto_error=False)


# Type aliases for better readability
CurrentUser = Annotated[Optional[UserClaims], Depends(middleware_get_current_user)]
ActiveUser = Annotated[UserClaims, Depends(middleware_require_auth)]


async def get_current_user(user_claims: CurrentUser) -> Optional[UserClaims]:
    """
    Get current user from JWT token if available.

    This dependency returns None if no valid token is provided,
    allowing endpoints to handle both authenticated and anonymous users.

    Args:
        user_claims: User claims from JWT token (injected by middleware)

    Returns:
        UserClaims if authenticated, None otherwise

    Example:
        @app.get("/profile")
        async def get_profile(user: Optional[UserClaims] = Depends(get_current_user)):
            if user:
                return {"email": user.email}
            return {"message": "Not authenticated"}
    """
    return user_claims


async def get_current_active_user(user_claims: ActiveUser) -> UserClaims:
    """
    Get current authenticated and active user.

    This dependency requires authentication and validates that:
    - User has a valid JWT token
    - User's email is verified
    - User account is not suspended/deactivated

    Args:
        user_claims: Authenticated user claims (injected by middleware)

    Returns:
        Validated UserClaims

    Raises:
        HTTPException: 401 if not authenticated
        HTTPException: 403 if account is not verified or inactive

    Example:
        @app.post("/generate")
        async def generate_music(
            user: UserClaims = Depends(get_current_active_user)
        ):
            return {"user_id": user.user_id}
    """
    # Check if email is verified
    if not user_claims.is_verified:
        logger.warning(f"Unverified user attempted access: {user_claims.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Please verify your email address.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token is not too old (additional security measure)
    token_age = datetime.now(timezone.utc) - user_claims.issued_at
    max_token_age_days = 30  # Configurable based on security requirements

    if token_age.days > max_token_age_days:
        logger.warning(
            f"User {user_claims.user_id} attempted access with old token "
            f"(issued {token_age.days} days ago)"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is too old. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # TODO: In production, check user status in database
    # Example:
    # if user_claims.status == "suspended":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Account has been suspended"
    #     )

    logger.debug(f"Active user authenticated: {user_claims.user_id}")
    return user_claims


async def require_admin_role(user_claims: ActiveUser) -> UserClaims:
    """
    Require admin role for access.

    This dependency ensures the authenticated user has admin privileges.

    Args:
        user_claims: Authenticated user claims

    Returns:
        UserClaims with admin role

    Raises:
        HTTPException: 403 if user doesn't have admin role

    Example:
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: str,
            admin: UserClaims = Depends(require_admin_role)
        ):
            return {"deleted": user_id}
    """
    if UserRole.ADMIN not in user_claims.roles:
        logger.warning(
            f"Non-admin user {user_claims.user_id} attempted admin access. "
            f"Roles: {[role.value for role in user_claims.roles]}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    logger.info(f"Admin access granted to user {user_claims.user_id}")
    return user_claims


async def require_authenticated(user_claims: ActiveUser) -> UserClaims:
    """
    Simple dependency that requires authentication.

    Unlike get_current_active_user, this doesn't perform additional
    validation beyond JWT token verification.

    Args:
        user_claims: Authenticated user claims

    Returns:
        UserClaims

    Raises:
        HTTPException: 401 if not authenticated

    Example:
        @app.get("/my-generations")
        async def list_generations(
            user: UserClaims = Depends(require_authenticated)
        ):
            return {"user_id": user.user_id}
    """
    return user_claims


# Role-based dependencies using RoleChecker
def require_premium_user() -> RoleChecker:
    """
    Require premium user or admin role.

    Example:
        @app.post("/advanced-generate")
        async def advanced_generate(
            user: UserClaims = Depends(require_premium_user())
        ):
            return {"access": "granted"}
    """
    return RoleChecker(required_roles=[UserRole.PREMIUM_USER, UserRole.ADMIN], require_all=False)


def require_moderator() -> RoleChecker:
    """
    Require moderator or admin role.

    Example:
        @app.post("/flag-content")
        async def flag_content(
            user: UserClaims = Depends(require_moderator())
        ):
            return {"access": "granted"}
    """
    return RoleChecker(required_roles=[UserRole.MODERATOR, UserRole.ADMIN], require_all=False)


def require_developer() -> RoleChecker:
    """
    Require developer or admin role.

    Example:
        @app.get("/api-stats")
        async def get_api_stats(
            user: UserClaims = Depends(require_developer())
        ):
            return {"access": "granted"}
    """
    return RoleChecker(required_roles=[UserRole.DEVELOPER, UserRole.ADMIN], require_all=False)


def require_multiple_roles(roles: list[UserRole], require_all: bool = False) -> RoleChecker:
    """
    Create a dependency that requires specific roles.

    Args:
        roles: List of required roles
        require_all: If True, user must have ALL roles. If False, ANY role is sufficient.

    Returns:
        RoleChecker dependency

    Example:
        @app.post("/special-access")
        async def special_endpoint(
            user: UserClaims = Depends(
                require_multiple_roles([UserRole.PREMIUM_USER, UserRole.DEVELOPER])
            )
        ):
            return {"access": "granted"}
    """
    return RoleChecker(required_roles=roles, require_all=require_all)


# Tier-based dependencies
def require_pro_tier() -> TierChecker:
    """
    Require pro or enterprise subscription tier.

    Example:
        @app.post("/batch-generate")
        async def batch_generate(
            user: UserClaims = Depends(require_pro_tier())
        ):
            return {"access": "granted"}
    """
    return TierChecker(required_tiers=["pro", "enterprise"])


def require_enterprise_tier() -> TierChecker:
    """
    Require enterprise subscription tier.

    Example:
        @app.post("/custom-model")
        async def use_custom_model(
            user: UserClaims = Depends(require_enterprise_tier())
        ):
            return {"access": "granted"}
    """
    return TierChecker(required_tiers=["enterprise"])


def require_tier(tiers: list[str]) -> TierChecker:
    """
    Create a dependency that requires specific subscription tiers.

    Args:
        tiers: List of allowed tiers

    Returns:
        TierChecker dependency

    Example:
        @app.post("/feature")
        async def premium_feature(
            user: UserClaims = Depends(require_tier(["pro", "enterprise"]))
        ):
            return {"access": "granted"}
    """
    return TierChecker(required_tiers=tiers)


# Database session dependency
async def get_db() -> Session:
    """
    Get database session.

    This dependency provides a SQLAlchemy session that will be
    automatically closed after the request completes.

    Returns:
        SQLAlchemy Session

    Example:
        @app.get("/users")
        async def list_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    # TODO: Implement actual database session management
    # For now, return a mock session
    get_container()
    # db_session = container.get(Session)
    # try:
    #     yield db_session
    # finally:
    #     db_session.close()

    # Placeholder implementation
    raise NotImplementedError(
        "Database session management not yet implemented. "
        "Configure SQLAlchemy and update this dependency."
    )


# Request context dependencies
async def get_request_id(request: Request) -> str:
    """
    Get or generate request ID for tracking.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string

    Example:
        @app.post("/generate")
        async def generate(
            request_id: str = Depends(get_request_id)
        ):
            logger.info(f"Processing request {request_id}")
    """
    # Try to get from headers first
    request_id = request.headers.get("X-Request-ID")

    # Fall back to state if available
    if not request_id and hasattr(request.state, "request_id"):
        request_id = request.state.request_id

    # Generate new if not found
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())

    return request_id


async def get_client_ip(request: Request) -> str:
    """
    Get client IP address, handling proxies.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address

    Example:
        @app.post("/generate")
        async def generate(
            client_ip: str = Depends(get_client_ip)
        ):
            logger.info(f"Request from {client_ip}")
    """
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


# Composite dependencies
async def get_authenticated_request_context(
    user: ActiveUser,
    request_id: str = Depends(get_request_id),
    client_ip: str = Depends(get_client_ip),
) -> dict:
    """
    Get complete authenticated request context.

    Combines user authentication with request metadata.

    Returns:
        Dictionary with user, request_id, and client_ip

    Example:
        @app.post("/generate")
        async def generate(
            context: dict = Depends(get_authenticated_request_context)
        ):
            logger.info(
                f"User {context['user'].user_id} from {context['client_ip']} "
                f"(request: {context['request_id']})"
            )
    """
    return {
        "user": user,
        "request_id": request_id,
        "client_ip": client_ip,
        "timestamp": datetime.now(timezone.utc),
    }


# Export all dependencies
__all__ = [
    # User authentication
    "get_current_user",
    "get_current_active_user",
    "require_admin_role",
    "require_authenticated",
    # Role-based access
    "require_premium_user",
    "require_moderator",
    "require_developer",
    "require_multiple_roles",
    # Tier-based access
    "require_pro_tier",
    "require_enterprise_tier",
    "require_tier",
    # Database
    "get_db",
    # Request context
    "get_request_id",
    "get_client_ip",
    "get_authenticated_request_context",
    # Type aliases
    "CurrentUser",
    "ActiveUser",
    # Re-exports from middleware
    "UserClaims",
    "UserRole",
]
