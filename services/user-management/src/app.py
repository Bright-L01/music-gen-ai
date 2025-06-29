"""
User Management Service API

Handles user authentication, profiles, and social features
for the music generation platform.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, generate_latest

from .models import (
    UserCreate,
    UserResponse,
    UserLogin,
    TokenResponse,
    UserProfile,
    UserUpdate,
    SocialProfile,
    FollowRequest,
    PlaylistCreate,
    PlaylistResponse
)
from .service import UserService
from .auth import get_current_user, create_access_token
from .database import database


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_counter = Counter(
    "user_requests_total",
    "Total user management requests",
    ["endpoint", "status"]
)

# Service instance will be created after database connection
user_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global user_service
    
    logger.info("Starting User Management Service...")
    await database.connect()
    
    # Initialize user service after database connection
    user_service = UserService(database)
    await user_service.initialize()
    logger.info("User Management Service started successfully")
    
    yield
    
    logger.info("Shutting down User Management Service...")
    await user_service.cleanup()
    await database.disconnect()
    logger.info("User Management Service stopped")


# Create FastAPI app
app = FastAPI(
    title="User Management Service",
    description="Microservice for user authentication and management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "user-management",
        "version": "1.0.0"
    }


@app.post("/register", response_model=TokenResponse)
async def register_user(user: UserCreate):
    """Register a new user"""
    request_counter.labels(endpoint="register", status="started").inc()
    
    try:
        # Check if user already exists
        existing_user = await user_service.get_user_by_email(user.email)
        if existing_user:
            request_counter.labels(endpoint="register", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        new_user = await user_service.create_user(user)
        
        # Generate tokens
        access_token = create_access_token(data={"user_id": new_user.id})
        
        request_counter.labels(endpoint="register", status="success").inc()
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=new_user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        request_counter.labels(endpoint="register", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@app.post("/login", response_model=TokenResponse)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user with email and password"""
    request_counter.labels(endpoint="login", status="started").inc()
    
    try:
        # Authenticate user
        user = await user_service.authenticate_user(
            form_data.username,  # Using username field for email
            form_data.password
        )
        
        if not user:
            request_counter.labels(endpoint="login", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Generate tokens
        access_token = create_access_token(data={"user_id": user.id})
        
        # Update last login
        await user_service.update_last_login(user.id)
        
        request_counter.labels(endpoint="login", status="success").inc()
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        request_counter.labels(endpoint="login", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile"""
    try:
        profile = await user_service.get_user_profile(current_user["id"])
        return profile
        
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch profile"
        )


@app.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update current user's profile"""
    request_counter.labels(endpoint="update_profile", status="started").inc()
    
    try:
        updated_user = await user_service.update_user(
            current_user["id"],
            user_update
        )
        
        request_counter.labels(endpoint="update_profile", status="success").inc()
        return updated_user
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        request_counter.labels(endpoint="update_profile", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@app.get("/social/profile", response_model=SocialProfile)
async def get_social_profile(current_user: dict = Depends(get_current_user)):
    """Get user's social profile with stats"""
    try:
        social_profile = await user_service.get_social_profile(current_user["id"])
        return social_profile
        
    except Exception as e:
        logger.error(f"Social profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch social profile"
        )


@app.post("/social/follow")
async def follow_user(
    follow_request: FollowRequest,
    current_user: dict = Depends(get_current_user)
):
    """Follow another user"""
    request_counter.labels(endpoint="follow", status="started").inc()
    
    try:
        # Check if target user exists
        target_user = await user_service.get_user_by_id(follow_request.user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Cannot follow yourself
        if follow_request.user_id == current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot follow yourself"
            )
        
        success = await user_service.follow_user(
            current_user["id"],
            follow_request.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Already following user"
            )
        
        request_counter.labels(endpoint="follow", status="success").inc()
        return {"message": "Successfully followed user"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Follow error: {e}")
        request_counter.labels(endpoint="follow", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Follow operation failed"
        )


@app.delete("/social/follow/{user_id}")
async def unfollow_user(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Unfollow a user"""
    try:
        success = await user_service.unfollow_user(
            current_user["id"],
            user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not following user"
            )
        
        return {"message": "Successfully unfollowed user"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unfollow error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unfollow operation failed"
        )


@app.get("/social/followers")
async def get_followers(current_user: dict = Depends(get_current_user)):
    """Get user's followers"""
    try:
        followers = await user_service.get_followers(current_user["id"])
        return {"followers": followers}
        
    except Exception as e:
        logger.error(f"Followers fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch followers"
        )


@app.get("/social/following")
async def get_following(current_user: dict = Depends(get_current_user)):
    """Get users that current user follows"""
    try:
        following = await user_service.get_following(current_user["id"])
        return {"following": following}
        
    except Exception as e:
        logger.error(f"Following fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch following"
        )


@app.post("/playlists", response_model=PlaylistResponse)
async def create_playlist(
    playlist: PlaylistCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new playlist"""
    request_counter.labels(endpoint="create_playlist", status="started").inc()
    
    try:
        new_playlist = await user_service.create_playlist(
            current_user["id"],
            playlist
        )
        
        request_counter.labels(endpoint="create_playlist", status="success").inc()
        return new_playlist
        
    except Exception as e:
        logger.error(f"Playlist creation error: {e}")
        request_counter.labels(endpoint="create_playlist", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Playlist creation failed"
        )


@app.get("/playlists")
async def get_user_playlists(current_user: dict = Depends(get_current_user)):
    """Get user's playlists"""
    try:
        playlists = await user_service.get_user_playlists(current_user["id"])
        return {"playlists": playlists}
        
    except Exception as e:
        logger.error(f"Playlists fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch playlists"
        )


@app.post("/playlists/{playlist_id}/tracks")
async def add_track_to_playlist(
    playlist_id: str,
    track_url: str,
    current_user: dict = Depends(get_current_user)
):
    """Add track to playlist"""
    try:
        success = await user_service.add_track_to_playlist(
            current_user["id"],
            playlist_id,
            track_url
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Playlist not found or access denied"
            )
        
        return {"message": "Track added to playlist"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add track error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not add track to playlist"
        )


@app.get("/discover/users")
async def discover_users(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Discover new users to follow"""
    try:
        users = await user_service.discover_users(
            current_user["id"],
            limit
        )
        return {"users": users}
        
    except Exception as e:
        logger.error(f"User discovery error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User discovery failed"
        )


@app.get("/activity/feed")
async def get_activity_feed(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get user's activity feed"""
    try:
        feed = await user_service.get_activity_feed(
            current_user["id"],
            limit
        )
        return {"activities": feed}
        
    except Exception as e:
        logger.error(f"Activity feed error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch activity feed"
        )


@app.post("/activity/share")
async def share_track(
    track_url: str,
    message: str = "",
    current_user: dict = Depends(get_current_user)
):
    """Share a track to activity feed"""
    try:
        await user_service.share_track(
            current_user["id"],
            track_url,
            message
        )
        
        return {"message": "Track shared successfully"}
        
    except Exception as e:
        logger.error(f"Share track error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not share track"
        )


@app.delete("/me")
async def delete_account(current_user: dict = Depends(get_current_user)):
    """Delete user account"""
    request_counter.labels(endpoint="delete_account", status="started").inc()
    
    try:
        success = await user_service.delete_user(current_user["id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Account deletion failed"
            )
        
        request_counter.labels(endpoint="delete_account", status="success").inc()
        return {"message": "Account deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        request_counter.labels(endpoint="delete_account", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)