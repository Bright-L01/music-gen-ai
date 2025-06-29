"""
Data models for User Management Service

Defines schemas for users, authentication, and social features.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator


class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class ActivityType(str, Enum):
    """Types of user activities"""
    TRACK_SHARED = "track_shared"
    PLAYLIST_CREATED = "playlist_created"
    USER_FOLLOWED = "user_followed"
    TRACK_LIKED = "track_liked"
    ACHIEVEMENT_EARNED = "achievement_earned"


class UserCreate(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=30, regex=r"^[a-zA-Z0-9_]+$")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    bio: Optional[str] = Field(None, max_length=500, description="User bio")
    
    @validator('username')
    def validate_username(cls, v):
        # Reserved usernames
        reserved = ['admin', 'root', 'api', 'www', 'mail', 'support']
        if v.lower() in reserved:
            raise ValueError('Username is reserved')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "username": "musiclover123",
                "email": "user@example.com",
                "password": "securepassword123",
                "full_name": "John Doe",
                "bio": "Music enthusiast and aspiring composer"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }


class UserResponse(BaseModel):
    """Schema for user data in responses"""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    bio: Optional[str] = Field(None, description="User bio")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    tier: UserTier = Field(UserTier.FREE, description="Subscription tier")
    status: UserStatus = Field(UserStatus.ACTIVE, description="Account status")
    created_at: datetime = Field(..., description="Account creation date")
    is_verified: bool = Field(False, description="Email verification status")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_123e4567",
                "username": "musiclover123",
                "email": "user@example.com",
                "full_name": "John Doe",
                "bio": "Music enthusiast",
                "tier": "free",
                "status": "active",
                "created_at": "2024-01-01T12:00:00Z",
                "is_verified": True
            }
        }


class TokenResponse(BaseModel):
    """Schema for authentication token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(86400, description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {
                    "id": "user_123",
                    "username": "musiclover123",
                    "email": "user@example.com"
                }
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    bio: Optional[str] = Field(None, max_length=500, description="User bio")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    settings: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    
    class Config:
        schema_extra = {
            "example": {
                "full_name": "John Smith",
                "bio": "Electronic music producer and sound designer",
                "settings": {
                    "email_notifications": True,
                    "public_playlists": False,
                    "default_generation_length": 30
                }
            }
        }


class UserProfile(BaseModel):
    """Extended user profile with statistics"""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    tier: UserTier
    status: UserStatus
    created_at: datetime
    last_login_at: Optional[datetime]
    is_verified: bool
    
    # Statistics
    tracks_generated: int = Field(0, description="Total tracks generated")
    playlists_count: int = Field(0, description="Number of playlists")
    followers_count: int = Field(0, description="Number of followers")
    following_count: int = Field(0, description="Number of users following")
    total_generation_time: float = Field(0, description="Total audio generated in seconds")
    
    # Settings
    settings: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_123",
                "username": "musiclover123",
                "email": "user@example.com",
                "full_name": "John Doe",
                "tier": "premium",
                "tracks_generated": 45,
                "playlists_count": 8,
                "followers_count": 123,
                "following_count": 89
            }
        }


class SocialProfile(BaseModel):
    """Social profile with public information"""
    id: str
    username: str
    full_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    tier: UserTier
    created_at: datetime
    
    # Public stats
    tracks_generated: int
    playlists_count: int
    followers_count: int
    following_count: int
    
    # Activity
    recent_tracks: List[str] = Field([], description="Recent track URLs")
    featured_playlists: List[str] = Field([], description="Featured playlist IDs")
    badges: List[str] = Field([], description="Achievement badges")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_123",
                "username": "musiclover123",
                "full_name": "John Doe",
                "bio": "Electronic music producer",
                "tracks_generated": 45,
                "playlists_count": 8,
                "followers_count": 123,
                "following_count": 89,
                "badges": ["early_adopter", "prolific_creator"]
            }
        }


class FollowRequest(BaseModel):
    """Request to follow a user"""
    user_id: str = Field(..., description="ID of user to follow")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_456"
            }
        }


class PlaylistCreate(BaseModel):
    """Schema for creating a playlist"""
    name: str = Field(..., min_length=1, max_length=100, description="Playlist name")
    description: Optional[str] = Field(None, max_length=500, description="Playlist description")
    is_public: bool = Field(True, description="Whether playlist is public")
    tags: List[str] = Field([], description="Playlist tags")
    
    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Chill Electronic",
                "description": "Relaxing electronic music for focus",
                "is_public": True,
                "tags": ["electronic", "ambient", "focus"]
            }
        }


class PlaylistResponse(BaseModel):
    """Schema for playlist data"""
    id: str = Field(..., description="Playlist ID")
    name: str = Field(..., description="Playlist name")
    description: Optional[str] = Field(None, description="Playlist description")
    is_public: bool = Field(..., description="Public visibility")
    tags: List[str] = Field([], description="Playlist tags")
    user_id: str = Field(..., description="Owner user ID")
    username: str = Field(..., description="Owner username")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    track_count: int = Field(0, description="Number of tracks")
    total_duration: float = Field(0, description="Total duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "playlist_789",
                "name": "Chill Electronic",
                "description": "Relaxing electronic music",
                "is_public": True,
                "tags": ["electronic", "ambient"],
                "user_id": "user_123",
                "username": "musiclover123",
                "created_at": "2024-01-01T12:00:00Z",
                "track_count": 12,
                "total_duration": 2340.5
            }
        }


class ActivityItem(BaseModel):
    """User activity item"""
    id: str = Field(..., description="Activity ID")
    user_id: str = Field(..., description="User who performed activity")
    username: str = Field(..., description="Username")
    avatar_url: Optional[str] = Field(None, description="User avatar")
    activity_type: ActivityType = Field(..., description="Type of activity")
    content: Dict[str, Any] = Field(..., description="Activity content")
    created_at: datetime = Field(..., description="Activity timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "activity_abc123",
                "user_id": "user_123",
                "username": "musiclover123",
                "activity_type": "track_shared",
                "content": {
                    "track_url": "https://storage.example.com/track.wav",
                    "track_title": "Ambient Journey",
                    "message": "Check out this cool track I generated!"
                },
                "created_at": "2024-01-01T12:00:00Z"
            }
        }


class UserStats(BaseModel):
    """Detailed user statistics"""
    user_id: str
    tracks_generated: int
    total_generation_time: float
    avg_track_length: float
    favorite_genres: List[str]
    generation_by_month: Dict[str, int]
    most_active_day: str
    achievement_points: int
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "tracks_generated": 45,
                "total_generation_time": 1350.0,
                "avg_track_length": 30.0,
                "favorite_genres": ["electronic", "ambient", "jazz"],
                "generation_by_month": {
                    "2024-01": 12,
                    "2024-02": 18,
                    "2024-03": 15
                },
                "most_active_day": "Tuesday",
                "achievement_points": 1250
            }
        }


class UserSearchResult(BaseModel):
    """User search result"""
    id: str
    username: str
    full_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    tier: UserTier
    followers_count: int
    tracks_generated: int
    is_following: bool = Field(False, description="Whether current user follows this user")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_456",
                "username": "producer_jane",
                "full_name": "Jane Producer",
                "bio": "Electronic music producer",
                "tier": "premium",
                "followers_count": 234,
                "tracks_generated": 89,
                "is_following": False
            }
        }


class NotificationPreferences(BaseModel):
    """User notification preferences"""
    email_notifications: bool = Field(True, description="Email notifications")
    push_notifications: bool = Field(True, description="Push notifications")
    new_followers: bool = Field(True, description="Notify on new followers")
    playlist_shares: bool = Field(True, description="Notify on playlist shares")
    system_updates: bool = Field(True, description="System update notifications")
    marketing: bool = Field(False, description="Marketing communications")
    
    class Config:
        schema_extra = {
            "example": {
                "email_notifications": True,
                "push_notifications": False,
                "new_followers": True,
                "playlist_shares": True,
                "system_updates": True,
                "marketing": False
            }
        }