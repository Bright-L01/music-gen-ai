"""
User Management Service Implementation

Core business logic for user operations, authentication, and social features.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import bcrypt
from databases import Database

from .models import (
    UserCreate,
    UserResponse,
    UserUpdate,
    UserProfile,
    SocialProfile,
    PlaylistCreate,
    PlaylistResponse,
    ActivityItem,
    UserSearchResult,
    UserTier,
    UserStatus,
    ActivityType
)


logger = logging.getLogger(__name__)


class UserService:
    """Main service for user management operations"""
    
    def __init__(self, database: Database):
        self.db = database
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service"""
        if self._initialized:
            return
            
        logger.info("Initializing User Service...")
        await self._create_tables()
        self._initialized = True
        logger.info("User Service initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        pass
        
    async def _create_tables(self):
        """Create database tables if they don't exist"""
        # Users table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(50) PRIMARY KEY,
                username VARCHAR(30) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                bio TEXT,
                avatar_url VARCHAR(500),
                tier VARCHAR(20) DEFAULT 'free',
                status VARCHAR(20) DEFAULT 'active',
                is_verified BOOLEAN DEFAULT FALSE,
                settings JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at TIMESTAMP
            )
        """)
        
        # Create indexes
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users(status)")
        
        # User follows table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_follows (
                id VARCHAR(50) PRIMARY KEY,
                follower_id VARCHAR(50) NOT NULL,
                following_id VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (follower_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (following_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE (follower_id, following_id)
            )
        """)
        
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_follows_follower ON user_follows(follower_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_follows_following ON user_follows(following_id)")
        
        # Playlists table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                user_id VARCHAR(50) NOT NULL,
                is_public BOOLEAN DEFAULT TRUE,
                tags JSONB,
                track_count INT DEFAULT 0,
                total_duration FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_playlists_user_id ON playlists(user_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_playlists_public ON playlists(is_public)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_playlists_created_at ON playlists(created_at)")
        
        # Playlist tracks table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                id VARCHAR(50) PRIMARY KEY,
                playlist_id VARCHAR(50) NOT NULL,
                track_url VARCHAR(500) NOT NULL,
                track_title VARCHAR(200),
                track_duration FLOAT,
                position INT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
            )
        """)
        
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_playlist_tracks_position ON playlist_tracks(position)")
        
        # User activities table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_activities (
                id VARCHAR(50) PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                activity_type VARCHAR(50) NOT NULL,
                content JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_activities_user_id ON user_activities(user_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_activities_type ON user_activities(activity_type)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_activities_created_at ON user_activities(created_at)")
        
        # User stats table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id VARCHAR(50) PRIMARY KEY,
                tracks_generated INT DEFAULT 0,
                total_generation_time FLOAT DEFAULT 0,
                playlists_count INT DEFAULT 0,
                followers_count INT DEFAULT 0,
                following_count INT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        password_hash = self._hash_password(user_data.password)
        
        # Insert user
        await self.db.execute("""
            INSERT INTO users (
                id, username, email, password_hash, full_name, bio, tier, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            user_id,
            user_data.username,
            user_data.email,
            password_hash,
            user_data.full_name,
            user_data.bio,
            UserTier.FREE.value,
            UserStatus.ACTIVE.value
        ])
        
        # Initialize user stats
        await self.db.execute("""
            INSERT INTO user_stats (user_id) VALUES (?)
        """, [user_id])
        
        # Create welcome activity
        await self._create_activity(
            user_id,
            ActivityType.USER_FOLLOWED,  # Using as "joined" activity
            {"message": "Welcome to the music generation platform!"}
        )
        
        return await self.get_user_by_id(user_id)
        
    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID"""
        row = await self.db.fetch_one("""
            SELECT * FROM users WHERE id = ? AND status != 'deleted'
        """, [user_id])
        
        if row:
            return UserResponse(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                status=UserStatus(row['status']),
                created_at=row['created_at'],
                is_verified=row['is_verified']
            )
        return None
        
    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email"""
        row = await self.db.fetch_one("""
            SELECT * FROM users WHERE email = ? AND status != 'deleted'
        """, [email])
        
        if row:
            return UserResponse(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                status=UserStatus(row['status']),
                created_at=row['created_at'],
                is_verified=row['is_verified']
            )
        return None
        
    async def authenticate_user(self, email: str, password: str) -> Optional[UserResponse]:
        """Authenticate user with email and password"""
        row = await self.db.fetch_one("""
            SELECT * FROM users 
            WHERE email = ? AND status = 'active'
        """, [email])
        
        if row and self._verify_password(password, row['password_hash']):
            return UserResponse(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                status=UserStatus(row['status']),
                created_at=row['created_at'],
                is_verified=row['is_verified']
            )
        return None
        
    async def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        await self.db.execute("""
            UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?
        """, [user_id])
        
    async def update_user(self, user_id: str, user_update: UserUpdate) -> UserResponse:
        """Update user profile"""
        # Build update query dynamically
        update_fields = []
        values = []
        
        if user_update.full_name is not None:
            update_fields.append("full_name = ?")
            values.append(user_update.full_name)
            
        if user_update.bio is not None:
            update_fields.append("bio = ?")
            values.append(user_update.bio)
            
        if user_update.avatar_url is not None:
            update_fields.append("avatar_url = ?")
            values.append(user_update.avatar_url)
            
        if user_update.settings is not None:
            update_fields.append("settings = ?")
            values.append(json.dumps(user_update.settings))
            
        if update_fields:
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(user_id)
            
            await self.db.execute(f"""
                UPDATE users SET {', '.join(update_fields)} WHERE id = ?
            """, values)
            
        return await self.get_user_by_id(user_id)
        
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get detailed user profile with statistics"""
        # Get user data
        user_row = await self.db.fetch_one("""
            SELECT * FROM users WHERE id = ? AND status != 'deleted'
        """, [user_id])
        
        if not user_row:
            return None
            
        # Get user stats
        stats_row = await self.db.fetch_one("""
            SELECT * FROM user_stats WHERE user_id = ?
        """, [user_id])
        
        stats = stats_row if stats_row else {}
        
        return UserProfile(
            id=user_row['id'],
            username=user_row['username'],
            email=user_row['email'],
            full_name=user_row['full_name'],
            bio=user_row['bio'],
            avatar_url=user_row['avatar_url'],
            tier=UserTier(user_row['tier']),
            status=UserStatus(user_row['status']),
            created_at=user_row['created_at'],
            last_login_at=user_row['last_login_at'],
            is_verified=user_row['is_verified'],
            tracks_generated=stats.get('tracks_generated', 0),
            playlists_count=stats.get('playlists_count', 0),
            followers_count=stats.get('followers_count', 0),
            following_count=stats.get('following_count', 0),
            total_generation_time=stats.get('total_generation_time', 0),
            settings=json.loads(user_row['settings']) if user_row['settings'] else {}
        )
        
    async def get_social_profile(self, user_id: str) -> Optional[SocialProfile]:
        """Get public social profile"""
        user_row = await self.db.fetch_one("""
            SELECT * FROM users WHERE id = ? AND status = 'active'
        """, [user_id])
        
        if not user_row:
            return None
            
        stats_row = await self.db.fetch_one("""
            SELECT * FROM user_stats WHERE user_id = ?
        """, [user_id])
        
        stats = stats_row if stats_row else {}
        
        # Get recent tracks (would be from generations service)
        recent_tracks = []
        
        # Get featured playlists
        featured_playlists = await self.db.fetch_all("""
            SELECT id FROM playlists 
            WHERE user_id = ? AND is_public = TRUE 
            ORDER BY track_count DESC, created_at DESC 
            LIMIT 3
        """, [user_id])
        
        return SocialProfile(
            id=user_row['id'],
            username=user_row['username'],
            full_name=user_row['full_name'],
            bio=user_row['bio'],
            avatar_url=user_row['avatar_url'],
            tier=UserTier(user_row['tier']),
            created_at=user_row['created_at'],
            tracks_generated=stats.get('tracks_generated', 0),
            playlists_count=stats.get('playlists_count', 0),
            followers_count=stats.get('followers_count', 0),
            following_count=stats.get('following_count', 0),
            recent_tracks=recent_tracks,
            featured_playlists=[row['id'] for row in featured_playlists],
            badges=self._get_user_badges(stats)
        )
        
    def _get_user_badges(self, stats: Dict) -> List[str]:
        """Get user achievement badges"""
        badges = []
        
        tracks_count = stats.get('tracks_generated', 0)
        followers_count = stats.get('followers_count', 0)
        
        if tracks_count >= 100:
            badges.append("prolific_creator")
        elif tracks_count >= 50:
            badges.append("active_creator")
        elif tracks_count >= 10:
            badges.append("budding_artist")
            
        if followers_count >= 1000:
            badges.append("influencer")
        elif followers_count >= 100:
            badges.append("popular_creator")
            
        # Early adopter badge (users created in first month)
        badges.append("early_adopter")
        
        return badges
        
    async def follow_user(self, follower_id: str, following_id: str) -> bool:
        """Follow a user"""
        try:
            follow_id = f"follow_{uuid.uuid4().hex[:8]}"
            
            await self.db.execute("""
                INSERT INTO user_follows (id, follower_id, following_id)
                VALUES (?, ?, ?)
            """, [follow_id, follower_id, following_id])
            
            # Update stats
            await self._update_follow_stats(follower_id, following_id, increment=True)
            
            # Create activity
            await self._create_activity(
                follower_id,
                ActivityType.USER_FOLLOWED,
                {"following_user_id": following_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Follow error: {e}")
            return False
            
    async def unfollow_user(self, follower_id: str, following_id: str) -> bool:
        """Unfollow a user"""
        try:
            result = await self.db.execute("""
                DELETE FROM user_follows 
                WHERE follower_id = ? AND following_id = ?
            """, [follower_id, following_id])
            
            if result:
                # Update stats
                await self._update_follow_stats(follower_id, following_id, increment=False)
                return True
                
        except Exception as e:
            logger.error(f"Unfollow error: {e}")
            
        return False
        
    async def _update_follow_stats(self, follower_id: str, following_id: str, increment: bool):
        """Update follower/following counts"""
        delta = 1 if increment else -1
        
        # Update follower's following count
        await self.db.execute("""
            UPDATE user_stats 
            SET following_count = following_count + ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, [delta, follower_id])
        
        # Update following user's followers count
        await self.db.execute("""
            UPDATE user_stats 
            SET followers_count = followers_count + ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, [delta, following_id])
        
    async def get_followers(self, user_id: str) -> List[UserSearchResult]:
        """Get user's followers"""
        rows = await self.db.fetch_all("""
            SELECT u.id, u.username, u.full_name, u.bio, u.avatar_url, u.tier,
                   s.tracks_generated, s.followers_count
            FROM user_follows f
            JOIN users u ON f.follower_id = u.id
            LEFT JOIN user_stats s ON u.id = s.user_id
            WHERE f.following_id = ? AND u.status = 'active'
            ORDER BY f.created_at DESC
        """, [user_id])
        
        return [
            UserSearchResult(
                id=row['id'],
                username=row['username'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                followers_count=row['followers_count'] or 0,
                tracks_generated=row['tracks_generated'] or 0,
                is_following=False  # Would need additional query
            )
            for row in rows
        ]
        
    async def get_following(self, user_id: str) -> List[UserSearchResult]:
        """Get users that current user follows"""
        rows = await self.db.fetch_all("""
            SELECT u.id, u.username, u.full_name, u.bio, u.avatar_url, u.tier,
                   s.tracks_generated, s.followers_count
            FROM user_follows f
            JOIN users u ON f.following_id = u.id
            LEFT JOIN user_stats s ON u.id = s.user_id
            WHERE f.follower_id = ? AND u.status = 'active'
            ORDER BY f.created_at DESC
        """, [user_id])
        
        return [
            UserSearchResult(
                id=row['id'],
                username=row['username'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                followers_count=row['followers_count'] or 0,
                tracks_generated=row['tracks_generated'] or 0,
                is_following=True
            )
            for row in rows
        ]
        
    async def create_playlist(
        self,
        user_id: str,
        playlist_data: PlaylistCreate
    ) -> PlaylistResponse:
        """Create a new playlist"""
        playlist_id = f"playlist_{uuid.uuid4().hex[:8]}"
        
        await self.db.execute("""
            INSERT INTO playlists (
                id, name, description, user_id, is_public, tags
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [
            playlist_id,
            playlist_data.name,
            playlist_data.description,
            user_id,
            playlist_data.is_public,
            json.dumps(playlist_data.tags)
        ])
        
        # Update user stats
        await self.db.execute("""
            UPDATE user_stats 
            SET playlists_count = playlists_count + 1, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, [user_id])
        
        # Create activity
        await self._create_activity(
            user_id,
            ActivityType.PLAYLIST_CREATED,
            {"playlist_id": playlist_id, "playlist_name": playlist_data.name}
        )
        
        return await self.get_playlist_by_id(playlist_id)
        
    async def get_playlist_by_id(self, playlist_id: str) -> Optional[PlaylistResponse]:
        """Get playlist by ID"""
        row = await self.db.fetch_one("""
            SELECT p.*, u.username
            FROM playlists p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
        """, [playlist_id])
        
        if row:
            return PlaylistResponse(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                is_public=row['is_public'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                user_id=row['user_id'],
                username=row['username'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                track_count=row['track_count'],
                total_duration=row['total_duration']
            )
        return None
        
    async def get_user_playlists(self, user_id: str) -> List[PlaylistResponse]:
        """Get user's playlists"""
        rows = await self.db.fetch_all("""
            SELECT p.*, u.username
            FROM playlists p
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ?
            ORDER BY p.created_at DESC
        """, [user_id])
        
        return [
            PlaylistResponse(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                is_public=row['is_public'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                user_id=row['user_id'],
                username=row['username'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                track_count=row['track_count'],
                total_duration=row['total_duration']
            )
            for row in rows
        ]
        
    async def add_track_to_playlist(
        self,
        user_id: str,
        playlist_id: str,
        track_url: str
    ) -> bool:
        """Add track to playlist"""
        # Verify playlist ownership
        playlist = await self.db.fetch_one("""
            SELECT user_id FROM playlists WHERE id = ?
        """, [playlist_id])
        
        if not playlist or playlist['user_id'] != user_id:
            return False
            
        try:
            # Get next position
            next_position = await self.db.fetch_val("""
                SELECT COALESCE(MAX(position), 0) + 1 
                FROM playlist_tracks WHERE playlist_id = ?
            """, [playlist_id])
            
            track_id = f"track_{uuid.uuid4().hex[:8]}"
            
            await self.db.execute("""
                INSERT INTO playlist_tracks (
                    id, playlist_id, track_url, position
                ) VALUES (?, ?, ?, ?)
            """, [track_id, playlist_id, track_url, next_position])
            
            # Update playlist stats
            await self.db.execute("""
                UPDATE playlists 
                SET track_count = track_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [playlist_id])
            
            return True
            
        except Exception as e:
            logger.error(f"Add track to playlist error: {e}")
            return False
            
    async def discover_users(self, user_id: str, limit: int = 10) -> List[UserSearchResult]:
        """Discover users to follow"""
        # Get users not already followed
        rows = await self.db.fetch_all("""
            SELECT u.id, u.username, u.full_name, u.bio, u.avatar_url, u.tier,
                   s.tracks_generated, s.followers_count
            FROM users u
            LEFT JOIN user_stats s ON u.id = s.user_id
            WHERE u.id != ? 
              AND u.status = 'active'
              AND u.id NOT IN (
                  SELECT following_id FROM user_follows WHERE follower_id = ?
              )
            ORDER BY s.followers_count DESC, u.created_at DESC
            LIMIT ?
        """, [user_id, user_id, limit])
        
        return [
            UserSearchResult(
                id=row['id'],
                username=row['username'],
                full_name=row['full_name'],
                bio=row['bio'],
                avatar_url=row['avatar_url'],
                tier=UserTier(row['tier']),
                followers_count=row['followers_count'] or 0,
                tracks_generated=row['tracks_generated'] or 0,
                is_following=False
            )
            for row in rows
        ]
        
    async def get_activity_feed(self, user_id: str, limit: int = 20) -> List[ActivityItem]:
        """Get activity feed for user"""
        # Get activities from followed users
        rows = await self.db.fetch_all("""
            SELECT a.*, u.username, u.avatar_url
            FROM user_activities a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id IN (
                SELECT following_id FROM user_follows WHERE follower_id = ?
                UNION SELECT ?
            )
            ORDER BY a.created_at DESC
            LIMIT ?
        """, [user_id, user_id, limit])
        
        return [
            ActivityItem(
                id=row['id'],
                user_id=row['user_id'],
                username=row['username'],
                avatar_url=row['avatar_url'],
                activity_type=ActivityType(row['activity_type']),
                content=json.loads(row['content']),
                created_at=row['created_at']
            )
            for row in rows
        ]
        
    async def share_track(self, user_id: str, track_url: str, message: str = ""):
        """Share a track to activity feed"""
        await self._create_activity(
            user_id,
            ActivityType.TRACK_SHARED,
            {
                "track_url": track_url,
                "message": message
            }
        )
        
    async def _create_activity(
        self,
        user_id: str,
        activity_type: ActivityType,
        content: Dict[str, Any]
    ):
        """Create user activity record"""
        activity_id = f"activity_{uuid.uuid4().hex[:8]}"
        
        await self.db.execute("""
            INSERT INTO user_activities (id, user_id, activity_type, content)
            VALUES (?, ?, ?, ?)
        """, [activity_id, user_id, activity_type.value, json.dumps(content)])
        
    async def delete_user(self, user_id: str) -> bool:
        """Soft delete user account"""
        try:
            await self.db.execute("""
                UPDATE users 
                SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [user_id])
            
            return True
            
        except Exception as e:
            logger.error(f"User deletion error: {e}")
            return False