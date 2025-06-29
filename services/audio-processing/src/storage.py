"""
Audio Storage Management

Handles file storage operations for audio files, supporting local storage
and cloud storage (S3/MinIO) in production.
"""

import hashlib
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import UploadFile


logger = logging.getLogger(__name__)


class AudioStorage:
    """
    Audio file storage management
    
    Supports:
    - Local file storage (development)
    - S3/MinIO storage (production)
    - File metadata tracking
    - Access control
    """
    
    def __init__(self):
        self.storage_path = Path(os.getenv("STORAGE_PATH", "/tmp/audio_storage"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Organize by type
        self.raw_path = self.storage_path / "raw"
        self.processed_path = self.storage_path / "processed"
        self.waveforms_path = self.storage_path / "waveforms"
        self.temp_path = self.storage_path / "temp"
        
        for path in [self.raw_path, self.processed_path, self.waveforms_path, self.temp_path]:
            path.mkdir(exist_ok=True)
            
        # In production, initialize S3/MinIO client
        self.s3_client = None
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "music-gen-audio")
        
        # File metadata (in production, use database)
        self.file_metadata = {}
        
    async def initialize(self):
        """Initialize storage backend"""
        logger.info("Initializing Audio Storage...")
        
        # In production, connect to S3/MinIO
        if os.getenv("USE_S3", "false").lower() == "true":
            await self._init_s3()
            
        logger.info("Audio Storage initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        # Clean temp files older than 1 hour
        await self._cleanup_temp_files()
        
    async def _init_s3(self):
        """Initialize S3/MinIO client"""
        try:
            import aioboto3
            
            session = aioboto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            
            async with session.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT_URL")
            ) as s3:
                # Create bucket if not exists
                try:
                    await s3.create_bucket(Bucket=self.bucket_name)
                except Exception:
                    pass  # Bucket already exists
                    
            self.s3_client = session.client("s3")
            logger.info("Connected to S3/MinIO storage")
            
        except Exception as e:
            logger.warning(f"S3 initialization failed, using local storage: {e}")
            
    async def upload_file(
        self,
        file: UploadFile,
        user_id: str,
        file_type: str = "raw"
    ) -> str:
        """
        Upload audio file to storage
        
        Args:
            file: Uploaded file
            user_id: User ID for organization
            file_type: Type of file (raw, processed, waveform)
            
        Returns:
            File URL/path
        """
        try:
            # Generate unique filename
            file_ext = Path(file.filename).suffix.lower()
            file_id = f"{user_id}_{uuid.uuid4().hex}{file_ext}"
            
            # Determine storage path
            if file_type == "raw":
                local_path = self.raw_path / file_id
            elif file_type == "processed":
                local_path = self.processed_path / file_id
            elif file_type == "waveform":
                local_path = self.waveforms_path / file_id
            else:
                local_path = self.temp_path / file_id
                
            # Save file locally
            content = await file.read()
            with open(local_path, "wb") as f:
                f.write(content)
                
            # Calculate file hash
            file_hash = hashlib.sha256(content).hexdigest()
            
            # Store metadata
            self.file_metadata[file_id] = {
                "user_id": user_id,
                "original_name": file.filename,
                "size": len(content),
                "content_type": file.content_type,
                "hash": file_hash,
                "uploaded_at": datetime.utcnow().isoformat(),
                "local_path": str(local_path),
                "file_type": file_type
            }
            
            # Upload to S3 if enabled
            if self.s3_client:
                s3_key = f"{file_type}/{user_id}/{file_id}"
                await self._upload_to_s3(local_path, s3_key, file.content_type)
                
                # Return S3 URL
                return f"s3://{self.bucket_name}/{s3_key}"
            else:
                # Return local URL
                return f"/storage/{file_type}/{file_id}"
                
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
            
    async def _upload_to_s3(
        self,
        local_path: Path,
        s3_key: str,
        content_type: str
    ):
        """Upload file to S3/MinIO"""
        try:
            async with self.s3_client as s3:
                with open(local_path, "rb") as f:
                    await s3.upload_fileobj(
                        f,
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={
                            "ContentType": content_type,
                            "Metadata": {
                                "uploaded_at": datetime.utcnow().isoformat()
                            }
                        }
                    )
                    
            logger.info(f"Uploaded to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            # Continue with local storage
            
    async def download_file(
        self,
        file_url: str,
        user_id: Optional[str] = None
    ) -> Path:
        """
        Download file from storage
        
        Args:
            file_url: File URL (local or S3)
            user_id: User ID for access control
            
        Returns:
            Local file path
        """
        try:
            if file_url.startswith("s3://"):
                # Download from S3
                return await self._download_from_s3(file_url, user_id)
            elif file_url.startswith("/storage/"):
                # Local storage
                file_type, file_id = file_url.replace("/storage/", "").split("/", 1)
                
                # Check access
                metadata = self.file_metadata.get(file_id, {})
                if user_id and metadata.get("user_id") != user_id:
                    raise PermissionError("Access denied")
                    
                return Path(metadata["local_path"])
            else:
                # Assume local path
                return Path(file_url)
                
        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise
            
    async def _download_from_s3(
        self,
        s3_url: str,
        user_id: Optional[str]
    ) -> Path:
        """Download file from S3/MinIO"""
        # Parse S3 URL
        parts = s3_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        
        # Local cache path
        cache_path = self.temp_path / f"s3_cache_{hashlib.md5(key.encode()).hexdigest()}"
        
        if cache_path.exists():
            return cache_path
            
        try:
            async with self.s3_client as s3:
                # Check access (in production, verify with metadata)
                if user_id and not key.startswith(f"raw/{user_id}/"):
                    raise PermissionError("Access denied")
                    
                # Download file
                with open(cache_path, "wb") as f:
                    await s3.download_fileobj(bucket, key, f)
                    
            logger.info(f"Downloaded from S3: {key}")
            return cache_path
            
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            raise
            
    async def delete_file(
        self,
        file_id: str,
        user_id: str
    ) -> bool:
        """
        Delete file from storage
        
        Args:
            file_id: File identifier
            user_id: User ID for access control
            
        Returns:
            Success status
        """
        try:
            # Check metadata
            metadata = self.file_metadata.get(file_id)
            if not metadata:
                return False
                
            # Verify ownership
            if metadata["user_id"] != user_id:
                raise PermissionError("Access denied")
                
            # Delete local file
            local_path = Path(metadata["local_path"])
            if local_path.exists():
                os.remove(local_path)
                
            # Delete from S3 if enabled
            if self.s3_client:
                s3_key = f"{metadata['file_type']}/{user_id}/{file_id}"
                async with self.s3_client as s3:
                    await s3.delete_object(Bucket=self.bucket_name, Key=s3_key)
                    
            # Remove metadata
            del self.file_metadata[file_id]
            
            logger.info(f"Deleted file: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False
            
    async def get_file_path(
        self,
        file_id: str,
        user_id: str
    ) -> str:
        """
        Get local file path with access control
        
        Args:
            file_id: File identifier
            user_id: User ID for access control
            
        Returns:
            Local file path
        """
        metadata = self.file_metadata.get(file_id)
        if not metadata:
            raise FileNotFoundError(f"File not found: {file_id}")
            
        # Verify access
        if metadata["user_id"] != user_id:
            raise PermissionError("Access denied")
            
        return metadata["local_path"]
        
    async def get_file_metadata(
        self,
        file_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get file metadata
        
        Args:
            file_id: File identifier
            user_id: User ID for access control
            
        Returns:
            File metadata or None
        """
        metadata = self.file_metadata.get(file_id)
        if metadata and metadata["user_id"] == user_id:
            return metadata
        return None
        
    async def list_user_files(
        self,
        user_id: str,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List user's files
        
        Args:
            user_id: User ID
            file_type: Optional filter by type
            
        Returns:
            List of file metadata
        """
        user_files = []
        
        for file_id, metadata in self.file_metadata.items():
            if metadata["user_id"] == user_id:
                if file_type is None or metadata["file_type"] == file_type:
                    user_files.append({
                        "file_id": file_id,
                        **metadata
                    })
                    
        # Sort by upload date
        user_files.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        return user_files
        
    async def get_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's storage statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Storage statistics
        """
        total_size = 0
        file_count = 0
        type_counts = {}
        
        for metadata in self.file_metadata.values():
            if metadata["user_id"] == user_id:
                total_size += metadata["size"]
                file_count += 1
                file_type = metadata["file_type"]
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
                
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "files_by_type": type_counts,
            "storage_limit_mb": 1000,  # 1GB limit per user
            "usage_percent": round((total_size / (1000 * 1024 * 1024)) * 100, 2)
        }
        
    async def _cleanup_temp_files(self):
        """Clean up old temporary files"""
        try:
            now = datetime.utcnow()
            
            for file_path in self.temp_path.iterdir():
                if file_path.is_file():
                    # Check file age
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age = now - mtime
                    
                    # Delete if older than 1 hour
                    if age.total_seconds() > 3600:
                        os.remove(file_path)
                        logger.info(f"Cleaned up temp file: {file_path.name}")
                        
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")