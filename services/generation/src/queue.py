"""
Generation Queue Management

Handles job queueing, prioritization, and state management using Redis.
Supports multiple priority levels and fair scheduling.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

import redis.asyncio as redis
from redis.exceptions import RedisError

from .models import GenerationStatus


logger = logging.getLogger(__name__)


class QueuePriority(str, Enum):
    """Queue priority levels"""
    HIGH = "high"       # Premium users, paid jobs
    NORMAL = "normal"   # Regular users
    LOW = "low"         # Free tier, batch jobs


class GenerationQueue:
    """
    Manages generation job queue with Redis
    
    Queue structure:
    - queue:generation:high -> List of high priority job IDs
    - queue:generation:normal -> List of normal priority job IDs  
    - queue:generation:low -> List of low priority job IDs
    - job:{job_id} -> Job data (JSON)
    - job:{job_id}:status -> Job status
    - stats:queue -> Queue statistics
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Queue keys
        self.QUEUE_HIGH = "queue:generation:high"
        self.QUEUE_NORMAL = "queue:generation:normal"
        self.QUEUE_LOW = "queue:generation:low"
        self.QUEUE_STATS = "stats:queue"
        
        # Job expiry (7 days)
        self.JOB_EXPIRY = 7 * 24 * 60 * 60
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            
            # Setup pub/sub for job updates
            self.pubsub = self.client.pubsub()
            
            logger.info("Connected to Redis queue")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.close()
            
    async def enqueue(
        self,
        job_id: str,
        job_data: Dict[str, Any],
        priority: str = "normal"
    ) -> int:
        """
        Add a job to the queue
        
        Returns:
            Position in queue
        """
        # Store job data
        job_key = f"job:{job_id}"
        await self.client.setex(
            job_key,
            self.JOB_EXPIRY,
            json.dumps(job_data)
        )
        
        # Add to appropriate queue
        queue_key = self._get_queue_key(priority)
        position = await self.client.rpush(queue_key, job_id)
        
        # Update stats
        await self._increment_stat("total_enqueued")
        await self._increment_stat(f"enqueued_{priority}")
        
        # Publish event
        await self.client.publish(
            "queue:events",
            json.dumps({
                "event": "job_enqueued",
                "job_id": job_id,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        logger.info(f"Job {job_id} enqueued with {priority} priority at position {position}")
        return position
        
    async def dequeue(self, priorities: List[str] = None) -> Optional[str]:
        """
        Get next job from queue (priority order)
        
        Args:
            priorities: List of priorities to check (default: all)
            
        Returns:
            Job ID or None if queue is empty
        """
        if not priorities:
            priorities = [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]
            
        for priority in priorities:
            queue_key = self._get_queue_key(priority)
            job_id = await self.client.lpop(queue_key)
            
            if job_id:
                # Update stats
                await self._increment_stat("total_dequeued")
                await self._increment_stat(f"dequeued_{priority}")
                
                logger.info(f"Dequeued job {job_id} from {priority} queue")
                return job_id
                
        return None
        
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data"""
        job_key = f"job:{job_id}"
        data = await self.client.get(job_key)
        
        if data:
            return json.loads(data)
        return None
        
    async def update_job(self, job_id: str, job_data: Dict[str, Any]):
        """Update job data"""
        job_key = f"job:{job_id}"
        
        # Get current TTL
        ttl = await self.client.ttl(job_key)
        if ttl > 0:
            await self.client.setex(
                job_key,
                ttl,
                json.dumps(job_data)
            )
            
            # Publish update event
            await self.client.publish(
                f"job:{job_id}:updates",
                json.dumps({
                    "job_id": job_id,
                    "status": job_data.get("status"),
                    "progress": job_data.get("progress", 0),
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
            
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        # Remove from all queues
        removed = False
        for priority in [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
            queue_key = self._get_queue_key(priority)
            count = await self.client.lrem(queue_key, 0, job_id)
            if count > 0:
                removed = True
                
        if removed:
            # Update job status
            job_data = await self.get_job(job_id)
            if job_data:
                job_data["status"] = GenerationStatus.CANCELLED
                await self.update_job(job_id, job_data)
                
            # Update stats
            await self._increment_stat("total_cancelled")
            
            logger.info(f"Job {job_id} cancelled")
            
        return removed
        
    async def position(self, job_id: str) -> Optional[int]:
        """Get position of job in queue"""
        position = 0
        
        for priority in [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
            queue_key = self._get_queue_key(priority)
            queue = await self.client.lrange(queue_key, 0, -1)
            
            if job_id in queue:
                return position + queue.index(job_id) + 1
                
            position += len(queue)
            
        return None
        
    async def size(self) -> int:
        """Get total queue size"""
        total = 0
        for priority in [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
            queue_key = self._get_queue_key(priority)
            total += await self.client.llen(queue_key)
        return total
        
    async def sizes_by_priority(self) -> Dict[str, int]:
        """Get queue sizes by priority"""
        sizes = {}
        for priority in [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
            queue_key = self._get_queue_key(priority)
            sizes[priority] = await self.client.llen(queue_key)
        return sizes
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = await self.client.hgetall(self.QUEUE_STATS)
        
        # Convert string values to int
        return {k: int(v) for k, v in stats.items()}
        
    async def clear_completed(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours"""
        # Implementation would scan for completed jobs and remove them
        pass
        
    async def subscribe_to_job(self, job_id: str):
        """Subscribe to job updates"""
        channel = f"job:{job_id}:updates"
        await self.pubsub.subscribe(channel)
        return self.pubsub
        
    def _get_queue_key(self, priority: str) -> str:
        """Get Redis key for priority queue"""
        priority_map = {
            "high": self.QUEUE_HIGH,
            "normal": self.QUEUE_NORMAL,
            "low": self.QUEUE_LOW
        }
        return priority_map.get(priority, self.QUEUE_NORMAL)
        
    async def _increment_stat(self, stat_name: str):
        """Increment a statistic counter"""
        await self.client.hincrby(self.QUEUE_STATS, stat_name, 1)
        

class QueueWorker:
    """
    Worker that processes jobs from the queue
    """
    
    def __init__(self, queue: GenerationQueue, generation_service):
        self.queue = queue
        self.generation_service = generation_service
        self.running = False
        
    async def start(self):
        """Start processing jobs"""
        self.running = True
        logger.info("Queue worker started")
        
        while self.running:
            try:
                # Get next job
                job_id = await self.queue.dequeue()
                
                if not job_id:
                    # No jobs, wait a bit
                    await asyncio.sleep(1)
                    continue
                    
                # Process job
                await self._process_job(job_id)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(5)
                
    async def stop(self):
        """Stop worker"""
        self.running = False
        
    async def _process_job(self, job_id: str):
        """Process a single job"""
        try:
            # Get job data
            job_data = await self.queue.get_job(job_id)
            if not job_data:
                logger.warning(f"Job {job_id} not found")
                return
                
            # Update status
            job_data["status"] = GenerationStatus.PROCESSING
            job_data["started_at"] = datetime.utcnow().isoformat()
            await self.queue.update_job(job_id, job_data)
            
            # Generate audio
            request = job_data["request"]
            
            # Progress callback
            async def update_progress(progress: float):
                job_data["progress"] = progress
                await self.queue.update_job(job_id, job_data)
                
            if request.get("structure"):
                audio_url = await self.generation_service.generate_structured(
                    prompt=request["prompt"],
                    structure=request["structure"],
                    duration=request["duration"],
                    progress_callback=update_progress
                )
            else:
                audio_url = await self.generation_service.generate_simple(
                    prompt=request["prompt"],
                    duration=request["duration"],
                    temperature=request.get("temperature", 1.0),
                    progress_callback=update_progress
                )
                
            # Update job with result
            job_data["status"] = GenerationStatus.COMPLETED
            job_data["audio_url"] = audio_url
            job_data["completed_at"] = datetime.utcnow().isoformat()
            job_data["progress"] = 100.0
            await self.queue.update_job(job_id, job_data)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Update job with error
            job_data["status"] = GenerationStatus.FAILED
            job_data["error"] = str(e)
            job_data["completed_at"] = datetime.utcnow().isoformat()
            await self.queue.update_job(job_id, job_data)