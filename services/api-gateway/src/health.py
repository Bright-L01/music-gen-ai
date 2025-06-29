"""
Health Checking System

Monitors health of all microservices and provides aggregated health status.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

import httpx

from .models import ServiceStatus


logger = logging.getLogger(__name__)


class HealthChecker:
    """Monitors health of all microservices"""
    
    def __init__(self):
        self.services = {
            "generation": "http://generation:8001",
            "audio-processing": "http://audio-processing:8002", 
            "user-management": "http://user-management:8003",
            "redis": "redis://redis:6379",
            "postgres": "postgresql://postgres:password@postgres:5432/user_management"
        }
        
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        self.health_history = {}
        self.check_interval = 30  # seconds
        self.background_task = None
        self._running = False
        
    async def initialize(self):
        """Start background health checking"""
        logger.info("Initializing Health Checker...")
        
        # Initial health check
        await self.check_all_services()
        
        # Start background monitoring
        self._running = True
        self.background_task = asyncio.create_task(self._background_check())
        
        logger.info("Health Checker initialized")
        
    async def cleanup(self):
        """Stop background health checking"""
        self._running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
                
        await self.client.aclose()
        
    async def _background_check(self):
        """Background task for periodic health checks"""
        while self._running:
            try:
                await self.check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background health check error: {e}")
                await asyncio.sleep(5)  # Short delay on error
                
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services"""
        results = {}
        
        # Check HTTP services concurrently
        http_services = {
            name: url for name, url in self.services.items()
            if url.startswith("http")
        }
        
        http_tasks = [
            self.check_http_service(name, url)
            for name, url in http_services.items()
        ]
        
        http_results = await asyncio.gather(*http_tasks, return_exceptions=True)
        
        for i, (service_name, _) in enumerate(http_services.items()):
            result = http_results[i]
            if isinstance(result, Exception):
                results[service_name] = ServiceStatus(
                    name=service_name,
                    status="unhealthy",
                    last_check=datetime.utcnow(),
                    error=str(result)
                ).dict()
            else:
                results[service_name] = result
                
        # Check database services
        results["redis"] = await self.check_redis()
        results["postgres"] = await self.check_postgres()
        
        # Store results
        self.health_history[datetime.utcnow()] = results
        
        # Clean old history (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.health_history = {
            timestamp: data
            for timestamp, data in self.health_history.items()
            if timestamp > cutoff
        }
        
        # Calculate overall health
        healthy_services = sum(1 for service in results.values() if service["status"] == "healthy")
        total_services = len(results)
        
        overall_status = {
            "overall_status": "healthy" if healthy_services == total_services else "degraded",
            "healthy_services": healthy_services,
            "total_services": total_services,
            "services": results,
            "last_check": datetime.utcnow().isoformat()
        }
        
        return overall_status
        
    async def check_http_service(self, name: str, url: str) -> Dict[str, Any]:
        """Check health of an HTTP service"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{url}/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                service_info = response.json()
                return ServiceStatus(
                    name=name,
                    status="healthy",
                    response_time=response_time,
                    last_check=datetime.utcnow()
                ).dict()
            else:
                return ServiceStatus(
                    name=name,
                    status="unhealthy",
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}"
                ).dict()
                
        except httpx.TimeoutException:
            return ServiceStatus(
                name=name,
                status="unhealthy",
                last_check=datetime.utcnow(),
                error="Timeout"
            ).dict()
            
        except httpx.ConnectError:
            return ServiceStatus(
                name=name,
                status="unhealthy", 
                last_check=datetime.utcnow(),
                error="Connection failed"
            ).dict()
            
        except Exception as e:
            return ServiceStatus(
                name=name,
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e)
            ).dict()
            
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            import redis.asyncio as redis
            
            client = redis.from_url("redis://redis:6379/0")
            
            start_time = time.time()
            await client.ping()
            end_time = time.time()
            
            response_time = end_time - start_time
            
            await client.close()
            
            return ServiceStatus(
                name="redis",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow()
            ).dict()
            
        except Exception as e:
            return ServiceStatus(
                name="redis",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e)
            ).dict()
            
    async def check_postgres(self) -> Dict[str, Any]:
        """Check PostgreSQL health"""
        try:
            import asyncpg
            
            start_time = time.time()
            conn = await asyncpg.connect(
                "postgresql://postgres:password@postgres:5432/user_management"
            )
            
            # Simple query
            await conn.fetchval("SELECT 1")
            await conn.close()
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return ServiceStatus(
                name="postgres",
                status="healthy",
                response_time=response_time,
                last_check=datetime.utcnow()
            ).dict()
            
        except Exception as e:
            return ServiceStatus(
                name="postgres",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e)
            ).dict()
            
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for a specific service"""
        if not self.health_history:
            return {"error": "No health data available"}
            
        latest_check = max(self.health_history.keys())
        latest_data = self.health_history[latest_check]
        
        return latest_data.get(service_name, {"error": "Service not found"})
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of service health over time"""
        if not self.health_history:
            return {"error": "No health data available"}
            
        # Get latest status
        latest_check = max(self.health_history.keys())
        latest_data = self.health_history[latest_check]
        
        # Calculate uptime percentages
        total_checks = len(self.health_history)
        service_stats = {}
        
        for service_name in latest_data.get("services", {}):
            healthy_checks = sum(
                1 for data in self.health_history.values()
                if data.get("services", {}).get(service_name, {}).get("status") == "healthy"
            )
            
            uptime_percent = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
            
            service_stats[service_name] = {
                "current_status": latest_data["services"][service_name]["status"],
                "uptime_percent": round(uptime_percent, 2),
                "total_checks": total_checks,
                "healthy_checks": healthy_checks
            }
            
        return {
            "summary": service_stats,
            "overall_status": latest_data.get("overall_status", "unknown"),
            "check_interval": self.check_interval,
            "data_points": total_checks,
            "time_range_hours": 1
        }