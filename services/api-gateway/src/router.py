"""
Service Router

Handles routing requests to different microservices.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any

import httpx
from fastapi import HTTPException, Request, status


logger = logging.getLogger(__name__)


class ServiceRouter:
    """Routes requests to appropriate microservices"""
    
    def __init__(self):
        self.generation_url = os.getenv("GENERATION_SERVICE_URL", "http://localhost:8001")
        self.audio_url = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8002")
        self.user_url = os.getenv("USER_SERVICE_URL", "http://localhost:8003")
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the router"""
        if self._initialized:
            return
            
        logger.info("Initializing Service Router...")
        
        # Test connectivity to all services
        services = {
            "generation": self.generation_url,
            "audio-processing": self.audio_url,
            "user-management": self.user_url
        }
        
        for service_name, url in services.items():
            try:
                response = await self.client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} service healthy at {url}")
                else:
                    logger.warning(f"⚠ {service_name} service returned {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠ Could not reach {service_name} service at {url}: {e}")
                
        self._initialized = True
        logger.info("Service Router initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()
        
    async def route_to_generation_service(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        form_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Route request to generation service"""
        return await self._route_request(
            self.generation_url,
            method,
            path,
            json_data,
            form_data,
            headers,
            params
        )
        
    async def route_to_audio_service(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        form_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Route request to audio processing service"""
        return await self._route_request(
            self.audio_url,
            method,
            path,
            json_data,
            form_data,
            headers,
            params
        )
        
    async def route_to_user_service(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        form_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Route request to user management service"""
        return await self._route_request(
            self.user_url,
            method,
            path,
            json_data,
            form_data,
            headers,
            params
        )
        
    async def stream_to_audio_service(
        self,
        method: str,
        path: str,
        request: Request,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Stream request to audio service (for file uploads)"""
        url = f"{self.audio_url}{path}"
        
        try:
            # Stream the request body
            async with self.client.stream(
                method,
                url,
                content=request.stream(),
                headers={
                    **request.headers,
                    **(headers or {})
                }
            ) as response:
                response.raise_for_status()
                return await response.aread()
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from audio service: {e.response.status_code}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Audio service error: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error to audio service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audio service unavailable"
            )
            
    async def _route_request(
        self,
        base_url: str,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        form_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Internal method to route requests"""
        url = f"{base_url}{path}"
        
        # Prepare request parameters
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "params": params or {}
        }
        
        if json_data is not None:
            request_kwargs["json"] = json_data
        elif form_data is not None:
            request_kwargs["data"] = form_data
            
        try:
            response = await self.client.request(**request_kwargs)
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                return response.json()
            else:
                return {"data": response.text}
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from service: {e.response.status_code} - {e.response.text}")
            
            # Try to parse error response
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = e.response.text or str(e)
                
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unavailable: {str(e)}"
            )
            
    async def health_check_service(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        url_map = {
            "generation": self.generation_url,
            "audio-processing": self.audio_url,
            "user-management": self.user_url
        }
        
        base_url = url_map.get(service_name)
        if not base_url:
            return {
                "service": service_name,
                "status": "unknown",
                "error": "Unknown service"
            }
            
        try:
            start_time = asyncio.get_event_loop().time()
            response = await self.client.get(f"{base_url}/health", timeout=5.0)
            end_time = asyncio.get_event_loop().time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                return {
                    "service": service_name,
                    "status": "healthy",
                    "response_time": response_time,
                    "details": response.json()
                }
            else:
                return {
                    "service": service_name,
                    "status": "unhealthy",
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "service": service_name,
                "status": "unhealthy",
                "error": str(e)
            }
            
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all services"""
        stats = {}
        
        services = ["generation", "audio-processing", "user-management"]
        
        for service in services:
            try:
                health = await self.health_check_service(service)
                stats[service] = health
            except Exception as e:
                stats[service] = {
                    "service": service,
                    "status": "error",
                    "error": str(e)
                }
                
        return stats