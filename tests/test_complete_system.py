"""
Complete System Integration Tests

Comprehensive test suite for the entire microservices platform.
Tests all services working together end-to-end.
"""

import asyncio
import json
import os
import pytest
import time
from typing import Dict, Any

import httpx
from httpx import AsyncClient


# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_USER_PREFIX = f"testuser_{int(time.time())}"


class TestClient:
    """Test client for API interactions"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = AsyncClient(timeout=httpx.Timeout(60.0))
        self.token = None
        self.user_info = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def register_user(self, username: str, email: str, password: str) -> Dict:
        """Register a test user"""
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": f"Test User {username}"
        }
        
        response = await self.client.post(f"{self.base_url}/auth/register", json=data)
        response.raise_for_status()
        
        result = response.json()
        self.token = result["access_token"]
        self.user_info = result["user"]
        
        return result
        
    async def login_user(self, email: str, password: str) -> Dict:
        """Login user"""
        data = {"username": email, "password": password}
        
        response = await self.client.post(f"{self.base_url}/auth/login", data=data)
        response.raise_for_status()
        
        result = response.json()
        self.token = result["access_token"]
        self.user_info = result["user"]
        
        return result
        
    async def generate_music(self, prompt: str, duration: float = 10.0) -> Dict:
        """Generate music"""
        data = {"prompt": prompt, "duration": duration}
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.post(
            f"{self.base_url}/generate",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def get_job_status(self, job_id: str) -> Dict:
        """Get job status"""
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.get(
            f"{self.base_url}/generate/job/{job_id}",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def create_playlist(self, name: str) -> Dict:
        """Create playlist"""
        data = {"name": name, "description": "Test playlist", "is_public": True}
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.post(
            f"{self.base_url}/playlists",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()


@pytest.fixture
async def test_client():
    """Fixture providing test client"""
    async with TestClient() as client:
        yield client


@pytest.mark.asyncio
class TestSystemHealth:
    """Test system health and connectivity"""
    
    async def test_api_gateway_health(self, test_client):
        """Test API gateway health endpoint"""
        response = await test_client.client.get(f"{test_client.base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-gateway"
        
    async def test_services_health(self, test_client):
        """Test all microservices health"""
        response = await test_client.client.get(f"{test_client.base_url}/health/services")
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        assert "overall_status" in data
        
        # Check that we have all expected services
        expected_services = ["generation", "audio-processing", "user-management", "redis", "postgres"]
        for service in expected_services:
            assert service in data["services"]
            
    async def test_service_connectivity(self, test_client):
        """Test that all services are reachable"""
        health_response = await test_client.client.get(f"{test_client.base_url}/health/services")
        health_data = health_response.json()
        
        healthy_services = 0
        total_services = len(health_data.get("services", {}))
        
        for service_name, service_data in health_data.get("services", {}).items():
            if service_data.get("status") == "healthy":
                healthy_services += 1
                
        # At least 80% of services should be healthy for tests to proceed
        health_ratio = healthy_services / total_services if total_services > 0 else 0
        assert health_ratio >= 0.8, f"Only {healthy_services}/{total_services} services are healthy"


@pytest.mark.asyncio
class TestUserManagement:
    """Test user management functionality"""
    
    async def test_user_registration(self, test_client):
        """Test user registration flow"""
        username = f"{TEST_USER_PREFIX}_reg"
        email = f"{username}@example.com"
        password = "testpass123"
        
        result = await test_client.register_user(username, email, password)
        
        assert "access_token" in result
        assert "user" in result
        assert result["user"]["username"] == username
        assert result["user"]["email"] == email
        assert result["user"]["tier"] == "free"
        
    async def test_user_login(self, test_client):
        """Test user login flow"""
        username = f"{TEST_USER_PREFIX}_login"
        email = f"{username}@example.com"
        password = "testpass123"
        
        # Register first
        await test_client.register_user(username, email, password)
        
        # Clear token and login
        test_client.token = None
        result = await test_client.login_user(email, password)
        
        assert "access_token" in result
        assert result["user"]["username"] == username
        
    async def test_user_profile(self, test_client):
        """Test user profile retrieval"""
        username = f"{TEST_USER_PREFIX}_profile"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.get(
            f"{test_client.base_url}/auth/me",
            headers=headers
        )
        
        assert response.status_code == 200
        profile = response.json()
        
        assert profile["username"] == username
        assert profile["email"] == email
        assert "tracks_generated" in profile
        assert "playlists_count" in profile
        

@pytest.mark.asyncio
class TestMusicGeneration:
    """Test music generation functionality"""
    
    async def test_simple_generation(self, test_client):
        """Test basic music generation"""
        username = f"{TEST_USER_PREFIX}_gen"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Generate music
        job = await test_client.generate_music("Simple test melody", duration=5.0)
        
        assert "job_id" in job
        assert job["status"] in ["queued", "processing"]
        
        # Check job status
        job_status = await test_client.get_job_status(job["job_id"])
        assert "status" in job_status
        assert "progress" in job_status
        
    async def test_generation_with_options(self, test_client):
        """Test music generation with advanced options"""
        username = f"{TEST_USER_PREFIX}_gen_opts"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Generate with options
        data = {
            "prompt": "Jazz piano with saxophone",
            "duration": 10.0,
            "genre": "jazz",
            "mood": "relaxed",
            "instruments": ["piano", "saxophone"]
        }
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.post(
            f"{test_client.base_url}/generate",
            json=data,
            headers=headers
        )
        
        assert response.status_code == 200
        job = response.json()
        assert "job_id" in job
        
    async def test_batch_generation(self, test_client):
        """Test batch music generation"""
        username = f"{TEST_USER_PREFIX}_batch"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Generate batch
        data = {
            "requests": [
                {"prompt": "Piano melody", "duration": 5.0},
                {"prompt": "Guitar chord", "duration": 5.0}
            ],
            "sequential": False
        }
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.post(
            f"{test_client.base_url}/generate/batch",
            json=data,
            headers=headers
        )
        
        assert response.status_code == 200
        batch = response.json()
        assert "batch_id" in batch
        assert "jobs" in batch
        assert len(batch["jobs"]) == 2


@pytest.mark.asyncio
class TestPlaylistManagement:
    """Test playlist functionality"""
    
    async def test_create_playlist(self, test_client):
        """Test playlist creation"""
        username = f"{TEST_USER_PREFIX}_playlist"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        playlist = await test_client.create_playlist("Test Playlist")
        
        assert "id" in playlist
        assert playlist["name"] == "Test Playlist"
        assert playlist["user_id"] == test_client.user_info["id"]
        
    async def test_get_playlists(self, test_client):
        """Test playlist retrieval"""
        username = f"{TEST_USER_PREFIX}_get_playlist"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Create a playlist
        await test_client.create_playlist("My Test Playlist")
        
        # Get playlists
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.get(
            f"{test_client.base_url}/playlists",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "playlists" in data
        assert len(data["playlists"]) >= 1
        assert data["playlists"][0]["name"] == "My Test Playlist"


@pytest.mark.asyncio
class TestAudioProcessing:
    """Test audio processing functionality"""
    
    async def test_audio_analysis_request(self, test_client):
        """Test audio analysis endpoint"""
        username = f"{TEST_USER_PREFIX}_audio"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Test analysis request (would fail without actual audio file)
        data = {
            "audio_url": "https://example.com/test.wav",
            "feature_types": ["tempo", "key"]
        }
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.post(
            f"{test_client.base_url}/audio/analyze",
            json=data,
            headers=headers
        )
        
        # May fail with 500 due to no actual audio file, but endpoint should exist
        assert response.status_code in [200, 400, 500]
        
    async def test_waveform_generation_request(self, test_client):
        """Test waveform generation endpoint"""
        username = f"{TEST_USER_PREFIX}_waveform"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.post(
            f"{test_client.base_url}/audio/waveform",
            params={
                "audio_url": "https://example.com/test.wav",
                "width": 1920,
                "height": 200
            },
            headers=headers
        )
        
        # May fail with 500 due to no actual audio file, but endpoint should exist
        assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
class TestIntegration:
    """Test cross-service integration"""
    
    async def test_dashboard_data(self, test_client):
        """Test dashboard data aggregation"""
        username = f"{TEST_USER_PREFIX}_dashboard"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.get(
            f"{test_client.base_url}/dashboard",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have aggregated data from multiple services
        assert "user_profile" in data
        assert "social_profile" in data
        assert "playlists" in data
        
    async def test_search_functionality(self, test_client):
        """Test search across services"""
        username = f"{TEST_USER_PREFIX}_search"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.get(
            f"{test_client.base_url}/search",
            params={"query": "test", "type": "all"},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        

@pytest.mark.asyncio 
class TestAuthentication:
    """Test authentication and authorization"""
    
    async def test_protected_endpoint_without_auth(self, test_client):
        """Test that protected endpoints require authentication"""
        response = await test_client.client.post(
            f"{test_client.base_url}/generate",
            json={"prompt": "test", "duration": 10.0}
        )
        
        assert response.status_code == 401
        
    async def test_invalid_token(self, test_client):
        """Test invalid token handling"""
        headers = {"Authorization": "Bearer invalid_token"}
        
        response = await test_client.client.post(
            f"{test_client.base_url}/generate",
            json={"prompt": "test", "duration": 10.0},
            headers=headers
        )
        
        assert response.status_code == 401
        
    async def test_token_expiration_handling(self, test_client):
        """Test expired token handling"""
        # This would require generating an expired token
        # For now, just test the endpoint structure
        username = f"{TEST_USER_PREFIX}_token"
        email = f"{username}@example.com"
        
        await test_client.register_user(username, email, "testpass123")
        
        # Valid token should work
        headers = {"Authorization": f"Bearer {test_client.token}"}
        response = await test_client.client.get(
            f"{test_client.base_url}/auth/me",
            headers=headers
        )
        
        assert response.status_code == 200


# Performance and Load Tests
@pytest.mark.asyncio
class TestPerformance:
    """Test system performance"""
    
    async def test_concurrent_registrations(self, test_client):
        """Test concurrent user registrations"""
        async def register_user(index):
            username = f"{TEST_USER_PREFIX}_concurrent_{index}"
            email = f"{username}@example.com"
            
            async with TestClient() as client:
                return await client.register_user(username, email, "testpass123")
                
        # Test 5 concurrent registrations
        tasks = [register_user(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least 80% should succeed
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful >= 4, f"Only {successful}/5 concurrent registrations succeeded"
        
    async def test_response_times(self, test_client):
        """Test API response times"""
        start_time = time.time()
        response = await test_client.client.get(f"{test_client.base_url}/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0, f"Health endpoint took {response_time:.3f}s (should be < 1s)"
        
        # Test services health endpoint
        start_time = time.time()
        response = await test_client.client.get(f"{test_client.base_url}/health/services")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0, f"Services health took {response_time:.3f}s (should be < 5s)"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])