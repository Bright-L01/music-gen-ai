#!/usr/bin/env python3
"""
Simple test script to verify the system works
"""

import asyncio
import sys
import time

import httpx


async def test_system():
    """Test basic system functionality"""
    
    print("🎵 Testing Music Generation Platform...")
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        
        # Test 1: Health check
        print("1. Testing API Gateway health...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("   ✅ API Gateway is healthy")
            else:
                print(f"   ❌ API Gateway returned {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Cannot reach API Gateway: {e}")
            print(f"   💡 Make sure to run: docker-compose -f docker-compose.microservices.yml up -d")
            return False
            
        # Test 2: Services health  
        print("2. Testing all services...")
        try:
            response = await client.get(f"{base_url}/health/services")
            if response.status_code == 200:
                data = response.json()
                healthy = data.get("healthy_services", 0)
                total = data.get("total_services", 0)
                print(f"   ✅ {healthy}/{total} services are healthy")
                
                if healthy < total:
                    print("   ⚠️  Some services may still be starting up")
                    
            else:
                print(f"   ❌ Services health check failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Services health check error: {e}")
            
        # Test 3: Try user registration
        print("3. Testing user registration...")
        try:
            test_user = f"testuser_{int(time.time())}"
            user_data = {
                "username": test_user,
                "email": f"{test_user}@example.com", 
                "password": "testpass123"
            }
            
            response = await client.post(f"{base_url}/auth/register", json=user_data)
            if response.status_code == 200:
                print("   ✅ User registration works")
                result = response.json()
                token = result.get("access_token")
                
                # Test 4: Try music generation
                print("4. Testing music generation...")
                headers = {"Authorization": f"Bearer {token}"}
                gen_data = {
                    "prompt": "Simple test melody",
                    "duration": 5.0
                }
                
                response = await client.post(
                    f"{base_url}/generate", 
                    json=gen_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    job = response.json()
                    print(f"   ✅ Music generation started (Job ID: {job.get('job_id')})")
                else:
                    print(f"   ❌ Music generation failed: {response.status_code}")
                    
            else:
                print(f"   ❌ User registration failed: {response.status_code}")
                if response.status_code == 500:
                    print("   💡 Database may not be ready yet")
                    
        except Exception as e:
            print(f"   ❌ User registration error: {e}")
            
    print("\n🎉 Basic system test completed!")
    return True


if __name__ == "__main__":
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)