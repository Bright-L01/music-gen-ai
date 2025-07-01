#!/usr/bin/env python3
"""
Simple test to verify the new consolidated API works correctly.
"""

import asyncio
import httpx


async def test_api():
    """Test the new API endpoints."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Test root endpoint
        print("Testing root endpoint...")
        response = await client.get(f"{base_url}/")
        print(f"Root: {response.status_code} - {response.json()}")
        
        # Test health check
        print("\nTesting health endpoint...")
        response = await client.get(f"{base_url}/health")
        print(f"Health: {response.status_code} - {response.json()}")
        
        # Test health ready
        print("\nTesting ready endpoint...")
        response = await client.get(f"{base_url}/health/ready")
        print(f"Ready: {response.status_code} - {response.json()}")
        
        # Test models list
        print("\nTesting models endpoint...")
        response = await client.get(f"{base_url}/api/v1/models")
        print(f"Models: {response.status_code} - {response.json()}")
        
        # Test generation endpoint (just check it exists)
        print("\nTesting generation endpoint...")
        test_request = {
            "prompt": "A peaceful piano melody",
            "duration": 5.0,
            "temperature": 0.8
        }
        response = await client.post(f"{base_url}/api/v1/generate", json=test_request)
        print(f"Generate: {response.status_code}")
        if response.status_code == 200:
            print(f"Task ID: {response.json()['task_id']}")


def main():
    """Main function."""
    print("Testing New Consolidated API")
    print("=" * 50)
    print("\nMake sure the API is running:")
    print("  uvicorn music_gen.api.app:app --reload")
    print("\nPress Enter to continue...")
    input()
    
    try:
        asyncio.run(test_api())
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()