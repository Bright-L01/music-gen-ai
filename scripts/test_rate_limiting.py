#!/usr/bin/env python3
"""
Test script for rate limiting middleware with various proxy configurations.
Tests proxy header handling, rate limit enforcement, and internal service bypass.
"""

import asyncio
import time
import argparse
from typing import Dict, List, Optional
import httpx
import json
from datetime import datetime

# Test configurations
TEST_CONFIGS = {
    "direct": {
        "name": "Direct Connection",
        "headers": {},
        "expected_ip": "client"
    },
    "nginx": {
        "name": "Behind Nginx",
        "headers": {
            "X-Real-IP": "203.0.113.42",
            "X-Forwarded-For": "203.0.113.42, 10.0.0.1"
        },
        "expected_ip": "203.0.113.42"
    },
    "cloudflare": {
        "name": "Behind Cloudflare",
        "headers": {
            "CF-Connecting-IP": "198.51.100.178",
            "X-Forwarded-For": "198.51.100.178, 172.71.255.1, 10.0.0.1"
        },
        "expected_ip": "198.51.100.178"
    },
    "multi_proxy": {
        "name": "Multiple Proxies",
        "headers": {
            "X-Forwarded-For": "192.0.2.123, 172.31.255.255, 10.0.0.1",
            "X-Real-IP": "192.0.2.123"
        },
        "expected_ip": "192.0.2.123"
    },
    "forwarded": {
        "name": "RFC 7239 Forwarded Header",
        "headers": {
            "Forwarded": 'for="[2001:db8:cafe::17]:4711"; proto=https; by=10.0.0.1'
        },
        "expected_ip": "2001:db8:cafe::17"
    },
    "internal": {
        "name": "Internal Service",
        "headers": {
            "X-Internal-Service": "monitoring-service",
            "X-API-Key": "test-internal-key"
        },
        "expected_ip": "internal"
    }
}


class RateLimitTester:
    """Test rate limiting functionality"""
    
    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url
        self.verbose = verbose
        self.results: List[Dict] = []
    
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting without proxy headers"""
        print("\n=== Testing Basic Rate Limiting ===")
        
        async with httpx.AsyncClient() as client:
            # Make requests up to the limit
            for i in range(35):  # Free tier: 30/min
                response = await client.get(f"{self.base_url}/api/v1/models")
                
                if i < 30:
                    assert response.status_code == 200, f"Request {i+1} should succeed"
                    if self.verbose:
                        print(f"Request {i+1}: OK - Remaining: {response.headers.get('X-RateLimit-Minute-Remaining')}")
                else:
                    assert response.status_code == 429, f"Request {i+1} should be rate limited"
                    print(f"Request {i+1}: Rate limited (expected)")
                    
                    # Check retry-after header
                    retry_after = response.headers.get("Retry-After")
                    assert retry_after, "Retry-After header should be present"
                    print(f"Retry-After: {retry_after} seconds")
                    break
                
                await asyncio.sleep(0.1)  # Small delay between requests
    
    async def test_proxy_headers(self):
        """Test IP extraction from various proxy headers"""
        print("\n=== Testing Proxy Header Handling ===")
        
        async with httpx.AsyncClient() as client:
            for config_name, config in TEST_CONFIGS.items():
                print(f"\nTesting: {config['name']}")
                
                response = await client.get(
                    f"{self.base_url}/api/v1/models",
                    headers=config["headers"]
                )
                
                # Check response
                assert response.status_code in [200, 429], f"Unexpected status: {response.status_code}"
                
                # Log rate limit info
                tier = response.headers.get("X-RateLimit-Tier", "unknown")
                remaining = response.headers.get("X-RateLimit-Minute-Remaining", "?")
                
                print(f"  - Status: {response.status_code}")
                print(f"  - Tier: {tier}")
                print(f"  - Remaining requests: {remaining}")
                
                self.results.append({
                    "test": config['name'],
                    "status": response.status_code,
                    "tier": tier,
                    "headers_sent": config["headers"]
                })
    
    async def test_burst_protection(self):
        """Test burst protection mechanism"""
        print("\n=== Testing Burst Protection ===")
        
        async with httpx.AsyncClient() as client:
            # Send rapid burst of requests
            burst_results = []
            
            for i in range(10):  # Free tier burst size: 5
                start_time = time.time()
                response = await client.get(f"{self.base_url}/api/v1/models")
                elapsed = time.time() - start_time
                
                burst_results.append({
                    "request": i + 1,
                    "status": response.status_code,
                    "elapsed": elapsed
                })
                
                if response.status_code == 429:
                    print(f"Burst limit reached at request {i+1}")
                    break
            
            # Print results
            for result in burst_results:
                status = "OK" if result["status"] == 200 else "BLOCKED"
                print(f"Request {result['request']}: {status} ({result['elapsed']:.3f}s)")
    
    async def test_tier_limits(self):
        """Test different tier rate limits"""
        print("\n=== Testing Tier-Based Rate Limiting ===")
        
        # Test with different authentication headers simulating different tiers
        tiers = [
            {"name": "Free", "headers": {}},
            {"name": "Premium", "headers": {"Authorization": "Bearer premium-token"}},
            {"name": "Admin", "headers": {"Authorization": "Bearer admin-token"}},
        ]
        
        async with httpx.AsyncClient() as client:
            for tier in tiers:
                print(f"\nTesting {tier['name']} tier:")
                
                response = await client.get(
                    f"{self.base_url}/api/v1/models",
                    headers=tier["headers"]
                )
                
                if response.status_code == 200:
                    tier_name = response.headers.get("X-RateLimit-Tier", "unknown")
                    minute_limit = response.headers.get("X-RateLimit-Minute-Limit", "?")
                    hour_limit = response.headers.get("X-RateLimit-Hour-Limit", "?")
                    day_limit = response.headers.get("X-RateLimit-Day-Limit", "?")
                    
                    print(f"  - Detected tier: {tier_name}")
                    print(f"  - Minute limit: {minute_limit}")
                    print(f"  - Hour limit: {hour_limit}")
                    print(f"  - Day limit: {day_limit}")
    
    async def test_internal_service_bypass(self):
        """Test internal service rate limit bypass"""
        print("\n=== Testing Internal Service Bypass ===")
        
        async with httpx.AsyncClient() as client:
            # Make many requests with internal service headers
            internal_headers = {
                "X-Internal-Service": "monitoring",
                "X-API-Key": "internal-test-key"
            }
            
            success_count = 0
            for i in range(100):  # Should all succeed
                response = await client.get(
                    f"{self.base_url}/api/v1/models",
                    headers=internal_headers
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"Request {i+1} failed with status {response.status_code}")
                    break
            
            print(f"Internal service requests succeeded: {success_count}/100")
            assert success_count == 100, "All internal service requests should succeed"
    
    async def test_rate_limit_headers(self):
        """Test rate limit response headers"""
        print("\n=== Testing Rate Limit Headers ===")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/models")
            
            # Check for required headers
            required_headers = [
                "X-RateLimit-Tier",
                "X-RateLimit-Minute-Limit",
                "X-RateLimit-Minute-Remaining",
                "X-RateLimit-Minute-Reset",
                "X-RateLimit-Hour-Limit",
                "X-RateLimit-Hour-Remaining",
                "X-RateLimit-Hour-Reset",
                "X-RateLimit-Day-Limit",
                "X-RateLimit-Day-Remaining",
                "X-RateLimit-Day-Reset",
            ]
            
            print("Rate limit headers:")
            for header in required_headers:
                value = response.headers.get(header)
                if value:
                    print(f"  - {header}: {value}")
                else:
                    print(f"  - {header}: MISSING")
            
            # Validate reset timestamps
            for window in ["Minute", "Hour", "Day"]:
                reset_header = f"X-RateLimit-{window}-Reset"
                reset_timestamp = response.headers.get(reset_header)
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    print(f"\n{window} reset time: {reset_time.isoformat()}")
    
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        print("\n=== Testing Concurrent Requests ===")
        
        async def make_request(session: httpx.AsyncClient, request_id: int) -> Dict:
            start_time = time.time()
            response = await session.get(f"{self.base_url}/api/v1/models")
            elapsed = time.time() - start_time
            
            return {
                "id": request_id,
                "status": response.status_code,
                "elapsed": elapsed,
                "remaining": response.headers.get("X-RateLimit-Minute-Remaining")
            }
        
        async with httpx.AsyncClient() as client:
            # Send 20 concurrent requests
            tasks = [make_request(client, i) for i in range(20)]
            results = await asyncio.gather(*tasks)
            
            # Analyze results
            successful = sum(1 for r in results if r["status"] == 200)
            rate_limited = sum(1 for r in results if r["status"] == 429)
            
            print(f"Concurrent requests sent: 20")
            print(f"Successful: {successful}")
            print(f"Rate limited: {rate_limited}")
            
            # Show timing
            avg_time = sum(r["elapsed"] for r in results) / len(results)
            print(f"Average response time: {avg_time:.3f}s")
    
    def print_summary(self):
        """Print test summary"""
        print("\n=== Test Summary ===")
        print(f"Total tests performed: {len(self.results)}")
        
        if self.results:
            print("\nProxy Configuration Tests:")
            for result in self.results:
                status = "✓" if result["status"] == 200 else "✗"
                print(f"  {status} {result['test']} - Tier: {result['tier']}")
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.test_rate_limit_headers()
            await self.test_basic_rate_limiting()
            await self.test_proxy_headers()
            await self.test_burst_protection()
            await self.test_tier_limits()
            await self.test_internal_service_bypass()
            await self.test_concurrent_requests()
            
            self.print_summary()
            print("\n✅ All tests completed!")
            
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Test rate limiting middleware")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test",
        choices=["all", "basic", "proxy", "burst", "tier", "internal", "headers", "concurrent"],
        default="all",
        help="Specific test to run"
    )
    
    args = parser.parse_args()
    
    tester = RateLimitTester(args.url, args.verbose)
    
    # Run specific test or all tests
    if args.test == "all":
        asyncio.run(tester.run_all_tests())
    else:
        test_method = getattr(tester, f"test_{args.test}_rate_limiting", None) or \
                      getattr(tester, f"test_{args.test}", None)
        if test_method:
            asyncio.run(test_method())
        else:
            print(f"Unknown test: {args.test}")


if __name__ == "__main__":
    main()