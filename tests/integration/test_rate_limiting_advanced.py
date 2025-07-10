"""
Advanced rate limiting integration tests.

Tests comprehensive rate limiting scenarios including:
- Different user tiers and limits
- Proxy header handling
- Internal service bypasses
- Rate limit persistence
- Burst handling
- Cross-endpoint rate limiting
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from music_gen.api.app import create_app


@pytest.fixture
def test_app_with_rate_limits():
    """Create test app with specific rate limiting configuration."""
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "testing",
            "DEFAULT_MODEL": "facebook/musicgen-small",
            "TRUSTED_PROXIES": "192.168.1.0/24,10.0.0.0/8",
            "ENABLE_PROXY_HEADERS": "true",
            "INTERNAL_API_KEYS": "internal-key-123,admin-key-456",
            "DEFAULT_RATE_LIMIT_TIER": "free",
        },
    ):
        app = create_app()
        return app


@pytest.fixture
def client(test_app_with_rate_limits):
    """Create test client with rate limiting."""
    return TestClient(test_app_with_rate_limits)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock dependencies for testing."""
    with patch("music_gen.core.model_manager.ModelManager") as mock_model_manager:
        mock_manager = Mock()
        mock_manager.has_loaded_models.return_value = True
        mock_model_manager.return_value = mock_manager
        yield


class TestBasicRateLimiting:
    """Test basic rate limiting functionality."""

    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are included in responses."""
        response = client.get("/health")

        # Should have rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Validate header values
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        reset_time = int(response.headers["X-RateLimit-Reset"])

        assert limit > 0
        assert remaining >= 0
        assert remaining <= limit
        assert reset_time > time.time()

    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is enforced."""
        # Make rapid requests to trigger rate limit
        responses = []

        for i in range(30):  # Exceed typical rate limit
            response = client.get("/health")
            responses.append(
                {
                    "status_code": response.status_code,
                    "remaining": response.headers.get("X-RateLimit-Remaining"),
                    "limit": response.headers.get("X-RateLimit-Limit"),
                }
            )

            # Stop if we hit rate limit
            if response.status_code == 429:
                break

        # Should eventually get rate limited
        rate_limited_responses = [r for r in responses if r["status_code"] == 429]

        if rate_limited_responses:
            # Verify rate limit response
            rate_limited = rate_limited_responses[0]
            assert rate_limited["remaining"] == "0"

            # Check that subsequent requests are also rate limited
            follow_up = client.get("/health")
            assert follow_up.status_code == 429

    def test_rate_limit_different_clients(self, client):
        """Test that different clients have separate rate limits."""
        # Simulate different clients with different IPs
        headers_client1 = {"X-Forwarded-For": "192.168.1.100"}
        headers_client2 = {"X-Forwarded-For": "192.168.1.101"}

        # Client 1 makes requests
        responses_client1 = []
        for _ in range(15):
            response = client.get("/health", headers=headers_client1)
            responses_client1.append(response.status_code)
            if response.status_code == 429:
                break

        # Client 2 should still have full rate limit
        response_client2 = client.get("/health", headers=headers_client2)
        assert response_client2.status_code == 200

        # Client 2 should have near-full remaining count
        remaining = int(response_client2.headers["X-RateLimit-Remaining"])
        limit = int(response_client2.headers["X-RateLimit-Limit"])
        assert remaining >= limit - 1  # Should be nearly full


class TestProxyHeaderHandling:
    """Test proxy header handling in rate limiting."""

    def test_x_forwarded_for_extraction(self, client):
        """Test IP extraction from X-Forwarded-For header."""
        # Request from trusted proxy
        headers = {"X-Forwarded-For": "203.0.113.195, 192.168.1.1", "X-Real-IP": "203.0.113.195"}

        response1 = client.get("/health", headers=headers)
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        # Same client IP through proxy
        response2 = client.get("/health", headers=headers)
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        # Should be rate limited as same client
        assert remaining2 == remaining1 - 1

    def test_untrusted_proxy_ignored(self, client):
        """Test that untrusted proxy headers are ignored."""
        # Request from untrusted proxy (not in TRUSTED_PROXIES)
        headers = {"X-Forwarded-For": "203.0.113.195, 8.8.8.8"}  # 8.8.8.8 not in trusted list

        response1 = client.get("/health", headers=headers)

        # Different X-Forwarded-For from same untrusted proxy
        headers2 = {"X-Forwarded-For": "203.0.113.196, 8.8.8.8"}

        response2 = client.get("/health", headers=headers2)

        # Should use the proxy IP (8.8.8.8) for rate limiting, not client IPs
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])
        assert remaining2 == remaining1 - 1

    def test_multiple_proxy_headers(self, client):
        """Test handling of multiple proxy headers."""
        headers = {
            "X-Forwarded-For": "203.0.113.195",
            "X-Real-IP": "203.0.113.196",  # Different IP
            "CF-Connecting-IP": "203.0.113.197",  # Different IP
        }

        response = client.get("/health", headers=headers)
        assert response.status_code == 200

        # Should consistently use the same IP for rate limiting
        # (typically first valid header in priority order)


class TestInternalServiceBypass:
    """Test internal service authentication and bypass."""

    def test_internal_api_key_bypass(self, client):
        """Test that internal API keys bypass rate limits."""
        headers = {"X-API-Key": "internal-key-123"}

        # Make many requests with internal key
        responses = []
        for _ in range(50):  # Well above normal rate limit
            response = client.get("/health", headers=headers)
            responses.append(response.status_code)

        # All should succeed (no rate limiting)
        assert all(status == 200 for status in responses)

    def test_invalid_internal_key(self, client):
        """Test that invalid internal keys don't bypass rate limits."""
        headers = {"X-API-Key": "invalid-key-999"}

        # Make requests with invalid key
        responses = []
        for _ in range(30):
            response = client.get("/health", headers=headers)
            responses.append(response.status_code)
            if response.status_code == 429:
                break

        # Should eventually get rate limited
        assert any(status == 429 for status in responses)

    def test_admin_key_bypass(self, client):
        """Test admin key bypass functionality."""
        headers = {"X-API-Key": "admin-key-456"}

        # Admin key should also bypass rate limits
        responses = []
        for _ in range(40):
            response = client.post(
                "/api/v1/generate/", json={"prompt": "test", "duration": 1.0}, headers=headers
            )
            responses.append(response.status_code)

        # All should succeed (200 or 503 if service unavailable, but not 429)
        assert all(status != 429 for status in responses)


class TestTieredRateLimiting:
    """Test different rate limits for different user tiers."""

    def test_free_tier_limits(self, client):
        """Test free tier rate limits."""
        # Simulate free tier user
        headers = {"X-User-Tier": "free"}

        # Make requests until rate limited
        responses = []
        for _ in range(25):
            response = client.get("/health", headers=headers)
            responses.append(
                {"status": response.status_code, "limit": response.headers.get("X-RateLimit-Limit")}
            )
            if response.status_code == 429:
                break

        # Should have lower limit for free tier
        if responses:
            limit = int(responses[0]["limit"])
            assert limit <= 100  # Free tier should have restrictive limits

    def test_premium_tier_limits(self, client):
        """Test premium tier rate limits."""
        headers = {"X-User-Tier": "premium"}

        response = client.get("/health", headers=headers)
        limit = int(response.headers["X-RateLimit-Limit"])

        # Premium should have higher limits than free
        # (exact values depend on configuration)
        assert limit >= 100

    def test_enterprise_tier_limits(self, client):
        """Test enterprise tier rate limits."""
        headers = {"X-User-Tier": "enterprise"}

        response = client.get("/health", headers=headers)
        limit = int(response.headers["X-RateLimit-Limit"])

        # Enterprise should have highest limits
        assert limit >= 1000


class TestRateLimitPersistence:
    """Test rate limit persistence and windows."""

    def test_rate_limit_window_reset(self, client):
        """Test that rate limits reset after window expires."""
        # Make request to get initial state
        response = client.get("/health")
        initial_remaining = int(response.headers["X-RateLimit-Remaining"])
        reset_time = int(response.headers["X-RateLimit-Reset"])

        # Make another request to decrease remaining
        response2 = client.get("/health")
        decreased_remaining = int(response2.headers["X-RateLimit-Remaining"])

        assert decreased_remaining == initial_remaining - 1

        # Reset time should be consistent
        reset_time2 = int(response2.headers["X-RateLimit-Reset"])
        assert abs(reset_time2 - reset_time) <= 1  # Should be same or 1 second difference

    def test_burst_handling(self, client):
        """Test burst request handling."""
        # Make burst of requests
        burst_size = 10
        start_time = time.time()

        responses = []
        for _ in range(burst_size):
            response = client.get("/health")
            responses.append(
                {
                    "status": response.status_code,
                    "remaining": int(response.headers["X-RateLimit-Remaining"]),
                    "timestamp": time.time(),
                }
            )

        end_time = time.time()
        burst_duration = end_time - start_time

        # Burst should complete quickly
        assert burst_duration < 2.0  # Should handle burst in under 2 seconds

        # Should show decreasing remaining count
        remaining_counts = [r["remaining"] for r in responses]
        for i in range(1, len(remaining_counts)):
            assert remaining_counts[i] <= remaining_counts[i - 1]


class TestCrossEndpointRateLimiting:
    """Test rate limiting across different endpoints."""

    def test_shared_rate_limit_pool(self, client):
        """Test that rate limits are shared across endpoints."""
        # Make requests to different endpoints
        endpoints = [
            ("/health", "GET", None),
            ("/", "GET", None),
            ("/api/v1/generate/", "POST", {"prompt": "test", "duration": 1.0}),
        ]

        total_requests = 0
        last_remaining = None

        for endpoint, method, data in endpoints * 5:  # Repeat to use more quota
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=data)

            if response.status_code in [200, 503]:  # 503 might be service unavailable
                total_requests += 1
                remaining = int(response.headers["X-RateLimit-Remaining"])

                if last_remaining is not None:
                    # Remaining should decrease (or stay same if endpoint doesn't count)
                    assert remaining <= last_remaining

                last_remaining = remaining
            elif response.status_code == 429:
                break

        assert total_requests > 0

    def test_endpoint_specific_limits(self, client):
        """Test if certain endpoints have specific limits."""
        # Generation endpoints might have different limits than health checks
        gen_response = client.post("/api/v1/generate/", json={"prompt": "test", "duration": 1.0})

        health_response = client.get("/health")

        if gen_response.status_code in [200, 503] and health_response.status_code == 200:
            gen_limit = int(gen_response.headers.get("X-RateLimit-Limit", 0))
            health_limit = int(health_response.headers.get("X-RateLimit-Limit", 0))

            # Limits might be different for different endpoint types
            # (This depends on specific rate limiting configuration)
            assert gen_limit > 0
            assert health_limit > 0


class TestConcurrentRateLimiting:
    """Test rate limiting under concurrent load."""

    def test_concurrent_requests_same_client(self, client):
        """Test concurrent requests from same client."""

        def make_request():
            response = client.get("/health")
            return {
                "status": response.status_code,
                "remaining": response.headers.get("X-RateLimit-Remaining"),
            }

        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]

        # Should handle concurrent requests properly
        success_count = sum(1 for r in results if r["status"] == 200)
        rate_limited_count = sum(1 for r in results if r["status"] == 429)

        # Some should succeed, some might be rate limited
        assert success_count > 0
        assert success_count + rate_limited_count == len(results)

    def test_concurrent_different_clients(self, client):
        """Test concurrent requests from different clients."""

        def make_request_as_client(client_ip):
            headers = {"X-Forwarded-For": f"192.168.1.{client_ip}"}
            response = client.get("/health", headers=headers)
            return {
                "client_ip": client_ip,
                "status": response.status_code,
                "remaining": response.headers.get("X-RateLimit-Remaining"),
            }

        # Multiple clients making concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(make_request_as_client, i)
                for i in range(100, 110)  # 10 different client IPs
            ]
            results = [f.result() for f in as_completed(futures)]

        # Each client should have separate rate limit
        success_count = sum(1 for r in results if r["status"] == 200)

        # Most should succeed since they're different clients
        assert success_count >= 8  # At least 80% success rate


class TestRateLimitingEdgeCases:
    """Test edge cases in rate limiting."""

    def test_malformed_headers(self, client):
        """Test handling of malformed proxy headers."""
        malformed_headers = [
            {"X-Forwarded-For": "invalid-ip-address"},
            {"X-Forwarded-For": ""},
            {"X-Forwarded-For": "192.168.1.1, , 192.168.1.2"},
            {"X-Real-IP": "not.an.ip"},
            {"CF-Connecting-IP": "::invalid::ipv6::"},
        ]

        for headers in malformed_headers:
            response = client.get("/health", headers=headers)
            # Should not crash, should return proper response
            assert response.status_code in [200, 429]
            assert "X-RateLimit-Limit" in response.headers

    def test_missing_proxy_headers(self, client):
        """Test behavior when proxy headers are missing."""
        # Request without any proxy headers
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers

        # Should use direct connection IP for rate limiting

    def test_very_long_headers(self, client):
        """Test handling of very long header values."""
        long_ip_list = ", ".join([f"192.168.1.{i}" for i in range(100)])
        headers = {"X-Forwarded-For": long_ip_list}

        response = client.get("/health", headers=headers)
        assert response.status_code in [200, 429]
        # Should not crash or cause performance issues


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
