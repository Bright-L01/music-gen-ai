# Rate Limiting Guide for Music Gen AI API

This guide explains how to configure and use the enhanced rate limiting middleware with proxy support.

## Overview

The rate limiting middleware provides:
- **Proxy-aware IP extraction** with support for various proxy headers
- **Tiered rate limits** based on user subscription level
- **Redis backend** with automatic fallback to in-memory storage
- **Internal service bypass** for monitoring and health checks
- **Burst protection** to prevent rapid-fire requests
- **Comprehensive security logging** for audit trails

## Configuration

### Environment Variables

```bash
# Enable proxy header parsing (default: true)
ENABLE_PROXY_HEADERS=true

# Trusted proxy IPs/CIDRs (comma-separated)
TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

# Internal service API keys (comma-separated)
INTERNAL_API_KEYS=monitoring-key-123,metrics-key-456

# Default rate limit tier for unauthenticated users
DEFAULT_RATE_LIMIT_TIER=free

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_RATELIMIT_DB=2

# Trust all proxies (DANGEROUS - only for development!)
TRUST_ALL_PROXIES=false
```

### Rate Limit Tiers

| Tier | Per Minute | Per Hour | Per Day | Burst Size |
|------|------------|----------|---------|------------|
| Free | 30 | 500 | 5,000 | 5 |
| Premium | 60 | 2,000 | 20,000 | 10 |
| Enterprise | 300 | 10,000 | 100,000 | 50 |
| Internal | 1,000 | 50,000 | 1,000,000 | 100 |
| Admin | 1,000 | 50,000 | 1,000,000 | 100 |

### Proxy Headers

The middleware checks the following headers in order:
1. `X-Forwarded-For` - Standard proxy header
2. `X-Real-IP` - Nginx
3. `CF-Connecting-IP` - Cloudflare
4. `X-Client-IP` - Some proxies
5. `X-Forwarded` - Variation
6. `Forwarded-For` - Variation
7. `Forwarded` - RFC 7239

## Deployment Configurations

### Behind Nginx

```nginx
location /api {
    proxy_pass http://localhost:8000;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Host $host;
}
```

### Behind Cloudflare

Add Cloudflare IP ranges to trusted proxies:

```bash
TRUSTED_PROXIES=173.245.48.0/20,103.21.244.0/22,103.22.200.0/22,103.31.4.0/22,141.101.64.0/18,108.162.192.0/18,190.93.240.0/20,188.114.96.0/20,197.234.240.0/22,198.41.128.0/17,162.158.0.0/15,172.64.0.0/13,131.0.72.0/22,104.16.0.0/13,104.24.0.0/14
```

### Kubernetes Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/use-forwarded-headers: "true"
    nginx.ingress.kubernetes.io/compute-full-forwarded-for: "true"
    nginx.ingress.kubernetes.io/enable-real-ip: "true"
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    image: musicgen-api:latest
    environment:
      - REDIS_HOST=redis
      - ENABLE_PROXY_HEADERS=true
      - TRUSTED_PROXIES=172.16.0.0/12
      - DEFAULT_RATE_LIMIT_TIER=free
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## Testing

### 1. Basic Rate Limiting Test

```bash
# Test rate limiting without proxy
python scripts/test_rate_limiting.py --test basic
```

### 2. Proxy Header Test

```bash
# Test with various proxy configurations
python scripts/test_rate_limiting.py --test proxy
```

### 3. Internal Service Bypass

```bash
# Test internal service headers
curl -H "X-Internal-Service: monitoring" \
     -H "X-API-Key: your-internal-key" \
     http://localhost:8000/api/v1/models
```

### 4. Burst Protection

```bash
# Test burst limit enforcement
python scripts/test_rate_limiting.py --test burst
```

### 5. Full Test Suite

```bash
# Run all tests
python scripts/test_rate_limiting.py --test all --verbose
```

## Response Headers

The API returns the following rate limit headers:

```
X-RateLimit-Tier: free
X-RateLimit-Minute-Limit: 30
X-RateLimit-Minute-Remaining: 25
X-RateLimit-Minute-Reset: 1704123660
X-RateLimit-Hour-Limit: 500
X-RateLimit-Hour-Remaining: 495
X-RateLimit-Hour-Reset: 1704126000
X-RateLimit-Day-Limit: 5000
X-RateLimit-Day-Remaining: 4995
X-RateLimit-Day-Reset: 1704153600
```

## Rate Limit Exceeded Response

When rate limit is exceeded:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 45,
  "limits": {
    "minute": {
      "limit": 30,
      "remaining": 0,
      "reset": 1704123660
    },
    "hour": {
      "limit": 500,
      "remaining": 470,
      "reset": 1704126000
    },
    "day": {
      "limit": 5000,
      "remaining": 4970,
      "reset": 1704153600
    }
  }
}
```

HTTP Status: `429 Too Many Requests`

## Security Best Practices

### 1. Configure Trusted Proxies

Never use `TRUST_ALL_PROXIES=true` in production. Always specify exact proxy IPs:

```bash
# Good - specific proxy IPs
TRUSTED_PROXIES=10.0.1.1,10.0.1.2,10.0.1.3

# Bad - trusting all proxies
TRUST_ALL_PROXIES=true
```

### 2. Validate Internal Service Keys

Use strong, randomly generated API keys for internal services:

```bash
# Generate secure API key
openssl rand -hex 32
```

### 3. Monitor Rate Limit Violations

Set up alerts for suspicious activity:

```python
# Example monitoring integration
if response.status_code == 429:
    alert_security_team(
        ip=client_ip,
        path=request.path,
        headers=request.headers
    )
```

### 4. Regular IP List Updates

Keep trusted proxy lists updated:

```bash
# Update Cloudflare IPs
curl https://www.cloudflare.com/ips-v4 > cloudflare-ips.txt
curl https://www.cloudflare.com/ips-v6 >> cloudflare-ips.txt
```

## Troubleshooting

### Issue: Wrong IP Detected

**Symptom**: Rate limiting is applied to proxy IP instead of client IP

**Solution**: 
1. Ensure proxy is in trusted list
2. Check proxy is sending correct headers
3. Verify header priority order

### Issue: Internal Services Rate Limited

**Symptom**: Monitoring or health checks are being rate limited

**Solution**:
1. Add service API key to `INTERNAL_API_KEYS`
2. Ensure service sends correct headers
3. Check exempt paths configuration

### Issue: Redis Connection Failed

**Symptom**: Falling back to in-memory storage

**Solution**:
1. Verify Redis is running
2. Check Redis connection settings
3. Ensure Redis is accessible from API

### Debug Mode

Enable debug logging to troubleshoot:

```python
import logging
logging.getLogger("music_gen.api.middleware.rate_limiting").setLevel(logging.DEBUG)
```

## Performance Considerations

1. **Redis Performance**: Use Redis pipeline for atomic operations
2. **Memory Usage**: In-memory storage cleans up expired entries every 5 minutes
3. **Header Parsing**: Proxy header extraction is optimized for performance
4. **Burst Protection**: Uses sliding window algorithm for accuracy

## Integration Examples

### Python Client with Retry

```python
import httpx
import time

class RateLimitedClient:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url)
    
    def request_with_retry(self, method: str, path: str, **kwargs):
        while True:
            response = self.client.request(method, path, **kwargs)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            return response
```

### JavaScript Client with Backoff

```javascript
async function fetchWithRateLimit(url, options = {}) {
  const response = await fetch(url, options);
  
  if (response.status === 429) {
    const retryAfter = parseInt(response.headers.get('Retry-After')) || 60;
    console.log(`Rate limited. Retrying after ${retryAfter} seconds...`);
    
    await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
    return fetchWithRateLimit(url, options);
  }
  
  return response;
}
```

## Monitoring and Analytics

Track rate limiting metrics:

```python
# Get rate limit statistics
stats = rate_limit_middleware.get_stats()
print(f"Allowed requests: {stats['allowed_requests']}")
print(f"Rate limited requests: {stats['rate_limited_requests']}")
print(f"Internal requests: {stats['internal_requests']}")
```

## Conclusion

The enhanced rate limiting middleware provides robust protection against abuse while supporting modern deployment architectures with proxies and load balancers. Proper configuration ensures accurate client identification and appropriate rate limits for different user tiers.