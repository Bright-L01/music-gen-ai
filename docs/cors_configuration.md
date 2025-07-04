# CORS Configuration Guide

This guide explains how to configure Cross-Origin Resource Sharing (CORS) for the Music Gen AI API.

## Overview

The API implements a secure CORS configuration that:
- Uses environment-based origin whitelisting
- Validates all origins against a strict whitelist
- Supports different configurations for development, staging, and production
- Allows credentials only for trusted origins
- Implements proper preflight request handling

## Environment Variables

### Core Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENVIRONMENT` | Current environment (development/staging/production) | `development` | `production` |
| `ALLOWED_ORIGINS` | Comma-separated list of additional allowed origins | Empty | `https://app.example.com,https://admin.example.com` |
| `ALLOWED_DOMAINS` | Comma-separated list of allowed domains (auto-generates https/www variants) | Empty | `example.com,app.example.com` |
| `ALLOW_SUBDOMAIN_WILDCARDS` | Enable wildcard subdomain matching | `false` | `true` |
| `STAGING_DEV_ORIGINS` | Additional development origins allowed in staging | Empty | `http://localhost:3000` |

### Environment-Specific Origins

#### Development (default)
```
http://localhost:3000      # Next.js dev server
http://localhost:3001      # Alternative Next.js port
http://localhost:8000      # API documentation
http://localhost:8080      # Alternative API port
http://127.0.0.1:3000      # IP-based localhost
http://127.0.0.1:8000
http://[::1]:3000         # IPv6 localhost
http://[::1]:8000
```

#### Staging
```
https://staging.musicgen.ai
https://staging-api.musicgen.ai
https://preview.musicgen.ai
https://beta.musicgen.ai
```

#### Production
```
https://musicgen.ai
https://www.musicgen.ai
https://app.musicgen.ai
https://api.musicgen.ai
```

## Configuration Examples

### Development Setup
```bash
# .env.development
ENVIRONMENT=development
# Development origins are automatically included
```

### Staging Setup
```bash
# .env.staging
ENVIRONMENT=staging
ALLOWED_ORIGINS=https://staging.musicgen.ai,https://preview.musicgen.ai
STAGING_DEV_ORIGINS=http://localhost:3000  # Allow local development against staging
```

### Production Setup
```bash
# .env.production
ENVIRONMENT=production
ALLOWED_ORIGINS=https://musicgen.ai,https://app.musicgen.ai
ALLOWED_DOMAINS=musicgen.ai  # Automatically allows https://musicgen.ai and https://www.musicgen.ai
```

### Multi-Domain Production Setup
```bash
# .env.production
ENVIRONMENT=production
ALLOWED_DOMAINS=musicgen.ai,musicgen.com,musicgenai.app
# This automatically allows:
# - https://musicgen.ai, https://www.musicgen.ai
# - https://musicgen.com, https://www.musicgen.com
# - https://musicgenai.app, https://www.musicgenai.app
```

### Wildcard Subdomain Setup
```bash
# .env.production
ENVIRONMENT=production
ALLOWED_ORIGINS=https://*.musicgen.ai
ALLOW_SUBDOMAIN_WILDCARDS=true
# This allows any subdomain of musicgen.ai
```

## Security Features

### 1. Origin Validation
- All origins are validated for proper format
- HTTP origins are rejected in production
- Empty or malformed origins are rejected

### 2. Credential Support
- Credentials (cookies, authorization headers) are only sent to whitelisted origins
- `Access-Control-Allow-Credentials: true` is set for valid origins

### 3. Preflight Handling
- OPTIONS requests are validated against the origin whitelist
- Only allowed methods and headers are permitted
- Preflight responses are cached for 24 hours

### 4. Headers Configuration

**Allowed Request Headers:**
- Accept
- Accept-Language
- Content-Language
- Content-Type
- Authorization
- X-Requested-With
- X-Request-ID
- X-CSRF-Token
- Cache-Control
- Pragma

**Exposed Response Headers:**
- X-Request-ID
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset
- Content-Disposition

**Allowed Methods:**
- GET, POST, PUT, DELETE, OPTIONS, PATCH

## Troubleshooting

### CORS Errors in Browser Console

1. **"CORS policy: No 'Access-Control-Allow-Origin' header"**
   - Origin is not in the whitelist
   - Check `ALLOWED_ORIGINS` environment variable
   - Verify the exact origin including protocol and port

2. **"CORS policy: The value of the 'Access-Control-Allow-Credentials' header"**
   - Credentials are being sent to a non-whitelisted origin
   - Add the origin to the whitelist or disable credentials

3. **"Preflight response is not successful"**
   - The OPTIONS request failed validation
   - Check server logs for specific validation errors
   - Ensure requested headers/methods are allowed

### Testing CORS Configuration

```bash
# Test preflight request
curl -X OPTIONS https://api.musicgen.ai/api/v1/generate \
  -H "Origin: https://app.musicgen.ai" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type, Authorization" \
  -v

# Test actual request
curl -X POST https://api.musicgen.ai/api/v1/generate \
  -H "Origin: https://app.musicgen.ai" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}' \
  -v
```

### Logging

CORS validation events are logged at the WARNING level for security monitoring:
- Unauthorized origin attempts
- Invalid origin formats
- Disallowed methods or headers in preflight

Check logs with:
```bash
grep "CORS" logs/api.log
```

## Best Practices

1. **Never use wildcard (*) origins in production**
   - Always specify exact origins
   - Use domain-based configuration for flexibility

2. **Minimize allowed origins**
   - Only whitelist origins that actually need API access
   - Regularly audit the whitelist

3. **Use HTTPS in production**
   - HTTP origins are automatically rejected in production
   - Ensures credential security

4. **Monitor CORS violations**
   - Set up alerts for repeated CORS failures
   - May indicate attempted attacks or misconfiguration

5. **Test thoroughly**
   - Test all client applications against the API
   - Verify preflight requests work correctly
   - Test with and without credentials

## Migration from Previous Configuration

If migrating from a wildcard CORS configuration:

1. **Identify all client origins**
   ```bash
   # Check access logs for Origin headers
   grep "Origin:" logs/access.log | sort | uniq
   ```

2. **Update environment configuration**
   ```bash
   # Add all legitimate origins
   ALLOWED_ORIGINS=https://app1.example.com,https://app2.example.com
   ```

3. **Test each client application**
   - Verify all API calls work correctly
   - Check for CORS errors in browser console

4. **Monitor for issues**
   - Watch logs for CORS rejections
   - Be ready to add missing legitimate origins