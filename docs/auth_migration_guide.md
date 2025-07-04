# Authentication Migration Guide

This guide explains the migration from localStorage-based authentication to secure httpOnly cookies in the Music Gen AI application.

## Overview

We've upgraded our authentication system to use httpOnly cookies instead of localStorage for storing authentication tokens. This provides several security benefits:

- **Protection against XSS attacks**: httpOnly cookies cannot be accessed by JavaScript, preventing malicious scripts from stealing tokens
- **CSRF protection**: Implemented with CSRF tokens for all state-changing operations
- **Automatic handling**: Browser automatically includes cookies in requests
- **Better session management**: Server-side session control and invalidation

## What's Changed

### Before (localStorage)
- JWT tokens stored in localStorage
- Tokens accessible to JavaScript
- Manual token management in requests
- Vulnerable to XSS attacks

### After (httpOnly cookies)
- JWT tokens stored in secure httpOnly cookies
- Tokens not accessible to JavaScript
- Automatic credential handling
- CSRF protection on all mutations
- Enhanced security against common attacks

## Migration Process

### Automatic Migration

When users first visit the application after the update, the system will:

1. **Check for existing tokens** in localStorage
2. **Display migration notification** if tokens are found
3. **Exchange tokens** for secure session cookies
4. **Clear localStorage** tokens after successful migration
5. **Reload the application** with new authentication

The migration happens automatically and takes only a few seconds.

### Manual Migration

If automatic migration fails, users can:

1. **Log out** of the application
2. **Clear browser data** for the site
3. **Log in again** with their credentials

## Technical Implementation

### Frontend Changes

#### API Client
All API calls now use `credentials: 'include'` to send cookies:

```typescript
// Before
fetch('/api/endpoint', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
})

// After
fetch('/api/endpoint', {
  credentials: 'include',
  headers: {
    'X-CSRF-Token': csrfToken
  }
})
```

#### AuthContext
The authentication context no longer manages tokens directly:

```typescript
// Before
const token = localStorage.getItem('auth_token')

// After
// Tokens are managed by the browser in httpOnly cookies
// Session checked via API call
```

### Backend Requirements

The backend must be configured to:

1. **Set httpOnly cookies** on login:
```python
response.set_cookie(
    key="auth_token",
    value=token,
    httponly=True,
    secure=True,  # HTTPS only
    samesite="lax",
    max_age=900  # 15 minutes
)
```

2. **Validate CSRF tokens** on mutations
3. **Support cookie-based authentication**
4. **Handle session refresh** via refresh tokens

## API Endpoints

### New/Modified Endpoints

#### GET /api/auth/csrf-token
Returns a CSRF token for the current session:
```json
{
  "csrfToken": "random-token-string"
}
```

#### GET /api/auth/session
Checks current session status:
```json
{
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "username": "user",
    "tier": "pro",
    "isVerified": true
  },
  "csrfToken": "token"
}
```

#### POST /api/auth/refresh
Refreshes the session using refresh token cookie.

#### POST /api/auth/migrate
Exchanges old localStorage tokens for secure cookies:
```json
{
  "accessToken": "old-access-token",
  "refreshToken": "old-refresh-token"
}
```

## Security Considerations

### CSRF Protection

All state-changing requests now require a CSRF token:

```typescript
const response = await fetch('/api/user/update', {
  method: 'POST',
  credentials: 'include',
  headers: {
    'Content-Type': 'application/json',
    'X-CSRF-Token': csrfToken
  },
  body: JSON.stringify(data)
})
```

### Cookie Configuration

Cookies are configured with:
- `httpOnly`: true (no JavaScript access)
- `secure`: true (HTTPS only in production)
- `sameSite`: "lax" (CSRF protection)
- `path`: "/" (available across the site)

### Session Management

- Sessions expire after 15 minutes of inactivity
- Refresh tokens last 7 days
- Automatic session refresh every 14 minutes
- Server-side session invalidation on logout

## Troubleshooting

### Common Issues

#### 1. "Authentication failed after migration"
- Clear all browser data for the site
- Log in again with credentials

#### 2. "CSRF token invalid"
- Refresh the page to get a new CSRF token
- Ensure cookies are enabled in browser

#### 3. "Session expired quickly"
- Check browser cookie settings
- Ensure third-party cookies aren't blocked for the site

#### 4. "Can't log in after update"
- Clear browser cache and cookies
- Try incognito/private browsing mode
- Check browser console for errors

### Browser Requirements

- Cookies must be enabled
- JavaScript must be enabled
- For development: localhost must be allowed

### Development Setup

For local development, ensure:

```bash
# Backend CORS configuration allows localhost
ALLOWED_ORIGINS=http://localhost:3000

# Frontend API URL is correct
NEXT_PUBLIC_API_URL=http://localhost:8000

# Cookies work on localhost (no secure flag in dev)
ENVIRONMENT=development
```

## Benefits for Users

1. **Enhanced Security**: Protection against token theft
2. **Seamless Experience**: No manual token management
3. **Automatic Session Handling**: Browser manages credentials
4. **Better Logout**: Server-side session invalidation
5. **Improved Privacy**: Tokens not visible in dev tools

## Benefits for Developers

1. **Simplified Code**: No manual token storage
2. **Better Security**: Reduced attack surface
3. **Automatic Handling**: Browser manages cookies
4. **Standard Practice**: Industry-standard approach
5. **Easier Testing**: Session management simplified

## Migration Timeline

1. **Phase 1**: Automatic migration for existing users
2. **Phase 2**: Remove localStorage code after 30 days
3. **Phase 3**: Deprecate migration endpoint after 60 days

## Support

If users experience issues:

1. Check the browser console for errors
2. Try logging out and back in
3. Clear site data and retry
4. Contact support with error details

## Code Examples

### Using the Secure API Client

```typescript
import { apiClient } from '@/lib/api-client'

// GET request
const data = await apiClient.get('/api/user/profile')

// POST request with CSRF protection
const result = await apiClient.post('/api/music/generate', {
  prompt: 'Jazz music'
})

// File upload with credentials
const formData = new FormData()
formData.append('file', audioFile)

const response = await fetch('/api/upload', {
  method: 'POST',
  credentials: 'include',
  headers: {
    'X-CSRF-Token': csrfToken
  },
  body: formData
})
```

### Protected Route Component

```typescript
import { withAuth } from '@/contexts/AuthContext'

function ProtectedPage() {
  const { user } = useAuth()
  
  return (
    <div>
      <h1>Welcome {user.username}</h1>
    </div>
  )
}

export default withAuth(ProtectedPage)
```

## Rollback Plan

If issues arise:

1. **Immediate**: Revert frontend changes
2. **Backend**: Continue supporting both auth methods
3. **Gradual**: Fix issues and re-deploy
4. **Communication**: Notify users of temporary revert

This migration enhances security while maintaining a smooth user experience. The automatic migration process ensures existing users can continue using the application without manual intervention.