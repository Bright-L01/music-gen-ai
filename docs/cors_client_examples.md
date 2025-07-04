# CORS Client Examples

This document provides examples of how to properly make CORS requests to the Music Gen AI API from various client applications.

## JavaScript/TypeScript (Browser)

### Basic Fetch Request
```javascript
// Simple GET request
async function getHealth() {
  try {
    const response = await fetch('https://api.musicgen.ai/health', {
      method: 'GET',
      credentials: 'include', // Include cookies for authentication
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Health check:', data);
  } catch (error) {
    console.error('CORS request failed:', error);
  }
}
```

### Authenticated POST Request
```javascript
async function generateMusic(prompt, authToken) {
  try {
    const response = await fetch('https://api.musicgen.ai/api/v1/generate', {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        prompt: prompt,
        duration: 30,
        temperature: 0.8
      })
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Authentication failed - please login again');
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Generation request failed:', error);
    throw error;
  }
}
```

### Handling Preflight Requests
```javascript
// The browser automatically handles preflight requests, but you can debug them:
async function debugCORS() {
  // This will trigger a preflight request due to custom headers
  const response = await fetch('https://api.musicgen.ai/api/v1/generate', {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer token123',
      'X-Custom-Header': 'value' // This will trigger preflight
    },
    body: JSON.stringify({ prompt: 'test' })
  });
  
  // Check response headers in browser DevTools Network tab
  console.log('CORS headers received:', {
    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
    'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials')
  });
}
```

## React with Axios

### Axios Configuration
```typescript
import axios from 'axios';

// Create axios instance with CORS configuration
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'https://api.musicgen.ai',
  withCredentials: true, // Enable credentials
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle CORS errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      if (error.response.status === 401) {
        // Handle authentication error
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
      }
    } else if (error.request) {
      // Request made but no response - could be CORS issue
      console.error('No response received - possible CORS issue:', error.request);
    }
    return Promise.reject(error);
  }
);

// Usage
export const musicGenAPI = {
  async generateMusic(prompt: string, options?: any) {
    const response = await apiClient.post('/api/v1/generate', {
      prompt,
      ...options
    });
    return response.data;
  },
  
  async getModels() {
    const response = await apiClient.get('/api/v1/models');
    return response.data;
  }
};
```

## Next.js (Server-Side and Client-Side)

### API Route Proxy (Avoid CORS)
```typescript
// pages/api/generate.ts or app/api/generate/route.ts
import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Forward request to actual API (server-to-server, no CORS)
  const response = await fetch('https://api.musicgen.ai/api/v1/generate', {
    method: req.method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': req.headers.authorization || '',
    },
    body: JSON.stringify(req.body),
  });
  
  const data = await response.json();
  res.status(response.status).json(data);
}
```

### Client-Side with SWR
```typescript
import useSWR from 'swr';

const fetcher = async (url: string) => {
  const response = await fetch(url, {
    credentials: 'include',
    headers: {
      'Authorization': `Bearer ${getAuthToken()}`,
    },
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  
  return response.json();
};

export function useModels() {
  const { data, error, isLoading } = useSWR(
    `${process.env.NEXT_PUBLIC_API_URL}/api/v1/models`,
    fetcher
  );
  
  return {
    models: data,
    isLoading,
    isError: error,
  };
}
```

## Angular with HttpClient

```typescript
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class MusicGenService {
  private apiUrl = 'https://api.musicgen.ai';
  
  constructor(private http: HttpClient) {}
  
  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('auth_token');
    return new HttpHeaders({
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : ''
    });
  }
  
  generateMusic(prompt: string): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/api/v1/generate`,
      { prompt },
      {
        headers: this.getHeaders(),
        withCredentials: true // Enable credentials
      }
    ).pipe(
      catchError(this.handleError)
    );
  }
  
  private handleError(error: HttpErrorResponse) {
    if (error.status === 0) {
      // Client-side or network error (possibly CORS)
      console.error('CORS or network error:', error.error);
    } else {
      // Backend returned an error response
      console.error(`Backend returned code ${error.status}:`, error.error);
    }
    return throwError(() => new Error('API request failed'));
  }
}
```

## Vue.js with Composition API

```javascript
import { ref } from 'vue';

export function useMusicGenAPI() {
  const loading = ref(false);
  const error = ref(null);
  
  const apiBase = import.meta.env.VITE_API_URL || 'https://api.musicgen.ai';
  
  async function request(endpoint, options = {}) {
    loading.value = true;
    error.value = null;
    
    try {
      const token = localStorage.getItem('auth_token');
      const response = await fetch(`${apiBase}${endpoint}`, {
        ...options,
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` }),
          ...options.headers,
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  }
  
  return {
    loading,
    error,
    generateMusic: (prompt) => request('/api/v1/generate', {
      method: 'POST',
      body: JSON.stringify({ prompt })
    }),
    getModels: () => request('/api/v1/models'),
  };
}
```

## Common CORS Issues and Solutions

### 1. "CORS policy: No 'Access-Control-Allow-Origin'"
```javascript
// Problem: Origin not whitelisted
// Solution: Ensure your domain is in ALLOWED_ORIGINS environment variable

// For development, ensure the API allows localhost:
// ALLOWED_ORIGINS=http://localhost:3000
```

### 2. "CORS policy: Credentials flag is true, but Access-Control-Allow-Credentials is not"
```javascript
// Problem: credentials: 'include' used but server doesn't allow credentials
// Solution: Ensure the API has allow_credentials=True in CORS config

// Correct client code:
fetch(url, {
  credentials: 'include', // or 'same-origin' if not cross-origin
  // ...
});
```

### 3. Preflight request fails
```javascript
// Problem: Custom headers trigger preflight that fails
// Solution: Ensure headers are in allowed list

// Debugging preflight:
// 1. Open browser DevTools
// 2. Go to Network tab
// 3. Look for OPTIONS request
// 4. Check response headers and status
```

### 4. Cookies not being sent
```javascript
// Problem: Cookies not included in cross-origin requests
// Solution: Set credentials and ensure SameSite cookie attribute allows it

fetch(url, {
  credentials: 'include', // Must include this
  // ...
});

// Server should set cookies with:
// Set-Cookie: session=...; SameSite=None; Secure; HttpOnly
```

## Testing CORS Locally

### Using Chrome with disabled security (development only)
```bash
# macOS
open -n -a /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --args --user-data-dir="/tmp/chrome_dev_test" --disable-web-security

# Windows
chrome.exe --user-data-dir="C:/Chrome dev session" --disable-web-security

# Linux
google-chrome --user-data-dir="/tmp/chrome_dev_test" --disable-web-security
```

### Using a local proxy
```javascript
// vite.config.js (Vite)
export default {
  server: {
    proxy: {
      '/api': {
        target: 'https://api.musicgen.ai',
        changeOrigin: true,
        secure: false,
      }
    }
  }
}

// webpack.config.js (Webpack)
module.exports = {
  devServer: {
    proxy: {
      '/api': {
        target: 'https://api.musicgen.ai',
        changeOrigin: true,
        secure: false,
      }
    }
  }
}
```

### Testing with cURL
```bash
# Test CORS preflight
curl -X OPTIONS https://api.musicgen.ai/api/v1/generate \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type, Authorization" \
  -v

# Test actual request with CORS
curl -X POST https://api.musicgen.ai/api/v1/generate \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"prompt": "upbeat jazz"}' \
  -v
```

## Security Best Practices

1. **Never use wildcards in production**
   ```javascript
   // BAD: allow_origins=["*"]
   // GOOD: allow_origins=["https://app.musicgen.ai"]
   ```

2. **Always use HTTPS in production**
   ```javascript
   // BAD: http://api.musicgen.ai
   // GOOD: https://api.musicgen.ai
   ```

3. **Validate origins server-side**
   ```javascript
   // The server should validate Origin header against whitelist
   ```

4. **Use appropriate SameSite cookie settings**
   ```javascript
   // For cross-site requests:
   // Set-Cookie: session=...; SameSite=None; Secure
   ```

5. **Implement CSRF protection**
   ```javascript
   // Use CSRF tokens for state-changing operations
   headers: {
     'X-CSRF-Token': csrfToken,
   }
   ```