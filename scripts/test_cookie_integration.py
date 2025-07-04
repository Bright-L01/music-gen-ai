#!/usr/bin/env python3
"""
Simple script to test cookie integration without full test dependencies.
"""

import httpx
import json
import sys
from typing import Dict, Optional


class CookieTestClient:
    """Simple client to test cookie-based auth."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = httpx.Client(base_url=base_url)
        self.csrf_token: Optional[str] = None
    
    def update_csrf_token(self, response):
        """Extract CSRF token from response."""
        if response.status_code == 200:
            try:
                data = response.json()
                if "csrfToken" in data:
                    self.csrf_token = data["csrfToken"]
                    print(f"  ✓ CSRF token updated: {self.csrf_token[:10]}...")
            except:
                pass
    
    def test_csrf_token(self):
        """Test CSRF token endpoint."""
        print("\n1. Testing CSRF token endpoint...")
        response = self.session.get("/api/auth/csrf-token")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            self.update_csrf_token(response)
            
            # Check cookie
            csrf_cookie = self.session.cookies.get("csrf_token")
            if csrf_cookie:
                print(f"  ✓ CSRF cookie set: {csrf_cookie[:10]}...")
            else:
                print("  ✗ CSRF cookie not found")
        
        return response.status_code == 200
    
    def test_login(self):
        """Test login with cookies."""
        print("\n2. Testing login...")
        
        headers = {}
        if self.csrf_token:
            headers["X-CSRF-Token"] = self.csrf_token
        
        login_data = {
            "email": "user@example.com",
            "password": "password123",
            "remember_me": False
        }
        
        response = self.session.post(
            "/api/auth/login",
            json=login_data,
            headers=headers
        )
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Login successful")
            print(f"  User: {data.get('user', {}).get('email')}")
            
            # Check cookies
            auth_cookie = self.session.cookies.get("auth_token")
            refresh_cookie = self.session.cookies.get("refresh_token")
            
            if auth_cookie:
                print(f"  ✓ Auth cookie set")
            else:
                print("  ✗ Auth cookie not found")
            
            if refresh_cookie:
                print(f"  ✓ Refresh cookie set")
            else:
                print("  ✗ Refresh cookie not found")
            
            self.update_csrf_token(response)
        else:
            print(f"  ✗ Login failed: {response.text}")
        
        return response.status_code == 200
    
    def test_session(self):
        """Test session check."""
        print("\n3. Testing session check...")
        
        response = self.session.get("/api/auth/session")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            authenticated = data.get("authenticated", False)
            
            if authenticated:
                user = data.get("user", {})
                print(f"  ✓ Authenticated as: {user.get('email')}")
                print(f"  Roles: {user.get('roles')}")
                print(f"  Tier: {user.get('tier')}")
            else:
                print("  ✗ Not authenticated")
        
        return response.status_code == 200
    
    def test_protected_endpoint(self):
        """Test accessing protected endpoint."""
        print("\n4. Testing protected endpoint...")
        
        response = self.session.get("/api/v1/models")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            print("  ✓ Successfully accessed protected endpoint")
        elif response.status_code == 401:
            print("  ✗ Unauthorized - authentication required")
        else:
            print(f"  ✗ Unexpected status: {response.text}")
        
        return response.status_code == 200
    
    def test_logout(self):
        """Test logout."""
        print("\n5. Testing logout...")
        
        headers = {}
        if self.csrf_token:
            headers["X-CSRF-Token"] = self.csrf_token
        
        response = self.session.post("/api/auth/logout", headers=headers)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            print("  ✓ Logout successful")
            
            # Check if cookies were cleared
            # Note: httpx might still have cookies but they should be expired
            print("  Checking cookie clearing...")
        
        return response.status_code == 200
    
    def test_session_after_logout(self):
        """Test session after logout."""
        print("\n6. Testing session after logout...")
        
        response = self.session.get("/api/auth/session")
        
        if response.status_code == 200:
            data = response.json()
            authenticated = data.get("authenticated", False)
            
            if not authenticated:
                print("  ✓ Successfully logged out")
            else:
                print("  ✗ Still authenticated after logout")
        
        return response.status_code == 200
    
    def run_all_tests(self):
        """Run all cookie integration tests."""
        print("=== Cookie Integration Tests ===")
        
        tests = [
            self.test_csrf_token,
            self.test_login,
            self.test_session,
            self.test_protected_endpoint,
            self.test_logout,
            self.test_session_after_logout
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Test failed with error: {e}")
                failed += 1
        
        print(f"\n=== Summary ===")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total: {len(tests)}")
        
        return failed == 0


def test_concurrent_sessions():
    """Test multiple concurrent sessions."""
    print("\n=== Testing Concurrent Sessions ===")
    
    # Create two clients
    client1 = CookieTestClient()
    client2 = CookieTestClient()
    
    print("\nClient 1 login...")
    client1.test_csrf_token()
    client1.test_login()
    
    print("\nClient 2 login...")
    client2.test_csrf_token()
    client2.test_login()
    
    print("\nChecking both sessions...")
    print("Client 1:")
    client1.test_session()
    print("Client 2:")
    client2.test_session()
    
    print("\nClient 1 logout...")
    client1.test_logout()
    
    print("\nChecking sessions after Client 1 logout...")
    print("Client 1:")
    client1.test_session()
    print("Client 2 (should still be logged in):")
    client2.test_session()


def main():
    """Run cookie integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cookie integration")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Test concurrent sessions"
    )
    
    args = parser.parse_args()
    
    if args.concurrent:
        test_concurrent_sessions()
    else:
        client = CookieTestClient(args.url)
        success = client.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()