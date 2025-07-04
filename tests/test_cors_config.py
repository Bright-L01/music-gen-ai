"""
Tests for CORS configuration.
"""

import pytest
import os
from unittest.mock import patch

from music_gen.api.cors_config import CORSConfig, cors_config


class TestCORSConfig:
    """Test CORS configuration functionality."""
    
    def test_development_environment_defaults(self):
        """Test development environment includes localhost origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            
            # Should include localhost origins
            assert "http://localhost:3000" in config.allowed_origins
            assert "http://localhost:8000" in config.allowed_origins
            assert "http://127.0.0.1:3000" in config.allowed_origins
            assert "http://[::1]:3000" in config.allowed_origins
    
    def test_staging_environment_defaults(self):
        """Test staging environment includes staging origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            config = CORSConfig()
            
            # Should include staging origins
            assert "https://staging.musicgen.ai" in config.allowed_origins
            assert "https://preview.musicgen.ai" in config.allowed_origins
            assert "https://beta.musicgen.ai" in config.allowed_origins
            
            # Should not include localhost by default
            assert "http://localhost:3000" not in config.allowed_origins
    
    def test_production_environment_defaults(self):
        """Test production environment includes production origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            config = CORSConfig()
            
            # Should include production origins
            assert "https://musicgen.ai" in config.allowed_origins
            assert "https://www.musicgen.ai" in config.allowed_origins
            assert "https://app.musicgen.ai" in config.allowed_origins
            
            # Should not include HTTP origins
            assert "http://musicgen.ai" not in config.allowed_origins
            assert "http://localhost:3000" not in config.allowed_origins
    
    def test_custom_allowed_origins(self):
        """Test adding custom allowed origins via environment variable."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://custom1.com,https://custom2.com"
        }, clear=True):
            config = CORSConfig()
            
            # Should include custom origins
            assert "https://custom1.com" in config.allowed_origins
            assert "https://custom2.com" in config.allowed_origins
            
            # Should still include defaults
            assert "https://musicgen.ai" in config.allowed_origins
    
    def test_allowed_domains(self):
        """Test domain-based origin generation."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_DOMAINS": "example.com,test.io"
        }, clear=True):
            config = CORSConfig()
            
            # Should generate HTTPS variants
            assert "https://example.com" in config.allowed_origins
            assert "https://www.example.com" in config.allowed_origins
            assert "https://test.io" in config.allowed_origins
            assert "https://www.test.io" in config.allowed_origins
    
    def test_origin_validation(self):
        """Test origin format validation."""
        config = CORSConfig()
        
        # Valid origins
        assert config._validate_origin("https://example.com")
        assert config._validate_origin("http://localhost:3000")
        assert config._validate_origin("https://sub.example.com:8080")
        
        # Invalid origins
        assert not config._validate_origin("")
        assert not config._validate_origin("example.com")  # No scheme
        assert not config._validate_origin("https://")  # No host
        assert not config._validate_origin("https://example.com/path")  # Has path
        assert not config._validate_origin("https://example.com?query=1")  # Has query
        assert not config._validate_origin("ftp://example.com")  # Wrong scheme
    
    def test_production_rejects_http(self):
        """Test that production environment rejects HTTP origins."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "http://insecure.com,https://secure.com"
        }, clear=True):
            config = CORSConfig()
            
            # Should reject HTTP origin
            assert "http://insecure.com" not in config.allowed_origins
            assert "https://secure.com" in config.allowed_origins
    
    def test_is_origin_allowed(self):
        """Test origin allowance checking."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "ALLOWED_ORIGINS": "https://custom.com"
        }, clear=True):
            config = CORSConfig()
            
            # Allowed origins
            assert config.is_origin_allowed("http://localhost:3000")
            assert config.is_origin_allowed("https://custom.com")
            
            # Disallowed origins
            assert not config.is_origin_allowed("https://evil.com")
            assert not config.is_origin_allowed("")
            assert not config.is_origin_allowed(None)
    
    def test_wildcard_subdomain_matching(self):
        """Test wildcard subdomain matching when enabled."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://*.example.com",
            "ALLOW_SUBDOMAIN_WILDCARDS": "true"
        }, clear=True):
            config = CORSConfig()
            
            # Should match subdomains
            assert config.is_origin_allowed("https://app.example.com")
            assert config.is_origin_allowed("https://api.example.com")
            assert config.is_origin_allowed("https://staging.example.com")
            assert config.is_origin_allowed("https://example.com")
            
            # Should not match different domains
            assert not config.is_origin_allowed("https://example.org")
            assert not config.is_origin_allowed("https://notexample.com")
    
    def test_cors_options(self):
        """Test CORS middleware options generation."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            options = config.get_cors_options()
            
            assert "allow_origins" in options
            assert isinstance(options["allow_origins"], list)
            assert options["allow_credentials"] is True
            assert "GET" in options["allow_methods"]
            assert "POST" in options["allow_methods"]
            assert "Authorization" in options["allow_headers"]
            assert options["max_age"] == 86400
    
    def test_preflight_headers_valid_origin(self):
        """Test preflight headers for valid origin."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_preflight_headers(
                origin="http://localhost:3000",
                request_method="POST",
                request_headers="Content-Type, Authorization"
            )
            
            assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
            assert headers["Access-Control-Allow-Credentials"] == "true"
            assert "Access-Control-Allow-Methods" in headers
            assert "Access-Control-Allow-Headers" in headers
            assert headers["Access-Control-Max-Age"] == "86400"
    
    def test_preflight_headers_invalid_origin(self):
        """Test preflight headers for invalid origin."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_preflight_headers(
                origin="https://evil.com",
                request_method="POST"
            )
            
            # Should return empty dict for invalid origin
            assert headers == {}
    
    def test_preflight_headers_invalid_method(self):
        """Test preflight headers with disallowed method."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_preflight_headers(
                origin="http://localhost:3000",
                request_method="TRACE"  # Not allowed
            )
            
            # Should return empty dict for invalid method
            assert headers == {}
    
    def test_preflight_headers_invalid_header(self):
        """Test preflight headers with disallowed header."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_preflight_headers(
                origin="http://localhost:3000",
                request_method="POST",
                request_headers="X-Evil-Header"  # Not allowed
            )
            
            # Should return empty dict for invalid header
            assert headers == {}
    
    def test_response_headers_valid_origin(self):
        """Test response headers for valid origin."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_response_headers("http://localhost:3000")
            
            assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
            assert headers["Access-Control-Allow-Credentials"] == "true"
            assert "Access-Control-Expose-Headers" in headers
            assert headers["Vary"] == "Origin"
    
    def test_response_headers_invalid_origin(self):
        """Test response headers for invalid origin."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            config = CORSConfig()
            
            headers = config.get_response_headers("https://evil.com")
            
            # Should return empty dict for invalid origin
            assert headers == {}
    
    def test_staging_dev_origins(self):
        """Test allowing development origins in staging."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "staging",
            "STAGING_DEV_ORIGINS": "http://localhost:3000,http://localhost:3001"
        }, clear=True):
            config = CORSConfig()
            
            # Should include specified dev origins in staging
            assert "http://localhost:3000" in config.allowed_origins
            assert "http://localhost:3001" in config.allowed_origins
            
            # Should still include staging origins
            assert "https://staging.musicgen.ai" in config.allowed_origins
    
    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": ",,https://valid.com,,"  # Empty values
        }, clear=True):
            config = CORSConfig()
            
            # Should filter out empty strings
            assert "" not in config.allowed_origins
            assert "https://valid.com" in config.allowed_origins
    
    def test_unknown_environment(self):
        """Test handling of unknown environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "unknown"}, clear=True):
            config = CORSConfig()
            
            # Should have no default origins for unknown environment
            assert len(config.allowed_origins) == 0
    
    def test_case_sensitivity(self):
        """Test that origin matching is case-sensitive."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "ALLOWED_ORIGINS": "https://Example.com"
        }, clear=True):
            config = CORSConfig()
            
            # Origin matching should be case-sensitive
            assert config.is_origin_allowed("https://Example.com")
            assert not config.is_origin_allowed("https://example.com")
            assert not config.is_origin_allowed("HTTPS://Example.com")


# Integration test with FastAPI
@pytest.mark.asyncio
async def test_cors_with_fastapi():
    """Test CORS configuration with FastAPI application."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from music_gen.api.app import create_app
    
    with patch.dict(os.environ, {
        "ENVIRONMENT": "development",
        "ALLOWED_ORIGINS": "https://custom-app.com"
    }, clear=True):
        app = create_app()
        client = TestClient(app)
        
        # Test preflight request from allowed origin
        response = client.options(
            "/api/v1/generate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert response.headers.get("Access-Control-Allow-Credentials") == "true"
        
        # Test request from disallowed origin
        response = client.get(
            "/health",
            headers={"Origin": "https://evil.com"}
        )
        
        # Should still return 200 but without CORS headers
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers
        
        # Test request from custom allowed origin
        response = client.get(
            "/health",
            headers={"Origin": "https://custom-app.com"}
        )
        
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://custom-app.com"