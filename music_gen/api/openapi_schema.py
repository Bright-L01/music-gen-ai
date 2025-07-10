"""
OpenAPI schema generation and TypeScript type generation.

This module provides utilities for generating OpenAPI documentation
and TypeScript types from the API schema.
"""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def generate_custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate a custom OpenAPI schema with enhanced documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Music Gen AI API",
        version="1.0.0",
        description="""
# Music Gen AI API Documentation

Production-ready AI music generation API with the following features:

## 🎵 Music Generation
- Text-to-music generation with customizable parameters
- Multiple genre and mood support
- Real-time progress tracking via WebSocket
- Batch generation capabilities

## 👤 User Management
- Secure authentication with JWT tokens
- User profiles and preferences
- Generation history tracking

## 🎼 Track Management
- Track creation, editing, and deletion
- Public/private visibility controls
- Like, share, and comment functionality
- Advanced search and filtering

## 🌐 Community Features
- Trending tracks and artists
- Collaborative playlists
- Social interactions

## 🔧 Audio Processing
- Post-processing effects
- Audio mixing and mastering
- Export in multiple formats

## 📊 Analytics
- Usage tracking and insights
- Performance metrics
- User behavior analysis

## 🔐 Security
- CSRF protection
- Rate limiting
- Secure cookie handling
- OAuth2 integration

## 🚀 Performance
- Redis-based caching
- Celery task queue
- Horizontal scaling support
- WebSocket for real-time updates
        """,
        routes=app.routes,
        tags=[
            {
                "name": "authentication",
                "description": "User authentication and authorization endpoints",
            },
            {"name": "generation", "description": "Music generation endpoints"},
            {"name": "tracks", "description": "Track management endpoints"},
            {"name": "users", "description": "User profile and management endpoints"},
            {"name": "community", "description": "Community and social features"},
            {"name": "audio", "description": "Audio processing endpoints"},
            {"name": "analytics", "description": "Analytics and tracking endpoints"},
            {"name": "monitoring", "description": "System monitoring and health checks"},
        ],
        servers=[
            {"url": "http://localhost:8000", "description": "Local development server"},
            {"url": "https://api.musicgen.ai", "description": "Production server"},
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "cookieAuth": {
            "type": "apiKey",
            "in": "cookie",
            "name": "session",
            "description": "Session cookie authentication",
        },
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token authentication",
        },
        "csrfToken": {
            "type": "apiKey",
            "in": "header",
            "name": "X-CSRF-Token",
            "description": "CSRF token for state-changing operations",
        },
    }

    # Add global security
    openapi_schema["security"] = [
        {"cookieAuth": []},
        {"csrfToken": []},
    ]

    # Add common responses
    openapi_schema["components"]["responses"] = {
        "UnauthorizedError": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Authentication required"},
                            "message": {"type": "string", "example": "Please log in to continue"},
                        },
                    }
                }
            },
        },
        "ForbiddenError": {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Forbidden"},
                            "message": {
                                "type": "string",
                                "example": "You don't have permission to access this resource",
                            },
                        },
                    }
                }
            },
        },
        "NotFoundError": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Not found"},
                            "message": {
                                "type": "string",
                                "example": "The requested resource was not found",
                            },
                        },
                    }
                }
            },
        },
        "ValidationError": {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Validation error"},
                            "message": {"type": "string", "example": "Invalid input data"},
                            "details": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string"},
                                        "message": {"type": "string"},
                                    },
                                },
                            },
                        },
                    }
                }
            },
        },
        "RateLimitError": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Rate limit exceeded"},
                            "message": {
                                "type": "string",
                                "example": "Too many requests. Please try again later.",
                            },
                            "retry_after": {"type": "integer", "example": 60},
                        },
                    }
                }
            },
        },
    }

    # Add examples for common schemas
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        schemas = openapi_schema["components"]["schemas"]

        # Add examples to GenerationRequest
        if "GenerationRequest" in schemas:
            schemas["GenerationRequest"]["example"] = {
                "prompt": "Create a relaxing lofi hip hop beat with soft piano and rain sounds",
                "genre": "lofi",
                "mood": "relaxing",
                "duration": 30,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
            }

        # Add examples to Track
        if "Track" in schemas:
            schemas["Track"]["example"] = {
                "id": "track-123",
                "title": "Midnight Lofi",
                "description": "A chill lofi beat perfect for late night studying",
                "genre": "lofi",
                "duration": 180,
                "audioUrl": "/api/v1/tracks/track-123/audio",
                "waveformData": [0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1],
                "isPublic": True,
                "tags": ["lofi", "study", "chill", "relaxing"],
                "user": {
                    "id": "user-456",
                    "username": "lofiproducer",
                    "name": "Lofi Producer",
                    "avatar": "/avatars/user-456.jpg",
                },
                "stats": {
                    "plays": 1523,
                    "likes": 234,
                    "comments": 45,
                    "shares": 12,
                },
                "isLiked": True,
                "createdAt": "2024-01-15T10:30:00Z",
                "updatedAt": "2024-01-15T10:30:00Z",
            }

    app.openapi_schema = openapi_schema
    return openapi_schema


def save_openapi_schema(app: FastAPI, output_path: Path):
    """Save OpenAPI schema to file."""
    schema = generate_custom_openapi_schema(app)

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"OpenAPI schema saved to {output_path}")


def generate_typescript_types(openapi_schema: Dict[str, Any], output_path: Path):
    """
    Generate TypeScript types from OpenAPI schema.

    This is a simplified version. For production, use tools like:
    - openapi-typescript
    - openapi-generator
    - swagger-typescript-api
    """

    typescript_content = """// Generated TypeScript types from OpenAPI schema
// DO NOT EDIT - This file is auto-generated

// Base API Response
export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
}

// Paginated Response
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

"""

    # Extract and convert schemas
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        schemas = openapi_schema["components"]["schemas"]

        for schema_name, schema_def in schemas.items():
            # Skip internal schemas
            if schema_name.startswith("HTTPValidationError") or schema_name.startswith(
                "ValidationError"
            ):
                continue

            typescript_content += f"// {schema_name}\n"
            typescript_content += f"export interface {schema_name} {{\n"

            if "properties" in schema_def:
                for prop_name, prop_def in schema_def["properties"].items():
                    # Determine TypeScript type
                    ts_type = openapi_to_typescript_type(prop_def)

                    # Check if required
                    is_required = "required" in schema_def and prop_name in schema_def["required"]
                    optional = "" if is_required else "?"

                    # Add description as comment
                    if "description" in prop_def:
                        typescript_content += f"  /** {prop_def['description']} */\n"

                    typescript_content += f"  {prop_name}{optional}: {ts_type};\n"

            typescript_content += "}\n\n"

    # Add API client interface
    typescript_content += """
// API Client Interface
export interface MusicGenAPI {
  // Generation
  generate(request: GenerationRequest): Promise<ApiResponse<GenerationResponse>>;
  getGenerationStatus(taskId: string): Promise<ApiResponse<GenerationResponse>>;
  cancelGeneration(taskId: string): Promise<ApiResponse<void>>;
  getGenerationHistory(page?: number, limit?: number): Promise<ApiResponse<PaginatedResponse<GenerationResponse>>>;
  saveGeneration(taskId: string, metadata?: Record<string, any>): Promise<ApiResponse<void>>;

  // User
  getUserProfile(): Promise<ApiResponse<UserProfile>>;
  updateUserProfile(updates: Partial<UserProfile>): Promise<ApiResponse<UserProfile>>;

  // Tracks
  getTrendingTracks(limit?: number): Promise<ApiResponse<Track[]>>;
  getRecentTracks(page?: number, limit?: number): Promise<ApiResponse<PaginatedResponse<Track>>>;
  searchTracks(query: string, filters?: TrackSearchFilters): Promise<ApiResponse<PaginatedResponse<Track>>>;

  // Community
  getCommunityStats(): Promise<ApiResponse<CommunityStats>>;
  getFeaturedUsers(limit?: number): Promise<ApiResponse<User[]>>;
  getTrendingTopics(): Promise<ApiResponse<TrendingTopic[]>>;

  // Analytics
  trackEvent(event: string, properties?: Record<string, any>): Promise<ApiResponse<void>>;
}

// Additional Types
export interface TrackSearchFilters {
  genre?: string;
  user?: string;
  tags?: string[];
  page?: number;
  limit?: number;
}

export interface CommunityStats {
  totalUsers: number;
  totalTracks: number;
  totalPlays: number;
  totalMinutes: number;
}

export interface TrendingTopic {
  tag: string;
  count: number;
}
"""

    # Save to file
    with open(output_path, "w") as f:
        f.write(typescript_content)

    print(f"TypeScript types saved to {output_path}")


def openapi_to_typescript_type(prop_def: Dict[str, Any]) -> str:
    """Convert OpenAPI type to TypeScript type."""
    if "type" not in prop_def:
        if "$ref" in prop_def:
            # Extract type name from $ref
            ref = prop_def["$ref"]
            if ref.startswith("#/components/schemas/"):
                return ref.split("/")[-1]
        return "any"

    type_mapping = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "array": "Array",
        "object": "Record<string, any>",
    }

    base_type = prop_def["type"]

    if base_type == "array":
        if "items" in prop_def:
            item_type = openapi_to_typescript_type(prop_def["items"])
            return f"{item_type}[]"
        return "any[]"

    if base_type == "object":
        if "additionalProperties" in prop_def:
            value_type = openapi_to_typescript_type(prop_def["additionalProperties"])
            return f"Record<string, {value_type}>"
        return "Record<string, any>"

    # Handle enums
    if "enum" in prop_def:
        enum_values = " | ".join(f"'{v}'" for v in prop_def["enum"])
        return enum_values

    return type_mapping.get(base_type, "any")


def generate_api_documentation(app: FastAPI, output_dir: Path):
    """Generate complete API documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate OpenAPI schema
    schema = generate_custom_openapi_schema(app)

    # Save OpenAPI JSON
    openapi_path = output_dir / "openapi.json"
    save_openapi_schema(app, openapi_path)

    # Generate TypeScript types
    types_path = output_dir / "api-types.ts"
    generate_typescript_types(schema, types_path)

    # Generate Markdown documentation
    markdown_path = output_dir / "API_DOCUMENTATION.md"
    generate_markdown_docs(schema, markdown_path)

    print(f"API documentation generated in {output_dir}")


def generate_markdown_docs(schema: Dict[str, Any], output_path: Path):
    """Generate Markdown documentation from OpenAPI schema."""
    content = f"""# {schema['info']['title']} - API Documentation

Version: {schema['info']['version']}

{schema['info']['description']}

## Base URL

"""

    # Add servers
    for server in schema.get("servers", []):
        content += f"- **{server.get('description', 'Server')}**: `{server['url']}`\n"

    content += "\n## Authentication\n\n"

    # Add security schemes
    for scheme_name, scheme in schema.get("components", {}).get("securitySchemes", {}).items():
        content += f"### {scheme_name}\n"
        content += f"- **Type**: {scheme['type']}\n"
        if "description" in scheme:
            content += f"- **Description**: {scheme['description']}\n"
        content += "\n"

    content += "## Endpoints\n\n"

    # Group endpoints by tag
    endpoints_by_tag = {}
    for path, methods in schema.get("paths", {}).items():
        for method, operation in methods.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                tags = operation.get("tags", ["Other"])
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append((method.upper(), path, operation))

    # Write endpoints by tag
    for tag, endpoints in sorted(endpoints_by_tag.items()):
        content += f"### {tag.title()}\n\n"

        for method, path, operation in endpoints:
            content += f"#### {method} `{path}`\n\n"

            if "summary" in operation:
                content += f"{operation['summary']}\n\n"

            if "description" in operation:
                content += f"{operation['description']}\n\n"

            # Parameters
            if "parameters" in operation:
                content += "**Parameters:**\n\n"
                for param in operation["parameters"]:
                    required = "required" if param.get("required", False) else "optional"
                    content += f"- `{param['name']}` ({param.get('in', 'query')}, {required}): {param.get('description', 'No description')}\n"
                content += "\n"

            # Request body
            if "requestBody" in operation:
                content += "**Request Body:**\n\n"
                if "description" in operation["requestBody"]:
                    content += f"{operation['requestBody']['description']}\n\n"
                # Add schema example if available
                content += "```json\n"
                content += "// See schema for details\n"
                content += "```\n\n"

            # Responses
            content += "**Responses:**\n\n"
            for status_code, response in operation.get("responses", {}).items():
                content += f"- `{status_code}`: {response.get('description', 'No description')}\n"
            content += "\n---\n\n"

    # Save to file
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Markdown documentation saved to {output_path}")
