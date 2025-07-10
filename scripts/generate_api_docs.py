#!/usr/bin/env python3
"""
Generate API documentation and TypeScript types from OpenAPI schema.

Usage:
    python generate_api_docs.py --output-dir docs/api
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from music_gen.api.app import create_app
from music_gen.api.openapi_schema import generate_api_documentation


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/api"),
        help="Output directory for documentation files",
    )
    parser.add_argument(
        "--format",
        choices=["all", "openapi", "typescript", "markdown"],
        default="all",
        help="Documentation format to generate",
    )

    args = parser.parse_args()

    # Create FastAPI app
    app = create_app()

    # Generate documentation
    if args.format == "all":
        generate_api_documentation(app, args.output_dir)
    elif args.format == "openapi":
        from music_gen.api.openapi_schema import save_openapi_schema

        save_openapi_schema(app, args.output_dir / "openapi.json")
    elif args.format == "typescript":
        from music_gen.api.openapi_schema import (
            generate_custom_openapi_schema,
            generate_typescript_types,
        )

        schema = generate_custom_openapi_schema(app)
        generate_typescript_types(schema, args.output_dir / "api-types.ts")
    elif args.format == "markdown":
        from music_gen.api.openapi_schema import (
            generate_custom_openapi_schema,
            generate_markdown_docs,
        )

        schema = generate_custom_openapi_schema(app)
        generate_markdown_docs(schema, args.output_dir / "API_DOCUMENTATION.md")

    print(f"Documentation generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main()
