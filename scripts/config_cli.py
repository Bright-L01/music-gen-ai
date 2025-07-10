#!/usr/bin/env python3
"""
Configuration CLI tool for MusicGen AI.

Provides command-line interface for configuration validation, schema generation,
and configuration management tasks.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from music_gen.core.config_manager import (
    ConfigManager,
    load_config,
    validate_config_file,
    ConfigurationError,
)
from music_gen.core.config_models import MusicGenConfig


def validate_command(args):
    """Validate configuration file or environment."""
    print("🔍 Validating configuration...")

    if args.file:
        # Validate specific file
        result = validate_config_file(args.file)
        print(f"\nValidating file: {args.file}")
    else:
        # Validate environment configuration
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config(
                config_name=args.config or "base", environment=args.environment, validate=True
            )
            result = config_manager.validate_current_config()
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return 1

    # Display results
    if result.is_valid:
        print("✅ Configuration is valid!")

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")

        if result.config:
            summary = result.config.get_environment_summary()
            print(f"\n📋 Configuration Summary:")
            print(f"  Environment: {summary['environment']}")
            print(f"  Debug Mode: {summary['debug_mode']}")
            print(f"  API Workers: {summary['api_workers']}")
            print(f"  Auth Enabled: {summary['auth_enabled']}")
            print(f"  Default Model: {summary['default_model']}")
            print(f"  Cache Size: {summary['cache_size']}")
            print(f"  Max Duration: {summary['max_duration']}")
            print(f"  Monitoring: {summary['monitoring_enabled']}")
    else:
        print(f"❌ Configuration validation failed!")

        if result.errors:
            print(f"\n🚨 Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")

        return 1

    return 0


def schema_command(args):
    """Generate configuration schema."""
    print("📝 Generating configuration schema...")

    try:
        config_manager = ConfigManager()

        output_path = args.output or "config_schema.json"
        config_manager.save_config_schema(output_path)

        print(f"✅ Schema saved to: {output_path}")

        # Show basic schema info
        schema = MusicGenConfig.schema()
        print(f"\n📊 Schema Information:")
        print(f"  Title: {schema.get('title', 'N/A')}")
        print(f"  Properties: {len(schema.get('properties', {}))}")
        print(f"  Required Fields: {len(schema.get('required', []))}")

        return 0

    except Exception as e:
        print(f"❌ Failed to generate schema: {e}")
        return 1


def example_command(args):
    """Generate example configuration."""
    print(f"📄 Generating example {args.environment} configuration...")

    try:
        config_manager = ConfigManager()

        output_path = args.output or f"example_{args.environment}.yaml"
        config_manager.generate_example_config(output_path, args.environment)

        print(f"✅ Example configuration saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"❌ Failed to generate example: {e}")
        return 1


def load_command(args):
    """Load and display configuration."""
    print("📂 Loading configuration...")

    try:
        config = load_config(
            config_name=args.config or "base",
            environment=args.environment,
            overrides=args.override or [],
        )

        print("✅ Configuration loaded successfully!")

        # Display summary
        summary = config.get_environment_summary()
        print(f"\n📋 Configuration Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        # Save full config if requested
        if args.save:
            output_path = args.save
            config_manager = ConfigManager()
            config_manager._config = config

            if output_path.endswith(".json"):
                config_manager.export_config(output_path, "json")
            else:
                config_manager.export_config(output_path, "yaml")

            print(f"\n💾 Full configuration saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1


def check_command(args):
    """Check configuration health and requirements."""
    print("🏥 Checking configuration health...")

    try:
        config = load_config(config_name=args.config or "base", environment=args.environment)

        print("✅ Configuration loaded successfully!")

        # Check paths
        print(f"\n📁 Checking paths...")
        path_issues = config.validate_paths()

        if path_issues:
            print(f"⚠️  Path issues found ({len(path_issues)}):")
            for issue in path_issues:
                print(f"  - {issue}")
        else:
            print("✅ All paths are valid")

        # Check environment consistency
        print(f"\n🌍 Environment checks...")
        env = config.app.environment

        if env == "production":
            issues = []
            if not config.auth.enabled:
                issues.append("Authentication should be enabled in production")
            if config.app.debug:
                issues.append("Debug mode should be disabled in production")
            if config.api.workers < 2:
                issues.append("Should use multiple API workers in production")

            if issues:
                print(f"⚠️  Production issues ({len(issues)}):")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ Production configuration looks good")
        else:
            print(f"✅ {env.title()} configuration is appropriate")

        # Check resource settings
        print(f"\n💾 Resource configuration...")
        resources = config.resources
        if resources.max_memory_usage_percent > 90:
            print("⚠️  High memory usage limit (>90%)")
        if resources.max_cpu_usage_percent > 95:
            print("⚠️  High CPU usage limit (>95%)")
        if not resources.auto_cleanup_enabled:
            print("⚠️  Auto cleanup is disabled")

        return 0

    except Exception as e:
        print(f"❌ Configuration health check failed: {e}")
        return 1


def list_command(args):
    """List available configurations."""
    print("📋 Available configurations:")

    config_manager = ConfigManager()
    config_dir = Path(config_manager.config_dir)

    if not config_dir.exists():
        print(f"❌ Configuration directory not found: {config_dir}")
        return 1

    # Find all YAML config files
    config_files = []

    for yaml_file in config_dir.rglob("*.yaml"):
        rel_path = yaml_file.relative_to(config_dir)
        config_name = str(rel_path).replace(".yaml", "").replace("/", ".")
        config_files.append((config_name, yaml_file))

    if not config_files:
        print("❌ No configuration files found")
        return 1

    print(f"\n📂 Configuration directory: {config_dir}")
    print(f"📄 Found {len(config_files)} configuration files:\n")

    for config_name, file_path in sorted(config_files):
        print(f"  {config_name}")
        if args.detailed:
            try:
                # Quick validation
                result = validate_config_file(str(file_path))
                status = "✅ Valid" if result.is_valid else "❌ Invalid"
                print(f"    Status: {status}")
                print(f"    Path: {file_path}")

                if not result.is_valid and result.errors:
                    print(f"    Errors: {len(result.errors)}")
            except Exception as e:
                print(f"    Status: ❌ Error - {e}")
        print()

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MusicGen AI Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate --environment production
  %(prog)s schema --output config_schema.json
  %(prog)s example --environment staging --output staging_config.yaml
  %(prog)s load --environment development --save current_config.yaml
  %(prog)s check --environment production
  %(prog)s list --detailed
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--file", "-f", help="Specific config file to validate")
    validate_parser.add_argument("--config", "-c", help="Configuration name (default: base)")
    validate_parser.add_argument("--environment", "-e", help="Environment to validate")
    validate_parser.set_defaults(func=validate_command)

    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Generate configuration schema")
    schema_parser.add_argument(
        "--output", "-o", help="Output file path (default: config_schema.json)"
    )
    schema_parser.set_defaults(func=schema_command)

    # Example command
    example_parser = subparsers.add_parser("example", help="Generate example configuration")
    example_parser.add_argument(
        "--environment",
        "-e",
        choices=["development", "staging", "production"],
        default="development",
        help="Environment type",
    )
    example_parser.add_argument("--output", "-o", help="Output file path")
    example_parser.set_defaults(func=example_command)

    # Load command
    load_parser = subparsers.add_parser("load", help="Load and display configuration")
    load_parser.add_argument("--config", "-c", help="Configuration name (default: base)")
    load_parser.add_argument("--environment", "-e", help="Environment override")
    load_parser.add_argument(
        "--override", action="append", help="Configuration overrides (key=value)"
    )
    load_parser.add_argument("--save", "-s", help="Save loaded config to file")
    load_parser.set_defaults(func=load_command)

    # Check command
    check_parser = subparsers.add_parser("check", help="Check configuration health")
    check_parser.add_argument("--config", "-c", help="Configuration name (default: base)")
    check_parser.add_argument("--environment", "-e", help="Environment to check")
    check_parser.set_defaults(func=check_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List available configurations")
    list_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed information"
    )
    list_parser.set_defaults(func=list_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
