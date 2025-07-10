#!/usr/bin/env python3
"""
Secrets management CLI tool for MusicGen AI.

Provides command-line interface for managing secrets across different backends.
"""

import sys
import argparse
import getpass
import json
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from music_gen.core.secrets_manager import (
    SecretsManager,
    SecretBackend,
    SecretType,
    get_secrets_manager,
    validate_production_secrets,
    SecretGenerationError,
    SecretAccessError,
    SecretRotationError,
)


def list_command(args):
    """List all available secrets."""
    print("🔐 Listing secrets...")

    try:
        manager = get_secrets_manager()
        secrets = manager.list_secrets()

        if not secrets:
            print("No secrets found.")
            return 0

        print(f"\n📋 Found {len(secrets)} secrets:\n")

        # Group by backend
        backend_groups = {}
        for secret in secrets:
            backend = secret.backend.value
            if backend not in backend_groups:
                backend_groups[backend] = []
            backend_groups[backend].append(secret)

        for backend, backend_secrets in backend_groups.items():
            print(f"🔧 {backend.upper()} Backend ({len(backend_secrets)} secrets):")
            for secret in backend_secrets:
                status = "✅" if secret.masked_value else "❌"
                print(f"  {status} {secret.name}: {secret.masked_value or 'Not set'}")
                if secret.description:
                    print(f"      Description: {secret.description}")
            print()

        return 0

    except Exception as e:
        print(f"❌ Failed to list secrets: {e}")
        return 1


def get_command(args):
    """Get a specific secret value."""
    print(f"🔍 Getting secret: {args.name}")

    try:
        manager = get_secrets_manager()
        value = manager.get_secret(args.name)

        if value is None:
            print(f"❌ Secret '{args.name}' not found")
            return 1

        if args.show_value:
            print(f"✅ Secret value: {value}")
        else:
            masked_value = manager._mask_value(value)
            print(f"✅ Secret found: {masked_value}")
            print("💡 Use --show-value to display the actual value")

        return 0

    except Exception as e:
        print(f"❌ Failed to get secret: {e}")
        return 1


def set_command(args):
    """Set a secret value."""
    print(f"🔑 Setting secret: {args.name}")

    try:
        # Get secret value
        if args.value:
            value = args.value
        else:
            value = getpass.getpass(f"Enter value for '{args.name}': ")

        if not value:
            print("❌ Secret value cannot be empty")
            return 1

        # Determine backend
        backend = None
        if args.backend:
            try:
                backend = SecretBackend(args.backend)
            except ValueError:
                print(f"❌ Invalid backend: {args.backend}")
                print(f"Available backends: {[b.value for b in SecretBackend]}")
                return 1

        # Set secret
        manager = get_secrets_manager()
        success = manager.set_secret(args.name, value, backend)

        if success:
            backend_name = backend.value if backend else "default"
            print(f"✅ Secret '{args.name}' set in {backend_name} backend")
            return 0
        else:
            print(f"❌ Failed to set secret '{args.name}'")
            return 1

    except Exception as e:
        print(f"❌ Failed to set secret: {e}")
        return 1


def delete_command(args):
    """Delete a secret."""
    print(f"🗑️  Deleting secret: {args.name}")

    try:
        manager = get_secrets_manager()

        if not args.force:
            confirm = input(f"Are you sure you want to delete '{args.name}'? (y/N): ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return 0

        success = manager.delete_secret(args.name, args.all_backends)

        if success:
            scope = "all backends" if args.all_backends else "found backend"
            print(f"✅ Secret '{args.name}' deleted from {scope}")
            return 0
        else:
            print(f"❌ Failed to delete secret '{args.name}'")
            return 1

    except Exception as e:
        print(f"❌ Failed to delete secret: {e}")
        return 1


def validate_command(args):
    """Validate required secrets."""
    print("🔍 Validating secrets...")

    try:
        if args.production:
            # Validate production secrets
            validation_results = validate_production_secrets()
            print("\n🏭 Production Secrets Validation:")
        else:
            # Validate custom list
            required_secrets = args.secrets or []
            if not required_secrets:
                print("❌ No secrets specified for validation")
                return 1

            manager = get_secrets_manager()
            validation_results = manager.validate_required_secrets(required_secrets)
            print(f"\n📋 Custom Secrets Validation ({len(required_secrets)} secrets):")

        # Display results
        all_valid = True
        for secret_name, is_valid in validation_results.items():
            status = "✅" if is_valid else "❌"
            print(f"  {status} {secret_name}: {'Available' if is_valid else 'Missing'}")
            if not is_valid:
                all_valid = False

        if all_valid:
            print(f"\n✅ All secrets are available!")
            return 0
        else:
            print(f"\n❌ Some secrets are missing!")
            return 1

    except Exception as e:
        print(f"❌ Failed to validate secrets: {e}")
        return 1


def health_command(args):
    """Check secrets management system health."""
    print("🏥 Checking secrets system health...")

    try:
        manager = get_secrets_manager()
        health = manager.get_secret_health()

        print(f"\n📊 Health Status:")
        print(f"  Total secrets: {health['secrets_count']}")
        print(f"  Available backends: {len(health['available_backends'])}")

        print(f"\n🔧 Backend Status:")
        for backend_name, status in health["backends"].items():
            initialized = "✅" if status["initialized"] else "❌"
            available = "✅" if status["available"] else "❌"
            print(f"  {backend_name}:")
            print(f"    Initialized: {initialized}")
            print(f"    Available: {available}")

        print(f"\n📋 Available Backends:")
        for backend in health["available_backends"]:
            print(f"  ✅ {backend}")

        return 0

    except Exception as e:
        print(f"❌ Failed to check health: {e}")
        return 1


def backup_command(args):
    """Backup secrets to file."""
    print(f"💾 Backing up secrets to: {args.output}")

    try:
        manager = get_secrets_manager()

        if args.include_values:
            print("⚠️  WARNING: This backup will include actual secret values!")
            if not args.force:
                confirm = input("Are you sure? (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return 0

        success = manager.backup_secrets(args.output, args.include_values)

        if success:
            print(f"✅ Secrets backed up to: {args.output}")
            if args.include_values:
                print("🔐 Backup contains actual secret values - keep it secure!")
            else:
                print("📋 Backup contains metadata only (no actual values)")
            return 0
        else:
            print(f"❌ Failed to backup secrets")
            return 1

    except Exception as e:
        print(f"❌ Failed to backup secrets: {e}")
        return 1


def import_command(args):
    """Import secrets from backup file."""
    print(f"📂 Importing secrets from: {args.file}")

    try:
        if not Path(args.file).exists():
            print(f"❌ Backup file not found: {args.file}")
            return 1

        manager = get_secrets_manager()

        if args.overwrite:
            print("⚠️  WARNING: This will overwrite existing secrets!")
            if not args.force:
                confirm = input("Are you sure? (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return 0

        success = manager.import_secrets(args.file, args.overwrite)

        if success:
            print(f"✅ Secrets imported from: {args.file}")
            return 0
        else:
            print(f"❌ Failed to import secrets")
            return 1

    except Exception as e:
        print(f"❌ Failed to import secrets: {e}")
        return 1


def generate_command(args):
    """Generate secure random secrets using enhanced generator."""
    print(f"🎲 Generating secure secret: {args.name}")

    try:
        manager = get_secrets_manager()

        # Parse secret type
        try:
            secret_type = SecretType(args.type)
        except ValueError:
            print(f"❌ Invalid secret type: {args.type}")
            print(f"Available types: {[t.value for t in SecretType]}")
            return 1

        # Parse backend
        backend = None
        if args.backend:
            try:
                backend = SecretBackend(args.backend)
            except ValueError:
                print(f"❌ Invalid backend: {args.backend}")
                return 1

        # Parse tags
        tags = {}
        if hasattr(args, "tags") and args.tags:
            for tag_pair in args.tags.split(","):
                if "=" in tag_pair:
                    key, value = tag_pair.split("=", 1)
                    tags[key] = value

        if args.show_only:
            # Just generate and show, don't store
            if secret_type == SecretType.PASSWORD:
                value = manager.generator.generate_password(args.length)
            elif secret_type == SecretType.API_KEY:
                value = manager.generator.generate_api_key(args.length)
            elif secret_type == SecretType.JWT_SECRET:
                value = manager.generator.generate_jwt_secret(args.length)
            elif secret_type == SecretType.DATABASE_PASSWORD:
                value = manager.generator.generate_database_password(args.length)
            elif secret_type == SecretType.ENCRYPTION_KEY:
                value = manager.generator.generate_encryption_key()
            else:
                value = manager.generator.generate_password(args.length)

            print(f"Generated value: {value}")
            return 0

        # Generate and store the secret
        value = manager.generate_and_store_secret(
            name=args.name,
            secret_type=secret_type,
            length=args.length,
            description=getattr(args, "description", ""),
            expires_days=getattr(args, "expires_days", None),
            rotation_days=getattr(args, "rotation_days", None),
            tags=tags,
            backend=backend,
        )

        backend_name = backend.value if backend else "default"
        print(f"✅ Secret '{args.name}' generated and stored in {backend_name} backend")
        if args.show_value:
            print(f"Generated value: {value}")
        else:
            print(f"Generated value: {manager._mask_value(value)} (masked)")
            print("💡 Use --show-value to display the actual value")

        return 0

    except (SecretGenerationError, SecretAccessError) as e:
        print(f"❌ Failed to generate secret: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def rotate_command(args):
    """Rotate a secret."""
    print(f"🔄 Rotating secret: {args.name}")

    try:
        manager = get_secrets_manager()

        # Check if secret exists
        current_value = manager.get_secret(args.name)
        if current_value is None:
            print(f"❌ Secret '{args.name}' not found")
            return 1

        # Rotate the secret
        if hasattr(args, "value") and args.value:
            new_value = manager.rotate_secret(args.name, args.value)
        else:
            new_value = manager.rotate_secret(args.name)

        print(f"✅ Secret '{args.name}' rotated successfully")
        if hasattr(args, "show_value") and args.show_value:
            print(f"New value: {new_value}")
        else:
            print(f"New value: {manager._mask_value(new_value)} (masked)")

        return 0

    except SecretRotationError as e:
        print(f"❌ Failed to rotate secret: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def health_advanced_command(args):
    """Check advanced health of secrets including expiration and rotation."""
    print("🏥 Checking advanced secrets health...")

    try:
        manager = get_secrets_manager()

        # Get system health
        system_health = manager.get_secret_health()
        print(f"\n📊 System Status:")
        print(f"  Total secrets: {system_health['secrets_count']}")
        print(f"  Available backends: {len(system_health['available_backends'])}")

        # Get secret-specific health
        if hasattr(manager, "check_secret_health"):
            secret_health = manager.check_secret_health()

            print(f"\n🔍 Secret Health Report:")
            print(f"  Total secrets: {secret_health['total_secrets']}")

            if secret_health["expired_secrets"]:
                print(
                    f"  ❌ Expired ({len(secret_health['expired_secrets'])}): {', '.join(secret_health['expired_secrets'])}"
                )

            if secret_health["expiring_soon"]:
                print(
                    f"  ⚠️  Expiring soon ({len(secret_health['expiring_soon'])}): {', '.join(secret_health['expiring_soon'])}"
                )

            if secret_health["rotation_needed"]:
                print(
                    f"  🔄 Need rotation ({len(secret_health['rotation_needed'])}): {', '.join(secret_health['rotation_needed'])}"
                )

            healthy_count = len(secret_health["healthy_secrets"])
            if healthy_count > 0:
                print(f"  ✅ Healthy: {healthy_count} secrets")

        # Backend status
        print(f"\n🔧 Backend Status:")
        for backend_name, status in system_health["backends"].items():
            initialized = "✅" if status["initialized"] else "❌"
            available = "✅" if status["available"] else "❌"
            print(f"  {backend_name}: Initialized {initialized}, Available {available}")

        return 0

    except Exception as e:
        print(f"❌ Failed to check health: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MusicGen AI Secrets Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list
  %(prog)s get JWT_SECRET --show-value
  %(prog)s set API_KEY --backend keyring
  %(prog)s delete OLD_SECRET --force
  %(prog)s validate --production
  %(prog)s backup secrets_backup.json
  %(prog)s generate JWT_SECRET --type password --length 64
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all secrets")
    list_parser.set_defaults(func=list_command)

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a secret value")
    get_parser.add_argument("name", help="Secret name")
    get_parser.add_argument(
        "--show-value", action="store_true", help="Show actual value (use with caution)"
    )
    get_parser.set_defaults(func=get_command)

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a secret value")
    set_parser.add_argument("name", help="Secret name")
    set_parser.add_argument("--value", help="Secret value (will prompt if not provided)")
    set_parser.add_argument(
        "--backend", choices=[b.value for b in SecretBackend], help="Storage backend"
    )
    set_parser.set_defaults(func=set_command)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a secret")
    delete_parser.add_argument("name", help="Secret name")
    delete_parser.add_argument(
        "--all-backends", action="store_true", help="Delete from all backends"
    )
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=delete_command)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate required secrets")
    validate_parser.add_argument(
        "--production", action="store_true", help="Validate production secrets"
    )
    validate_parser.add_argument("--secrets", nargs="+", help="Custom list of secrets to validate")
    validate_parser.set_defaults(func=validate_command)

    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")
    health_parser.set_defaults(func=health_command)

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup secrets to file")
    backup_parser.add_argument("output", help="Output backup file")
    backup_parser.add_argument(
        "--include-values", action="store_true", help="Include actual secret values"
    )
    backup_parser.add_argument("--force", action="store_true", help="Skip confirmations")
    backup_parser.set_defaults(func=backup_command)

    # Import command
    import_parser = subparsers.add_parser("import", help="Import secrets from backup")
    import_parser.add_argument("file", help="Backup file to import")
    import_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing secrets"
    )
    import_parser.add_argument("--force", action="store_true", help="Skip confirmations")
    import_parser.set_defaults(func=import_command)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate secure secrets")
    generate_parser.add_argument("name", help="Secret name")
    generate_parser.add_argument(
        "--type", choices=[t.value for t in SecretType], default="password", help="Secret type"
    )
    generate_parser.add_argument("--length", type=int, default=32, help="Secret length")
    generate_parser.add_argument(
        "--backend", choices=[b.value for b in SecretBackend], help="Storage backend"
    )
    generate_parser.add_argument("--description", help="Secret description")
    generate_parser.add_argument("--expires-days", type=int, help="Days until expiration")
    generate_parser.add_argument("--rotation-days", type=int, help="Days between rotations")
    generate_parser.add_argument("--tags", help="Tags (key=value,key2=value2)")
    generate_parser.add_argument("--show-value", action="store_true", help="Show generated value")
    generate_parser.add_argument(
        "--show-only", action="store_true", help="Only show value, don't store"
    )
    generate_parser.set_defaults(func=generate_command)

    # Rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate a secret")
    rotate_parser.add_argument("name", help="Secret name")
    rotate_parser.add_argument("--value", help="New value (generated if not provided)")
    rotate_parser.add_argument("--show-value", action="store_true", help="Show new value")
    rotate_parser.set_defaults(func=rotate_command)

    # Advanced health command
    health_advanced_parser = subparsers.add_parser("health-advanced", help="Advanced health check")
    health_advanced_parser.set_defaults(func=health_advanced_command)

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
