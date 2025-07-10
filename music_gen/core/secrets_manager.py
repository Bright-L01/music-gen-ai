"""
Production-ready secrets management for MusicGen AI.

Provides secure handling of sensitive configuration data including API keys,
database passwords, and other secrets with support for multiple backends,
automatic rotation, and comprehensive security features.
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretBackend(Enum):
    """Available secret storage backends."""

    ENVIRONMENT = "environment"
    FILE = "file"
    KEYRING = "keyring"
    ENCRYPTED_FILE = "encrypted_file"
    VAULT = "vault"  # HashiCorp Vault
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager
    AZURE_KEYVAULT = "azure_keyvault"  # Azure Key Vault


@dataclass
class SecretMetadata:
    """Enhanced metadata for a secret."""

    name: str
    backend: SecretBackend
    description: Optional[str] = None
    required: bool = True
    masked_value: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    secret_type: str = "generic"

    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def needs_rotation(self) -> bool:
        """Check if secret needs rotation."""
        if self.rotation_interval_days is None:
            return False
        rotation_date = self.updated_at + timedelta(days=self.rotation_interval_days)
        return datetime.utcnow() > rotation_date


class SecretType(Enum):
    """Types of secrets with specific generation rules."""

    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_SECRET = "jwt_secret"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    CERTIFICATE = "certificate"
    GENERIC = "generic"


class SecretGenerationError(Exception):
    """Raised when secret generation fails."""

    pass


class SecretAccessError(Exception):
    """Raised when secret access fails."""

    pass


class SecretRotationError(Exception):
    """Raised when secret rotation fails."""

    pass


class SecretGenerator:
    """Generates secure secrets of various types."""

    @staticmethod
    def generate_password(length: int = 32, include_symbols: bool = True) -> str:
        """Generate a secure random password."""
        if length < 8:
            raise SecretGenerationError("Password length must be at least 8 characters")

        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""

        # Ensure at least one character from each required set
        password = [secrets.choice(lowercase), secrets.choice(uppercase), secrets.choice(digits)]

        if include_symbols:
            password.append(secrets.choice(symbols))

        # Fill remaining length with random characters
        all_chars = lowercase + uppercase + digits + symbols
        for _ in range(length - len(password)):
            password.append(secrets.choice(all_chars))

        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(password)

        return "".join(password)

    @staticmethod
    def generate_api_key(length: int = 64) -> str:
        """Generate a secure API key."""
        if length < 32:
            raise SecretGenerationError("API key length must be at least 32 characters")

        # Use URL-safe characters for API keys
        alphabet = string.ascii_letters + string.digits + "-_"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def generate_jwt_secret(length: int = 64) -> str:
        """Generate a secure JWT signing secret."""
        if length < 32:
            raise SecretGenerationError("JWT secret length must be at least 32 characters")

        # Generate random bytes and encode as base64
        random_bytes = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(random_bytes).decode("utf-8")

    @staticmethod
    def generate_encryption_key() -> str:
        """Generate a Fernet encryption key."""
        return Fernet.generate_key().decode("utf-8")

    @staticmethod
    def generate_database_password(length: int = 24) -> str:
        """Generate a database-safe password (no special characters that might cause issues)."""
        if length < 12:
            raise SecretGenerationError("Database password length must be at least 12 characters")

        # Use alphanumeric characters only to avoid shell/SQL escaping issues
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))


class SecretBackendInterface(ABC):
    """Abstract interface for secret storage backends."""

    @abstractmethod
    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret by name."""

    @abstractmethod
    def set_secret(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        """Store a secret with optional metadata."""

    @abstractmethod
    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secret names."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""

    def get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret (optional)."""
        return None

    def backup_secrets(self) -> Dict[str, Any]:
        """Create backup of all secrets (optional)."""
        return {}

    def restore_secrets(self, backup_data: Dict[str, Any]) -> bool:
        """Restore secrets from backup (optional)."""
        return False


class EnvironmentBackend(SecretBackendInterface):
    """Environment variable backend for secrets."""

    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment variable."""
        return os.getenv(name)

    def set_secret(self, name: str, value: str) -> bool:
        """Set environment variable (session only)."""
        try:
            os.environ[name] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set environment variable {name}: {e}")
            return False

    def delete_secret(self, name: str) -> bool:
        """Remove environment variable."""
        try:
            if name in os.environ:
                del os.environ[name]
            return True
        except Exception as e:
            logger.error(f"Failed to delete environment variable {name}: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List environment variables (filtered for secrets)."""
        secret_patterns = ["_KEY", "_SECRET", "_PASSWORD", "_TOKEN", "_API_KEY"]
        return [
            key
            for key in os.environ.keys()
            if any(pattern in key.upper() for pattern in secret_patterns)
        ]

    def is_available(self) -> bool:
        """Environment backend is always available."""
        return True


class FileBackend(SecretBackendInterface):
    """Plain file backend for secrets (development only)."""

    def __init__(self, secrets_file: str = ".secrets.json"):
        """Initialize file backend."""
        self.secrets_file = Path(secrets_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure secrets file exists."""
        if not self.secrets_file.exists():
            self._save_secrets({})

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from file."""
        try:
            with open(self.secrets_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load secrets file: {e}")
            return {}

    def _save_secrets(self, secrets: Dict[str, str]) -> bool:
        """Save secrets to file."""
        try:
            with open(self.secrets_file, "w") as f:
                json.dump(secrets, f, indent=2)

            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")
            return False

    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from file."""
        secrets = self._load_secrets()
        return secrets.get(name)

    def set_secret(self, name: str, value: str) -> bool:
        """Store secret in file."""
        secrets = self._load_secrets()
        secrets[name] = value
        return self._save_secrets(secrets)

    def delete_secret(self, name: str) -> bool:
        """Delete secret from file."""
        secrets = self._load_secrets()
        if name in secrets:
            del secrets[name]
            return self._save_secrets(secrets)
        return True

    def list_secrets(self) -> List[str]:
        """List secret names from file."""
        secrets = self._load_secrets()
        return list(secrets.keys())

    def is_available(self) -> bool:
        """File backend is always available."""
        return True


class KeyringBackend(SecretBackendInterface):
    """System keyring backend for secrets."""

    def __init__(self, service_name: str = "musicgen-ai"):
        """Initialize keyring backend."""
        self.service_name = service_name

    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from system keyring."""
        if not KEYRING_AVAILABLE:
            return None

        try:
            return keyring.get_password(self.service_name, name)
        except Exception as e:
            logger.error(f"Failed to get secret from keyring: {e}")
            return None

    def set_secret(self, name: str, value: str) -> bool:
        """Store secret in system keyring."""
        if not KEYRING_AVAILABLE:
            return False

        try:
            keyring.set_password(self.service_name, name, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret in keyring: {e}")
            return False

    def delete_secret(self, name: str) -> bool:
        """Delete secret from system keyring."""
        if not KEYRING_AVAILABLE:
            return False

        try:
            keyring.delete_password(self.service_name, name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from keyring: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List secrets from keyring (not directly supported)."""
        # Keyring doesn't provide listing functionality
        logger.warning("Keyring backend doesn't support listing secrets")
        return []

    def is_available(self) -> bool:
        """Check if keyring is available."""
        return KEYRING_AVAILABLE


class EncryptedFileBackend(SecretBackendInterface):
    """Encrypted file backend for secrets."""

    def __init__(self, secrets_file: str = ".secrets.encrypted", password: Optional[str] = None):
        """Initialize encrypted file backend."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required for encrypted file backend")

        self.secrets_file = Path(secrets_file)
        self.password = password or os.getenv("SECRETS_PASSWORD")

        if not self.password:
            raise ValueError("Password required for encrypted file backend")

        self._cipher = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create encryption cipher from password."""
        password_bytes = self.password.encode()
        salt = b"musicgen_ai_salt"  # In production, use random salt stored separately

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)

    def _load_secrets(self) -> Dict[str, str]:
        """Load and decrypt secrets from file."""
        if not self.secrets_file.exists():
            return {}

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self._cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to load encrypted secrets: {e}")
            return {}

    def _save_secrets(self, secrets: Dict[str, str]) -> bool:
        """Encrypt and save secrets to file."""
        try:
            data = json.dumps(secrets, indent=2).encode()
            encrypted_data = self._cipher.encrypt(data)

            with open(self.secrets_file, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to save encrypted secrets: {e}")
            return False

    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from encrypted file."""
        secrets = self._load_secrets()
        return secrets.get(name)

    def set_secret(self, name: str, value: str) -> bool:
        """Store secret in encrypted file."""
        secrets = self._load_secrets()
        secrets[name] = value
        return self._save_secrets(secrets)

    def delete_secret(self, name: str) -> bool:
        """Delete secret from encrypted file."""
        secrets = self._load_secrets()
        if name in secrets:
            del secrets[name]
            return self._save_secrets(secrets)
        return True

    def list_secrets(self) -> List[str]:
        """List secret names from encrypted file."""
        secrets = self._load_secrets()
        return list(secrets.keys())

    def is_available(self) -> bool:
        """Check if encrypted file backend is available."""
        return CRYPTOGRAPHY_AVAILABLE and self.password is not None


class HashiCorpVaultBackend(SecretBackendInterface):
    """HashiCorp Vault backend for production secret management."""

    def __init__(self, vault_url: str, token: Optional[str] = None, mount_path: str = "secret"):
        """Initialize Vault backend.

        Args:
            vault_url: Vault server URL
            token: Vault authentication token (or use VAULT_TOKEN env var)
            mount_path: KV secrets engine mount path
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library required for Vault backend")

        self.vault_url = vault_url.rstrip("/")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.mount_path = mount_path
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session with proper headers."""
        if self._session is None or self._session.closed:
            headers = {"X-Vault-Token": self.token, "Content-Type": "application/json"}
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _make_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Vault."""
        # For synchronous compatibility, we'll use a simple approach
        # In production, consider using async methods
        try:
            import requests

            headers = {"X-Vault-Token": self.token, "Content-Type": "application/json"}

            url = f"{self.vault_url}/v1/{path}"

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            elif method.upper() == "LIST":
                response = requests.request("LIST", url, headers=headers, timeout=30)
            else:
                raise SecretAccessError(f"Unsupported HTTP method: {method}")

            if response.status_code == 404:
                return {}

            if not (200 <= response.status_code < 300):
                raise SecretAccessError(
                    f"Vault request failed: {response.status_code} - {response.text}"
                )

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except ImportError:
            raise SecretAccessError("requests library required for Vault backend")
        except Exception as e:
            raise SecretAccessError(f"Vault connection failed: {e}")

    def _secret_path(self, name: str) -> str:
        """Get full secret path in Vault."""
        return f"{self.mount_path}/data/musicgen/{name}"

    def _metadata_path(self, name: str) -> str:
        """Get metadata path in Vault."""
        return f"{self.mount_path}/metadata/musicgen/{name}"

    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from Vault."""
        try:
            # Get secret data
            data_response = self._make_request("GET", self._secret_path(name))
            if not data_response or "data" not in data_response:
                return None

            secret_data = data_response["data"]["data"]
            return secret_data.get("value")

        except Exception as e:
            logger.error(f"Failed to get secret '{name}' from Vault: {e}")
            return None

    def set_secret(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        """Store secret in Vault."""
        try:
            # Prepare secret data
            secret_data = {
                "data": {
                    "value": value,
                    "created_by": "musicgen-secrets-manager",
                    "created_at": datetime.utcnow().isoformat(),
                }
            }

            # Store secret
            self._make_request("POST", self._secret_path(name), secret_data)

            # Update metadata if provided
            if metadata:
                self._update_metadata(name, metadata)

            logger.info(f"Secret '{name}' stored in Vault")
            return True

        except Exception as e:
            logger.error(f"Failed to store secret '{name}' in Vault: {e}")
            return False

    def _update_metadata(self, name: str, metadata: SecretMetadata) -> None:
        """Update secret metadata in Vault."""
        custom_metadata = {
            "description": metadata.description or "",
            "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else "",
            "rotation_interval_days": str(metadata.rotation_interval_days)
            if metadata.rotation_interval_days
            else "",
            "tags": json.dumps(metadata.tags),
            "access_count": str(metadata.access_count),
            "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else "",
            "secret_type": metadata.secret_type,
        }

        metadata_payload = {"custom_metadata": custom_metadata}

        try:
            self._make_request("POST", self._metadata_path(name), metadata_payload)
        except Exception as e:
            logger.warning(f"Failed to update metadata for '{name}': {e}")

    def delete_secret(self, name: str) -> bool:
        """Delete secret from Vault."""
        try:
            self._make_request("DELETE", self._metadata_path(name))
            logger.info(f"Secret '{name}' deleted from Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{name}' from Vault: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List all secrets in Vault."""
        try:
            response = self._make_request("LIST", f"{self.mount_path}/metadata/musicgen")
            return response.get("data", {}).get("keys", [])
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []

    def is_available(self) -> bool:
        """Check if Vault backend is available."""
        if not self.token:
            return False

        try:
            # Try to access Vault health endpoint
            response = self._make_request("GET", "sys/health")
            return "cluster_id" in response
        except Exception:
            return False

    def get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret from Vault."""
        try:
            metadata_response = self._make_request("GET", self._metadata_path(name))
            if not metadata_response:
                return None

            vault_metadata = metadata_response.get("data", {})
            custom_metadata = vault_metadata.get("custom_metadata", {})

            return SecretMetadata(
                name=name,
                backend=SecretBackend.VAULT,
                description=custom_metadata.get("description", ""),
                created_at=datetime.fromisoformat(
                    vault_metadata.get("created_time", datetime.utcnow().isoformat()).replace(
                        "Z", "+00:00"
                    )
                ),
                updated_at=datetime.fromisoformat(
                    vault_metadata.get("updated_time", datetime.utcnow().isoformat()).replace(
                        "Z", "+00:00"
                    )
                ),
                expires_at=datetime.fromisoformat(custom_metadata["expires_at"])
                if custom_metadata.get("expires_at")
                else None,
                rotation_interval_days=int(custom_metadata["rotation_interval_days"])
                if custom_metadata.get("rotation_interval_days")
                else None,
                tags=json.loads(custom_metadata.get("tags", "{}")),
                access_count=int(custom_metadata.get("access_count", "0")),
                secret_type=custom_metadata.get("secret_type", "generic"),
            )

        except Exception as e:
            logger.error(f"Failed to get metadata for '{name}' from Vault: {e}")
            return None

    def backup_secrets(self) -> Dict[str, Any]:
        """Create backup of all secrets from Vault."""
        try:
            secrets_list = self.list_secrets()
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "vault_url": self.vault_url,
                "mount_path": self.mount_path,
                "secrets": {},
            }

            for secret_name in secrets_list:
                secret_value = self.get_secret(secret_name)
                secret_metadata = self.get_secret_metadata(secret_name)

                if secret_value and secret_metadata:
                    backup_data["secrets"][secret_name] = {
                        "value": secret_value,
                        "metadata": {
                            "description": secret_metadata.description,
                            "created_at": secret_metadata.created_at.isoformat(),
                            "updated_at": secret_metadata.updated_at.isoformat(),
                            "expires_at": secret_metadata.expires_at.isoformat()
                            if secret_metadata.expires_at
                            else None,
                            "rotation_interval_days": secret_metadata.rotation_interval_days,
                            "tags": secret_metadata.tags,
                            "secret_type": secret_metadata.secret_type,
                        },
                    }

            logger.info(f"Created backup of {len(secrets_list)} secrets from Vault")
            return backup_data

        except Exception as e:
            logger.error(f"Failed to backup secrets from Vault: {e}")
            return {}

    def restore_secrets(self, backup_data: Dict[str, Any]) -> bool:
        """Restore secrets to Vault from backup."""
        try:
            secrets_data = backup_data.get("secrets", {})

            for secret_name, secret_info in secrets_data.items():
                metadata_dict = secret_info.get("metadata", {})
                metadata = SecretMetadata(
                    name=secret_name,
                    backend=SecretBackend.VAULT,
                    description=metadata_dict.get("description", ""),
                    created_at=datetime.fromisoformat(
                        metadata_dict.get("created_at", datetime.utcnow().isoformat())
                    ),
                    updated_at=datetime.fromisoformat(
                        metadata_dict.get("updated_at", datetime.utcnow().isoformat())
                    ),
                    expires_at=datetime.fromisoformat(metadata_dict["expires_at"])
                    if metadata_dict.get("expires_at")
                    else None,
                    rotation_interval_days=metadata_dict.get("rotation_interval_days"),
                    tags=metadata_dict.get("tags", {}),
                    secret_type=metadata_dict.get("secret_type", "generic"),
                )

                self.set_secret(secret_name, secret_info["value"], metadata)

            logger.info(f"Restored {len(secrets_data)} secrets to Vault")
            return True

        except Exception as e:
            logger.error(f"Failed to restore secrets to Vault: {e}")
            return False

    def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            # For sync compatibility, we'll just mark as closed
            # In async version, would await self._session.close()
            pass


class SecretsManager:
    """
    Centralized secrets management system.

    Provides unified interface for managing secrets across multiple backends
    with fallback support and security best practices.
    """

    def __init__(
        self,
        backends: Optional[List[SecretBackend]] = None,
        default_backend: SecretBackend = SecretBackend.ENVIRONMENT,
    ):
        """
        Initialize secrets manager.

        Args:
            backends: List of backends to use (in priority order)
            default_backend: Default backend for storing new secrets
        """
        self.backends = backends or [
            SecretBackend.VAULT,
            SecretBackend.ENVIRONMENT,
            SecretBackend.KEYRING,
            SecretBackend.ENCRYPTED_FILE,
            SecretBackend.FILE,
        ]
        self.default_backend = default_backend
        self._backend_instances: Dict[SecretBackend, SecretBackendInterface] = {}
        self._secret_metadata: Dict[str, SecretMetadata] = {}
        self.generator = SecretGenerator()
        self._rotation_tasks: Dict[str, asyncio.Task] = {}

        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize backend instances."""
        for backend in self.backends:
            try:
                if backend == SecretBackend.ENVIRONMENT:
                    self._backend_instances[backend] = EnvironmentBackend()
                elif backend == SecretBackend.FILE:
                    self._backend_instances[backend] = FileBackend()
                elif backend == SecretBackend.KEYRING:
                    self._backend_instances[backend] = KeyringBackend()
                elif backend == SecretBackend.ENCRYPTED_FILE:
                    # Only initialize if password is available
                    password = os.getenv("SECRETS_PASSWORD")
                    if password:
                        self._backend_instances[backend] = EncryptedFileBackend(password=password)
                    else:
                        logger.debug("Encrypted file backend skipped: no password")
                elif backend == SecretBackend.VAULT:
                    # Initialize Vault backend if URL is available
                    vault_url = os.getenv("VAULT_URL")
                    vault_token = os.getenv("VAULT_TOKEN")
                    if vault_url and vault_token:
                        self._backend_instances[backend] = HashiCorpVaultBackend(
                            vault_url, vault_token
                        )
                    else:
                        logger.debug("Vault backend skipped: missing URL or token")

            except Exception as e:
                logger.warning(f"Failed to initialize {backend.value} backend: {e}")

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret value from any available backend.

        Args:
            name: Secret name
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        for backend_type in self.backends:
            backend = self._backend_instances.get(backend_type)
            if not backend or not backend.is_available():
                continue

            try:
                value = backend.get_secret(name)
                if value is not None:
                    logger.debug(f"Secret '{name}' found in {backend_type.value} backend")
                    return value
            except Exception as e:
                logger.warning(f"Error getting secret from {backend_type.value}: {e}")

        logger.debug(f"Secret '{name}' not found in any backend, using default")
        return default

    def set_secret(self, name: str, value: str, backend: Optional[SecretBackend] = None) -> bool:
        """
        Store secret in specified or default backend.

        Args:
            name: Secret name
            value: Secret value
            backend: Specific backend to use

        Returns:
            True if successful
        """
        target_backend = backend or self.default_backend
        backend_instance = self._backend_instances.get(target_backend)

        if not backend_instance or not backend_instance.is_available():
            logger.error(f"Backend {target_backend.value} not available")
            return False

        try:
            success = backend_instance.set_secret(name, value)
            if success:
                # Update metadata
                self._secret_metadata[name] = SecretMetadata(
                    name=name, backend=target_backend, masked_value=self._mask_value(value)
                )
                logger.info(f"Secret '{name}' stored in {target_backend.value} backend")
            return success
        except Exception as e:
            logger.error(f"Error storing secret in {target_backend.value}: {e}")
            return False

    def delete_secret(self, name: str, all_backends: bool = False) -> bool:
        """
        Delete secret from backends.

        Args:
            name: Secret name
            all_backends: If True, delete from all backends

        Returns:
            True if successful
        """
        success = False

        if all_backends:
            # Delete from all backends
            for backend_type, backend in self._backend_instances.items():
                if backend and backend.is_available():
                    try:
                        if backend.delete_secret(name):
                            logger.info(f"Secret '{name}' deleted from {backend_type.value}")
                            success = True
                    except Exception as e:
                        logger.warning(f"Error deleting from {backend_type.value}: {e}")
        else:
            # Find and delete from the backend that has the secret
            for backend_type in self.backends:
                backend = self._backend_instances.get(backend_type)
                if not backend or not backend.is_available():
                    continue

                try:
                    if backend.get_secret(name) is not None:
                        success = backend.delete_secret(name)
                        if success:
                            logger.info(f"Secret '{name}' deleted from {backend_type.value}")
                        break
                except Exception as e:
                    logger.warning(f"Error deleting from {backend_type.value}: {e}")

        # Remove from metadata
        if success and name in self._secret_metadata:
            del self._secret_metadata[name]

        return success

    def list_secrets(self) -> List[SecretMetadata]:
        """
        List all available secrets from all backends.

        Returns:
            List of secret metadata
        """
        all_secrets = set()
        secret_backends = {}

        for backend_type, backend in self._backend_instances.items():
            if not backend or not backend.is_available():
                continue

            try:
                secrets = backend.list_secrets()
                for secret_name in secrets:
                    all_secrets.add(secret_name)
                    if secret_name not in secret_backends:
                        secret_backends[secret_name] = backend_type
            except Exception as e:
                logger.warning(f"Error listing secrets from {backend_type.value}: {e}")

        # Create metadata for each secret
        metadata_list = []
        for secret_name in sorted(all_secrets):
            backend = secret_backends.get(secret_name, SecretBackend.ENVIRONMENT)

            # Get value to create masked version
            value = self.get_secret(secret_name)
            masked_value = self._mask_value(value) if value else None

            metadata = SecretMetadata(name=secret_name, backend=backend, masked_value=masked_value)
            metadata_list.append(metadata)

        return metadata_list

    def validate_required_secrets(self, required_secrets: List[str]) -> Dict[str, bool]:
        """
        Validate that all required secrets are available.

        Args:
            required_secrets: List of required secret names

        Returns:
            Dict mapping secret names to availability status
        """
        validation_results = {}

        for secret_name in required_secrets:
            value = self.get_secret(secret_name)
            validation_results[secret_name] = value is not None

        return validation_results

    def get_secret_health(self) -> Dict[str, Any]:
        """
        Get health status of secrets management system.

        Returns:
            Health status information
        """
        backend_status = {}
        for backend_type, backend in self._backend_instances.items():
            backend_status[backend_type.value] = {
                "available": backend.is_available() if backend else False,
                "initialized": backend is not None,
            }

        secrets_count = len(self.list_secrets())

        return {
            "backends": backend_status,
            "secrets_count": secrets_count,
            "available_backends": [
                bt.value for bt, b in self._backend_instances.items() if b and b.is_available()
            ],
        }

    def _mask_value(self, value: str) -> str:
        """Create masked version of secret value."""
        if not value:
            return ""

        if len(value) <= 4:
            return "*" * len(value)

        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def generate_and_store_secret(
        self,
        name: str,
        secret_type: SecretType = SecretType.PASSWORD,
        length: int = 32,
        description: str = "",
        expires_days: Optional[int] = None,
        rotation_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        backend: Optional[SecretBackend] = None,
    ) -> str:
        """Generate and store a new secret."""
        # Generate the secret based on type
        if secret_type == SecretType.PASSWORD:
            value = self.generator.generate_password(length, True)
        elif secret_type == SecretType.API_KEY:
            value = self.generator.generate_api_key(length)
        elif secret_type == SecretType.JWT_SECRET:
            value = self.generator.generate_jwt_secret(length)
        elif secret_type == SecretType.DATABASE_PASSWORD:
            value = self.generator.generate_database_password(length)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            value = self.generator.generate_encryption_key()
        else:
            value = self.generator.generate_password(length, False)

        # Create metadata
        now = datetime.utcnow()
        expires_at = now + timedelta(days=expires_days) if expires_days else None

        metadata = SecretMetadata(
            name=name,
            backend=backend or self.default_backend,
            description=description,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            rotation_interval_days=rotation_days,
            tags=tags or {},
            secret_type=secret_type.value,
            masked_value=self._mask_value(value),
        )

        # Store the secret
        target_backend = backend or self.default_backend
        backend_instance = self._backend_instances.get(target_backend)

        if not backend_instance or not backend_instance.is_available():
            raise SecretAccessError(f"Backend {target_backend.value} not available")

        success = backend_instance.set_secret(name, value, metadata)
        if not success:
            raise SecretAccessError(f"Failed to store secret '{name}'")

        # Store metadata locally
        self._secret_metadata[name] = metadata

        # Schedule rotation if specified
        if rotation_days:
            self._schedule_rotation(name, rotation_days)

        logger.info(f"Generated and stored secret '{name}' with type {secret_type.value}")
        return value

    def rotate_secret(self, name: str, new_value: Optional[str] = None) -> str:
        """Rotate a secret (generate new value or use provided one)."""
        # Get current secret metadata
        current_metadata = self._get_secret_metadata(name)
        if not current_metadata:
            raise SecretRotationError(f"Secret '{name}' not found")

        # Generate new value if not provided
        if new_value is None:
            secret_type = (
                SecretType(current_metadata.secret_type)
                if current_metadata.secret_type
                else SecretType.PASSWORD
            )

            if secret_type == SecretType.PASSWORD:
                new_value = self.generator.generate_password()
            elif secret_type == SecretType.API_KEY:
                new_value = self.generator.generate_api_key()
            elif secret_type == SecretType.JWT_SECRET:
                new_value = self.generator.generate_jwt_secret()
            elif secret_type == SecretType.DATABASE_PASSWORD:
                new_value = self.generator.generate_database_password()
            elif secret_type == SecretType.ENCRYPTION_KEY:
                new_value = self.generator.generate_encryption_key()
            else:
                new_value = self.generator.generate_password()

        # Update metadata
        current_metadata.updated_at = datetime.utcnow()
        current_metadata.masked_value = self._mask_value(new_value)

        # Store the new secret
        backend_instance = self._backend_instances.get(current_metadata.backend)
        if not backend_instance or not backend_instance.is_available():
            raise SecretRotationError(f"Backend {current_metadata.backend.value} not available")

        success = backend_instance.set_secret(name, new_value, current_metadata)
        if not success:
            raise SecretRotationError(f"Failed to rotate secret '{name}'")

        # Update local metadata
        self._secret_metadata[name] = current_metadata

        logger.info(f"Rotated secret '{name}' successfully")
        return new_value

    def _get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret from backend or cache."""
        # Try local cache first
        if name in self._secret_metadata:
            return self._secret_metadata[name]

        # Try to get from backend
        for backend_type in self.backends:
            backend = self._backend_instances.get(backend_type)
            if not backend or not backend.is_available():
                continue

            try:
                if hasattr(backend, "get_secret_metadata"):
                    metadata = backend.get_secret_metadata(name)
                    if metadata:
                        self._secret_metadata[name] = metadata
                        return metadata
            except Exception as e:
                logger.warning(f"Error getting metadata from {backend_type.value}: {e}")

        return None

    def _schedule_rotation(self, name: str, rotation_days: int) -> None:
        """Schedule automatic secret rotation."""

        async def rotation_task():
            while True:
                await asyncio.sleep(rotation_days * 24 * 3600)  # Convert days to seconds
                try:
                    self.rotate_secret(name)
                    logger.info(f"Automatically rotated secret '{name}'")
                except Exception as e:
                    logger.error(f"Automatic rotation failed for '{name}': {e}")

        # Cancel existing rotation task if any
        if name in self._rotation_tasks:
            self._rotation_tasks[name].cancel()

        # Start new rotation task (would need to run in event loop)
        try:
            loop = asyncio.get_event_loop()
            self._rotation_tasks[name] = loop.create_task(rotation_task())
        except RuntimeError:
            # No event loop running - rotation will need to be triggered manually
            logger.warning(f"No event loop available for automatic rotation of '{name}'")

    def check_secret_health(self) -> Dict[str, Any]:
        """Check health of all secrets (expiration, rotation needs)."""
        secrets_list = self.list_secrets()
        health_report = {
            "total_secrets": len(secrets_list),
            "expired_secrets": [],
            "expiring_soon": [],
            "rotation_needed": [],
            "healthy_secrets": [],
        }

        for metadata in secrets_list:
            try:
                if metadata.is_expired():
                    health_report["expired_secrets"].append(metadata.name)
                elif metadata.expires_at and metadata.expires_at <= datetime.utcnow() + timedelta(
                    days=7
                ):
                    health_report["expiring_soon"].append(metadata.name)
                elif metadata.needs_rotation():
                    health_report["rotation_needed"].append(metadata.name)
                else:
                    health_report["healthy_secrets"].append(metadata.name)
            except Exception as e:
                logger.error(f"Failed to check health of secret '{metadata.name}': {e}")

        return health_report

    def get_secrets_by_tag(self, tag_key: str, tag_value: str) -> List[str]:
        """Get secrets that have a specific tag."""
        matching_secrets = []

        for metadata in self.list_secrets():
            if metadata.tags.get(tag_key) == tag_value:
                matching_secrets.append(metadata.name)

        return matching_secrets

    def export_secrets_for_environment(self, environment: str) -> Dict[str, str]:
        """Export secrets formatted for environment variables."""
        env_vars = {}
        env_secrets = self.get_secrets_by_tag("environment", environment)

        for secret_name in env_secrets:
            # Convert secret name to environment variable format
            env_name = secret_name.upper().replace("-", "_").replace(".", "_")
            value = self.get_secret(secret_name)
            if value:
                env_vars[env_name] = value

        return env_vars

    def backup_secrets(self, backup_file: str, include_values: bool = False) -> bool:
        """
        Backup secrets metadata (and optionally values) to file.

        Args:
            backup_file: Path to backup file
            include_values: Whether to include actual secret values

        Returns:
            True if successful
        """
        try:
            secrets_data = []

            for metadata in self.list_secrets():
                secret_data = {
                    "name": metadata.name,
                    "backend": metadata.backend.value,
                    "description": metadata.description,
                    "required": metadata.required,
                }

                if include_values:
                    # WARNING: This includes actual secret values
                    secret_data["value"] = self.get_secret(metadata.name)
                else:
                    secret_data["masked_value"] = metadata.masked_value

                secrets_data.append(secret_data)

            backup_data = {
                "timestamp": str(os.path.getctime(backup_file)),
                "include_values": include_values,
                "secrets": secrets_data,
            }

            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            # Set restrictive permissions
            os.chmod(backup_file, 0o600)

            logger.info(f"Secrets backup created: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup secrets: {e}")
            return False

    def import_secrets(self, backup_file: str, overwrite: bool = False) -> bool:
        """
        Import secrets from backup file.

        Args:
            backup_file: Path to backup file
            overwrite: Whether to overwrite existing secrets

        Returns:
            True if successful
        """
        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            secrets = backup_data.get("secrets", [])
            imported_count = 0

            for secret_data in secrets:
                name = secret_data["name"]
                value = secret_data.get("value")

                if not value:
                    logger.warning(f"No value for secret '{name}', skipping")
                    continue

                # Check if secret already exists
                if not overwrite and self.get_secret(name) is not None:
                    logger.info(f"Secret '{name}' already exists, skipping")
                    continue

                # Import secret
                backend_name = secret_data.get("backend", self.default_backend.value)
                try:
                    backend = SecretBackend(backend_name)
                except ValueError:
                    backend = self.default_backend

                if self.set_secret(name, value, backend):
                    imported_count += 1
                    logger.info(f"Imported secret '{name}'")
                else:
                    logger.warning(f"Failed to import secret '{name}'")

            logger.info(f"Imported {imported_count} secrets from {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to import secrets: {e}")
            return False


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    manager = get_secrets_manager()
    return manager.get_secret(name, default)


def set_secret(name: str, value: str, backend: Optional[SecretBackend] = None) -> bool:
    """Convenience function to set a secret."""
    manager = get_secrets_manager()
    return manager.set_secret(name, value, backend)


def delete_secret(name: str, all_backends: bool = False) -> bool:
    """Convenience function to delete a secret."""
    manager = get_secrets_manager()
    return manager.delete_secret(name, all_backends)


def validate_production_secrets() -> Dict[str, bool]:
    """Validate that all production secrets are available."""
    required_secrets = ["JWT_SECRET", "POSTGRES_PASSWORD", "REDIS_PASSWORD", "WANDB_API_KEY"]

    manager = get_secrets_manager()
    return manager.validate_required_secrets(required_secrets)
