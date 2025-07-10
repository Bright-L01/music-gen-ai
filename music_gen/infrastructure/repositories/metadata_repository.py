"""
Metadata repository implementations.

Provides concrete implementations for metadata storage and retrieval.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from music_gen.core.exceptions import MetadataNotFoundError
from music_gen.core.interfaces.repositories import MetadataRepository

logger = logging.getLogger(__name__)


class FileSystemMetadataRepository(MetadataRepository):
    """File system based metadata repository."""

    def __init__(self, base_path: Path):
        """Initialize repository with base path.

        Args:
            base_path: Base directory for metadata storage
        """
        self.base_path = Path(base_path)
        self.metadata_dir = self.base_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self, dataset_id: str) -> Path:
        """Get path for dataset metadata."""
        safe_id = dataset_id.replace("/", "_").replace(":", "_")
        return self.metadata_dir / f"{safe_id}.json"

    async def save_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """Save dataset metadata."""
        try:
            metadata_path = self._get_metadata_path(dataset_id)

            # Add timestamps
            metadata["dataset_id"] = dataset_id
            metadata["saved_at"] = datetime.utcnow().isoformat()

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved for dataset: {dataset_id}")

        except Exception as e:
            logger.error(f"Failed to save metadata for {dataset_id}: {e}")
            raise

    async def load_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata_path = self._get_metadata_path(dataset_id)

        if not metadata_path.exists():
            raise MetadataNotFoundError(f"Metadata not found for dataset: {dataset_id}")

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {dataset_id}: {e}")
            raise

    async def update_metadata(self, dataset_id: str, updates: Dict[str, Any]) -> None:
        """Update existing metadata."""
        # Load existing metadata
        metadata = await self.load_metadata(dataset_id)

        # Update fields
        metadata.update(updates)
        metadata["updated_at"] = datetime.utcnow().isoformat()

        # Save back
        await self.save_metadata(dataset_id, metadata)

    async def search_metadata(
        self, query: Dict[str, Any], limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search metadata by criteria."""
        results = []

        # Scan all metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Check if metadata matches query
                if self._matches_query(metadata, query):
                    results.append(metadata)

                    if len(results) >= limit:
                        break

            except Exception as e:
                logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                continue

        return results

    def _matches_query(self, metadata: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if metadata matches search query."""
        for key, value in query.items():
            if key not in metadata:
                return False

            # Handle different query types
            if isinstance(value, dict):
                # Range query
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False

        return True

    async def delete_metadata(self, dataset_id: str) -> None:
        """Delete metadata."""
        metadata_path = self._get_metadata_path(dataset_id)

        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Metadata deleted for dataset: {dataset_id}")
        else:
            logger.warning(f"Metadata not found for deletion: {dataset_id}")
