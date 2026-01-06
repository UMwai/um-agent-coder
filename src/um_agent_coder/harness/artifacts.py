"""
Artifact management for cross-harness file sharing.

Provides a mechanism for sub-harnesses to publish and consume artifacts
(files, code, configs) that can be shared across harnesses.
"""

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """Shared file artifact between harnesses."""

    name: str
    source_harness: str
    path: Path
    artifact_type: str = "file"  # file, code, config, data
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "source_harness": self.source_harness,
            "path": str(self.path),
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            source_harness=data["source_harness"],
            path=Path(data["path"]),
            artifact_type=data.get("artifact_type", "file"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


class ArtifactManager:
    """
    Manage artifacts shared between harnesses.

    Artifacts are stored in a central directory and can be published
    by one harness and consumed by others.

    Example:
        manager = ArtifactManager()

        # Publisher harness creates artifact
        manager.publish(
            name="auth-config",
            source_harness="auth-harness",
            source_path=Path("./config/auth.json"),
            artifact_type="config",
            metadata={"version": "1.0"}
        )

        # Consumer harness gets artifact
        artifact = manager.get("auth-config")
        config_path = manager.consume(
            artifact,
            dest_path=Path("./local/auth.json")
        )
    """

    def __init__(self, artifacts_dir: Path = Path(".harness/shared/artifacts")):
        """Initialize artifact manager.

        Args:
            artifacts_dir: Directory for storing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.artifacts_dir / "registry.json"

        # Initialize registry if it doesn't exist
        if not self.registry_file.exists():
            self._write_registry({})

    def _read_registry(self) -> Dict[str, Dict[str, Any]]:
        """Read artifact registry from file."""
        try:
            if self.registry_file.exists():
                return json.loads(self.registry_file.read_text())
            return {}
        except json.JSONDecodeError:
            logger.warning("Registry file corrupted, returning empty")
            return {}

    def _write_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        """Write artifact registry to file."""
        self.registry_file.write_text(json.dumps(registry, indent=2))

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def publish(
        self,
        name: str,
        source_harness: str,
        source_path: Path,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Publish an artifact.

        Args:
            name: Unique artifact name
            source_harness: ID of publishing harness
            source_path: Path to source file
            artifact_type: Type of artifact (file, code, config, data)
            metadata: Optional metadata dict

        Returns:
            Created Artifact object
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Create artifact directory
        artifact_dir = self.artifacts_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to artifacts directory
        dest_path = artifact_dir / source_path.name
        shutil.copy2(source_path, dest_path)

        # Compute checksum
        checksum = self._compute_checksum(dest_path)

        # Create artifact object
        artifact = Artifact(
            name=name,
            source_harness=source_harness,
            path=dest_path,
            artifact_type=artifact_type,
            metadata=metadata or {},
            checksum=checksum,
        )

        # Update registry
        registry = self._read_registry()
        registry[name] = artifact.to_dict()
        self._write_registry(registry)

        logger.info(f"Published artifact: {name} from {source_harness}")
        return artifact

    def publish_content(
        self,
        name: str,
        source_harness: str,
        content: str,
        filename: str = "content.txt",
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Publish content as an artifact.

        Args:
            name: Unique artifact name
            source_harness: ID of publishing harness
            content: String content to publish
            filename: Name for the content file
            artifact_type: Type of artifact
            metadata: Optional metadata dict

        Returns:
            Created Artifact object
        """
        # Create artifact directory
        artifact_dir = self.artifacts_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Write content to file
        dest_path = artifact_dir / filename
        dest_path.write_text(content)

        # Compute checksum
        checksum = self._compute_checksum(dest_path)

        # Create artifact object
        artifact = Artifact(
            name=name,
            source_harness=source_harness,
            path=dest_path,
            artifact_type=artifact_type,
            metadata=metadata or {},
            checksum=checksum,
        )

        # Update registry
        registry = self._read_registry()
        registry[name] = artifact.to_dict()
        self._write_registry(registry)

        logger.info(f"Published content artifact: {name} from {source_harness}")
        return artifact

    def get(self, name: str) -> Optional[Artifact]:
        """Get an artifact by name.

        Args:
            name: Artifact name

        Returns:
            Artifact object or None
        """
        registry = self._read_registry()
        if name in registry:
            return Artifact.from_dict(registry[name])
        return None

    def consume(
        self,
        artifact: Artifact,
        dest_path: Optional[Path] = None,
    ) -> Path:
        """Consume an artifact (copy to destination).

        Args:
            artifact: Artifact to consume
            dest_path: Optional destination path (defaults to current dir)

        Returns:
            Path to consumed file
        """
        if dest_path is None:
            dest_path = Path.cwd() / artifact.path.name
        else:
            dest_path = Path(dest_path)

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(artifact.path, dest_path)

        logger.info(f"Consumed artifact: {artifact.name} -> {dest_path}")
        return dest_path

    def read_content(self, artifact: Artifact) -> str:
        """Read artifact content as string.

        Args:
            artifact: Artifact to read

        Returns:
            File content as string
        """
        return artifact.path.read_text()

    def list_artifacts(
        self,
        source_harness: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[Artifact]:
        """List artifacts with optional filtering.

        Args:
            source_harness: Filter by source harness
            artifact_type: Filter by artifact type

        Returns:
            List of matching Artifact objects
        """
        registry = self._read_registry()
        artifacts = []

        for name, data in registry.items():
            artifact = Artifact.from_dict(data)

            if source_harness and artifact.source_harness != source_harness:
                continue
            if artifact_type and artifact.artifact_type != artifact_type:
                continue

            artifacts.append(artifact)

        return artifacts

    def get_by_harness(self, harness_id: str) -> List[Artifact]:
        """Get all artifacts published by a harness.

        Args:
            harness_id: Harness identifier

        Returns:
            List of Artifact objects
        """
        return self.list_artifacts(source_harness=harness_id)

    def delete(self, name: str) -> bool:
        """Delete an artifact.

        Args:
            name: Artifact name

        Returns:
            True if deleted
        """
        registry = self._read_registry()
        if name not in registry:
            return False

        # Remove directory
        artifact_dir = self.artifacts_dir / name
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        # Update registry
        del registry[name]
        self._write_registry(registry)

        logger.info(f"Deleted artifact: {name}")
        return True

    def verify_checksum(self, artifact: Artifact) -> bool:
        """Verify artifact checksum.

        Args:
            artifact: Artifact to verify

        Returns:
            True if checksum matches
        """
        if not artifact.path.exists():
            return False

        current_checksum = self._compute_checksum(artifact.path)
        return current_checksum == artifact.checksum

    def cleanup_orphaned(self) -> int:
        """Remove artifact directories not in registry.

        Returns:
            Number of orphaned directories removed
        """
        registry = self._read_registry()
        removed = 0

        for item in self.artifacts_dir.iterdir():
            if item.is_dir() and item.name not in registry:
                shutil.rmtree(item)
                removed += 1
                logger.info(f"Removed orphaned artifact directory: {item.name}")

        return removed

    def clear_all(self) -> None:
        """Remove all artifacts."""
        for item in self.artifacts_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

        self._write_registry({})
        logger.info("Cleared all artifacts")
