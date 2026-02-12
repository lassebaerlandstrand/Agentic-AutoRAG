"""Index registry for caching built indices by structural fingerprint."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from agentic_autorag.config.models import StructuralConfig


class IndexRegistry:
    """Caches built indices keyed by structural fingerprint."""

    def __init__(self, registry_dir: str = "./experiments/indices") -> None:
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest: dict[str, dict] = self._load_manifest()

    def has(self, fingerprint: str) -> bool:
        return fingerprint in self.manifest

    def get(self, fingerprint: str) -> Path:
        """Return the path to a cached index snapshot."""
        if fingerprint not in self.manifest:
            raise KeyError(f"Fingerprint '{fingerprint}' not found in index registry")
        return Path(self.manifest[fingerprint]["path"])

    def register(self, fingerprint: str, index_path: Path, config: StructuralConfig) -> None:
        """Copy a built index into the registry and update manifest metadata."""
        if not index_path.exists() or not index_path.is_dir():
            raise FileNotFoundError(f"Index path does not exist or is not a directory: {index_path}")

        destination = self.dir / fingerprint
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(index_path, destination)

        self.manifest[fingerprint] = {
            "path": str(destination),
            "config": config.model_dump(mode="json"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._save_manifest()

    def _load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def _save_manifest(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)
