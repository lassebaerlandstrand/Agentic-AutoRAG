"""LanceDB wrapper for vector and hybrid retrieval."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import lancedb
import numpy as np
from lancedb.rerankers import CrossEncoderReranker


class LanceDBStore:
    """Thin wrapper around LanceDB for vector and hybrid search."""

    def __init__(self, db_path: str | Path = "./data/lancedb") -> None:
        self.db = lancedb.connect(str(db_path))
        self.table = None

    def create_index(
        self,
        documents: list[dict],
        table_name: str = "documents",
        mode: str = "overwrite",
    ) -> None:
        """Create or replace a LanceDB table and its BM25 index."""
        self.table = self.db.create_table(table_name, data=documents, mode=mode)
        self.table.create_fts_index("text", replace=True)

    def search_vector(self, query_embedding: np.ndarray | Sequence[float], top_k: int = 5) -> list[dict]:
        """Run dense-vector retrieval using a precomputed query embedding."""
        table = self._require_table()
        vector = self._normalize_vector(query_embedding)
        return table.search(vector).limit(top_k).to_list()

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Run hybrid BM25 + vector retrieval."""
        vector = self._normalize_vector(query_embedding)
        return self._build_hybrid_query(query, vector).limit(top_k).to_list()

    def search_hybrid_reranked(
        self,
        query: str,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 5,
        reranker_model: str | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval and optionally rerank with a cross-encoder."""
        vector = self._normalize_vector(query_embedding)
        candidate_count = max(top_k * 3, top_k)
        query_builder = self._build_hybrid_query(query, vector).limit(candidate_count)

        if reranker_model:
            reranker = CrossEncoderReranker(model_name=reranker_model)
            query_builder = query_builder.rerank(reranker=reranker)

        return query_builder.limit(top_k).to_list()

    def snapshot(self, path: str | Path) -> None:
        """Copy the current Lance table directory to *path*."""
        table = self._require_table()
        if hasattr(table, "checkout_latest"):
            table.checkout_latest()

        source = self._resolve_table_path()
        destination = Path(path)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    def _require_table(self):
        if self.table is None:
            raise RuntimeError("Index table is not initialized. Call create_index() first.")
        return self.table

    @staticmethod
    def _normalize_vector(query_embedding: np.ndarray | Sequence[float]) -> list[float]:
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim != 1:
                raise ValueError("query_embedding must be a 1D array")
            return query_embedding.astype(float).tolist()
        return [float(v) for v in query_embedding]

    def _resolve_table_path(self) -> Path:
        db_root = self._resolve_db_root()
        table_name = self._require_table().name
        candidates = [db_root / table_name, db_root / f"{table_name}.lance"]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        for child in sorted(db_root.iterdir()):
            if child.is_dir() and child.name.startswith(table_name):
                return child

        raise FileNotFoundError(f"Could not locate table directory for '{table_name}' under {db_root}")

    def _resolve_db_root(self) -> Path:
        uri = str(self.db.uri)
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return Path(parsed.path)
        return Path(uri)

    def _build_hybrid_query(self, query: str, vector: list[float]):
        table = self._require_table()
        return table.search(query_type="hybrid").vector(vector).text(query)
