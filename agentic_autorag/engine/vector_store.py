"""LanceDB wrapper for vector and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.rerankers import Reranker


class HybridAlphaReranker(Reranker):
    """Custom hybrid reranker that blends vector and FTS scores with alpha."""

    def __init__(self, alpha: float, return_score: str = "all") -> None:
        super().__init__(return_score=return_score)
        self.alpha = float(min(1.0, max(0.0, alpha)))

    def rerank_hybrid(self, query: str, vector_results: pa.Table, fts_results: pa.Table) -> pa.Table:
        del query
        vector_rows = vector_results.to_pylist()
        fts_rows = fts_results.to_pylist()
        merged_rows = self._merge_rows(vector_rows, fts_rows)

        vector_scores = self._normalize_vector_scores(vector_rows)
        fts_scores = self._normalize_fts_scores(fts_rows)
        default_score = 0.0

        for index, row in enumerate(merged_rows):
            key = self._row_key(row, index=index)
            vector_score = vector_scores.get(key, default_score)
            fts_score = fts_scores.get(key, default_score)
            row["_relevance_score"] = self.alpha * vector_score + (1.0 - self.alpha) * fts_score

        merged_rows.sort(key=lambda row: float(row.get("_relevance_score", 0.0)), reverse=True)
        return pa.Table.from_pylist(merged_rows)

    @classmethod
    def _merge_rows(cls, vector_rows: list[dict], fts_rows: list[dict]) -> list[dict]:
        """Merge vector/FTS rows by key while preserving first-seen row payload."""
        merged: dict[str, dict] = {}
        for index, row in enumerate(vector_rows):
            key = cls._row_key(row, index=index)
            merged[key] = dict(row)
        offset = len(vector_rows)
        for index, row in enumerate(fts_rows):
            key = cls._row_key(row, index=offset + index)
            if key in merged:
                existing = merged[key]
                for field, value in row.items():
                    if field not in existing or existing[field] is None:
                        existing[field] = value
            else:
                merged[key] = dict(row)
        return list(merged.values())

    @staticmethod
    def _row_key(row: dict, index: int) -> str:
        doc_id = row.get("id")
        if doc_id:
            return str(doc_id)
        doc_text = row.get("text")
        if doc_text:
            return f"text::{doc_text}"
        return f"idx::{index}"

    @classmethod
    def _normalize_vector_scores(cls, rows: list[dict]) -> dict[str, float]:
        distances: list[float] = []
        keys: list[str] = []
        for index, row in enumerate(rows):
            distance = cls._as_float(row.get("_distance"))
            if distance is None:
                continue
            keys.append(cls._row_key(row, index=index))
            distances.append(distance)
        return cls._normalize_low_is_better(keys, distances)

    @classmethod
    def _normalize_fts_scores(cls, rows: list[dict]) -> dict[str, float]:
        scores: list[float] = []
        keys: list[str] = []
        for index, row in enumerate(rows):
            score = cls._as_float(row.get("score"))
            if score is None:
                continue
            keys.append(cls._row_key(row, index=index))
            scores.append(score)
        return cls._normalize_high_is_better(keys, scores)

    @staticmethod
    def _as_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_high_is_better(keys: list[str], values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        min_val = min(values)
        max_val = max(values)
        if np.isclose(min_val, max_val):
            return {key: 1.0 for key in keys}
        scale = max_val - min_val
        return {key: (value - min_val) / scale for key, value in zip(keys, values, strict=True)}

    @staticmethod
    def _normalize_low_is_better(keys: list[str], values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        min_val = min(values)
        max_val = max(values)
        if np.isclose(min_val, max_val):
            return {key: 1.0 for key in keys}
        scale = max_val - min_val
        return {key: (max_val - value) / scale for key, value in zip(keys, values, strict=True)}


class LanceDBStore:
    """Thin wrapper around LanceDB for vector and hybrid search."""

    def __init__(self, db_path: str | Path) -> None:
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

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Run pure BM25 (full-text) retrieval against the FTS index.

        Used by the gate-2 validator baseline — a deliberately weak retrieval
        flavour whose successes mark a question as too easy to discriminate
        between RAG configurations.
        """
        table = self._require_table()
        return table.search(query, query_type="fts").limit(top_k).to_list()

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 5,
        hybrid_alpha: float = 0.5,
    ) -> list[dict]:
        """Run hybrid BM25 + vector retrieval."""
        vector = self._normalize_vector(query_embedding)
        reranker = HybridAlphaReranker(alpha=hybrid_alpha, return_score="all")
        return self._build_hybrid_query(query, vector).rerank(reranker=reranker).limit(top_k).to_list()

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

    def _build_hybrid_query(self, query: str, vector: list[float]):
        table = self._require_table()
        return table.search(query_type="hybrid").vector(vector).text(query)
