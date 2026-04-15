"""Tests for the LanceDB vector store wrapper."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from agentic_autorag.engine.vector_store import HybridAlphaReranker, LanceDBStore


def _make_records() -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((4, 32)).astype(np.float32)
    records = [
        {"id": "doc_0", "text": "banana retrieval signal", "vector": embeddings[0].tolist()},
        {"id": "doc_1", "text": "apple context chunk", "vector": embeddings[1].tolist()},
        {"id": "doc_2", "text": "orange answer evidence", "vector": embeddings[2].tolist()},
        {"id": "doc_3", "text": "grape fallback passage", "vector": embeddings[3].tolist()},
    ]
    return records, embeddings


class TestLanceDBStore:
    def test_search_vector_sorted_by_relevance(self, tmp_path: Path) -> None:
        store = LanceDBStore(db_path=tmp_path / "lancedb")
        records, embeddings = _make_records()
        store.create_index(records, table_name="docs")

        results = store.search_vector(embeddings[0], top_k=3)

        assert len(results) == 3
        assert results[0]["id"] == "doc_0"
        distances = [r.get("_distance") for r in results]
        if all(d is not None for d in distances):
            assert distances == sorted(distances)

    def test_search_hybrid_returns_results(self, tmp_path: Path) -> None:
        store = LanceDBStore(db_path=tmp_path / "lancedb")
        records, embeddings = _make_records()
        store.create_index(records, table_name="docs")

        results = store.search_hybrid("banana retrieval", embeddings[0], top_k=3)

        assert len(results) > 0
        assert any(result["id"] == "doc_0" for result in results)

    def test_create_index_overwrite_replaces_existing_data(self, tmp_path: Path) -> None:
        store = LanceDBStore(db_path=tmp_path / "lancedb")
        records, _ = _make_records()
        store.create_index(records, table_name="docs", mode="overwrite")

        replacement_vector = np.ones(32, dtype=np.float32)
        replacement_records = [{"id": "new_doc", "text": "brand new corpus", "vector": replacement_vector.tolist()}]
        store.create_index(replacement_records, table_name="docs", mode="overwrite")

        results = store.search_vector(replacement_vector, top_k=5)
        ids = {row["id"] for row in results}

        assert "new_doc" in ids
        assert "doc_0" not in ids
        assert len(ids) == 1


class TestHybridAlphaReranker:
    def _make_tables(self) -> tuple[pa.Table, pa.Table]:
        vector_results = pa.Table.from_pylist(
            [
                {"id": "doc_vector", "text": "vector-favored", "_distance": 0.1},
                {"id": "doc_fts", "text": "fts-favored", "_distance": 0.8},
                {"id": "doc_vector_only", "text": "vector-only", "_distance": 0.3},
            ]
        )
        fts_results = pa.Table.from_pylist(
            [
                {"id": "doc_vector", "text": "vector-favored", "score": 0.2},
                {"id": "doc_fts", "text": "fts-favored", "score": 1.2},
                {"id": "doc_fts_only", "text": "fts-only", "score": 0.9},
            ]
        )
        return vector_results, fts_results

    @staticmethod
    def _ordered_ids(table: pa.Table) -> list[str]:
        return [str(row["id"]) for row in table.to_pylist()]

    def test_alpha_1_prefers_vector_scores(self) -> None:
        vector_results, fts_results = self._make_tables()
        reranker = HybridAlphaReranker(alpha=1.0)

        ranked = reranker.rerank_hybrid("query", vector_results, fts_results)
        ids = self._ordered_ids(ranked)

        assert ids[0] == "doc_vector"

    def test_alpha_0_prefers_fts_scores(self) -> None:
        vector_results, fts_results = self._make_tables()
        reranker = HybridAlphaReranker(alpha=0.0)

        ranked = reranker.rerank_hybrid("query", vector_results, fts_results)
        ids = self._ordered_ids(ranked)

        assert ids[0] == "doc_fts"

    def test_alpha_half_blends_and_keeps_unique_docs(self) -> None:
        vector_results, fts_results = self._make_tables()
        reranker = HybridAlphaReranker(alpha=0.5)

        ranked = reranker.rerank_hybrid("query", vector_results, fts_results)
        rows = ranked.to_pylist()
        ids = {str(row["id"]) for row in rows}

        assert ids == {"doc_vector", "doc_fts", "doc_vector_only", "doc_fts_only"}
        assert all("_relevance_score" in row for row in rows)

    def test_handles_missing_scores_without_crashing(self) -> None:
        vector_results = pa.Table.from_pylist(
            [
                {"id": "doc_a", "text": "a", "_distance": 0.2},
                {"id": "doc_b", "text": "b"},
            ]
        )
        fts_results = pa.Table.from_pylist(
            [
                {"id": "doc_a", "text": "a"},
                {"id": "doc_c", "text": "c", "score": 0.5},
            ]
        )
        reranker = HybridAlphaReranker(alpha=0.4)

        ranked = reranker.rerank_hybrid("query", vector_results, fts_results)
        ids = self._ordered_ids(ranked)

        assert set(ids) == {"doc_a", "doc_b", "doc_c"}
