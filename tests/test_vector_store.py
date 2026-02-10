"""Tests for the LanceDB vector store wrapper."""

from pathlib import Path

import numpy as np

from agentic_autorag.engine.vector_store import LanceDBStore


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

    def test_snapshot_copies_table_directory(self, tmp_path: Path) -> None:
        store = LanceDBStore(db_path=tmp_path / "lancedb")
        records, _ = _make_records()
        store.create_index(records, table_name="docs")

        snapshot_path = tmp_path / "snapshot_docs"
        store.snapshot(snapshot_path)

        assert snapshot_path.exists()
        assert snapshot_path.is_dir()
        assert any(snapshot_path.iterdir())
