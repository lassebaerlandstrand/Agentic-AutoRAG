"""Tests for embedding-similarity chunk pairing.

The seeder now takes a pre-computed (n, d) embedding matrix; tests build the
matrix from a per-substring vector map then pass it directly.
"""

from __future__ import annotations

import logging

import numpy as np

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.embedding_pair_index import emit_embedding_pairs


def _chunks(*specs: tuple[str, str, str]) -> list[ChunkRecord]:
    return [ChunkRecord(chunk_id=cid, doc_id=did, text=t) for cid, did, t in specs]


def _embed(chunks: list[ChunkRecord], mapping: dict[str, list[float]]) -> np.ndarray:
    """L2-normalised matrix of vectors looked up by substring of chunk text."""
    vectors = []
    for c in chunks:
        for key, vec in mapping.items():
            if key in c.text:
                vectors.append(vec)
                break
        else:
            raise AssertionError(f"no stub vector for chunk text: {c.text!r}")
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.where(norms > 0, norms, 1.0)
    return arr


class TestEmitEmbeddingPairs:
    def test_cross_doc_only(self) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "marker_A0 ..."),
            ("docA::c1", "docA", "marker_A1 ..."),
            ("docB::c0", "docB", "marker_B0 ..."),
        )
        embeddings = _embed(chunks, {"marker_A0": [1.0, 0.0], "marker_A1": [1.0, 0.0], "marker_B0": [1.0, 0.0]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=2, target_count=10)
        for seed in seeds:
            assert seed.chunk_a.doc_id != seed.chunk_b.doc_id

    def test_pair_dedup_canonical_order(self) -> None:
        chunks = _chunks(("docA::c0", "docA", "marker_A "), ("docB::c0", "docB", "marker_B "))
        embeddings = _embed(chunks, {"marker_A": [1.0, 0.0], "marker_B": [1.0, 0.0]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=1, target_count=10)
        assert len(seeds) == 1

    def test_top_k_per_chunk_respected(self) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "topic "),
            ("docB::c0", "docB", "topic "),
            ("docC::c0", "docC", "topic "),
            ("docD::c0", "docD", "topic "),
        )
        embeddings = _embed(chunks, {"topic": [1.0, 0.0]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=1, target_count=10)
        assert 2 <= len(seeds) <= 4
        for seed in seeds:
            assert seed.chunk_a.doc_id != seed.chunk_b.doc_id

    def test_score_is_cosine(self) -> None:
        chunks = _chunks(("docA::c0", "docA", "marker_A "), ("docB::c0", "docB", "marker_B "))
        embeddings = _embed(chunks, {"marker_A": [1.0, 0.0], "marker_B": [0.5, np.sqrt(3) / 2]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=1, target_count=10)
        assert len(seeds) == 1
        assert abs(seeds[0].score - 0.5) < 1e-5

    def test_target_count_truncation(self) -> None:
        chunks = _chunks(*[(f"d{i}::c0", f"d{i}", "marker ") for i in range(6)])
        embeddings = _embed(chunks, {"marker": [1.0, 0.0]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=2, target_count=2)
        assert len(seeds) == 2

    def test_empty_input(self) -> None:
        seeds = emit_embedding_pairs([], np.zeros((0, 2), dtype=np.float32), top_k_per_chunk=2, target_count=10)
        assert seeds == []

    def test_single_doc_yields_no_pairs(self) -> None:
        chunks = _chunks(("docA::c0", "docA", "marker "), ("docA::c1", "docA", "marker "))
        embeddings = _embed(chunks, {"marker": [1.0, 0.0]})
        seeds = emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=1, target_count=10)
        assert seeds == []

    def test_cosine_histogram_logged(self, caplog) -> None:
        chunks = _chunks(("docA::c0", "docA", "marker_A "), ("docB::c0", "docB", "marker_B "))
        embeddings = _embed(chunks, {"marker_A": [1.0, 0.0], "marker_B": [0.0, 1.0]})
        with caplog.at_level(logging.INFO, logger="agentic_autorag.examiner.embedding_pair_index"):
            emit_embedding_pairs(chunks, embeddings, top_k_per_chunk=1, target_count=10)
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Pair cosine distribution" in joined

    def test_misaligned_embeddings_raises(self) -> None:
        chunks = _chunks(("docA::c0", "docA", "x "), ("docB::c0", "docB", "y "))
        bad_embeddings = np.zeros((3, 2), dtype=np.float32)
        try:
            emit_embedding_pairs(chunks, bad_embeddings, top_k_per_chunk=1, target_count=10)
        except ValueError as e:
            assert "align" in str(e)
        else:
            raise AssertionError("expected ValueError")
