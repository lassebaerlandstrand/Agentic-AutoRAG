"""Tests for the seed generators (single-chunk, same-doc, cross-doc).

Embedding-using seeders now take a pre-computed (n, d) matrix aligned to the
chunk list; tests build the matrix from a per-text vector map.
"""

from __future__ import annotations

import numpy as np

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.seeders import (
    emit_cross_doc_pair_seeds,
    emit_same_doc_pair_seeds,
    emit_single_chunk_seeds,
)


def _chunk(chunk_id: str, doc_id: str, text: str, section: SectionLabel | None = None) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text, section=section)


def _embed(chunks: list[ChunkRecord], mapping: dict[str, list[float]]) -> np.ndarray:
    """L2-normalised matrix matching chunks by exact text key."""
    arr = np.asarray([mapping[c.text] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.where(norms > 0, norms, 1.0)
    return arr


class TestSingleChunkSeeds:
    def test_keeps_long_chunks(self) -> None:
        chunks = [
            _chunk("a::0", "a", "short", SectionLabel.BODY),
            _chunk("a::1", "a", "x" * 300, SectionLabel.BODY),
            _chunk("a::2", "a", "y" * 300, SectionLabel.BODY),
        ]
        seeds = emit_single_chunk_seeds(chunks, target_count=10, min_text_chars=200)
        assert len(seeds) == 2
        assert all(s.chunk_b is None for s in seeds)
        assert all(s.origin == "single_chunk" for s in seeds)

    def test_truncates_to_target(self) -> None:
        chunks = [_chunk(f"d::{i}", "d", "x" * 600, SectionLabel.BODY) for i in range(10)]
        seeds = emit_single_chunk_seeds(chunks, target_count=3)
        assert len(seeds) == 3

    def test_empty_target_returns_empty(self) -> None:
        chunks = [_chunk("a::0", "a", "x" * 600)]
        assert emit_single_chunk_seeds(chunks, target_count=0) == []


class TestSameDocPairSeeds:
    def test_pairs_within_doc_in_band(self) -> None:
        chunks = [
            _chunk("a::0", "a", "intro text", SectionLabel.ABSTRACT),
            _chunk("a::1", "a", "results text", SectionLabel.RESULTS),
            _chunk("b::0", "b", "unrelated text", SectionLabel.BODY),
        ]
        embeddings = _embed(
            chunks,
            {
                "intro text": [1.0, 0.6, 0.0],
                "results text": [0.6, 1.0, 0.0],
                "unrelated text": [0.0, 0.0, 1.0],
            },
        )
        seeds = emit_same_doc_pair_seeds(chunks, embeddings, target_count=5, cos_min=0.4, cos_max=0.95)
        assert len(seeds) == 1
        assert seeds[0].origin == "same_doc_pair"
        assert seeds[0].chunk_a.doc_id == seeds[0].chunk_b.doc_id == "a"
        assert seeds[0].chunk_a.section != seeds[0].chunk_b.section

    def test_drops_same_section_pairs(self) -> None:
        chunks = [
            _chunk("a::0", "a", "a body 1", SectionLabel.BODY),
            _chunk("a::1", "a", "a body 2", SectionLabel.BODY),
            _chunk("a::2", "a", "a methods", SectionLabel.METHODS),
        ]
        embeddings = _embed(chunks, {"a body 1": [1.0, 0.0], "a body 2": [1.0, 0.0], "a methods": [0.6, 0.6]})
        seeds = emit_same_doc_pair_seeds(chunks, embeddings, target_count=5, cos_min=0.0, cos_max=1.0)
        assert all(s.chunk_a.section != s.chunk_b.section for s in seeds)

    def test_skips_section_filter_on_single_section_corpus(self) -> None:
        """Wikipedia / single-section corpora used to silently emit 0 same-doc
        seeds because every chunk classifies as ``body``. With the fix, the
        filter is skipped and pairs are kept based on cosine band only."""
        chunks = [
            _chunk("a::0", "a", "intro p1", SectionLabel.BODY),
            _chunk("a::1", "a", "intro p2", SectionLabel.BODY),
        ]
        embeddings = _embed(chunks, {"intro p1": [1.0, 0.5], "intro p2": [0.5, 1.0]})
        seeds = emit_same_doc_pair_seeds(chunks, embeddings, target_count=5, cos_min=0.4, cos_max=0.95)
        assert len(seeds) == 1
        assert seeds[0].chunk_a.doc_id == seeds[0].chunk_b.doc_id == "a"

    def test_drops_pairs_outside_band(self) -> None:
        chunks = [
            _chunk("a::0", "a", "p1", SectionLabel.ABSTRACT),
            _chunk("a::1", "a", "p2", SectionLabel.RESULTS),
        ]
        embeddings = _embed(chunks, {"p1": [1.0, 0.0], "p2": [1.0, 0.0]})
        seeds = emit_same_doc_pair_seeds(chunks, embeddings, target_count=5, cos_min=0.4, cos_max=0.85)
        assert seeds == []

    def test_empty_when_no_pairs_in_doc(self) -> None:
        chunks = [_chunk("a::0", "a", "x", SectionLabel.BODY), _chunk("b::0", "b", "y", SectionLabel.BODY)]
        embeddings = _embed(chunks, {"x": [1.0, 0.0], "y": [0.0, 1.0]})
        seeds = emit_same_doc_pair_seeds(chunks, embeddings, target_count=5, cos_min=0.0, cos_max=1.0)
        assert seeds == []


class TestCrossDocPairSeeds:
    def test_origin_tag_set(self) -> None:
        chunks = [
            _chunk("a::0", "a", "topic_one chunk", SectionLabel.BODY),
            _chunk("b::0", "b", "topic_one chunk", SectionLabel.BODY),
        ]
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        seeds = emit_cross_doc_pair_seeds(chunks, embeddings, top_k_per_chunk=1, target_count=5)
        assert len(seeds) == 1
        assert seeds[0].origin == "cross_doc_pair"

    def test_fusion_finds_pair_only_idf_ranks(self) -> None:
        """Embedding gives all pairs ~equal cosine; IDF discriminates by shared rare tokens."""
        chunks = [
            _chunk("a::0", "a", "phoenix protocol distributed systems details", SectionLabel.BODY),
            _chunk("b::0", "b", "phoenix protocol followup section content", SectionLabel.BODY),
            _chunk("c::0", "c", "generic introduction content about other topics", SectionLabel.BODY),
        ]
        # All chunks nearly tied in embedding space — fusion must rely on IDF.
        embeddings = np.array([[1.0, 0.01 * i] for i in range(len(chunks))], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        seeds = emit_cross_doc_pair_seeds(chunks, embeddings, top_k_per_chunk=2, target_count=3)
        assert len(seeds) >= 1
        top_ids = {seeds[0].chunk_a.chunk_id, seeds[0].chunk_b.chunk_id}
        assert top_ids == {"a::0", "b::0"}

    def test_fusion_score_is_rrf(self) -> None:
        """Fused score is RRF-shaped (< 1.0 / (k+1) * 2), not raw cosine."""
        chunks = [
            _chunk("a::0", "a", "phoenix protocol distributed", SectionLabel.BODY),
            _chunk("b::0", "b", "phoenix protocol followup", SectionLabel.BODY),
        ]
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        seeds = emit_cross_doc_pair_seeds(chunks, embeddings, top_k_per_chunk=1, target_count=5)
        assert len(seeds) == 1
        # RRF for a pair at rank-1 in both rankings: 2 * 1/(60+1) ≈ 0.0328.
        # Empty IDF case (no shared content) → only the embedding term: 1/61 ≈ 0.0164.
        # Either way the fused score is below 0.1 and strictly positive.
        assert 0.0 < seeds[0].score < 0.1
