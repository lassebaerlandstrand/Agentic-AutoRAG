"""Tests for the seed generators (single-chunk, same-doc, cross-doc)."""

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


class TestSingleChunkSeeds:
    def test_keeps_eligible_long_chunks(self) -> None:
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
        chunks = [_chunk(f"d::{i}", "d", "x" * 300, SectionLabel.BODY) for i in range(10)]
        seeds = emit_single_chunk_seeds(chunks, target_count=3)
        assert len(seeds) == 3

    def test_eligible_section_filter(self) -> None:
        chunks = [
            _chunk("a::0", "a", "x" * 300, SectionLabel.BODY),
            _chunk("a::1", "a", "y" * 300, SectionLabel.REFERENCES),
        ]
        seeds = emit_single_chunk_seeds(
            chunks,
            target_count=10,
            eligible_sections=[SectionLabel.BODY],
        )
        assert len(seeds) == 1
        assert seeds[0].chunk_a.chunk_id == "a::0"

    def test_empty_target_returns_empty(self) -> None:
        chunks = [_chunk("a::0", "a", "x" * 300)]
        assert emit_single_chunk_seeds(chunks, target_count=0) == []


class TestSameDocPairSeeds:
    def _stub_embedder(self, mapping: dict[str, list[float]]):
        def _encode(texts: list[str]) -> np.ndarray:
            arr = np.asarray([mapping[t] for t in texts], dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

        return _encode

    def test_pairs_within_doc_in_band(self) -> None:
        # Two chunks in doc_a are similar but section-disjoint; one in doc_b.
        # Stub embeddings put doc_a chunks at cos ≈ 0.6, both far from doc_b.
        chunks = [
            _chunk("a::0", "a", "intro text", SectionLabel.ABSTRACT),
            _chunk("a::1", "a", "results text", SectionLabel.RESULTS),
            _chunk("b::0", "b", "unrelated text", SectionLabel.BODY),
        ]
        embed = self._stub_embedder(
            {
                "intro text": [1.0, 0.6, 0.0],
                "results text": [0.6, 1.0, 0.0],
                "unrelated text": [0.0, 0.0, 1.0],
            }
        )

        # Build same-doc-only pairing: cosine ≈ 0.86 between a::0/a::1.
        # Use band [0.4, 0.95] to capture it.
        seeds = emit_same_doc_pair_seeds(
            chunks,
            embed,
            target_count=5,
            cos_min=0.4,
            cos_max=0.95,
        )
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
        embed = self._stub_embedder(
            {
                "a body 1": [1.0, 0.0],
                "a body 2": [1.0, 0.0],
                "a methods": [0.6, 0.6],
            }
        )
        # Two distinct sections present → section-disjoint filter active →
        # body+body pair is dropped, body+methods pair survives.
        seeds = emit_same_doc_pair_seeds(chunks, embed, target_count=5, cos_min=0.0, cos_max=1.0)
        assert all(s.chunk_a.section != s.chunk_b.section for s in seeds)

    def test_skips_section_filter_on_single_section_corpus(self) -> None:
        """Wikipedia / single-section corpora used to silently emit 0 same-doc
        seeds because every chunk classifies as ``body``. With the fix, the
        filter is skipped and pairs are kept based on cosine band only."""
        chunks = [
            _chunk("a::0", "a", "intro p1", SectionLabel.BODY),
            _chunk("a::1", "a", "intro p2", SectionLabel.BODY),
        ]
        embed = self._stub_embedder(
            {
                "intro p1": [1.0, 0.5],
                "intro p2": [0.5, 1.0],
            }
        )
        seeds = emit_same_doc_pair_seeds(chunks, embed, target_count=5, cos_min=0.4, cos_max=0.95)
        assert len(seeds) == 1
        assert seeds[0].chunk_a.doc_id == seeds[0].chunk_b.doc_id == "a"

    def test_drops_pairs_outside_band(self) -> None:
        chunks = [
            _chunk("a::0", "a", "p1", SectionLabel.ABSTRACT),
            _chunk("a::1", "a", "p2", SectionLabel.RESULTS),
        ]
        # Cosine = 1.0; band excludes 1.0.
        embed = self._stub_embedder({"p1": [1.0, 0.0], "p2": [1.0, 0.0]})
        seeds = emit_same_doc_pair_seeds(chunks, embed, target_count=5, cos_min=0.4, cos_max=0.85)
        assert seeds == []

    def test_empty_when_no_pairs_in_doc(self) -> None:
        chunks = [
            _chunk("a::0", "a", "x", SectionLabel.BODY),
            _chunk("b::0", "b", "y", SectionLabel.BODY),
        ]
        embed = self._stub_embedder({"x": [1.0, 0.0], "y": [0.0, 1.0]})
        seeds = emit_same_doc_pair_seeds(chunks, embed, target_count=5, cos_min=0.0, cos_max=1.0)
        assert seeds == []


class TestCrossDocPairSeeds:
    def test_origin_tag_set(self) -> None:
        chunks = [
            _chunk("a::0", "a", "topic_one chunk", SectionLabel.BODY),
            _chunk("b::0", "b", "topic_one chunk", SectionLabel.BODY),
        ]

        def stub_embedder(texts: list[str]) -> np.ndarray:
            arr = np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

        seeds = emit_cross_doc_pair_seeds(chunks, stub_embedder, top_k_per_chunk=1, target_count=5)
        assert len(seeds) == 1
        assert seeds[0].origin == "cross_doc_pair"
