"""Tests for the anchor seeder."""

from __future__ import annotations

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.seeders import emit_anchor_seeds


def _chunk(chunk_id: str, doc_id: str, text: str, section: SectionLabel | None = None) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text, section=section)


class TestAnchorSeeds:
    def test_drops_short_chunks(self) -> None:
        chunks = [
            _chunk("a::0", "a", "short"),
            _chunk("a::1", "a", "x" * 300),
            _chunk("a::2", "a", "y" * 300),
        ]
        anchors = emit_anchor_seeds(chunks, target_count=10, min_text_chars=200, rng_seed=42)
        assert len(anchors) == 2
        anchor_ids = {a.chunk.chunk_id for a in anchors}
        assert anchor_ids == {"a::1", "a::2"}

    def test_truncates_to_target_count(self) -> None:
        chunks = [_chunk(f"d::{i}", "d", "x " * 200) for i in range(10)]
        anchors = emit_anchor_seeds(chunks, target_count=3, rng_seed=0)
        assert len(anchors) == 3

    def test_zero_target_returns_empty(self) -> None:
        chunks = [_chunk("a::0", "a", "x" * 600)]
        assert emit_anchor_seeds(chunks, target_count=0) == []

    def test_empty_input_returns_empty(self) -> None:
        assert emit_anchor_seeds([], target_count=5) == []

    def test_deterministic_with_seed(self) -> None:
        chunks = [_chunk(f"d::{i}", "d", "x " * (50 + i * 10)) for i in range(20)]
        a1 = emit_anchor_seeds(chunks, target_count=5, rng_seed=123)
        a2 = emit_anchor_seeds(chunks, target_count=5, rng_seed=123)
        a3 = emit_anchor_seeds(chunks, target_count=5, rng_seed=456)
        ids1 = [a.chunk.chunk_id for a in a1]
        ids2 = [a.chunk.chunk_id for a in a2]
        ids3 = [a.chunk.chunk_id for a in a3]
        assert ids1 == ids2
        # Different seed should produce a different sample (probabilistically;
        # with 5 of 20 picked and length-weighted sampling, near-certain).
        assert ids1 != ids3

    def test_weights_favor_longer_chunks(self) -> None:
        """Length-weighted sampling: across many runs, long chunks dominate."""
        chunks = [_chunk(f"d::{i}", "d", "x " * 50) for i in range(5)] + [
            _chunk("long::0", "long", "y " * 5000),
        ]
        # 100 trials with target=1; the much longer chunk should win the
        # vast majority. Tolerate variance.
        wins = 0
        for seed in range(100):
            anchors = emit_anchor_seeds(chunks, target_count=1, rng_seed=seed)
            if anchors and anchors[0].chunk.chunk_id == "long::0":
                wins += 1
        assert wins >= 80, f"length-weighted sampling failed: long chunk won only {wins}/100"
