"""Tests for embedding-similarity chunk pairing."""

from __future__ import annotations

import logging

import numpy as np

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.embedding_pair_index import emit_embedding_pairs


def _chunks(*specs: tuple[str, str, str]) -> list[ChunkRecord]:
    """Build ChunkRecords from (chunk_id, doc_id, text) tuples."""
    return [ChunkRecord(chunk_id=cid, doc_id=did, text=t) for cid, did, t in specs]


def _stub_embedder_from_vectors(mapping: dict[str, list[float]]):
    """Returns L2-normalised vectors keyed by chunk text (or substring match).

    Tests can pin precise relationships between chunks by giving them text
    that's a key into the mapping.
    """

    def encode(texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            for key, vec in mapping.items():
                if key in t:
                    vectors.append(vec)
                    break
            else:
                raise AssertionError(f"no stub vector for text: {t!r}")
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.where(norms > 0, norms, 1.0)
        return arr

    return encode


class TestEmitEmbeddingPairs:
    def test_cross_doc_only(self) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "marker_A0 ..."),
            ("docA::c1", "docA", "marker_A1 ..."),
            ("docB::c0", "docB", "marker_B0 ..."),
        )
        embed = _stub_embedder_from_vectors(
            {
                "marker_A0": [1.0, 0.0],
                "marker_A1": [1.0, 0.0],
                "marker_B0": [1.0, 0.0],
            }
        )
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=2,
            target_count=10,
            eligible_sections=None,
        )
        for seed in seeds:
            assert seed.chunk_a.doc_id != seed.chunk_b.doc_id

    def test_pair_dedup_canonical_order(self) -> None:
        # Two docs, two chunks each. Top-K cross-doc pairs from both sides
        # should collapse to unique pairs (no (a,b) and (b,a) both kept).
        chunks = _chunks(
            ("docA::c0", "docA", "marker_A "),
            ("docB::c0", "docB", "marker_B "),
        )
        embed = _stub_embedder_from_vectors({"marker_A": [1.0, 0.0], "marker_B": [1.0, 0.0]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=1,
            target_count=10,
            eligible_sections=None,
        )
        assert len(seeds) == 1

    def test_top_k_per_chunk_respected(self) -> None:
        # 4 docs, all close in embedding space. With top_k=1 each chunk
        # picks its single best neighbour; selection round-robin proceeds
        # one rank at a time. We expect target_count distinct pairs up to
        # the per-rank ceiling.
        chunks = _chunks(
            ("docA::c0", "docA", "topic "),
            ("docB::c0", "docB", "topic "),
            ("docC::c0", "docC", "topic "),
            ("docD::c0", "docD", "topic "),
        )
        embed = _stub_embedder_from_vectors({"topic": [1.0, 0.0]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=1,
            target_count=10,
            eligible_sections=None,
        )
        # With K=1 round-robin only takes rank-1 from each chunk; 4 chunks
        # give 4 rank-1 picks but each pair is bidirectional (A→B and B→A
        # collapse to one), so worst case 2 unique pairs, best case 4.
        assert 2 <= len(seeds) <= 4
        for seed in seeds:
            assert seed.chunk_a.doc_id != seed.chunk_b.doc_id

    def test_section_filter_excludes_ineligible(self) -> None:
        # docA chunk is in references; should not appear in any seed.
        chunks = [
            ChunkRecord(chunk_id="docA::c0", doc_id="docA", text="t1", section=SectionLabel.REFERENCES),
            ChunkRecord(chunk_id="docB::c0", doc_id="docB", text="t1", section=SectionLabel.BODY),
            ChunkRecord(chunk_id="docC::c0", doc_id="docC", text="t1", section=SectionLabel.BODY),
        ]
        embed = _stub_embedder_from_vectors({"t1": [1.0, 0.0]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=2,
            target_count=10,
            eligible_sections=frozenset({SectionLabel.BODY}),
        )
        for seed in seeds:
            assert seed.chunk_a.doc_id != "docA"
            assert seed.chunk_b.doc_id != "docA"

    def test_score_is_cosine(self) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "marker_A "),
            ("docB::c0", "docB", "marker_B "),
        )
        # Vectors at 60 degrees → cosine = 0.5.
        embed = _stub_embedder_from_vectors({"marker_A": [1.0, 0.0], "marker_B": [0.5, np.sqrt(3) / 2]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=1,
            target_count=10,
            eligible_sections=None,
        )
        assert len(seeds) == 1
        assert abs(seeds[0].score - 0.5) < 1e-5

    def test_target_count_truncation(self) -> None:
        # Build 6 chunks across 6 docs all topic-tied; round-robin will
        # produce up to 6 unique pairs at K=2 but we cap to target=2.
        chunks = _chunks(*[(f"d{i}::c0", f"d{i}", "marker ") for i in range(6)])
        embed = _stub_embedder_from_vectors({"marker": [1.0, 0.0]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=2,
            target_count=2,
            eligible_sections=None,
        )
        assert len(seeds) == 2

    def test_empty_input(self) -> None:
        embed = _stub_embedder_from_vectors({})
        seeds = emit_embedding_pairs(
            [],
            embed,
            top_k_per_chunk=2,
            target_count=10,
            eligible_sections=None,
        )
        assert seeds == []

    def test_single_doc_yields_no_pairs(self) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "marker "),
            ("docA::c1", "docA", "marker "),
        )
        embed = _stub_embedder_from_vectors({"marker": [1.0, 0.0]})
        seeds = emit_embedding_pairs(
            chunks,
            embed,
            top_k_per_chunk=1,
            target_count=10,
            eligible_sections=None,
        )
        assert seeds == []

    def test_cosine_histogram_logged(self, caplog) -> None:
        chunks = _chunks(
            ("docA::c0", "docA", "marker_A "),
            ("docB::c0", "docB", "marker_B "),
        )
        embed = _stub_embedder_from_vectors({"marker_A": [1.0, 0.0], "marker_B": [0.0, 1.0]})
        with caplog.at_level(logging.INFO, logger="agentic_autorag.examiner.embedding_pair_index"):
            emit_embedding_pairs(
                chunks,
                embed,
                top_k_per_chunk=1,
                target_count=10,
                eligible_sections=None,
            )
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Pair cosine distribution" in joined
