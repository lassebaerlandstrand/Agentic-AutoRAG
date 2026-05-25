"""Tests for the neighborhood builder."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.neighborhoods import build_neighborhood, build_tfidf_matrix


def _chunk(chunk_id: str, doc_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text)


def _tfidf(rows: list[list[float]]) -> csr_matrix:
    """L2-normalised sparse matrix from explicit rows.

    Stands in for the TF-IDF output in unit tests so we can express
    similarity structure directly without depending on the vocabulary
    sklearn would learn from real text.
    """
    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms > 0, norms, 1.0)
    return csr_matrix(arr)


class TestBuildNeighborhood:
    def test_anchor_at_position_zero(self) -> None:
        chunks = [_chunk(f"d::{i}", "d", "x") for i in range(20)]
        tfidf = _tfidf([[float(i), 1.0] for i in range(20)])
        nh = build_neighborhood(5, chunks, tfidf, min_chunks=3, min_words=99999)
        assert nh.chunks[0].chunk_id == "d::5"
        assert nh.anchor.chunk_id == "d::5"

    def test_small_chunk_corpus_hits_chunk_floor(self) -> None:
        """100-word chunks: 12-chunk floor binds before 5000-word floor."""
        chunks = [_chunk(f"doc::{i}", f"doc_{i % 4}", "word " * 100) for i in range(50)]
        tfidf = _tfidf([[float(i), 1.0] for i in range(50)])
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
        assert len(nh.chunks) == 12

    def test_large_chunk_corpus_hits_word_floor(self) -> None:
        """1000-word chunks: 5000-word floor binds before 12-chunk floor."""
        chunks = [_chunk(f"doc::{i}", f"doc_{i % 4}", "word " * 1000) for i in range(50)]
        tfidf = _tfidf([[float(i), 1.0] for i in range(50)])
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
        total_words = sum(len(c.text.split()) for c in nh.chunks)
        assert len(nh.chunks) < 12
        assert total_words >= 5000
        # 1000 words/chunk × ~5 chunks = ~5000 words.
        assert len(nh.chunks) <= 6

    def test_same_doc_fraction_low_biases_cross_doc(self) -> None:
        """Low same_doc_fraction draws more cross-doc cosine neighbors."""
        chunks: list[ChunkRecord] = []
        for i in range(20):
            chunks.append(_chunk(f"doc_0::c{i}", "doc_0", "word " * 50))
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=10, min_words=99999, same_doc_fraction=0.1)
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        n_cross = sum(1 for c in nh.chunks[1:] if c.doc_id != "doc_0")
        # With same_doc_fraction=0.1 and 9 non-anchor slots, we expect ~1
        # same-doc and ~8 cross-doc. Tolerate ±1.
        assert n_cross >= 6
        assert n_same <= 3

    def test_same_doc_fraction_high_biases_same_doc(self) -> None:
        chunks: list[ChunkRecord] = []
        for i in range(20):
            chunks.append(_chunk(f"doc_0::c{i}", "doc_0", "word " * 50))
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=10, min_words=99999, same_doc_fraction=0.8)
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        assert n_same >= 6

    def test_same_doc_pool_exhausted_falls_back_to_cross_doc(self) -> None:
        """When same-doc pool runs out, the remainder fills from cross-doc."""
        chunks: list[ChunkRecord] = [
            _chunk("doc_0::c0", "doc_0", "word " * 50),
            _chunk("doc_0::c1", "doc_0", "word " * 50),
        ]
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=8, min_words=99999, same_doc_fraction=0.9)
        assert len(nh.chunks) == 8
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        n_cross = sum(1 for c in nh.chunks[1:] if c.doc_id != "doc_0")
        assert n_same == 1
        assert n_cross == 6

    def test_cosine_order_picks_most_similar_cross_doc(self) -> None:
        """Cross-doc pool is sorted by cosine to the anchor (descending)."""
        anchor = _chunk("a::0", "a", "anchor text")
        chunks = [
            anchor,
            _chunk("b::0", "b", "text 1"),
            _chunk("c::0", "c", "text 2"),
            _chunk("d::0", "d", "text 3"),
            _chunk("e::0", "e", "text 4"),
            _chunk("f::0", "f", "text 5"),
        ]
        tfidf = _tfidf(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=4, min_words=99999, same_doc_fraction=0.0)
        picked = [c.chunk_id for c in nh.chunks]
        # Anchor first, then most-similar cross-doc chunks.
        assert picked[0] == "a::0"
        assert picked[1] == "b::0"
        assert picked[2] == "c::0"
        assert picked[3] == "d::0"

    def test_anchor_only_when_corpus_has_no_other_chunks(self) -> None:
        chunks = [_chunk("a::0", "a", "anchor only")]
        tfidf = _tfidf([[1.0, 0.0]])
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
        assert nh.chunks == [chunks[0]]

    def test_rejects_misaligned_tfidf(self) -> None:
        chunks = [_chunk("a::0", "a", "x"), _chunk("b::0", "b", "y")]
        tfidf = _tfidf([[1.0, 0.0]])
        with pytest.raises(ValueError, match="tfidf"):
            build_neighborhood(0, chunks, tfidf, min_chunks=2, min_words=99999)

    def test_rejects_invalid_anchor_idx(self) -> None:
        chunks = [_chunk("a::0", "a", "x")]
        tfidf = _tfidf([[1.0, 0.0]])
        with pytest.raises(IndexError):
            build_neighborhood(5, chunks, tfidf, min_chunks=1, min_words=99999)

    def test_rejects_invalid_same_doc_fraction(self) -> None:
        chunks = [_chunk("a::0", "a", "x")]
        tfidf = _tfidf([[1.0, 0.0]])
        with pytest.raises(ValueError, match="same_doc_fraction"):
            build_neighborhood(0, chunks, tfidf, min_chunks=1, min_words=99999, same_doc_fraction=1.5)


class TestBuildTfidfMatrix:
    def test_shared_rare_term_outranks_unrelated_chunks(self) -> None:
        """Doc A and B share a rare term; other docs share nothing distinctive.

        With TF-IDF cosine ranking, doc B's chunk lands first in doc A's
        cross-doc neighborhood — the bridge signal we want for multi-hop
        construction.
        """
        chunks = [
            _chunk(
                "doc_a::c0",
                "doc_a",
                # Anchor: rare bridge term `brookhaven`.
                "the accelerator facility brookhaven hosts physics research projects",
            ),
            _chunk(
                "doc_b::c0",
                "doc_b",
                # Bridges via shared rare term `brookhaven`, different topic.
                "naval logistics during wartime made brookhaven a critical depot",
            ),
            _chunk("doc_c::c0", "doc_c", "paris cafes serve coffee in elegant streetside settings"),
            _chunk("doc_d::c0", "doc_d", "vienna cafes rival those of paris in coffee culture"),
            _chunk("doc_e::c0", "doc_e", "the garden contained roses and the soil rich with nutrients"),
            _chunk("doc_f::c0", "doc_f", "garden roses bloomed last summer in warm weather"),
        ]
        tfidf, _ = build_tfidf_matrix(chunks)
        nh = build_neighborhood(0, chunks, tfidf, min_chunks=3, min_words=99999, same_doc_fraction=0.0)
        picked = [c.chunk_id for c in nh.chunks]
        # Anchor first; doc_b — the only cross-doc chunk sharing a rare
        # term with the anchor — is the top cross-doc neighbor.
        assert picked[0] == "doc_a::c0"
        assert picked[1] == "doc_b::c0"
