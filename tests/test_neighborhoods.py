"""Tests for the neighborhood builder."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.neighborhoods import (
    NeighborhoodDiagnostic,
    build_neighborhood,
    build_tfidf_matrix,
)


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
        nh, diag = build_neighborhood(5, chunks, tfidf, min_chunks=3, min_words=99999)
        assert nh.anchor.chunk_id == "d::5"
        assert nh.chunks[0].chunk_id == "d::5"
        assert diag.position_kinds[0] == "anchor"
        assert len(diag.position_kinds) == len(nh.chunks)

    def test_same_doc_picks_preserve_document_natural_order(self) -> None:
        """Sibling chunks are returned in chunker emission order, not lex chunk_id.

        Lex-sorted chunk ids ``chunk_0, chunk_1, chunk_10, chunk_11, ..., chunk_2``
        would put chunk_10 before chunk_2 — wrong. With document-natural order
        the first three same-doc picks after the anchor are chunk_1, chunk_2, chunk_3.
        """
        chunks = [_chunk(f"d::chunk_{i}", "d", "word " * 50) for i in range(13)]
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=4,
            min_words=99999,
            same_doc_weight=1.0,
            cross_doc_weight=0.0,
        )
        picked = [c.chunk_id for c in nh.chunks]
        assert picked == ["d::chunk_0", "d::chunk_1", "d::chunk_2", "d::chunk_3"]

    def test_target_size_hits_chunk_floor_on_small_chunks(self) -> None:
        """100-word chunks: 12-chunk floor binds before 5000-word floor."""
        chunks = [_chunk(f"doc::{i}", f"doc_{i % 4}", "word " * 100) for i in range(50)]
        tfidf = _tfidf([[float(i), 1.0] for i in range(50)])
        nh, _ = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
        assert len(nh.chunks) == 12

    def test_target_size_hits_word_floor_on_large_chunks(self) -> None:
        """1000-word chunks: 5000-word floor binds before 12-chunk floor."""
        chunks = [_chunk(f"doc::{i}", f"doc_{i % 4}", "word " * 1000) for i in range(50)]
        tfidf = _tfidf([[float(i), 1.0] for i in range(50)])
        nh, _ = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
        total_words = sum(len(c.text.split()) for c in nh.chunks)
        assert len(nh.chunks) < 12
        assert total_words >= 5000
        assert len(nh.chunks) <= 6

    def test_target_size_uses_median_not_max_for_word_floor(self) -> None:
        """Small-median pool with a few large outliers: chunk floor must still bind.

        Mirrors the HotpotQA shape (median≈81, max≈999). The earlier
        descending-by-size estimator hit min_words at ~5 large chunks and
        returned target_n=6, silently ignoring min_chunks=18. With the
        median-based estimator the chunk floor binds as intended.
        """
        chunks = [_chunk(f"doc::{i}", f"doc_{i % 4}", "word " * 50) for i in range(100)] + [
            _chunk(f"big::{i}", "doc_big", "word " * 1000) for i in range(10)
        ]
        tfidf = _tfidf([[float(i), 1.0] for i in range(len(chunks))])
        nh, _ = build_neighborhood(0, chunks, tfidf, min_chunks=18, min_words=5000)
        assert len(nh.chunks) == 18

    def test_low_same_weight_biases_cross_doc(self) -> None:
        """Low same_doc_weight draws the palette toward cross-doc neighbors."""
        chunks: list[ChunkRecord] = []
        for i in range(20):
            chunks.append(_chunk(f"doc_0::c{i}", "doc_0", "word " * 50))
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=10,
            min_words=99999,
            same_doc_weight=0.1,
            cross_doc_weight=0.9,
        )
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        n_cross = sum(1 for c in nh.chunks[1:] if c.doc_id != "doc_0")
        assert n_cross >= 7
        assert n_same <= 2

    def test_high_same_weight_biases_same_doc(self) -> None:
        chunks: list[ChunkRecord] = []
        for i in range(20):
            chunks.append(_chunk(f"doc_0::c{i}", "doc_0", "word " * 50))
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=10,
            min_words=99999,
            same_doc_weight=0.8,
            cross_doc_weight=0.2,
        )
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        assert n_same >= 6

    def test_weights_normalize_freely(self) -> None:
        """Weights need not sum to 1; ratio is what matters."""
        chunks: list[ChunkRecord] = []
        for i in range(20):
            chunks.append(_chunk(f"doc_0::c{i}", "doc_0", "word " * 50))
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh_a, _ = build_neighborhood(
            0, chunks, tfidf, min_chunks=10, min_words=99999, same_doc_weight=4.0, cross_doc_weight=1.0
        )
        nh_b, _ = build_neighborhood(
            0, chunks, tfidf, min_chunks=10, min_words=99999, same_doc_weight=0.8, cross_doc_weight=0.2
        )
        ids_a = [c.chunk_id for c in nh_a.chunks]
        ids_b = [c.chunk_id for c in nh_b.chunks]
        assert ids_a == ids_b

    def test_deficit_redirects_to_cross_doc(self) -> None:
        """When same-doc pool runs out, the deficit fills from cross-doc."""
        chunks: list[ChunkRecord] = [
            _chunk("doc_0::c0", "doc_0", "word " * 50),
            _chunk("doc_0::c1", "doc_0", "word " * 50),
        ]
        for i in range(20):
            chunks.append(_chunk(f"doc_{i + 1}::c0", f"doc_{i + 1}", "word " * 50))
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=8,
            min_words=99999,
            same_doc_weight=0.9,
            cross_doc_weight=0.1,
        )
        assert len(nh.chunks) == 8
        n_same = sum(1 for c in nh.chunks[1:] if c.doc_id == "doc_0")
        n_cross = sum(1 for c in nh.chunks[1:] if c.doc_id != "doc_0")
        assert n_same == 1
        assert n_cross == 6

    def test_cosine_to_anchor_picks_most_similar_cross_doc(self) -> None:
        """With no same-doc, centroid = anchor; cross-doc ranks by cosine to anchor."""
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
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=4,
            min_words=99999,
            same_doc_weight=0.0,
            cross_doc_weight=1.0,
        )
        picked = [c.chunk_id for c in nh.chunks]
        assert picked[0] == "a::0"
        assert picked[1] == "b::0"
        assert picked[2] == "c::0"
        assert picked[3] == "d::0"

    def test_centroid_differs_from_anchor_when_same_doc_present(self) -> None:
        """Same-doc picks shift the centroid; cross-doc ranking follows.

        Anchor TF-IDF leans on axis 0; same-doc sibling leans on axis 1.
        Anchor-only cosine would pick cross-doc chunks aligned with axis 0.
        Centroid cosine (anchor + sibling) picks chunks balanced across
        both axes — a different top cross-doc pick.
        """
        chunks = [
            _chunk("doc_a::0", "doc_a", "anchor word " * 50),
            _chunk("doc_a::1", "doc_a", "sibling word " * 50),
            _chunk("doc_b::0", "doc_b", "axis-0 cross-doc " * 50),
            _chunk("doc_c::0", "doc_c", "axis-1 cross-doc " * 50),
            _chunk("doc_d::0", "doc_d", "balanced cross-doc " * 50),
        ]
        tfidf = _tfidf(
            [
                [1.0, 0.0],  # anchor
                [0.0, 1.0],  # same-doc sibling — orthogonal to anchor
                [1.0, 0.0],  # cross-doc B: aligned with anchor only
                [0.0, 1.0],  # cross-doc C: aligned with sibling only
                [0.7, 0.7],  # cross-doc D: aligned with the centroid (a+s)/√2
            ]
        )
        nh_anchor_only, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=3,
            min_words=99999,
            same_doc_weight=0.0,
            cross_doc_weight=1.0,
        )
        nh_centroid, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=3,
            min_words=99999,
            same_doc_weight=0.5,
            cross_doc_weight=0.5,
        )
        ids_anchor_only = [c.chunk_id for c in nh_anchor_only.chunks]
        ids_centroid = [c.chunk_id for c in nh_centroid.chunks]
        assert ids_anchor_only[0] == "doc_a::0"
        assert ids_anchor_only[1] == "doc_b::0"
        assert ids_centroid[0] == "doc_a::0"
        assert ids_centroid[1] == "doc_a::1"
        assert ids_centroid[2] == "doc_d::0"

    def test_anchor_only_when_corpus_has_no_other_chunks(self) -> None:
        chunks = [_chunk("a::0", "a", "anchor only")]
        tfidf = _tfidf([[1.0, 0.0]])
        nh, _ = build_neighborhood(0, chunks, tfidf, min_chunks=12, min_words=5000)
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

    def test_rejects_both_weights_zero(self) -> None:
        chunks = [_chunk("a::0", "a", "x")]
        tfidf = _tfidf([[1.0, 0.0]])
        with pytest.raises(ValueError, match="must be > 0"):
            build_neighborhood(
                0,
                chunks,
                tfidf,
                min_chunks=1,
                min_words=99999,
                same_doc_weight=0.0,
                cross_doc_weight=0.0,
            )

    def test_rejects_negative_weight(self) -> None:
        chunks = [_chunk("a::0", "a", "x")]
        tfidf = _tfidf([[1.0, 0.0]])
        with pytest.raises(ValueError, match="weights"):
            build_neighborhood(
                0,
                chunks,
                tfidf,
                min_chunks=1,
                min_words=99999,
                same_doc_weight=-0.1,
                cross_doc_weight=1.0,
            )

    def test_even_length_pool_keeps_target_size_integer(self) -> None:
        """``statistics.median`` returns float for even-length pools.

        Without an int-cast on the word-floor ceil-div, the float leaks
        into ``target_n`` → ``cross_budget`` and trips the cross-doc
        slice with ``TypeError: slice indices must be integers``. Anchor
        at index 0, doc_id=``a`` (no same-doc siblings), 8 cross-doc
        chunks → pool length 8 (even).
        """
        chunks = [_chunk("a::0", "a", "word " * 800)] + [
            _chunk(f"d{i}::0", f"d{i}", "word " * 1000) for i in range(8)
        ]
        tfidf = _tfidf([[float(i), 1.0] for i in range(len(chunks))])
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=18,
            min_words=5000,
            same_doc_weight=0.3,
            cross_doc_weight=0.7,
        )
        assert sum(len(c.text.split()) for c in nh.chunks) >= 5000


class TestNeighborhoodDiagnostic:
    def test_position_kinds_label_anchor_same_doc_and_cross_doc(self) -> None:
        chunks: list[ChunkRecord] = [
            _chunk("doc_0::c0", "doc_0", "word " * 50),
            _chunk("doc_0::c1", "doc_0", "word " * 50),
            _chunk("doc_0::c2", "doc_0", "word " * 50),
            _chunk("doc_1::c0", "doc_1", "word " * 50),
            _chunk("doc_2::c0", "doc_2", "word " * 50),
        ]
        tfidf = _tfidf([[1.0, 0.0]] * len(chunks))
        nh, diag = build_neighborhood(
            0, chunks, tfidf, min_chunks=4, min_words=99999, same_doc_weight=0.5, cross_doc_weight=0.5
        )
        assert isinstance(diag, NeighborhoodDiagnostic)
        assert diag.position_kinds[0] == "anchor"
        for kind in diag.position_kinds[1:]:
            assert kind in {"same_doc", "cross_doc"}
        assert len(diag.position_kinds) == len(nh.chunks)
        assert diag.centroid.shape == (tfidf.shape[1],)

    def test_centroid_is_l2_normalized(self) -> None:
        chunks = [
            _chunk("doc_a::0", "doc_a", "anchor"),
            _chunk("doc_a::1", "doc_a", "sibling"),
            _chunk("doc_b::0", "doc_b", "other"),
        ]
        tfidf = _tfidf([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        _, diag = build_neighborhood(
            0, chunks, tfidf, min_chunks=3, min_words=99999, same_doc_weight=1.0, cross_doc_weight=0.5
        )
        norm = float(np.linalg.norm(diag.centroid))
        assert abs(norm - 1.0) < 1e-5


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
                "the accelerator facility brookhaven hosts physics research projects",
            ),
            _chunk(
                "doc_b::c0",
                "doc_b",
                "naval logistics during wartime made brookhaven a critical depot",
            ),
            _chunk("doc_c::c0", "doc_c", "paris cafes serve coffee in elegant streetside settings"),
            _chunk("doc_d::c0", "doc_d", "vienna cafes rival those of paris in coffee culture"),
            _chunk("doc_e::c0", "doc_e", "the garden contained roses and the soil rich with nutrients"),
            _chunk("doc_f::c0", "doc_f", "garden roses bloomed last summer in warm weather"),
        ]
        tfidf, _ = build_tfidf_matrix(chunks)
        nh, _ = build_neighborhood(
            0,
            chunks,
            tfidf,
            min_chunks=3,
            min_words=99999,
            same_doc_weight=0.0,
            cross_doc_weight=1.0,
        )
        picked = [c.chunk_id for c in nh.chunks]
        assert picked[0] == "doc_a::c0"
        assert picked[1] == "doc_b::c0"
