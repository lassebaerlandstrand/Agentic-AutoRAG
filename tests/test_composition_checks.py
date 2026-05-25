"""Tests for the deterministic structural check on composer output."""

from __future__ import annotations

from agentic_autorag.examiner.composition_checks import check_selected_chunk_ids


class TestCheckSelectedChunkIds:
    def test_accepts_valid_selection(self) -> None:
        r = check_selected_chunk_ids([0, 2, 5], ["span zero", "span two", "span five"], neighborhood_size=10)
        assert r.ok is True
        assert r.reason == ""

    def test_accepts_single_chunk_selection(self) -> None:
        r = check_selected_chunk_ids([3], ["only span"], neighborhood_size=10)
        assert r.ok is True

    def test_accepts_intra_chunk_multi_hop(self) -> None:
        """Same chunk_id with distinct spans is legitimate intra-chunk multi-hop."""
        r = check_selected_chunk_ids(
            [5, 5], ["first sentence in chunk", "different later sentence"], neighborhood_size=10
        )
        assert r.ok is True

    def test_rejects_empty_selection(self) -> None:
        r = check_selected_chunk_ids([], [], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "empty_selected_chunk_ids"

    def test_rejects_out_of_range_index(self) -> None:
        r = check_selected_chunk_ids([0, 10], ["a", "b"], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "uncited_chunk"

    def test_rejects_negative_index(self) -> None:
        r = check_selected_chunk_ids([-1, 2], ["a", "b"], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "uncited_chunk"

    def test_rejects_misaligned_spans(self) -> None:
        r = check_selected_chunk_ids([0, 2, 5], ["a", "b"], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "spans_misaligned"

    def test_rejects_duplicate_span_same_chunk(self) -> None:
        """Same chunk_id with identical span text is parallel restatement."""
        r = check_selected_chunk_ids([4, 4], ["identical text", "identical text"], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "duplicate_selected_spans"

    def test_rejects_duplicate_span_across_chunks(self) -> None:
        """Two different chunks emitting identical span text is parallel restatement."""
        r = check_selected_chunk_ids([0, 7], ["shared fact", "shared fact"], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "duplicate_selected_spans"

    def test_duplicate_check_is_whitespace_insensitive(self) -> None:
        r = check_selected_chunk_ids([0, 7], ["Shared Fact", "  shared   fact  "], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "duplicate_selected_spans"
