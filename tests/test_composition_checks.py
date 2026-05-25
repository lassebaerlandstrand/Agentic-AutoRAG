"""Tests for the deterministic structural check on composer output."""

from __future__ import annotations

from agentic_autorag.examiner.composition_checks import check_selected_chunk_ids


class TestCheckSelectedChunkIds:
    def test_accepts_valid_selection(self) -> None:
        r = check_selected_chunk_ids([0, 2, 5], neighborhood_size=10)
        assert r.ok is True
        assert r.reason == ""

    def test_accepts_single_chunk_selection(self) -> None:
        r = check_selected_chunk_ids([3], neighborhood_size=10)
        assert r.ok is True

    def test_rejects_empty_selection(self) -> None:
        r = check_selected_chunk_ids([], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "empty_selected_chunk_ids"

    def test_rejects_out_of_range_index(self) -> None:
        r = check_selected_chunk_ids([0, 10], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "uncited_chunk"

    def test_rejects_negative_index(self) -> None:
        r = check_selected_chunk_ids([-1, 2], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "uncited_chunk"

    def test_rejects_duplicates(self) -> None:
        r = check_selected_chunk_ids([0, 2, 0], neighborhood_size=10)
        assert r.ok is False
        assert r.reason == "duplicate_selected_chunk_ids"
