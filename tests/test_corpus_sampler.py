"""Tests for the deterministic word-budget corpus sampler."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_autorag.engine.corpus_sampler import (
    SELECTION_CACHE_FILENAME,
    sample_corpus,
)


def _make_md_file(dirpath: Path, name: str, n_bytes: int) -> Path:
    """Create a .md file of exactly ``n_bytes`` bytes."""
    path = dirpath / name
    path.write_text("a" * n_bytes, encoding="utf-8")
    return path


class TestSampleCorpus:
    def test_returns_all_files_when_budget_disabled(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(5):
            _make_md_file(corpus, f"doc_{i}.md", 1000)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        files = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=None,
            sample_seed=0,
            cache_dir=cache_dir,
        )
        assert len(files) == 5
        assert not (cache_dir / SELECTION_CACHE_FILENAME).exists()

    def test_walk_stops_when_budget_hit(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # 100 files × ~5000 bytes ≈ 90k estimated words total.
        for i in range(100):
            _make_md_file(corpus, f"doc_{i:03d}.md", 5000)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Budget of 9000 words ≈ 10 files at 900 words/file.
        files = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=9_000,
            sample_seed=42,
            cache_dir=cache_dir,
        )
        # Walk stops at first file pushing accumulated_words ≥ budget.
        # Each file contributes 5000 * 0.18 = 900 words, so ≈ 10 files.
        assert 5 <= len(files) <= 20
        assert len(files) < 100

        cache_path = cache_dir / SELECTION_CACHE_FILENAME
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text())
        assert payload["stats"]["limiter_kicked_in"] is True
        assert payload["sample_seed"] == 42

    def test_cache_hit_returns_same_selection(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(50):
            _make_md_file(corpus, f"doc_{i:02d}.md", 5000)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        first = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=9_000,
            sample_seed=42,
            cache_dir=cache_dir,
        )
        second = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=9_000,
            sample_seed=42,
            cache_dir=cache_dir,
        )
        assert first == second

    def test_cache_invalidates_when_listing_changes(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(20):
            _make_md_file(corpus, f"doc_{i:02d}.md", 5000)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        first = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=9_000,
            sample_seed=42,
            cache_dir=cache_dir,
        )
        cache_path = cache_dir / SELECTION_CACHE_FILENAME
        first_payload = json.loads(cache_path.read_text())

        # Add a new file → corpus_listing_hash changes → cache miss.
        _make_md_file(corpus, "new_doc.md", 5000)
        second = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=9_000,
            sample_seed=42,
            cache_dir=cache_dir,
        )
        second_payload = json.loads(cache_path.read_text())
        assert first_payload["corpus_listing_hash"] != second_payload["corpus_listing_hash"]
        # Selections may overlap but are not guaranteed identical (re-shuffle).
        assert second is not first

    def test_unsupported_extensions_filtered_out(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _make_md_file(corpus, "a.md", 1000)
        # .xyz is neither text-like nor in parser_extensions → filtered out.
        (corpus / "b.xyz").write_text("ignored", encoding="utf-8")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        files = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=10_000,
            sample_seed=0,
            cache_dir=cache_dir,
        )
        assert len(files) == 1
        assert files[0].suffix == ".md"

    def test_no_trimming_when_corpus_fits_budget(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(3):
            _make_md_file(corpus, f"doc_{i}.md", 1000)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        files = sample_corpus(
            corpus_path=corpus,
            parser_extensions=frozenset(),
            word_budget=10_000_000,
            sample_seed=0,
            cache_dir=cache_dir,
        )
        assert len(files) == 3
        payload = json.loads((cache_dir / SELECTION_CACHE_FILENAME).read_text())
        assert payload["stats"]["limiter_kicked_in"] is False
