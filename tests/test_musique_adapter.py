"""Tests for the MuSiQue adapter: offline-fixture, no HF download."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_autorag.benchmarks.base import slugify
from agentic_autorag.benchmarks.musique import MuSiQueAdapter


def _fake_rows() -> list[dict]:
    # Three rows: two answerable (kept) and one unanswerable (filtered out).
    # The two answerable rows share "Grant Green" so we exercise corpus
    # dedup-by-title across rows.
    return [
        {
            "id": "2hop__row1",
            "question": "Who is the spouse of the Green performer?",
            "answer": "Miquette Giraudy",
            "answer_aliases": [],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Grant's First Stand",
                    "paragraph_text": "Debut album by jazz guitarist Grant Green released in 1961.",
                    "is_supporting": False,
                },
                {
                    "idx": 1,
                    "title": "Grant Green",
                    "paragraph_text": "Grant Green was an American jazz guitarist.",
                    "is_supporting": True,
                },
                {
                    "idx": 2,
                    "title": "Miquette Giraudy",
                    "paragraph_text": "Miquette Giraudy is a French musician.",
                    "is_supporting": True,
                },
            ],
            "question_decomposition": [
                {"id": 1, "question": "Green >> performer", "answer": "Grant Green", "paragraph_support_idx": 1},
                {"id": 2, "question": "Spouse of #1", "answer": "Miquette Giraudy", "paragraph_support_idx": 2},
            ],
        },
        {
            "id": "2hop__row2",
            "question": "What instrument did the jazz musician play?",
            "answer": "guitar",
            "answer_aliases": ["jazz guitar"],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Grant Green",
                    "paragraph_text": "Grant Green was an American jazz guitarist.",
                    "is_supporting": True,
                },
                {
                    "idx": 1,
                    "title": "Some Other Page",
                    "paragraph_text": "Unrelated distractor text.",
                    "is_supporting": False,
                },
            ],
            "question_decomposition": [
                {"id": 3, "question": "Green >> instrument", "answer": "guitar", "paragraph_support_idx": 0},
            ],
        },
        {
            # MuSiQue-Full unanswerable contrast — must be filtered out.
            "id": "2hop__row3",
            "question": "Where does Grant Green's cousin live?",
            "answer": "",
            "answer_aliases": [],
            "answerable": False,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Filler Distractor",
                    "paragraph_text": "Distractor that should not land in the corpus.",
                    "is_supporting": False,
                },
            ],
            "question_decomposition": [],
        },
    ]


def _patched_load_dataset(*args, **kwargs):
    return _fake_rows()


def _patched_hf_api():
    api = MagicMock()
    api.dataset_info.return_value = MagicMock(sha="deadbeefcafe")
    return api


def test_prepare_writes_expected_artefacts(tmp_path: Path) -> None:
    adapter = MuSiQueAdapter()
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        manifest = adapter.prepare(output_dir=tmp_path, split="validation", sample_size=None, seed=42)

    corpus_dir = tmp_path / "corpus"
    qa_path = tmp_path / "qa.json"
    meta_path = tmp_path / "metadata.json"

    # 4 unique titles from the two answerable rows. The "Filler Distractor"
    # title is bundled with the unanswerable row and must not appear.
    md_files = sorted(p.name for p in corpus_dir.glob("*.md"))
    assert len(md_files) == 4
    assert f"{slugify('Filler Distractor')}.md" not in md_files

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    # Unanswerable row dropped pre-sample.
    assert len(qa) == 2
    ids = sorted(q["id"] for q in qa)
    assert ids == ["2hop__row1", "2hop__row2"]

    row1 = next(q for q in qa if q["id"] == "2hop__row1")
    assert row1["gold_answers"] == ["Miquette Giraudy"]
    assert set(row1["supporting_doc_ids"]) == {
        slugify("Grant Green"),
        slugify("Miquette Giraudy"),
    }
    assert row1["metadata"]["n_hops"] == 2

    row2 = next(q for q in qa if q["id"] == "2hop__row2")
    # answer + answer_aliases collapse into gold_answers.
    assert row2["gold_answers"] == ["guitar", "jazz guitar"]
    assert row2["supporting_doc_ids"] == [slugify("Grant Green")]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["name"] == "musique"
    assert meta["corpus_doc_count"] == 4
    assert meta["sample_size"] == 2  # post-filter
    assert meta["hf_revision"] == "deadbeefcafe"
    assert manifest.hf_revision == "deadbeefcafe"


def test_deterministic_sampling(tmp_path: Path) -> None:
    """Same seed → identical qa.json across runs."""
    adapter = MuSiQueAdapter()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        adapter.prepare(out_a, split="validation", sample_size=1, seed=7)
        adapter.prepare(out_b, split="validation", sample_size=1, seed=7)

    assert (out_a / "qa.json").read_text() == (out_b / "qa.json").read_text()


def test_sample_size_overflow_raises(tmp_path: Path) -> None:
    """sample_size > available answerable rows must raise, not silently truncate."""
    adapter = MuSiQueAdapter()
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        # Only 2 answerable rows in the fixture; asking for 5 must raise.
        with pytest.raises(ValueError, match="exceeds available answerable rows"):
            adapter.prepare(tmp_path, split="validation", sample_size=5, seed=42)


def test_unanswerable_rows_excluded_from_sample_size(tmp_path: Path) -> None:
    """sample_size=2 must return both answerable rows, not 'first two of all three'."""
    adapter = MuSiQueAdapter()
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        adapter.prepare(tmp_path, split="validation", sample_size=2, seed=42)

    qa = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))
    assert len(qa) == 2
    assert all(q["id"].startswith("2hop__row") for q in qa)
    assert all(q["id"] != "2hop__row3" for q in qa)
