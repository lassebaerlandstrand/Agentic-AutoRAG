"""Tests for the HotpotQA adapter: offline-fixture, no HF download."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_autorag.benchmarks.base import slugify
from agentic_autorag.benchmarks.hotpot_qa import HotpotQAAdapter


def _fake_rows() -> list[dict]:
    return [
        {
            "id": "ex_001",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
            "context": {
                "title": ["Scott Derrickson", "Ed Wood", "Woodson, Arkansas"],
                "sentences": [
                    ["Scott Derrickson (born July 16, 1966) is an American director.", " He lives in LA."],
                    ["Ed Wood was an American filmmaker.", " Known for low-budget horror films."],
                    ["Woodson is a CDP in Pulaski County.", " Pop. 403."],
                ],
            },
        },
        {
            "id": "ex_002",
            "question": "What nationality was the director of Doctor Strange?",
            "answer": "American",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Scott Derrickson"], "sent_id": [0]},
            "context": {
                "title": ["Scott Derrickson", "Doctor Strange (2016 film)"],
                "sentences": [
                    ["Scott Derrickson (born July 16, 1966) is an American director.", " He lives in LA."],
                    ["Doctor Strange is a 2016 Marvel Studios film.", " Directed by Scott Derrickson."],
                ],
            },
        },
    ]


def _patched_load_dataset(*args, **kwargs):
    return _fake_rows()


def _patched_hf_api():
    api = MagicMock()
    api.dataset_info.return_value = MagicMock(sha="deadbeefcafe")
    return api


def test_prepare_writes_expected_artefacts(tmp_path: Path) -> None:
    adapter = HotpotQAAdapter()
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        manifest = adapter.prepare(output_dir=tmp_path, split="validation", sample_size=None, seed=42)

    corpus_dir = tmp_path / "corpus"
    qa_path = tmp_path / "qa.json"
    meta_path = tmp_path / "metadata.json"

    assert corpus_dir.exists()
    # 4 unique titles: Scott Derrickson, Ed Wood, Woodson Arkansas, Doctor Strange.
    md_files = sorted(p.name for p in corpus_dir.glob("*.md"))
    assert len(md_files) == 4

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert len(qa) == 2
    assert qa[0]["id"] == "ex_001"
    assert qa[0]["gold_answers"] == ["yes"]
    assert set(qa[0]["supporting_doc_ids"]) == {
        slugify("Scott Derrickson"),
        slugify("Ed Wood"),
    }
    assert qa[1]["metadata"]["type"] == "bridge"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["name"] == "hotpot_qa"
    assert meta["corpus_doc_count"] == 4
    assert meta["hf_revision"] == "deadbeefcafe"
    assert manifest.hf_revision == "deadbeefcafe"


def test_deterministic_sampling(tmp_path: Path) -> None:
    """Same seed → identical qa.json across runs."""
    adapter = HotpotQAAdapter()
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
    """sample_size > available rows must raise, not silently truncate."""
    adapter = HotpotQAAdapter()
    with (
        patch("datasets.load_dataset", side_effect=_patched_load_dataset),
        patch("huggingface_hub.HfApi", side_effect=_patched_hf_api),
    ):
        with pytest.raises(ValueError, match="exceeds available rows"):
            adapter.prepare(tmp_path, split="validation", sample_size=999, seed=42)


def test_slugify_collision_suffix() -> None:
    """Distinct titles that slugify to the same base get a sha suffix."""
    used: set[str] = set()
    a = slugify("Foo Bar", used=used)
    b = slugify("foo-bar", used=used)  # same slug base
    assert a == "foo_bar"
    assert b != a
    assert b.startswith("foo_bar__")
