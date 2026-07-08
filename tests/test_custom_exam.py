"""Tests for the tiered OpenEndedQuestion schema and the custom-exam loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.examiner.custom_exam import load_custom_exam

# --- schema: the three grounding tiers ------------------------------------


def test_tier_c_spans_accepted_and_offsets_defaulted() -> None:
    q = OpenEndedQuestion(
        id="c1",
        question="q?",
        canonical_answer="a",
        reasoning_type="bridge",
        source_doc_ids=["d1", "d2"],
        source_spans=["span one", "span two"],
    )
    assert q.grounding_tier == "C"
    assert q.num_hops == 2
    assert q.source_span_offsets == [None, None]


def test_tier_b_doc_ids_only_accepted() -> None:
    q = OpenEndedQuestion(id="b1", question="q?", canonical_answer="a", supporting_doc_ids=["d1", "d2"])
    assert q.grounding_tier == "B"
    assert q.reasoning_type is None
    assert q.num_hops == 0
    assert q.source_span_offsets == []


def test_tier_a_bare_accepted() -> None:
    q = OpenEndedQuestion(id="a1", question="q?", canonical_answer="a")
    assert q.grounding_tier == "A"


def test_empty_canonical_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        OpenEndedQuestion(id="x", question="q?", canonical_answer="   ")


def test_misaligned_spans_rejected() -> None:
    with pytest.raises(ValidationError):
        OpenEndedQuestion(
            id="x",
            question="q?",
            canonical_answer="a",
            source_spans=["only one span"],
            source_doc_ids=["d1", "d2"],
        )


def test_doc_ids_without_spans_rejected() -> None:
    # source_doc_ids belongs to the tier-C span lane; without spans the
    # doc-level gold must live in supporting_doc_ids instead.
    with pytest.raises(ValidationError):
        OpenEndedQuestion(id="x", question="q?", canonical_answer="a", source_doc_ids=["d1"])


def test_blank_span_rejected() -> None:
    with pytest.raises(ValidationError):
        OpenEndedQuestion(
            id="x",
            question="q?",
            canonical_answer="a",
            source_spans=["  "],
            source_doc_ids=["d1"],
        )


# --- BenchmarkQAPair.to_open_ended ----------------------------------------


def test_benchmark_qa_pair_converts_to_tier_b() -> None:
    qa = BenchmarkQAPair(id="qa1", question="q?", gold_answers=["gold", "alt"], supporting_doc_ids=["d1"])
    oe = qa.to_open_ended()
    assert oe.grounding_tier == "B"
    assert oe.canonical_answer == "gold"
    assert oe.answer_variants == ["alt"]
    assert oe.supporting_doc_ids == ["d1"]


def test_benchmark_qa_pair_no_docs_is_tier_a() -> None:
    qa = BenchmarkQAPair(id="qa1", question="q?", gold_answers=["gold"])
    assert qa.to_open_ended().grounding_tier == "A"


def test_benchmark_qa_pair_no_gold_raises() -> None:
    qa = BenchmarkQAPair(id="qa1", question="q?", gold_answers=[""])
    with pytest.raises(ValueError):
        qa.to_open_ended()


# --- loader: never drops, both shapes, malformed raises -------------------


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "exam.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


def test_loader_accepts_mixed_shapes_and_preserves_count(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "c1",
                "question": "q1",
                "canonical_answer": "a1",
                "reasoning_type": "bridge",
                "source_doc_ids": ["d1", "d2"],
                "source_spans": ["span one", "span two"],
            },
            {"id": "b1", "question": "q2", "canonical_answer": "a2", "supporting_doc_ids": ["d3"]},
            {"id": "qa1", "question": "q3", "gold_answers": ["gold", "alt"], "supporting_doc_ids": ["d4"]},
            {"id": "a1", "question": "q4", "canonical_answer": "a4"},
        ],
    )
    exam = load_custom_exam(path)
    assert len(exam) == 4  # count in == count out
    assert [q.grounding_tier for q in exam] == ["C", "B", "B", "A"]


def test_loader_malformed_record_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "x"}])
    with pytest.raises(ValueError):
        load_custom_exam(path)


def test_loader_duplicate_id_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"id": "z", "question": "q", "canonical_answer": "a"},
            {"id": "z", "question": "q", "canonical_answer": "a"},
        ],
    )
    with pytest.raises(ValueError):
        load_custom_exam(path)


def test_loader_non_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "exam.json"
    p.write_text(json.dumps({"id": "z"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_custom_exam(p)


def test_loader_empty_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_custom_exam(_write(tmp_path, []))


def test_loader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_custom_exam(tmp_path / "nope.json")


def test_bare_abstention_record_loads_as_tier_a(tmp_path: Path) -> None:
    # A user marks a question unanswerable by giving it a gold statement of
    # insufficiency with no spans and no docs. It loads as a tier-A question
    # with no reasoning_type — the answerer may abstain and the judge grades it.
    path = _write(
        tmp_path,
        [{"id": "ua1", "question": "q?", "canonical_answer": "Insufficient information."}],
    )
    exam = load_custom_exam(path)
    assert len(exam) == 1
    q = exam[0]
    assert q.grounding_tier == "A"
    assert q.reasoning_type is None
    assert q.gold_answers == ["Insufficient information."]
