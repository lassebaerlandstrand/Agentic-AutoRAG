"""Test ``qa_mcq.export_mcq_exam_to_parquet``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from agentic_autorag.baselines.autorag.qa_mcq import export_mcq_exam_to_parquet
from agentic_autorag.config.models import MCQQuestion


def _make_exam(n: int = 3) -> list[MCQQuestion]:
    return [
        MCQQuestion(
            id=f"Q{i:02d}",
            question=f"Question {i}?",
            options={"A": f"a{i}", "B": f"b{i}", "C": f"c{i}", "D": f"d{i}"},
            correct_answer="A",
            source_doc_ids=[f"doc{i}.md"],
            cluster_id=0,
        )
        for i in range(n)
    ]


def test_writes_one_row_per_question(tmp_path: Path) -> None:
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in _make_exam(5)]))

    out = tmp_path / "qa.parquet"
    n = export_mcq_exam_to_parquet(exam_path, out)
    assert n == 5

    df = pd.read_parquet(out)
    assert list(df.columns) >= ["qid", "query", "retrieval_gt", "generation_gt", "metadata"]
    assert len(df) == 5
    assert df["qid"].tolist() == [f"Q{i:02d}" for i in range(5)]


def test_generation_gt_is_correct_option_text(tmp_path: Path) -> None:
    exam = [
        MCQQuestion(
            id="Q01",
            question="What is the capital of France?",
            options={"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"},
            correct_answer="A",
            source_doc_ids=["france.md"],
            cluster_id=0,
        )
    ]
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    gen_gt = df["generation_gt"].iloc[0]
    # AutoRAG's schema is list[str]; pyarrow may round-trip as np.ndarray.
    assert list(gen_gt) == ["Paris"]


def test_retrieval_gt_uses_doc_stems(tmp_path: Path) -> None:
    """``retrieval_gt`` must use ``Path.stem`` so it aligns with corpus.parquet."""
    exam = [
        MCQQuestion(
            id="Q01",
            question="?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
            source_doc_ids=["alpha.md", "beta.txt"],
            cluster_id=0,
        )
    ]
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    retrieval_gt = df["retrieval_gt"].iloc[0]
    # list[list[str]] — top-level list per question, inner list holds gold doc set.
    inner = list(retrieval_gt[0])
    assert inner == ["alpha", "beta"]


def test_options_inlined_into_query(tmp_path: Path) -> None:
    """AutoRAG's fstring only sees ``{query}`` — options must live in query text."""
    exam = _make_exam(1)
    exam[0].options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}
    exam[0].correct_answer = "B"
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    query = df["query"].iloc[0]
    assert "A. Paris" in query
    assert "B. London" in query
    assert "C. Berlin" in query
    assert "D. Madrid" in query


def test_metadata_dict_has_last_modified(tmp_path: Path) -> None:
    """AutoRAG schema requires metadata as a dict with last_modified_datetime."""
    exam = _make_exam(1)
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    md = df["metadata"].iloc[0]
    assert isinstance(md, dict)
    assert isinstance(md["last_modified_datetime"], datetime)
