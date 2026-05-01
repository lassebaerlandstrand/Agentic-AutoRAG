"""Test ``qa_mcq.export_mcq_exam_to_parquet`` for the open-ended exam format."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from agentic_autorag.baselines.autorag.qa_mcq import export_mcq_exam_to_parquet
from agentic_autorag.config.models import OpenEndedQuestion


def _make_question(
    qid: str,
    canonical: str = "Sarah Smith",
    variants: list[str] | None = None,
    docs: list[str] | None = None,
) -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"Q {qid}?",
        canonical_answer=canonical,
        answer_variants=variants or [],
        chunk_A_id=f"{(docs or ['doc_a', 'doc_b'])[0]}::chunk_0",
        chunk_B_id=f"{(docs or ['doc_a', 'doc_b'])[1]}::chunk_0",
        source_span_A="span A",
        source_span_B="span B",
        source_doc_ids=docs or ["doc_a", "doc_b"],
        bridge_entity="bridge",
        cluster_id=0,
    )


def _make_exam(n: int = 3) -> list[OpenEndedQuestion]:
    return [_make_question(f"Q{i:02d}") for i in range(n)]


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


def test_generation_gt_includes_canonical_and_variants(tmp_path: Path) -> None:
    exam = [
        _make_question(
            "Q01",
            canonical="Paris",
            variants=["the capital of France"],
            docs=["france.md", "europe.md"],
        )
    ]
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    gen_gt = df["generation_gt"].iloc[0]
    assert list(gen_gt) == ["Paris", "the capital of France"]


def test_retrieval_gt_uses_doc_stems(tmp_path: Path) -> None:
    exam = [_make_question("Q01", docs=["alpha.md", "beta.txt"])]
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    retrieval_gt = df["retrieval_gt"].iloc[0]
    inner = list(retrieval_gt[0])
    assert inner == ["alpha", "beta"]


def test_query_is_question_text_verbatim(tmp_path: Path) -> None:
    exam = _make_exam(1)
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    assert df["query"].iloc[0] == exam[0].question


def test_metadata_dict_has_last_modified(tmp_path: Path) -> None:
    exam = _make_exam(1)
    exam_path = tmp_path / "exam.json"
    exam_path.write_text(json.dumps([q.model_dump(mode="json") for q in exam]))

    out = tmp_path / "qa.parquet"
    export_mcq_exam_to_parquet(exam_path, out)
    df = pd.read_parquet(out)

    md = df["metadata"].iloc[0]
    assert isinstance(md, dict)
    assert isinstance(md["last_modified_datetime"], datetime)
