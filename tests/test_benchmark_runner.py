"""Tests for benchmark_eval.runner helpers and guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_autorag.benchmark_eval.evaluator import is_error_sentinel
from agentic_autorag.benchmark_eval.models import QAResult
from agentic_autorag.benchmark_eval.runner import _aggregate, _corpus_hash, _load_corpus
from agentic_autorag.benchmark_eval.scoring import retrieval_metrics  # noqa: F401 (import check)
from agentic_autorag.config.models import ParsingConfig


class TestCorpusHashParity:
    """The runner's corpus_hash must match orchestrator._corpus_cache_key byte-for-byte."""

    def _orchestrator_hash(self, corpus_path: Path, parsing: ParsingConfig) -> str:
        """Replicate orchestrator._corpus_cache_key; if the orchestrator changes we see it here."""
        import hashlib

        sigs: list[tuple[str, int, int]] = []
        for file_path in sorted(corpus_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith(".") or file_path.name == "metadata.json":
                continue
            stat = file_path.stat()
            sigs.append((str(file_path.relative_to(corpus_path)), stat.st_mtime_ns, stat.st_size))
        key = json.dumps(
            {
                "parser": parsing.parser,
                "ocr": parsing.ocr,
                "table_structure": parsing.table_structure,
                "files": sigs,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def test_hashes_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Doc A contents", encoding="utf-8")
        (tmp_path / "b.md").write_text("Doc B contents", encoding="utf-8")
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")  # must be skipped

        parsing = ParsingConfig(parser="docling", ocr=False, table_structure=False)
        assert _corpus_hash(tmp_path, parsing) == self._orchestrator_hash(tmp_path, parsing)

    def test_parser_change_invalidates(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Doc A", encoding="utf-8")
        parsing_a = ParsingConfig(parser="docling", ocr=False, table_structure=False)
        parsing_b = ParsingConfig(parser="pymupdf4llm", ocr=False, table_structure=False)
        assert _corpus_hash(tmp_path, parsing_a) != _corpus_hash(tmp_path, parsing_b)


class TestLoadCorpusGuards:
    def test_errors_on_unsupported_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"\x00PDF-ish")

        with pytest.raises(RuntimeError, match="only supports .md/.txt"):
            _load_corpus(tmp_path)

    def test_errors_on_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No .md/.txt files"):
            _load_corpus(tmp_path)

    def test_uses_stem_as_doc_id(self, tmp_path: Path) -> None:
        (tmp_path / "aloysia.md").write_text("Some text", encoding="utf-8")
        (tmp_path / "ed_wood.md").write_text("Other text", encoding="utf-8")

        filenames, texts = _load_corpus(tmp_path)
        assert sorted(filenames) == ["aloysia", "ed_wood"]
        assert len(texts) == 2

    def test_skips_metadata_json(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("hi", encoding="utf-8")
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

        filenames, _ = _load_corpus(tmp_path)
        assert filenames == ["doc"]


def _qa_result(**kwargs) -> QAResult:
    defaults = {
        "id": "q",
        "question": "Q?",
        "gold_answers": ["a"],
        "pred": "a",
        "em": 1.0,
        "f1": 1.0,
    }
    defaults.update(kwargs)
    return QAResult(**defaults)


class TestAggregate:
    def test_basic_metrics(self) -> None:
        results = [
            _qa_result(id="q1", em=1.0, f1=1.0, retrieved_doc_ids=["a", "b"], supporting_doc_ids=["a"]),
            _qa_result(id="q2", em=0.0, f1=0.5, retrieved_doc_ids=["c", "d"], supporting_doc_ids=["a"]),
        ]
        agg = _aggregate(results, supporting_present=True, judge_enabled=False)
        assert agg["n_valid"] == 2
        assert agg["em"] == 0.5
        assert agg["f1"] == 0.75
        assert agg["recall_at_1"] == 0.5  # q1 hit, q2 miss
        assert agg["mrr"] == 0.5  # q1 rank 1 (1.0), q2 miss (0), avg 0.5
        assert agg["llm_judge_accuracy"] is None
        assert agg["n_judge_invalid"] == 0

    def test_judge_accuracy_excludes_invalid(self) -> None:
        results = [
            _qa_result(id="q1", judge=1),
            _qa_result(id="q2", judge=0),
            _qa_result(id="q3", judge=None),  # parse-fail
        ]
        agg = _aggregate(results, supporting_present=False, judge_enabled=True)
        assert agg["llm_judge_accuracy"] == 0.5  # 1 correct of 2 judged
        assert agg["n_judge_invalid"] == 1

    def test_all_error_sentinels(self) -> None:
        results = [
            _qa_result(id="q1", error="TRANSIENT_LLM_ERROR"),
            _qa_result(id="q2", error="PERMANENT_LLM_ERROR"),
        ]
        assert all(is_error_sentinel(r) for r in results)
        agg = _aggregate(results, supporting_present=False, judge_enabled=False)
        assert agg["n_valid"] == 0
        assert agg["em"] == 0.0
        assert agg["recall_at_5"] is None

    def test_no_supporting_docs_no_retrieval_metrics(self) -> None:
        results = [_qa_result(id="q1", supporting_doc_ids=[])]
        agg = _aggregate(results, supporting_present=False, judge_enabled=False)
        assert agg["recall_at_1"] is None
        assert agg["mrr"] is None
