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


class TestSharedCorpusLoader:
    """The optimizer's retrieval index and the held-out runner share one
    .md/.txt loader, so both index the identical doc-id universe."""

    def test_stems_all_docs_headings_sorted_with_skips(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "beta.md").write_text("# Beta Title\n\nbody about beta", encoding="utf-8")
        (corpus / "alpha.md").write_text("# Alpha Title\n\nbody about alpha", encoding="utf-8")
        (corpus / "blank.md").write_text("   ", encoding="utf-8")  # empty -> dropped
        (corpus / "metadata.json").write_text("{}", encoding="utf-8")  # skipped
        (corpus / ".hidden").write_text("secret", encoding="utf-8")  # skipped

        stems, texts = _load_corpus(corpus)

        # doc-ids are stems, sorted; empty / metadata / hidden files dropped.
        assert stems == ["alpha", "beta"]
        # raw text keeps the markdown heading (high-signal retrieval term).
        assert "Alpha Title" in texts[0]
        assert "body about beta" in texts[1]


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
                "schema": 3,
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

        parsing = ParsingConfig(ocr=False, table_structure=False)
        assert _corpus_hash(tmp_path, parsing) == self._orchestrator_hash(tmp_path, parsing)

    def test_ocr_change_invalidates(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Doc A", encoding="utf-8")
        parsing_a = ParsingConfig(ocr=False, table_structure=False)
        parsing_b = ParsingConfig(ocr=True, table_structure=False)
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
        # Single-gold questions → joint_recall == recall at every k.
        assert agg["joint_recall_at_1"] == 0.5
        assert agg["mrr_first"] == 0.5  # q1 rank 1, q2 miss → mean 0.5
        # Single-gold → mrr_complete == mrr_first.
        assert agg["mrr_complete"] == 0.5
        assert agg["llm_judge_accuracy"] is None
        assert agg["n_judge_invalid"] == 0

    def test_multi_hop_separates_first_from_complete(self) -> None:
        # q1: both gold at ranks 1,2 → first_rank=1, complete_rank=2.
        # q2: only one of two gold ever retrieved → first_rank=1, complete=None.
        results = [
            _qa_result(
                id="q1",
                retrieved_doc_ids=["a", "b", "c"],
                supporting_doc_ids=["a", "b"],
            ),
            _qa_result(
                id="q2",
                retrieved_doc_ids=["a", "x", "y"],
                supporting_doc_ids=["a", "b"],
            ),
        ]
        agg = _aggregate(results, supporting_present=True, judge_enabled=False)
        # Partial recall: q1=1.0 @2, q2=0.5 @anything → mean@2 = 0.75.
        assert agg["recall_at_2"] == 0.75
        # Joint recall: q1=1 @2, q2=0 @anything → mean@2 = 0.5.
        assert agg["joint_recall_at_2"] == 0.5
        # Both questions hit gold at rank 1 → mrr_first = 1.0.
        assert agg["mrr_first"] == 1.0
        # Only q1 ever completes (at rank 2 → 0.5); q2 never → 0.
        assert agg["mrr_complete"] == pytest.approx(0.25)

    def test_judge_accuracy_excludes_invalid(self) -> None:
        results = [
            _qa_result(id="q1", judge=1),
            _qa_result(id="q2", judge=0),
            _qa_result(id="q3", judge=None),  # parse-fail
        ]
        agg = _aggregate(results, supporting_present=False, judge_enabled=True)
        assert agg["llm_judge_accuracy"] == 0.5  # 1 correct of 2 judged
        assert agg["n_judge_invalid"] == 1

    def test_judge_accuracy_counts_abstention_as_incorrect(self) -> None:
        """judge=-1 is NO_ANSWER (abstention). It counts as incorrect — in the
        denominator, not the numerator — never as a negative penalty."""
        results = [
            _qa_result(id="q1", judge=1),
            _qa_result(id="q2", judge=-1),  # abstained
            _qa_result(id="q3", judge=-1),  # abstained
            _qa_result(id="q4", judge=0),
        ]
        agg = _aggregate(results, supporting_present=False, judge_enabled=True)
        # 1 correct of 4 judged; the two -1s must not drag this to (1-2)/4.
        assert agg["llm_judge_accuracy"] == 0.25
        assert agg["n_judge_invalid"] == 0

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
        assert agg["joint_recall_at_5"] is None
        assert agg["mrr_first"] is None
        assert agg["mrr_complete"] is None

    def test_no_supporting_docs_no_retrieval_metrics(self) -> None:
        results = [_qa_result(id="q1", supporting_doc_ids=[])]
        agg = _aggregate(results, supporting_present=False, judge_enabled=False)
        assert agg["recall_at_1"] is None
        assert agg["joint_recall_at_1"] is None
        assert agg["mrr_first"] is None
        assert agg["mrr_complete"] is None
