"""Tests for the open-ended oracle-gate validator and source-fact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.engine.pipeline import RetrievedDocument
from agentic_autorag.examiner.exam_validator import (
    chunk_contains_source_fact,
    gate_oracle_pass,
    verify_source_facts,
)


def _q(qid: str, span_a: str = "Span A text", span_b: str = "Span B text") -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"Q {qid}?",
        canonical_answer="Sarah Smith",
        answer_variants=["S. Smith"],
        reasoning_type="bridge",
        source_chunk_ids=["a::0", "b::0"],
        source_doc_ids=["doc_a", "doc_b"],
        source_spans=[span_a, span_b],
    )


class TestVerifySourceFacts:
    def test_records_offsets_when_spans_present(self) -> None:
        documents = {
            "doc_a": "prefix Span A text suffix",
            "doc_b": "prefix Span B text suffix",
        }
        passed = verify_source_facts([_q("q1")], documents)
        assert len(passed) == 1
        assert passed[0].source_span_offsets == [(7, 18), (7, 18)]

    def test_rejects_when_span_missing(self) -> None:
        documents = {
            "doc_a": "prefix Span A text suffix",
            "doc_b": "prefix completely different content",
        }
        passed = verify_source_facts([_q("q1")], documents)
        assert passed == []

    def test_finds_span_with_raw_gt_lt_in_chunk_text_corpus(self) -> None:
        """The chunk-text-concat corpus preserves raw ``>``/``<`` (no HTML escaping).

        Regression for the C0003 failure where ``export_to_markdown`` escaped
        ``P > 0.10`` to ``P &gt; 0.10``, breaking verbatim match.
        """
        documents = {
            "doc_a": "Result one Span A text exact match here.",
            "doc_b": "No significant effect on mortality were found (8.77%; P > 0.10).",
        }
        q = _q("q1", span_a="Span A text", span_b="P > 0.10")
        passed = verify_source_facts([q], documents)
        assert len(passed) == 1
        start_b, end_b = passed[0].source_span_offsets[1]
        assert documents["doc_b"][start_b:end_b] == "P > 0.10"

    def test_finds_span_extracted_from_flattened_table_prose(self) -> None:
        """Table-derived spans match when the corpus carries HybridChunker's
        flattened-prose form rather than markdown pipe rows. Regression for
        C0004 where ``| cell | cell |`` couldn't match comma/equals prose.
        """
        documents = {
            "doc_a": "Result one Span A text exact match here.",
            "doc_b": (
                "coronary atherosclerotic heart disease, Transfer Task = G → N C → N R → N. "
                "coronary atherosclerotic heart disease, # disease term in source domain training set "
                "= 5 136 23."
            ),
        }
        span = "coronary atherosclerotic heart disease, Transfer Task = G → N C → N R → N."
        q = _q("q1", span_a="Span A text", span_b=span)
        passed = verify_source_facts([q], documents)
        assert len(passed) == 1
        start_b, end_b = passed[0].source_span_offsets[1]
        assert documents["doc_b"][start_b:end_b] == span

    def test_writes_report_with_kept_and_rejected_details(self, tmp_path: Path) -> None:
        documents = {
            "doc_a": "prefix Span A text suffix",
            "doc_b": "prefix Span B text suffix",
            "doc_c": "prefix Span A text suffix",
            "doc_d": "prefix completely different content",
        }
        kept_q = _q("q_kept")
        rejected_q = _q("q_rejected").model_copy(
            update={"source_doc_ids": ["doc_c", "doc_d"]},
        )
        report_path = tmp_path / "span_verification.json"

        passed = verify_source_facts([kept_q, rejected_q], documents, report_path=report_path)

        assert [q.id for q in passed] == ["q_kept"]
        report = json.loads(report_path.read_text())
        assert report["summary"]["n_total"] == 2
        assert report["summary"]["n_kept"] == 1
        assert report["summary"]["n_rejected"] == 1
        assert report["summary"]["match_modes"]["verbatim"] == 1
        assert report["summary"]["rejection_reasons"] == {"span_not_found": 1}

        kept_entry = next(r for r in report["questions"] if r["id"] == "q_kept")
        assert kept_entry["status"] == "kept"
        assert kept_entry["match_mode"] == "verbatim"
        assert [s["outcome"] for s in kept_entry["spans"]] == ["verbatim", "verbatim"]

        rejected_entry = next(r for r in report["questions"] if r["id"] == "q_rejected")
        assert rejected_entry["status"] == "rejected"
        assert rejected_entry["rejection_reason"] == "span_not_found"
        assert rejected_entry["failing_span_index"] == 1
        outcomes = [s["outcome"] for s in rejected_entry["spans"]]
        assert outcomes == ["verbatim", "not_found"]
        assert rejected_entry["spans"][1]["span"] == "Span B text"
        assert rejected_entry["spans"][1]["doc_len"] == len(documents["doc_d"])


def _q_single(qid: str, span: str = "Span A text") -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"Q {qid}?",
        canonical_answer="Sarah Smith",
        answer_variants=["S. Smith"],
        reasoning_type="extraction",
        source_chunk_ids=["a::0"],
        source_doc_ids=["doc_a"],
        source_spans=[span],
    )


def _multihop_response(
    *,
    sufficient_spans: list[int],
    answer: str,
    quotes: dict[str, str] | None = None,
    reasoning: str = "Span 0 names X; Span 1 supplies the bridge.",
) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "supporting_quotes": quotes or {"0": "Span A text", "1": "Span B text"},
            "sufficient_spans": sufficient_spans,
            "answer": answer,
        }
    )


@pytest.mark.asyncio
class TestGateOraclePass:
    async def test_keeps_multihop_when_not_decomposable_and_answer_correct(self) -> None:
        response = _multihop_response(sufficient_spans=[0, 1], answer="Sarah Smith")
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value=response),
        ):
            kept = await gate_oracle_pass(
                [_q("q1"), _q("q2")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert len(kept) == 2

    async def test_rejects_multihop_when_oracle_answer_wrong(self) -> None:
        response = _multihop_response(sufficient_spans=[0, 1], answer="completely wrong")
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value=response),
        ):
            kept = await gate_oracle_pass(
                [_q("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert kept == []

    async def test_rejects_multihop_when_decomposable(self) -> None:
        response = _multihop_response(
            sufficient_spans=[0],
            answer="Sarah Smith",
            quotes={"0": "Span A text"},
        )
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value=response),
        ):
            kept = await gate_oracle_pass(
                [_q("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert kept == []

    async def test_rejects_multihop_on_parse_error(self) -> None:
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value="not json at all"),
        ):
            kept = await gate_oracle_pass(
                [_q("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert kept == []

    async def test_writes_multi_hop_rejections_json_when_cache_dir_supplied(self, tmp_path: Path) -> None:
        decomposable_response = _multihop_response(
            sufficient_spans=[1],
            answer="Sarah Smith",
            quotes={"1": "Span B text"},
            reasoning="Span 1 alone has the answer.",
        )
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value=decomposable_response),
        ):
            kept = await gate_oracle_pass(
                [_q("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
                cache_dir=tmp_path,
            )
        assert kept == []
        path = tmp_path / "multi_hop_rejections.json"
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["validator_model"] == "test/model"
        assert payload["prompt_version"] == "multihop_dependency_oracle_v1"
        assert len(payload["rejections"]) == 1
        record = payload["rejections"][0]
        assert record["id"] == "q1"
        assert record["reject_reason"] == "decomposable"
        assert record["llm_sufficient_spans"] == [1]
        assert record["llm_answer"] == "Sarah Smith"
        assert record["oracle_judge_verdict"] is None
        assert [s["idx"] for s in record["spans"]] == [0, 1]

    async def test_single_hop_uses_legacy_oracle_prompt(self) -> None:
        captured_prompts: list[str] = []

        async def fake_call(model: str, prompt: str, temperature: float = 0.0) -> str:
            captured_prompts.append(prompt)
            return "Sarah Smith"

        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=fake_call,
        ):
            kept = await gate_oracle_pass(
                [_q_single("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert len(kept) == 1
        assert len(captured_prompts) == 1
        assert "JSON" not in captured_prompts[0]
        assert "STRUCTURAL JUDGMENT" not in captured_prompts[0]

    async def test_multihop_check_invokes_json_mode(self) -> None:
        captured_kwargs: list[dict] = []
        multihop_response = _multihop_response(sufficient_spans=[0, 1], answer="Sarah Smith")

        async def fake_call(model: str, prompt: str, **kwargs: object) -> str:
            captured_kwargs.append(dict(kwargs))
            return multihop_response if "STRUCTURAL JUDGMENT" in prompt else "Sarah Smith"

        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=fake_call,
        ):
            kept = await gate_oracle_pass(
                [_q("q_multi"), _q_single("q_single")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert len(kept) == 2
        assert len(captured_kwargs) == 2
        multihop_kwargs = next(kw for kw in captured_kwargs if kw.get("response_format") is not None)
        single_hop_kwargs = next(kw for kw in captured_kwargs if kw.get("response_format") is None)
        assert multihop_kwargs["response_format"] == {"type": "json_object"}
        assert "response_format" not in single_hop_kwargs or single_hop_kwargs["response_format"] is None


class TestChunkContainsSourceFact:
    """Doc-id canonicalization: a duplicate alias counts as the canonical."""

    @staticmethod
    def _question() -> OpenEndedQuestion:
        # Question's source doc is the canonical paper.pdf; offsets cover
        # bytes 100..150 of doc_a (the canonical doc text).
        q = _q("q01", span_a="Span A text exact content here", span_b="Span B exact content here")
        return q.model_copy(
            update={
                "source_span_offsets": [(100, 150), (200, 250)],
                "source_doc_ids": ["paper.pdf", "other.pdf"],
            }
        )

    @staticmethod
    def _retrieved(doc_id: str, char_range: tuple[int, int] = (100, 200)) -> RetrievedDocument:
        return RetrievedDocument(
            id="chunk_xyz",
            text="span A text " * 5,
            score=0.5,
            metadata={"doc_id": doc_id},
            char_range=char_range,
        )

    def test_alias_doc_id_counts_when_canonicalized(self) -> None:
        q = self._question()
        # Retrieved chunk's doc_id is paper_page_001.png — an alias of paper.pdf.
        chunk = self._retrieved("paper_page_001.png", char_range=(100, 200))
        alias_map = {"paper_page_001.png": "paper.pdf", "paper.pdf": "paper.pdf"}
        assert chunk_contains_source_fact(q, chunk, duplicate_alias_map=alias_map) is True

    def test_alias_does_not_count_without_map(self) -> None:
        q = self._question()
        chunk = self._retrieved("paper_page_001.png", char_range=(100, 200))
        # Same chunk, no alias map — offset-overlap path requires exact doc_id
        # match and falls through to ngram_relevance which here has no overlap.
        assert chunk_contains_source_fact(q, chunk, duplicate_alias_map=None) is False

    def test_unrelated_doc_still_rejected_with_map(self) -> None:
        q = self._question()
        chunk = self._retrieved("totally_other.pdf", char_range=(100, 200))
        alias_map = {"paper_page_001.png": "paper.pdf"}
        assert chunk_contains_source_fact(q, chunk, duplicate_alias_map=alias_map) is False
