"""Tests for the open-ended oracle-gate validator and source-fact helpers."""

from __future__ import annotations

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
        chunk_A_id="a::0",
        chunk_B_id="b::0",
        source_span_A=span_a,
        source_span_B=span_b,
        source_doc_ids=["doc_a", "doc_b"],
        bridge_entity="beta inc",
    )


class TestVerifySourceFacts:
    def test_records_offsets_when_spans_present(self) -> None:
        documents = {
            "doc_a": "prefix Span A text suffix",
            "doc_b": "prefix Span B text suffix",
        }
        passed = verify_source_facts([_q("q1")], documents)
        assert len(passed) == 1
        assert passed[0].source_span_A_offset == (7, 18)
        assert passed[0].source_span_B_offset == (7, 18)

    def test_rejects_when_span_missing(self) -> None:
        documents = {
            "doc_a": "prefix Span A text suffix",
            "doc_b": "prefix completely different content",
        }
        passed = verify_source_facts([_q("q1")], documents)
        assert passed == []


@pytest.mark.asyncio
class TestGateOraclePass:
    async def test_keeps_questions_oracle_answers_correctly(self) -> None:
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value="Sarah Smith"),
        ):
            kept = await gate_oracle_pass(
                [_q("q1"), _q("q2")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert len(kept) == 2

    async def test_rejects_when_oracle_fails(self) -> None:
        with patch(
            "agentic_autorag.examiner.exam_validator._call_completion",
            new=AsyncMock(return_value="completely wrong"),
        ):
            kept = await gate_oracle_pass(
                [_q("q1")],
                validator_model="test/model",
                judge_model=None,
                concurrency=1,
            )
        assert kept == []


class TestChunkContainsSourceFact:
    """Doc-id canonicalization: a duplicate alias counts as the canonical."""

    @staticmethod
    def _question() -> OpenEndedQuestion:
        # Question's source doc is the canonical paper.pdf; offsets cover
        # bytes 100..150 of doc_a (the canonical doc text).
        q = _q("q01", span_a="Span A text exact content here", span_b="Span B exact content here")
        return q.model_copy(
            update={
                "source_span_A_offset": (100, 150),
                "source_span_B_offset": (200, 250),
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
