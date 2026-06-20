"""Tests for the open-ended ExamAgent and its composition pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from agentic_autorag.config.models import ExaminerConfig, OpenEndedQuestion
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Neighborhood
from agentic_autorag.examiner.exam_agent import (
    CompositionResult,
    ExamAgent,
    _greedy_merge_chunks,
    dl_doc_to_chunk_text,
    self_containment_failure,
)

_MD_CONVERTER = DocumentConverter()


def _md_to_dl(markdown: str) -> DoclingDocument:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(markdown)
        path = Path(f.name)
    try:
        return _MD_CONVERTER.convert(str(path)).document
    finally:
        path.unlink()


def _chunk(chunk_id: str, doc_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text)


def _neighborhood(*chunks: ChunkRecord) -> Neighborhood:
    return Neighborhood(chunks=list(chunks))


def _agent() -> ExamAgent:
    return ExamAgent(
        config=ExaminerConfig(exam_size=10),
        examiner_model="test/model",
        corpus_description="test",
        concurrency=1,
    )


class TestDlDocToChunkText:
    def test_returns_concatenation_of_hybrid_chunker_outputs(self) -> None:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        from agentic_autorag.examiner.exam_agent import _WordCountTokenizer

        dl_doc = _md_to_dl("# Methods\n\nWe describe procedure A.\n\n## Results\n\nProcedure A yielded 42%.\n")
        max_words = 1000
        chunker = HybridChunker(tokenizer=_WordCountTokenizer(max_tokens=max_words))
        expected = "\n\n".join(c.text for c in chunker.chunk(dl_doc=dl_doc) if c.text.strip())

        assert dl_doc_to_chunk_text(dl_doc, max_chunk_words=max_words) == expected

    def test_every_chunk_is_a_verbatim_substring(self) -> None:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        from agentic_autorag.examiner.exam_agent import _WordCountTokenizer

        dl_doc = _md_to_dl(
            "# Section 1\n\nThe stroke screen detected cognitive deficits in 73 of 90 patients.\n\n"
            "# Section 2\n\nAt 6 months, improvement reached 41% (P > 0.05).\n"
        )
        max_words = 1000
        doc_text = dl_doc_to_chunk_text(dl_doc, max_chunk_words=max_words)
        chunker = HybridChunker(tokenizer=_WordCountTokenizer(max_tokens=max_words))
        for chunk in chunker.chunk(dl_doc=dl_doc):
            if chunk.text.strip():
                assert chunk.text in doc_text

    def test_does_not_html_escape_gt_lt(self) -> None:
        dl_doc = _md_to_dl("# Stats\n\nNo significant effect on mortality (P > 0.10).\n")
        doc_text = dl_doc_to_chunk_text(dl_doc, max_chunk_words=1000)
        assert "P > 0.10" in doc_text
        assert "&gt;" not in doc_text


class TestSelfContainment:
    def test_rejects_document_proxy(self) -> None:
        assert self_containment_failure("According to the document, what is X?") is not None

    def test_accepts_self_contained_question(self) -> None:
        assert self_containment_failure("Who founded the company that Acme acquired?") is None

    @pytest.mark.parametrize(
        "question",
        [
            "What did the authors conclude about the study?",
            "How many subjects participated in the trial?",
            "What were the results, after correction?",
        ],
    )
    def test_fires_on_bare_reference(self, question: str) -> None:
        assert self_containment_failure(question) is not None

    @pytest.mark.parametrize(
        "question",
        [
            "What were the findings of the Phase 3 trial of selumetinib in NF1?",
            "Did the results of the trial of compound X show benefit?",
            "Which analysis of carbon flux in tropical peatland used eddy-covariance?",
        ],
    )
    def test_does_not_fire_on_qualified_reference(self, question: str) -> None:
        assert self_containment_failure(question) is None

    @pytest.mark.parametrize(
        "question",
        [
            "What is the topic of Chunk 1?",
            "Compare the first chunk and the second chunk.",
            "What is described in the neighborhood?",
        ],
    )
    def test_fires_on_scaffolding_label_leak(self, question: str) -> None:
        assert self_containment_failure(question) is not None


class TestCallCompositionRendering:
    """The rendered user prompt enumerates chunks with [Chunk N] labels."""

    def test_chunk_blocks_use_positional_labels(self) -> None:
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_USER_PROMPT

        nh = _neighborhood(
            _chunk("docA::c0", "docA", "first chunk text"),
            _chunk("docB::c1", "docB", "second chunk text"),
            _chunk("docC::c2", "docC", "third chunk text"),
        )
        chunk_blocks = []
        for pos, c in enumerate(nh.chunks):
            chunk_blocks.append(f"[Chunk {pos}] (doc_id={c.doc_id}, chunk_id={c.chunk_id})\n{c.text}")
        rendered = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description="test",
            anchor_chunk_id=nh.anchor.chunk_id,
            chunk_blocks="\n\n".join(chunk_blocks),
        )
        assert "[Chunk 0]" in rendered
        assert "[Chunk 1]" in rendered
        assert "[Chunk 2]" in rendered
        assert "first chunk text" in rendered
        assert "third chunk text" in rendered


class TestParseNeighborhoodResponse:
    """The parser converts the LLM's JSON array into a list of CompositionResults."""

    def _agent_parse(self, raw: str, nh: Neighborhood) -> list[CompositionResult]:
        agent = _agent()
        return agent._parse_composition_neighborhood(raw, nh)

    def test_parses_valid_multi_question_response_cited_chunks(self) -> None:
        """New schema: cited_chunks is a list of {chunk_id, span} objects."""
        nh = _neighborhood(
            _chunk("a::0", "a", "text A"),
            _chunk("b::0", "b", "text B"),
        )
        raw = json.dumps(
            [
                {
                    "linkable": True,
                    "reasoning_type": "bridge",
                    "cited_chunks": [
                        {"chunk_id": 0, "span": "text A"},
                        {"chunk_id": 1, "span": "text B"},
                    ],
                    "question": "Q1?",
                    "canonical_answer": "A1",
                    "answer_variants": [],
                    "formula": None,
                    "formula_kind": None,
                },
                {
                    "linkable": True,
                    "reasoning_type": "extraction",
                    "cited_chunks": [{"chunk_id": 0, "span": "text A"}],
                    "question": "Q2?",
                    "canonical_answer": "A2",
                    "answer_variants": [],
                    "formula": None,
                    "formula_kind": None,
                },
            ]
        )
        results = self._agent_parse(raw, nh)
        assert len(results) == 2
        assert all(r.linkable for r in results)
        assert results[0].selected_chunk_ids == [0, 1]
        assert results[1].selected_chunk_ids == [0]
        assert results[0].source_spans == ["text A", "text B"]
        assert results[1].source_spans == ["text A"]

    def test_parses_legacy_flat_shape_as_fallback(self) -> None:
        """Transitional: cached composition_log.json may still carry the old
        flat shape (selected_chunk_ids + source_spans). Parser must still
        accept it until the fallback is removed."""
        nh = _neighborhood(
            _chunk("a::0", "a", "text A"),
            _chunk("b::0", "b", "text B"),
        )
        raw = json.dumps(
            [
                {
                    "linkable": True,
                    "reasoning_type": "bridge",
                    "selected_chunk_ids": [0, 1],
                    "source_spans": ["text A", "text B"],
                    "question": "Q?",
                    "canonical_answer": "A",
                },
            ]
        )
        results = self._agent_parse(raw, nh)
        assert len(results) == 1
        assert results[0].linkable
        assert results[0].selected_chunk_ids == [0, 1]
        assert results[0].source_spans == ["text A", "text B"]

    def test_cited_chunks_with_missing_span_skips_that_entry(self) -> None:
        """If an inner cited_chunks object lacks `span`, skip it but keep
        valid entries. The structural checker downstream will reject the
        composition if no valid citations remain."""
        nh = _neighborhood(_chunk("a::0", "a", "x"), _chunk("b::0", "b", "y"))
        raw = json.dumps(
            [
                {
                    "linkable": True,
                    "reasoning_type": "bridge",
                    "cited_chunks": [
                        {"chunk_id": 0, "span": "x"},
                        {"chunk_id": 1},  # missing span
                    ],
                    "question": "Q?",
                    "canonical_answer": "A",
                },
            ]
        )
        results = self._agent_parse(raw, nh)
        assert len(results) == 1
        # Only the valid object is kept.
        assert results[0].selected_chunk_ids == [0]
        assert results[0].source_spans == ["x"]

    def test_handles_refusal_entry(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        raw = json.dumps([{"linkable": False, "explanation": "all boilerplate"}])
        results = self._agent_parse(raw, nh)
        assert len(results) == 1
        assert results[0].linkable is False
        assert "boilerplate" in results[0].rejection_explanation

    def test_parse_failure_returns_single_error_result(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        results = self._agent_parse("not json", nh)
        assert len(results) == 1
        assert results[0].linkable is False
        assert results[0].reason == "parse_error"

    def test_rejects_long_canonical_answer(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        long_answer = " ".join(["word"] * 20)
        raw = json.dumps(
            [
                {
                    "linkable": True,
                    "reasoning_type": "definitional",
                    "cited_chunks": [{"chunk_id": 0, "span": "x"}],
                    "question": "Q?",
                    "canonical_answer": long_answer,
                }
            ]
        )
        results = self._agent_parse(raw, nh)
        assert len(results) == 1
        assert results[0].linkable is False
        assert results[0].reason == "answer_too_long"

    def test_empty_response_returns_empty_response_result(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        results = self._agent_parse("[]", nh)
        assert len(results) == 1
        assert results[0].linkable is False
        assert results[0].reason == "empty_response"

    def test_invalid_reasoning_type_defaults_to_bridge(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        raw = json.dumps(
            [
                {
                    "linkable": True,
                    "reasoning_type": "made_up_type",
                    "cited_chunks": [{"chunk_id": 0, "span": "x"}],
                    "question": "Q?",
                    "canonical_answer": "A",
                }
            ]
        )
        results = self._agent_parse(raw, nh)
        assert results[0].reasoning_type == "bridge"


class TestCompositionsToQuestions:
    """End-to-end of the in-process structural + self-contained + formula gates."""

    def _agent(self) -> ExamAgent:
        return _agent()

    def _result(
        self,
        nh: Neighborhood,
        *,
        question: str = "Who founded Beta Inc?",
        canonical: str = "Sarah Smith",
        selected: list[int] | None = None,
        spans: list[str] | None = None,
        reasoning_type: str = "bridge",
        formula: str | None = None,
        formula_kind: str | None = None,
    ) -> CompositionResult:
        if selected is None:
            selected = [0]
        if spans is None:
            spans = [nh.chunks[i].text for i in selected]
        return CompositionResult(
            neighborhood=nh,
            linkable=True,
            reasoning_type=reasoning_type,
            question=question,
            canonical_answer=canonical,
            selected_chunk_ids=selected,
            source_spans=spans,
            formula=formula,
            formula_kind=formula_kind,
        )

    def test_keeps_clean_question(self) -> None:
        nh = _neighborhood(
            _chunk("a::0", "a", "Beta Inc was founded by Sarah Smith in 1985."),
            _chunk("b::0", "b", "Acme acquired Beta Inc in 1998."),
        )
        r = self._result(nh, selected=[0, 1])
        kept = self._agent()._compositions_to_questions([r])
        assert len(kept) == 1
        q = kept[0]
        assert q.question == "Who founded Beta Inc?"
        assert q.source_chunk_ids == ["a::0", "b::0"]
        assert q.num_hops == 2

    def test_rejects_self_contained_violations(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "text"))
        r = self._result(nh, question="According to the document, who founded it?", spans=["text"])
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["self_contained"] == 1

    def test_rejects_uncited_chunk(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        # Cite position 5 — out of range for a 1-chunk neighborhood.
        r = self._result(nh, selected=[5], spans=["x"])
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["uncited_chunk"] == 1

    def test_rejects_empty_selected_chunk_ids(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        r = self._result(nh, selected=[], spans=[])
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["empty_selected_chunk_ids"] == 1

    def test_rejects_misaligned_spans(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"), _chunk("b::0", "b", "y"))
        r = self._result(nh, selected=[0, 1], spans=["x"])  # 1 span for 2 cited chunks
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["spans_misaligned"] == 1

    def test_rejects_empty_span(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        r = self._result(nh, selected=[0], spans=[""])
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["empty_span"] == 1

    def test_numeric_missing_formula_rejected(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "100 and 200"))
        r = self._result(
            nh,
            selected=[0],
            spans=["100 and 200"],
            reasoning_type="numeric_single",
            canonical="300",
        )
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["formula_missing"] == 1

    def test_numeric_formula_mismatch_rejected(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "100 and 200"))
        r = self._result(
            nh,
            selected=[0],
            spans=["100 and 200"],
            reasoning_type="numeric_single",
            canonical="500",  # 100 + 200 = 300, not 500
            formula="100 + 200",
            formula_kind="arithmetic",
        )
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["formula_mismatch"] == 1

    def test_numeric_clean_formula_accepted(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "100 and 200"))
        r = self._result(
            nh,
            selected=[0],
            spans=["100 and 200"],
            reasoning_type="numeric_single",
            canonical="300",
            formula="100 + 200",
            formula_kind="arithmetic",
        )
        kept = self._agent()._compositions_to_questions([r])
        assert len(kept) == 1
        assert kept[0].formula == "100 + 200"

    def test_llm_refusal_counts_in_rejections(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "x"))
        r = CompositionResult(
            neighborhood=nh,
            linkable=False,
            rejection_explanation="boilerplate only",
        )
        agent = self._agent()
        kept = agent._compositions_to_questions([r])
        assert kept == []
        assert agent.last_composition_rejections["llm_refused"] == 1

    def test_downstream_rejections_records_persisted(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "text"))
        r = self._result(nh, question="According to the document, what is X?", spans=["text"])
        agent = self._agent()
        agent._compositions_to_questions([r])
        assert len(agent.last_downstream_rejections) == 1
        assert agent.last_downstream_rejections[0]["reason"] == "self_contained"
        assert agent.last_downstream_rejections[0]["anchor_chunk_id"] == "a::0"

    def test_downstream_rejections_reset_between_calls(self) -> None:
        nh = _neighborhood(_chunk("a::0", "a", "text"))
        agent = self._agent()
        agent._compositions_to_questions(
            [self._result(nh, question="According to the document, what?", spans=["text"])]
        )
        assert len(agent.last_downstream_rejections) == 1
        # Second call should reset.
        agent._compositions_to_questions([])
        assert agent.last_downstream_rejections == []

    def test_yields_open_ended_question_with_correct_provenance(self) -> None:
        nh = _neighborhood(
            _chunk("docA::0", "docA", "Beta Inc was founded by Sarah Smith in 1985."),
            _chunk("docB::0", "docB", "Acme acquired Beta Inc in 1998."),
        )
        r = self._result(nh, selected=[0, 1])
        kept = self._agent()._compositions_to_questions([r])
        assert len(kept) == 1
        q = kept[0]
        assert isinstance(q, OpenEndedQuestion)
        assert q.source_chunk_ids == ["docA::0", "docB::0"]
        assert q.source_doc_ids == ["docA", "docB"]
        assert q.is_multi_doc is True
        assert q.num_hops == 2


class TestGreedyMergeChunks:
    def _c(self, chunk_id: str, doc_id: str, words: int) -> ChunkRecord:
        return _chunk(chunk_id=chunk_id, doc_id=doc_id, text=" ".join(["w"] * words))

    def test_combines_small_chunks_within_budget(self) -> None:
        chunks = [self._c(f"d::{i}", "d", 10) for i in range(5)]
        merged = _greedy_merge_chunks(chunks, max_words=100)
        assert len(merged) == 1
        assert len(merged[0].text.split()) == 50

    def test_splits_when_budget_exceeded(self) -> None:
        chunks = [self._c(f"d::{i}", "d", 60) for i in range(4)]
        merged = _greedy_merge_chunks(chunks, max_words=100)
        assert len(merged) == 4  # each chunk > half budget, can't pair

    def test_preserves_doc_boundaries(self) -> None:
        chunks = [
            self._c("a::0", "a", 30),
            self._c("a::1", "a", 30),
            self._c("b::0", "b", 30),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=200)
        assert len(merged) == 2
        assert merged[0].doc_id == "a"
        assert merged[1].doc_id == "b"

    def test_inherits_first_chunk_id_and_section(self) -> None:
        chunks = [
            ChunkRecord(chunk_id="a::0", doc_id="a", text="first", section=None),
            ChunkRecord(chunk_id="a::1", doc_id="a", text="second", section=None),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=100)
        assert merged[0].chunk_id == "a::0"

    def test_handles_oversized_input_chunk(self) -> None:
        chunks = [self._c("a::0", "a", 500)]
        merged = _greedy_merge_chunks(chunks, max_words=100)
        # Already over budget — left as-is.
        assert len(merged) == 1
        assert len(merged[0].text.split()) == 500

    def test_empty_input(self) -> None:
        assert _greedy_merge_chunks([], max_words=100) == []

    def test_joins_with_double_newline(self) -> None:
        chunks = [self._c("a::0", "a", 3), self._c("a::1", "a", 3)]
        merged = _greedy_merge_chunks(chunks, max_words=100)
        assert "\n\n" in merged[0].text
