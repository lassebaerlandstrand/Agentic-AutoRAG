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


class TestCompositionPromptShape:
    """Prompt-shape guards on the system prompt and the user template."""

    @property
    def system_prompt(self) -> str:
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        return COMPOSITION_BATCH_SYSTEM_PROMPT

    @property
    def user_prompt(self) -> str:
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_USER_PROMPT

        return COMPOSITION_BATCH_USER_PROMPT

    def test_opens_with_hardness_goal_then_rag_preamble(self) -> None:
        p = self.system_prompt
        # The first 1500 chars must establish the DIFFICULT-exam goal.
        head = p[:1500]
        assert "DIFFICULT" in head
        assert "RAG" in head
        # And the closed-book reader framing follows.
        assert "closed-book" in p
        assert "load-bearing" in p

    def test_advertises_full_taxonomy(self) -> None:
        for t in ("extraction", "definitional", "numeric_single", "inference", "bridge", "comparison", "numeric"):
            assert t in self.system_prompt

    def test_describes_neighborhood_input_format(self) -> None:
        assert "NEIGHBORHOOD" in self.system_prompt or "neighborhood" in self.system_prompt
        # New co-located schema replaces the old parallel-array shape.
        assert "cited_chunks" in self.system_prompt
        # And shows the [Chunk N] labelling convention in the user prompt.
        assert "[Chunk " in self.user_prompt or "[Chunk" in self.system_prompt

    def test_schema_uses_cited_chunks_co_located_shape(self) -> None:
        """source_spans/selected_chunk_ids must be replaced by the nested
        cited_chunks shape so the LLM can't drop spans independently."""
        text = self.system_prompt
        assert "cited_chunks" in text
        assert "chunk_id" in text and "span" in text
        # Old parallel-array shape must not appear in the schema description.
        # We allow incidental mentions ("source span" prose) but the structured
        # top-level field names should be gone.
        assert "selected_chunk_ids" not in text
        assert '"source_spans"' not in text

    def test_preamble_pushes_three_plus_hop_scan(self) -> None:
        """The 3+ hop scan instruction must land BEFORE the HARD RULES so
        the composer reads it as operational stance, not a preference."""
        text = self.system_prompt
        assert "3+" in text or "three" in text.lower()
        hop_idx = text.find("3+ load-bearing SPANS")
        hard_idx = text.find("HARD RULES")
        assert hop_idx != -1, "3+ hop instruction not found in preamble"
        assert hard_idx != -1, "HARD RULES section not found"
        assert hop_idx < hard_idx, "3+ hop instruction must appear before HARD RULES"

    def test_p1_includes_anti_keyword_guidance(self) -> None:
        """P1 must explicitly tell the composer to avoid copying proper nouns
        from cited chunks into the question text."""
        text = self.system_prompt
        # The substantive marker phrasing.
        assert "proper noun" in text.lower()
        assert "lexical" in text.lower() or "verbatim" in text.lower()

    def test_includes_hard_rules_section(self) -> None:
        for tag in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
            assert tag in self.system_prompt
        assert "HARD RULES" in self.system_prompt

    def test_includes_difficulty_preferences_section(self) -> None:
        for tag in ("P1", "P2", "P3", "P4", "P5"):
            assert tag in self.system_prompt
        assert "DIFFICULTY PREFERENCES" in self.system_prompt
        assert "NEVER refuse for failing" in self.system_prompt or "never grounds to refuse" in self.system_prompt

    def test_indirect_descriptor_preference_has_inline_example(self) -> None:
        # P1 — short worked inline example showing indirect vs direct framing.
        text = self.system_prompt
        assert "indirect" in text.lower()
        assert "Beta Inc" in text or "biotech" in text.lower()

    def test_p2_anchor_plus_paraphrase(self) -> None:
        """P2 now teaches 'anchor the question + paraphrase the anchor +
        indirect-describe the answer'. The previous 'let the chunks
        resolve any ambiguity' framing taught the composer to strip
        retrieval anchors entirely — questions like 'in what month did
        the cohort close?' have no corpus signal and are unanswerable
        regardless of RAG quality. The new P2 must (a) demand an anchor,
        (b) demand paraphrase of that anchor, (c) keep the answer entity
        indirect, and (d) present both contrastive failure modes."""
        text = self.system_prompt
        # Lead — direct rule statement.
        assert "Anchor the question" in text
        # Paraphrase requirement.
        assert "paraphrase the anchor" in text.lower()
        # Both failure modes must be shown as contrasts.
        assert "Under-anchored" in text
        assert "Over-disambiguated" in text
        # Verbatim-anchor "weaker" case calls out BM25 wins trivially.
        assert "BM25" in text
        # Smoking-gun phrases from prior iterations are gone.
        assert "combine multiple distinguishing attributes" not in text.lower()
        assert "Let the cited chunks resolve any ambiguity" not in text

    def test_p1_anti_stacking_guidance(self) -> None:
        """P1 absorbs the anti-stacking rule (one indirect descriptor,
        not three chained attributes). Verify the operational sentence
        appears."""
        text = self.system_prompt
        assert "One indirect descriptor is enough" in text or "one indirect descriptor is enough" in text.lower()
        # Word-count guideline appears.
        assert "25 words" in text or "30 words" in text

    def test_reasoning_field_requires_corpus_anchor(self) -> None:
        """The reasoning-field description previously told the composer
        the question's clues must 'uniquely identify one answer in the
        broader corpus' — that framing drove clue stacking. A subsequent
        pass replaced it with 'need NOT be globally unambiguous', which
        overcorrected and taught the composer to strip the retrieval
        anchor entirely. The current framing demands a corpus-distinctive
        anchor (paraphrased where possible) so retrieval has something
        to lock onto, while keeping the answer entity indirect."""
        text = self.system_prompt
        # Old smoking-gun (stacking incentive) stays gone.
        assert "uniquely identify one answer in the broader corpus" not in text
        # Mid-iteration overcorrection ("need NOT be globally unambiguous")
        # stays gone — that wording stripped anchors.
        assert "need NOT be globally unambiguous" not in text
        # Current framing requires an anchor and mentions paraphrase.
        assert "corpus-distinctive anchor" in text
        assert "paraphrased" in text.lower()

    def test_p1_distinguishes_anchor_from_answer_entity(self) -> None:
        """P1 must explicitly separate the retrieval anchor (a corpus-
        distinctive identifier that SHOULD be in the question) from the
        answer entity (described indirectly). Without this distinction,
        the composer reads 'don't copy distinctive proper nouns' as
        'strip every proper noun' — which kills the retrieval anchor."""
        text = self.system_prompt
        assert "two entity roles" in text.lower()
        assert "retrieval anchor" in text.lower()
        assert "answer entity" in text.lower()
        # The contrast — anchor SHOULD be in the question; without it,
        # retrieval has no signal.
        assert "retrieval has no signal" in text or "no signal" in text.lower()

    def test_strong_not_templates_disclaimer(self) -> None:
        # LLMs over-copy examples; the disclaimer must be loud.
        text = self.system_prompt
        assert "NOT TEMPLATES" in text or "NOT A TEMPLATE" in text
        # And it must instruct against surface-form copying.
        assert "Do NOT copy" in text or "do not copy" in text.lower()
        # The strengthened ban explicitly calls out the templates the
        # composer was copying (the "Between X and Y, which has more Z?"
        # construction) — guards against the templating regression.
        assert "Vary phrasing" in text or "vary phrasing" in text.lower()
        assert "Between" in text and "Of the" in text  # alternative shape examples appear

    def test_disallows_day_precision_long_arithmetic(self) -> None:
        text = self.system_prompt
        assert "365" in text  # mentions the multiply-by-365 anti-pattern
        assert "unit" in text.lower()

    def test_no_upper_cap_on_questions(self) -> None:
        text = self.system_prompt
        assert "no upper cap" in text.lower() or "NO UPPER CAP" in text

    def test_no_mention_of_downstream_filters(self) -> None:
        """Composer should not be told about probe selector / trivial cap / etc."""
        text = self.system_prompt.lower()
        for forbidden in ("probe selector", "trivial cap", "kendall", "tier3", "discrimination scorer"):
            assert forbidden not in text

    def test_worked_example_questions_are_short(self) -> None:
        """Each worked example's ``question:`` line must be <= 25 words.
        The composer mimics example sentence structure AND length; the
        cap exists to catch true descriptor stacking (>30 words). 25 is
        the natural ceiling for anchored + paraphrased questions —
        paraphrased anchors ('treatment-resistant hypertension') run
        2-4 words longer than verbatim ones, and the lean question
        guideline (P1) targets ≤ 25 words."""
        import re

        text = self.system_prompt
        question_lines = re.findall(r'question:\s*"([^"]+)"', text)
        assert len(question_lines) >= 5, f"expected at least 5 example questions, found {len(question_lines)}"
        for q in question_lines:
            word_count = len(q.split())
            assert word_count <= 25, f"example question is {word_count} words (>25): {q!r}"

    def test_examples_have_no_diagnostic_note_lines(self) -> None:
        """Earlier iterations attached 'Note: 15 words. ...' meta-commentary
        lines to several worked examples. Those notes were scaffolding
        ABOUT the example rather than letting the example teach — the
        reasoning field already plays the per-example role, and extra
        prose dilutes attention from the actual rules (P1/P2). All such
        lines must be gone."""
        text = self.system_prompt
        # The note pattern: lines beginning with two-space-indented "Note:"
        # that appeared inside example blocks.
        assert "\n  Note:" not in text, "diagnostic 'Note:' lines must not appear in examples"

    def test_example_10_is_removed(self) -> None:
        """Example 10 (Italian Claudios) demonstrated a 'corpus-rare
        property as anchor' pattern that's fragile — it works only on
        corpora that genuinely contain a small specific set matching
        the property. On dense corpora the composer misapplies it
        ('the two cohorts mentioned') and produces ambiguous questions.
        Example dropped; the lesson it tried to teach is covered by
        Examples 1 and 5 (anchor + indirect descriptor)."""
        text = self.system_prompt
        assert "Example 10" not in text
        assert "Italian Claudio" not in text

    def test_inference_no_longer_includes_calendar_date(self) -> None:
        """Calendar-date arithmetic was a listed inference case but
        tests the LLM's mental math, not retrieval — bad RAG
        discriminator. Both the inference bullet's cases and Example 9
        (previously a calendar-date inference) must be free of it."""
        text = self.system_prompt
        # The old case description and the vamorolone calendar-date example
        # must both be gone.
        assert "temporal arithmetic producing a calendar date" not in text
        assert "vamorolone" not in text
        assert "September 2019" not in text  # was the old Example 9 canonical answer

    def test_inference_allows_multi_hop_with_disambiguation(self) -> None:
        """Inference should be usable single- OR multi-hop, with explicit
        disambiguation against bridge / comparison / numeric so the
        composer doesn't grab it as a catch-all."""
        text = self.system_prompt
        # New subsection header that opens up multi-hop scope.
        assert "Single- or multi-hop types:" in text
        # The three allowed cases survive.
        assert "causal chain" in text.lower()
        assert "implicit-referent" in text.lower()
        assert "qualitative direction" in text.lower()
        # Disambiguation: composer told to prefer bridge/comparison/numeric
        # when those fit, and use inference only as the fallback.
        assert "Use ``inference`` only when none of those fits." in text

    def test_example_9_is_qualitative_direction_inference(self) -> None:
        """Example 9 must now demonstrate qualitative-direction inference
        across chunks (one chunk = baseline, another = follow-up,
        question asks the direction). Anchors the new multi-hop scope."""
        text = self.system_prompt
        # Header signals multi-hop qualitative-direction inference.
        assert "qualitative direction from quantitative facts" in text
        # The new chunks + question are present.
        assert "Greenland summit" in text
        assert "grow or shrink" in text
        assert "+0.4°C" in text and "+2.1°C" in text

    def test_user_prompt_opens_with_hardness_reminder(self) -> None:
        assert "hardest" in self.user_prompt.lower()
        assert "weak and strong" in self.user_prompt.lower() or "widen the gap" in self.user_prompt.lower()

    def test_user_prompt_references_cited_chunks_schema(self) -> None:
        assert "cited_chunks" in self.user_prompt
        assert "chunk_id" in self.user_prompt

    def test_user_prompt_reminds_about_three_plus_hop(self) -> None:
        text = self.user_prompt.lower()
        assert "3+" in text or "three" in text or "deepest" in text


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
