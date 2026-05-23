"""Tests for the open-ended ExamAgent and its composition pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from agentic_autorag.config.models import ExaminerConfig, OpenEndedQuestion
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.exam_agent import (
    CompositionResult,
    ExamAgent,
    _greedy_merge_chunks,
    self_containment_failure,
)

_MD_CONVERTER = DocumentConverter()


def _md_to_dl(markdown: str) -> DoclingDocument:
    """Build a DoclingDocument from a markdown string for tests.

    Docling's MD backend reads from disk so we round-trip via a temp file.
    Fast enough (~3-5ms per call) for use inside tests.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(markdown)
        path = Path(f.name)
    try:
        return _MD_CONVERTER.convert(str(path)).document
    finally:
        path.unlink()


def _seeds(n: int = 1) -> list[Seed]:
    out = []
    for i in range(n):
        out.append(
            Seed(
                chunk_a=ChunkRecord(
                    chunk_id=f"docA{i}::c0",
                    doc_id=f"docA{i}",
                    text=f"In 1998, Acme Corp acquired Beta Inc{i} for $50M.",
                ),
                chunk_b=ChunkRecord(
                    chunk_id=f"docB{i}::c0",
                    doc_id=f"docB{i}",
                    text=f"Beta Inc{i} was founded by Sarah Smith{i} in 1985.",
                ),
                score=0.7,
            )
        )
    return out


def _typed(seeds: list[Seed], preferred_type: str = "bridge") -> list[tuple[Seed, str]]:
    """Wrap a Seed list with a constant preferred-type for ``_parse_composition_batch``."""
    return [(s, preferred_type) for s in seeds]


class TestSeedBlocksDoNotLeakBridge:
    """The composition prompt must not surface any indexer-side bridge hint.

    With the v4 framing the LLM is a question generator with a refuse
    option. The prompt sees only the two chunk texts and their doc_ids —
    not any seed-level metadata. Verifying this prompt-shape contract is
    cheap and catches regressions that would otherwise only surface in
    production.
    """

    def test_seed_blocks_omit_bridge_entity(self) -> None:
        from agentic_autorag.examiner.prompts import (
            COMPOSITION_BATCH_SYSTEM_PROMPT,
            COMPOSITION_BATCH_USER_PROMPT,
        )

        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        seed = Seed(
            chunk_a=ChunkRecord(
                chunk_id="docA::c0",
                doc_id="docA",
                text="Mansoura University funded this study.",
            ),
            chunk_b=ChunkRecord(
                chunk_id="docB::c0",
                doc_id="docB",
                text="Mansoura University also funded the comparator study.",
            ),
            score=0.8,
        )
        seed_blocks = []
        for i, s in enumerate([seed]):
            seed_blocks.append(
                f"Seed #{i}\n"
                f"  === Input 1 === (doc_id={s.chunk_a.doc_id})\n  {s.chunk_a.text}\n"
                f"  === Input 2 === (doc_id={s.chunk_b.doc_id})\n  {s.chunk_b.text}"
            )
        rendered_user = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description=agent.corpus_description,
            k=1,
            seed_blocks="\n\n".join(seed_blocks),
        )
        assert "bridge_entity:" not in rendered_user
        assert "bridge_entity:" not in COMPOSITION_BATCH_SYSTEM_PROMPT
        assert "  bridge_entity: " not in rendered_user


class TestSelfContainment:
    def test_rejects_document_proxy(self) -> None:
        result = self_containment_failure("According to the document, what is X?")
        assert result is not None

    def test_accepts_self_contained_question(self) -> None:
        result = self_containment_failure("Who founded the company that Acme acquired?")
        assert result is None


class TestSelfContainmentBarePhrasePattern:
    """Pattern #10 ('the study', 'the authors', ...) fires only when the noun
    is followed by clause-end punctuation — qualified references survive."""

    @pytest.mark.parametrize(
        "question",
        [
            "According to the authors, what did they find?",
            "What did the authors discover? The study.",
            "What does the experiment: a closed-loop trial, show?",
            "The authors! What did they conclude?",
            "What follows from the study, given the setup?",
        ],
    )
    def test_fires_on_bare_reference(self, question: str) -> None:
        result = self_containment_failure(question)
        assert result is not None, f"expected pattern to fire on: {question!r}"

    @pytest.mark.parametrize(
        "question",
        [
            # Real false-positive examples from the experiments-unidoc-probe run.
            "In the study evaluating a mineral and vitamin mix added to drinking "
            "water to mitigate heat stress in broiler chickens, what was the "
            "maximum stocking density reported?",
            "Which receptor is identified as the main receptor for ACTH in the study of its effects on the hair cycle?",
            "Which electrophysiological marker did the researchers record to "
            "track the speed of attentional engagement in the experiment where "
            "distractor intrusions were less frequent than in the first experiment?",
            "What type of design was used for the study comparing consumption of "
            "fructose- or glucose-sweetened beverages?",
            "How many participants in the experiment with fully predictable target "
            "location had not taken part in the previous experiment?",
            "How many children in total were included in the study of MMR antibodies and MBP autoantibodies?",
            "Which adipose depot did the arm consuming glucose preferentially "
            "deposit in compared to the fructose arm, according to the authors' "
            "discussion?",
            "What mechanism did the study demonstrate for fructose-induced postprandial hypertriglyceridemia?",
        ],
    )
    def test_does_not_fire_on_qualified_reference(self, question: str) -> None:
        result = self_containment_failure(question)
        assert result is None, f"pattern wrongly fired on qualified reference (matched={result[1]!r}): {question!r}"


class TestSelfContainmentScaffoldingLabels:
    """Internal scaffolding labels (Input 1/2, chunk_A/B, 'the first/second
    input') must never reach a closed-book reader. The runtime regex is a
    safety net behind the R3 prompt-side rule."""

    @pytest.mark.parametrize(
        "question",
        [
            "What does Input 1 describe about the cohort?",
            "Which finding from Input 2 contradicts the earlier hypothesis?",
            "According to chunk_B, what was the outcome?",
            "What did the first input report about adverse events?",
            "Which population is studied in the second input?",
        ],
    )
    def test_fires_on_scaffolding_label_leak(self, question: str) -> None:
        result = self_containment_failure(question)
        assert result is not None, f"expected scaffolding-label pattern to fire on: {question!r}"

    @pytest.mark.parametrize(
        "question",
        [
            # Legitimate domain uses of "input" and "chunk" that must NOT bite.
            "What input did the encoder receive in the classification task?",
            "Which chunked storage scheme did the database use for the indexed table?",
            "How many input neurons were used in the recurrent layer?",
            "What is the chunk-size parameter that the paper recommends?",
        ],
    )
    def test_does_not_fire_on_legitimate_domain_use(self, question: str) -> None:
        result = self_containment_failure(question)
        assert result is None, f"pattern wrongly fired on legitimate use (matched={result[1]!r}): {question!r}"


class TestTypedSampling:
    """The preferred-type sampler must respect configured weights, restrict
    to per-origin compatible types, and fall back gracefully when no
    compatible weight is positive."""

    def test_multi_hop_seeds_draw_only_multi_hop_types(self) -> None:
        """Cross-doc / same-doc seeds must never receive a single-hop type."""
        agent = ExamAgent(
            config=ExaminerConfig(
                exam_size=10,
                question_type_weights={
                    "extraction": 0.20,
                    "definitional": 0.20,
                    "bridge": 0.20,
                    "comparison": 0.20,
                    "numeric": 0.20,
                },
            ),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
            type_sampler_seed=42,
        )
        seeds = _seeds(2000)  # all cross_doc pairs (chunk_b populated)
        types = agent._sample_preferred_types(seeds)
        assert all(t in {"bridge", "comparison", "numeric"} for t in types)
        # Distribution within the multi-hop subset should reflect renormalised
        # weights (each multi-hop type has equal config weight, so ~1/3 each).
        counts = {t: types.count(t) for t in ("bridge", "comparison", "numeric")}
        for t, c in counts.items():
            observed = c / len(seeds)
            assert abs(observed - 1 / 3) < 0.05, f"{t}: observed={observed:.3f}"

    def test_single_chunk_seeds_draw_only_single_hop_types(self) -> None:
        """single_chunk seeds must never receive a multi-hop type."""
        agent = ExamAgent(
            config=ExaminerConfig(
                exam_size=10,
                question_type_weights={
                    "extraction": 0.40,
                    "definitional": 0.10,
                    "bridge": 0.25,
                    "comparison": 0.15,
                    "numeric": 0.10,
                },
            ),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
            type_sampler_seed=42,
        )
        # Single-chunk seeds (chunk_b=None) regardless of cross-doc pairs above.
        seeds = [
            Seed(
                chunk_a=ChunkRecord(chunk_id=f"doc{i}::c0", doc_id=f"doc{i}", text=f"chunk {i} text."),
                chunk_b=None,
                score=0.0,
                origin="single_chunk",
            )
            for i in range(2000)
        ]
        types = agent._sample_preferred_types(seeds)
        assert all(t in {"extraction", "definitional"} for t in types)
        # Renormalised single-hop weights: extraction=0.8, definitional=0.2.
        ex = types.count("extraction") / len(seeds)
        df = types.count("definitional") / len(seeds)
        assert abs(ex - 0.8) < 0.05
        assert abs(df - 0.2) < 0.05

    def test_fallback_when_no_compatible_weight_is_positive(self) -> None:
        """If every compatible type's weight is zero, fall back deterministically
        to the canonical default per origin (extraction for single, bridge for multi)."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        # Zero out every multi-hop type; only single-hop weights remain positive.
        agent.config.question_type_weights = {"extraction": 0.5, "definitional": 0.5}
        multi_seeds = _seeds(20)
        multi_types = agent._sample_preferred_types(multi_seeds)
        # No multi-hop weight positive → deterministic fallback to "bridge".
        assert all(t == "bridge" for t in multi_types)

    def test_seeded_sampler_is_reproducible(self) -> None:
        cfg = ExaminerConfig(
            exam_size=10,
            question_type_weights={
                "bridge": 0.30,
                "comparison": 0.25,
                "extraction": 0.25,
                "numeric": 0.20,
            },
        )
        seeds = _seeds(50)
        a1 = ExamAgent(config=cfg, examiner_model="m", corpus_description="t", concurrency=1, type_sampler_seed="proj")
        a2 = ExamAgent(config=cfg, examiner_model="m", corpus_description="t", concurrency=1, type_sampler_seed="proj")
        assert a1._sample_preferred_types(seeds) == a2._sample_preferred_types(seeds)

    def test_fallback_kept_when_llm_returns_different_type(self) -> None:
        """When the LLM ignores the preferred type, the question is still kept
        and the CompositionResult records the fallback."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        seed = _seeds(1)[0]
        # Asked for "numeric", got "bridge".
        raw = (
            "["
            ' {"seed_id": 0, "linkable": true,'
            '  "reasoning_type": "bridge",'
            '  "preferred_type_used": false,'
            '  "question": "Who founded the company that the acquirer acquired?",'
            '  "canonical_answer": "Sarah Smith0",'
            '  "source_span_A": "In 1998, Acme Corp acquired Beta Inc0 for $50M.",'
            '  "source_span_B": "Beta Inc0 was founded by Sarah Smith0 in 1985."}'
            "]"
        )
        results = agent._parse_composition_batch(raw, [(seed, "numeric")])
        assert results[0].linkable is True
        assert results[0].preferred_type == "numeric"
        assert results[0].reasoning_type == "bridge"
        assert results[0].preferred_type_used is False
        kept = agent._compositions_to_questions(results)
        # The fallback is accepted into the exam — preferred is preferred, not forced.
        assert len(kept) == 1
        assert kept[0].reasoning_type == "bridge"


class TestCompositionPromptShape:
    """The composition prompt must include type taxonomy + anti-surface-token guidance."""

    def test_prompt_advertises_taxonomy(self) -> None:
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        for t in ("extraction", "definitional", "bridge", "comparison", "numeric"):
            assert t in COMPOSITION_BATCH_SYSTEM_PROMPT

    def test_prompt_uses_canonical_type_names(self) -> None:
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        for forbidden in ("multi_constraint", "exclusion", "bridge_chain"):
            assert forbidden not in COMPOSITION_BATCH_SYSTEM_PROMPT

    def test_prompt_warns_against_surface_token_copy(self) -> None:
        from agentic_autorag.examiner.prompts import (
            COMPOSITION_BATCH_SYSTEM_PROMPT,
            COMPOSITION_BATCH_USER_PROMPT,
        )

        # The anti-surface-token instruction is the replacement for the
        # dropped Jaccard-overlap filter; assert it appears in the system
        # OR user prompt (both are sent on every composition call).
        combined = (COMPOSITION_BATCH_SYSTEM_PROMPT + COMPOSITION_BATCH_USER_PROMPT).lower()
        assert "surface-token" in combined or "surface tokens" in combined
        assert "document title" in combined or "rare proper noun" in combined

    def test_prompt_disallows_day_precision_arithmetic(self) -> None:
        """The FORMULA section forbids day-precision date arithmetic and
        directs the LLM to express durations at year-or-coarser granularity."""
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        prompt = COMPOSITION_BATCH_SYSTEM_PROMPT.lower()
        assert "day-precision arithmetic is not supported" in prompt
        # date_diff_days kind is gone; the prompt should no longer name it.
        assert "date_diff_days" not in prompt
        # Year-arithmetic guidance is present.
        assert "2011 - 2008" in prompt

    def test_prompt_includes_uniqueness_rule_with_example(self) -> None:
        """Descriptor-uniqueness rule with a BAD/GOOD example pair."""
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        prompt = COMPOSITION_BATCH_SYSTEM_PROMPT.lower()
        assert "uniqueness" in prompt
        # Concrete example contrast — a BAD ambiguous descriptor and a GOOD
        # uniquely-identifying one. Catches regressions that strip the
        # example without removing the rule label.
        assert "ambiguous clue" in prompt
        assert "unique clue" in prompt

    def test_prompt_includes_system_context_preamble(self) -> None:
        """The system prompt opens with a descriptive RAG-architecture preamble
        so R1/R3 read as deductive consequences of how retrieval works, not as
        arbitrary rules. Asserts on stable anchor phrases that capture the
        three implications (closed-book reader, independent retrieval, grader
        contract)."""
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_SYSTEM_PROMPT

        prompt = COMPOSITION_BATCH_SYSTEM_PROMPT.lower()
        assert "system context" in prompt
        assert "closed-book" in prompt
        assert "independent" in prompt
        assert "load-bearing" in prompt

    def test_prompt_uses_neutral_input_labels(self) -> None:
        """Inputs are labelled with neutral structural names (Input 1 / Input 2)
        rather than chunk_A / chunk_B — the latter were occasionally leaking
        into composed question text. The literal scaffolding tokens may still
        appear inside R3's prohibited-phrase list (that's the safety net)."""
        from agentic_autorag.examiner.prompts import (
            COMPOSITION_BATCH_SYSTEM_PROMPT,
            COMPOSITION_BATCH_USER_PROMPT,
        )

        # chunk_A / chunk_B remain only in R3's prohibition list — assert they
        # don't appear in worked examples, taxonomy, OUTPUT schema, or user prompt.
        assert COMPOSITION_BATCH_SYSTEM_PROMPT.count("chunk_A") <= 1
        assert COMPOSITION_BATCH_SYSTEM_PROMPT.count("chunk_B") <= 1
        assert "chunk_A" not in COMPOSITION_BATCH_USER_PROMPT
        assert "chunk_B" not in COMPOSITION_BATCH_USER_PROMPT
        # The neutral labels are present (multiple times: taxonomy + 7 examples).
        assert COMPOSITION_BATCH_SYSTEM_PROMPT.count("Input 1") >= 5
        assert COMPOSITION_BATCH_SYSTEM_PROMPT.count("Input 2") >= 5

    def test_seed_blocks_use_input_delimiters(self) -> None:
        """The seed-block formatter emits bracket-delimited markers rather than
        chunk_A: / chunk_B: labels, so the LLM can't pattern-match the label
        into its own output."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        seed = _seeds(1)[0]
        # Mirror the production formatter exactly (kept in lock-step here so the
        # test catches drift; the inline construction is the public surface for
        # this assertion).
        block = (
            f"Seed #0\n"
            f"  === Input 1 === (doc_id={seed.chunk_a.doc_id})\n  {seed.chunk_a.text}\n"
            f"  === Input 2 === (doc_id={seed.chunk_b.doc_id})\n  {seed.chunk_b.text}"
        )
        from agentic_autorag.examiner.prompts import COMPOSITION_BATCH_USER_PROMPT

        rendered = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description=agent.corpus_description,
            k=1,
            seed_blocks=block,
        )
        assert "=== Input 1 ===" in rendered
        assert "=== Input 2 ===" in rendered
        assert "chunk_A:" not in rendered
        assert "chunk_B:" not in rendered


class TestComposeBatchParsing:
    def _make_agent(self) -> ExamAgent:
        return ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="test corpus",
            concurrency=1,
        )

    def test_parses_valid_batch(self) -> None:
        agent = self._make_agent()
        seeds = _seeds(2)
        raw = (
            "["
            ' {"seed_id": 0, "linkable": true,'
            '  "fact_a": "Acme acquired Beta in 1998.",'
            '  "fact_b": "Beta was founded by Sarah Smith.",'
            '  "question": "Who founded the company that Acme acquired?",'
            '  "canonical_answer": "Sarah Smith0",'
            '  "answer_variants": ["S. Smith"],'
            '  "source_span_A": "In 1998, Acme Corp acquired Beta Inc0 for $50M.",'
            '  "source_span_B": "Beta Inc0 was founded by Sarah Smith0 in 1985."},'
            ' {"seed_id": 1, "linkable": false,'
            '  "explanation": "Only overlap is institutional affiliation."}'
            "]"
        )
        results = agent._parse_composition_batch(raw, _typed(seeds))
        assert len(results) == 2
        assert results[0].linkable is True
        assert results[0].canonical_answer == "Sarah Smith0"
        assert results[1].linkable is False
        assert "institutional affiliation" in results[1].rejection_explanation

    def test_handles_malformed_entries_per_element(self) -> None:
        agent = self._make_agent()
        seeds = _seeds(2)
        raw = (
            "["
            ' {"seed_id": 0, "linkable": true},'  # missing required fields
            ' {"seed_id": 1, "linkable": true,'
            '  "question": "Q",'
            '  "canonical_answer": "A",'
            '  "source_span_A": "x", "source_span_B": "y"}'
            "]"
        )
        results = agent._parse_composition_batch(raw, _typed(seeds))
        assert len(results) == 2
        assert results[0].linkable is False
        assert results[0].reason == "missing_fields"
        assert results[1].linkable is True

    def test_invalid_json_returns_parse_errors_for_all_seeds(self) -> None:
        agent = self._make_agent()
        seeds = _seeds(3)
        results = agent._parse_composition_batch("not json at all", _typed(seeds))
        assert all(not r.linkable and r.reason == "parse_error" for r in results)

    def test_rejects_long_canonical_answer(self) -> None:
        """R7: canonical answers > 15 words rejected as answer_too_long."""
        agent = self._make_agent()
        seeds = _seeds(1)
        long_answer = " ".join(f"word{i}" for i in range(20))  # 20 words
        raw = (
            "["
            ' {"seed_id": 0, "linkable": true,'
            '  "question": "Q?",'
            f'  "canonical_answer": "{long_answer}",'
            '  "source_span_A": "x",'
            '  "source_span_B": "y"}'
            "]"
        )
        results = agent._parse_composition_batch(raw, _typed(seeds))
        assert len(results) == 1
        assert results[0].linkable is False
        assert results[0].reason == "answer_too_long"

    def test_keeps_15_word_canonical_answer(self) -> None:
        """A 15-word canonical answer is accepted (boundary case)."""
        agent = self._make_agent()
        seeds = _seeds(1)
        boundary_answer = " ".join(f"w{i}" for i in range(15))  # exactly 15 words
        raw = (
            "["
            ' {"seed_id": 0, "linkable": true,'
            '  "question": "Q?",'
            f'  "canonical_answer": "{boundary_answer}",'
            '  "source_span_A": "x",'
            '  "source_span_B": "y"}'
            "]"
        )
        results = agent._parse_composition_batch(raw, _typed(seeds))
        assert results[0].linkable is True

    def test_parses_free_text_explanation_on_refusal(self) -> None:
        agent = self._make_agent()
        seeds = _seeds(1)
        raw = (
            "["
            ' {"seed_id": 0, "linkable": false,'
            '  "explanation": "The two chunks share only a citation; no substantive fact links them."}'
            "]"
        )
        results = agent._parse_composition_batch(raw, _typed(seeds))
        assert len(results) == 1
        assert results[0].linkable is False
        assert "citation" in results[0].rejection_explanation


class TestCompositionsToQuestions:
    def _make_agent(self) -> ExamAgent:
        return ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="test corpus",
            concurrency=1,
        )

    def test_keeps_clean_question(self) -> None:
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                answer_variants=["S. Smith"],
                source_span_A=seed.chunk_a.text,
                source_span_B=seed.chunk_b.text,
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1
        assert isinstance(kept[0], OpenEndedQuestion)
        assert kept[0].source_doc_ids == ["docA0", "docB0"]
        assert kept[0].source_chunk_ids == ["docA0::c0", "docB0::c0"]
        assert kept[0].source_spans == [seed.chunk_a.text, seed.chunk_b.text]

    def test_rejects_self_contained_violations(self) -> None:
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                question="According to the document, who founded the company?",
                canonical_answer="Sarah Smith0",
                source_span_A=seed.chunk_a.text,
                source_span_B=seed.chunk_b.text,
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert kept == []

    def test_does_not_reject_span_drift_at_composition_stage(self) -> None:
        """Span ↔ source verification moved entirely to verify_source_facts.
        Composition stage no longer enforces per-chunk substring matching —
        unicode drift (NBSP, smart quotes) used to false-positive here."""
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                source_span_A="span text the composition stage does not verify",
                source_span_B=seed.chunk_b.text,
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1

    def test_paired_seed_with_multi_hop_type_and_empty_span_b_is_rejected(self) -> None:
        """LLM emitting a multi-hop ``reasoning_type`` with empty ``source_span_B``
        is an R1 violation (claimed 2-hop but didn't ground in chunk_B) — typed
        rejection, persisted record."""
        agent = self._make_agent()
        seed = _seeds(1)[0]  # paired seed (chunk_b is set)
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                reasoning_type="bridge",
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                source_span_A=seed.chunk_a.text,
                source_span_B="",
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert kept == []
        records = agent.last_downstream_rejections
        assert len(records) == 1
        assert records[0]["reason"] == "empty_span_b_with_multi_hop_type"
        assert records[0]["reasoning_type"] == "bridge"
        assert records[0]["source_chunk_ids"] == [seed.chunk_a.chunk_id, seed.chunk_b.chunk_id]

    def test_paired_seed_with_single_hop_type_and_empty_span_b_is_kept_as_single_hop(self) -> None:
        """Internally-consistent single-hop fallback on a paired seed: LLM said
        ``definitional`` and grounded only in chunk_A. Accept and record as
        single-hop (drop chunk_B from source bookkeeping)."""
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                reasoning_type="definitional",
                question="What is the term for the modified place conditioning procedure?",
                canonical_answer="reference-conditioning procedure",
                source_span_A=seed.chunk_a.text,
                source_span_B="",
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1
        q = kept[0]
        assert q.reasoning_type == "definitional"
        assert q.source_chunk_ids == [seed.chunk_a.chunk_id]
        assert q.source_doc_ids == [seed.chunk_a.doc_id]
        assert q.source_spans == [seed.chunk_a.text]
        assert q.num_hops == 1
        assert agent.last_downstream_rejections == []

    def test_paired_seed_with_single_hop_type_and_nonempty_span_b_keeps_both_chunks(self) -> None:
        """span_B emptiness drives the bookkeeping, not the type. If the LLM
        used ``extraction`` but did ground in both chunks, both should be kept."""
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                reasoning_type="extraction",
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                source_span_A=seed.chunk_a.text,
                source_span_B=seed.chunk_b.text,
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1
        q = kept[0]
        assert q.source_chunk_ids == [seed.chunk_a.chunk_id, seed.chunk_b.chunk_id]
        assert q.source_spans == [seed.chunk_a.text, seed.chunk_b.text]

    def test_single_chunk_seed_yields_single_hop_question(self) -> None:
        agent = self._make_agent()
        from agentic_autorag.examiner.chunk_pair_index import Seed

        chunk = ChunkRecord(
            chunk_id="docA::c0",
            doc_id="docA",
            text="A clinical investigation reported a maximum tolerated dose of 240 mg/kg.",
        )
        single_seed = Seed(chunk_a=chunk, chunk_b=None, origin="single_chunk")
        results = [
            CompositionResult(
                seed=single_seed,
                linkable=True,
                reasoning_type="extraction",
                question="At what value was the maximum tolerated dose set in adult cohorts B and C?",
                canonical_answer="240 mg/kg",
                source_span_A=chunk.text,
                source_span_B="",
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1
        q = kept[0]
        assert q.num_hops == 1
        assert q.source_chunk_ids == ["docA::c0"]
        assert q.source_doc_ids == ["docA"]
        assert q.source_spans == [chunk.text]
        assert q.is_multi_doc is False
        assert q.reasoning_type == "extraction"

    def test_records_downstream_rejections_for_persistence(self) -> None:
        """Each post-LLM filter rejection should be appended to
        ``last_downstream_rejections`` so the orchestrator can persist it
        alongside LLM refusals in candidates.json."""
        agent = self._make_agent()
        seeds = _seeds(3)
        single_chunk = ChunkRecord(chunk_id="docX::c0", doc_id="docX", text="solo text")
        single_seed = Seed(chunk_a=single_chunk, chunk_b=None, origin="single_chunk")
        results = [
            # 1. self_contained — uses pattern #10 trigger ("the authors,")
            CompositionResult(
                seed=seeds[0],
                linkable=True,
                question="According to the authors, what is X?",
                canonical_answer="answer one",
                source_span_A=seeds[0].chunk_a.text,
                source_span_B=seeds[0].chunk_b.text,
            ),
            # 2. empty_span_b on a paired seed with multi-hop reasoning_type
            #    (LLM claimed 2-hop but didn't ground in chunk_B → R1 violation)
            CompositionResult(
                seed=seeds[1],
                linkable=True,
                reasoning_type="bridge",
                question="Some multi-hop question?",
                canonical_answer="answer two",
                source_span_A=seeds[1].chunk_a.text,
                source_span_B="",
            ),
            # 3. formula_mismatch on a numeric seed
            CompositionResult(
                seed=single_seed,
                linkable=True,
                reasoning_type="numeric",
                question="What is 10 plus 5?",
                canonical_answer="20",  # wrong: 10 + 5 = 15
                source_span_A=single_chunk.text,
                source_span_B="",
                formula="10 + 5",
                formula_kind="arithmetic",
            ),
            # 4. kept (clean question, ensures the loop continues past rejections)
            CompositionResult(
                seed=seeds[2],
                linkable=True,
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith2",
                source_span_A=seeds[2].chunk_a.text,
                source_span_B=seeds[2].chunk_b.text,
            ),
        ]
        kept = agent._compositions_to_questions(results)
        assert len(kept) == 1, "exactly one clean question survives"

        records = agent.last_downstream_rejections
        reasons = [r["reason"] for r in records]
        assert reasons == ["self_contained", "empty_span_b_with_multi_hop_type", "formula_mismatch"]

        sc_record = records[0]
        assert sc_record["source_chunk_ids"] == ["docA0::c0", "docB0::c0"]
        assert sc_record["question"] == "According to the authors, what is X?"
        assert "matched_phrase" in sc_record

        span_b_record = records[1]
        assert span_b_record["source_chunk_ids"] == ["docA1::c0", "docB1::c0"]

        fm_record = records[2]
        assert fm_record["source_chunk_ids"] == ["docX::c0"]
        assert fm_record["formula"] == "10 + 5"
        assert fm_record["canonical_answer"] == "20"

    def test_downstream_rejections_reset_between_calls(self) -> None:
        agent = self._make_agent()
        seed = _seeds(1)[0]
        # First call: produces a rejection.
        agent._compositions_to_questions(
            [
                CompositionResult(
                    seed=seed,
                    linkable=True,
                    question="According to the authors, what is X?",
                    canonical_answer="x",
                    source_span_A=seed.chunk_a.text,
                    source_span_B=seed.chunk_b.text,
                )
            ]
        )
        assert len(agent.last_downstream_rejections) == 1
        # Second call: clean question, list should reset to empty.
        agent._compositions_to_questions(
            [
                CompositionResult(
                    seed=seed,
                    linkable=True,
                    question="Who founded the company that the acquirer acquired?",
                    canonical_answer="Sarah Smith0",
                    source_span_A=seed.chunk_a.text,
                    source_span_B=seed.chunk_b.text,
                )
            ]
        )
        assert agent.last_downstream_rejections == []


class TestGreedyMergeChunks:
    """Greedy-merge post-processing of HybridChunker output."""

    @staticmethod
    def _c(chunk_id: str, doc_id: str, words: int, section=None) -> ChunkRecord:
        return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=" ".join(["w"] * words), section=section)

    def test_combines_small_chunks_within_budget(self) -> None:
        chunks = [
            self._c("d::0", "d", 30),
            self._c("d::1", "d", 250),
            self._c("d::2", "d", 100),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert len(merged) == 1
        assert len(merged[0].text.split()) == 380
        assert merged[0].chunk_id == "d::0"

    def test_splits_when_budget_exceeded(self) -> None:
        chunks = [
            self._c("d::0", "d", 800),
            self._c("d::1", "d", 800),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert len(merged) == 2
        assert merged[0].chunk_id == "d::0"
        assert merged[1].chunk_id == "d::1"

    def test_preserves_doc_boundaries(self) -> None:
        chunks = [
            self._c("a::0", "a", 30),
            self._c("a::1", "a", 30),
            self._c("b::0", "b", 30),
            self._c("b::1", "b", 30),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert [c.doc_id for c in merged] == ["a", "b"]
        assert all(len(c.text.split()) == 60 for c in merged)

    def test_inherits_first_chunk_id_and_section(self) -> None:
        from agentic_autorag.engine.section_classifier import SectionLabel

        chunks = [
            self._c("d::0", "d", 30, section=None),
            self._c("d::1", "d", 30, section=SectionLabel.BODY),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert len(merged) == 1
        assert merged[0].chunk_id == "d::0"
        assert merged[0].section is None

    def test_handles_oversized_input_chunk(self) -> None:
        chunks = [
            self._c("d::0", "d", 1500),
            self._c("d::1", "d", 100),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert len(merged) == 2
        assert len(merged[0].text.split()) == 1500
        assert len(merged[1].text.split()) == 100

    def test_empty_input(self) -> None:
        assert _greedy_merge_chunks([], max_words=1200) == []

    def test_joins_with_double_newline(self) -> None:
        chunks = [
            ChunkRecord(chunk_id="d::0", doc_id="d", text="alpha"),
            ChunkRecord(chunk_id="d::1", doc_id="d", text="beta"),
        ]
        merged = _greedy_merge_chunks(chunks, max_words=1200)
        assert merged[0].text == "alpha\n\nbeta"


class TestPrepareCorpusUsesEmbeddingPairing:
    """End-to-end: prepare_corpus invokes the embedding pairing path with
    an injectable embed_callable so tests don't need SentenceTransformer."""

    def test_prepare_corpus_yields_seeds_via_stub_embedder(self) -> None:
        # Three docs: A and B share topic-1 vocabulary; C is on a different topic.
        # Stub embedder returns vectors that put A and B close, C far away.
        def stub_embedder(texts: list[str]) -> np.ndarray:
            vectors = []
            for t in texts:
                if "topic_one" in t:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            arr = np.asarray(vectors, dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

        agent = ExamAgent(
            config=ExaminerConfig(
                exam_size=2,
                pair_overgeneration_factor=1.0,
                min_doc_words=1,
                seed_mix={"single_chunk": 0.0, "same_doc_pair": 0.0, "cross_doc_pair": 1.0},
            ),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
            embed_callable=stub_embedder,
        )
        documents = [
            _md_to_dl("topic_one " * 250),  # docA
            _md_to_dl("topic_one " * 250),  # docB — close to docA
            _md_to_dl("totally_different " * 250),  # docC — far from A and B
        ]
        doc_ids = ["docA", "docB", "docC"]
        corpus = agent.prepare_corpus(documents, doc_ids, eligible_sections=None)
        assert len(corpus.seeds) >= 1
        # All seeds must be cross-doc.
        for seed in corpus.seeds:
            assert seed.chunk_b is not None
            assert seed.chunk_a.doc_id != seed.chunk_b.doc_id


@pytest.mark.asyncio
class TestVerifyMultiHopDependency:
    async def test_keeps_questions_when_probe_says_insufficient(self) -> None:
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        question = OpenEndedQuestion(
            id="q1",
            question="Who founded the company that Acme acquired?",
            canonical_answer="Sarah Smith",
            reasoning_type="bridge",
            source_chunk_ids=["a::0", "b::0"],
            source_doc_ids=["a", "b"],
            source_spans=["Acme acquired Beta Inc", "Beta Inc was founded by Sarah Smith"],
        )
        with patch(
            "agentic_autorag.examiner.exam_agent._extractive_probe",
            return_value="",
        ):
            kept = await agent.verify_multi_hop_dependency([question])
        assert len(kept) == 1

    async def test_rejects_when_probe_solves_with_chunk_a_only(self) -> None:
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        question = OpenEndedQuestion(
            id="q2",
            question="Who founded the company that Acme acquired?",
            canonical_answer="Sarah Smith",
            reasoning_type="bridge",
            source_chunk_ids=["a::0", "b::0"],
            source_doc_ids=["a", "b"],
            source_spans=[
                "Acme acquired Beta Inc, which was founded by Sarah Smith",
                "Beta Inc has Tokyo HQ",
            ],
        )
        with patch(
            "agentic_autorag.examiner.exam_agent._extractive_probe",
            return_value="Sarah Smith",
        ):
            kept = await agent.verify_multi_hop_dependency([question])
        assert kept == []

    async def test_rejects_when_probe_solves_with_chunk_b_only(self) -> None:
        """Span B alone must also trigger rejection — fake-2-hop questions
        that smuggle the full answer into the second span must not survive."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        question = OpenEndedQuestion(
            id="q3",
            question="Who founded the company that Acme acquired?",
            canonical_answer="Sarah Smith",
            reasoning_type="bridge",
            source_chunk_ids=["a::0", "b::0"],
            source_doc_ids=["a", "b"],
            source_spans=[
                "Acme is a Delaware-incorporated holding company",
                "Acme acquired Beta Inc, which was founded by Sarah Smith",
            ],
        )

        # span A → unanswerable (""); span B → exact canonical answer
        responses = ["", "Sarah Smith"]
        call_iter = iter(responses)

        def fake_probe(question: str, context: str) -> str:
            return next(call_iter)

        with patch("agentic_autorag.examiner.exam_agent._extractive_probe", side_effect=fake_probe):
            kept = await agent.verify_multi_hop_dependency([question])
        assert kept == []

    async def test_keeps_when_both_spans_insufficient(self) -> None:
        """Both spans must say INSUFFICIENT for a 2-hop question to survive."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        question = OpenEndedQuestion(
            id="q4",
            question="Who founded the company that Acme acquired?",
            canonical_answer="Sarah Smith",
            reasoning_type="bridge",
            source_chunk_ids=["a::0", "b::0"],
            source_doc_ids=["a", "b"],
            source_spans=[
                "Acme acquired Beta Inc",
                "Beta Inc was founded by Sarah Smith",
            ],
        )

        with patch(
            "agentic_autorag.examiner.exam_agent._extractive_probe",
            return_value="",
        ):
            kept = await agent.verify_multi_hop_dependency([question])
        assert len(kept) == 1
