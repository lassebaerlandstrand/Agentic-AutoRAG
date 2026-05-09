"""Tests for the open-ended ExamAgent and its composition pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import ExaminerConfig, OpenEndedQuestion
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.exam_agent import (
    CompositionResult,
    ExamAgent,
    self_containment_failure,
)


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
                f"  chunk_A (doc_id={s.chunk_a.doc_id}):\n{s.chunk_a.text}\n"
                f"  chunk_B (doc_id={s.chunk_b.doc_id}):\n{s.chunk_b.text}"
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


class TestTypedSampling:
    """The preferred-type sampler must respect the configured weights and
    fall back gracefully when the dict is empty / all-zero."""

    def test_sample_distribution_matches_weights(self) -> None:
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
        n = 5000
        types = agent._sample_preferred_types(n)
        counts = {t: types.count(t) for t in set(types)}
        # Within ±5% of the configured weights for each type.
        for t, w in agent.config.question_type_weights.items():
            observed = counts.get(t, 0) / n
            assert abs(observed - w) < 0.05, f"{t}: observed={observed:.3f} expected={w:.3f}"

    def test_fallback_when_all_zero_at_runtime(self) -> None:
        """The Pydantic validator rejects an all-zero dict, but ``_sample_preferred_types``
        must still degrade gracefully if a downstream hand-mutated the dict (defence in
        depth — keeps the sampler from crashing the whole exam build)."""
        agent = ExamAgent(
            config=ExaminerConfig(exam_size=10),
            examiner_model="test/model",
            corpus_description="t",
            concurrency=1,
        )
        # Hand-mutate to simulate a downstream zero-out.
        agent.config.question_type_weights = {"comparison": 0.0}
        types = agent._sample_preferred_types(20)
        assert all(t == "bridge" for t in types)

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
        a1 = ExamAgent(config=cfg, examiner_model="m", corpus_description="t", concurrency=1, type_sampler_seed="proj")
        a2 = ExamAgent(config=cfg, examiner_model="m", corpus_description="t", concurrency=1, type_sampler_seed="proj")
        assert a1._sample_preferred_types(50) == a2._sample_preferred_types(50)

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

    def test_rejects_when_source_span_not_in_chunk(self) -> None:
        agent = self._make_agent()
        seed = _seeds(1)[0]
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                source_span_A="completely fabricated text not in the chunk",
                source_span_B=seed.chunk_b.text,
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert kept == []

    def test_empty_span_b_on_multi_hop_seed_rejected_typed(self) -> None:
        """LLM returning empty source_span_B for a multi-hop seed gets a typed
        rejection, not a Pydantic exception."""
        agent = self._make_agent()
        seed = _seeds(1)[0]  # Cross-doc multi-hop seed (chunk_b is set).
        results = [
            CompositionResult(
                seed=seed,
                linkable=True,
                question="Who founded the company that the acquirer acquired?",
                canonical_answer="Sarah Smith0",
                source_span_A=seed.chunk_a.text,
                source_span_B="",
            )
        ]
        kept = agent._compositions_to_questions(results)
        assert kept == []

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
            "topic_one " * 250,  # docA
            "topic_one " * 250,  # docB — close to docA
            "totally_different " * 250,  # docC — far from A and B
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
            "agentic_autorag.examiner.exam_agent._call_completion",
            new=AsyncMock(return_value="INSUFFICIENT"),
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
            "agentic_autorag.examiner.exam_agent._call_completion",
            new=AsyncMock(return_value="Sarah Smith"),
        ):
            kept = await agent.verify_multi_hop_dependency([question])
        assert kept == []
