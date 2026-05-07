"""Exam Agent — generates open-ended 2-hop questions via batched LLM composition.

Pipeline:
  1. Chunk the corpus into ``ChunkRecord``s and label sections.
  2. ``embedding_pair_index.emit_embedding_pairs()`` embeds every eligible
     chunk and emits cross-doc seeds via per-chunk top-K + round-robin
     selection over rank.
  3. ``ExamAgent.compose_multihop_batched()`` packs K seeds per LLM call and
     parses out per-seed outcomes — either a candidate question or a
     refusal with a free-text explanation.
  4. ``verify_multi_hop_dependency()`` runs one LLM probe per surviving
     multi-hop candidate to confirm the question can NOT be answered
     with chunk_A's span alone (skipped for single-hop seeds).

The two validation gates (oracle-pass + naive-RAG-fail) live in
``exam_validator``; this module hands off candidates that have cleared the
composition + dependency-verification stages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from agentic_autorag.config.models import (
    QUESTION_TYPES,
    ExaminerConfig,
    OpenEndedQuestion,
)
from agentic_autorag.engine.section_classifier import (
    DEFAULT_ELIGIBLE_SECTIONS,
    SectionLabel,
    classify_chunks_in_document,
)
from agentic_autorag.examiner._errors import format_llm_error, is_transient_llm_error
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord, Seed
from agentic_autorag.examiner.embedding_pair_index import make_pair_embedder
from agentic_autorag.examiner.formula_verify import verify_formula
from agentic_autorag.examiner.prompts import (
    COMPOSITION_BATCH_SYSTEM_PROMPT,
    COMPOSITION_BATCH_USER_PROMPT,
    SINGLE_HOP_SUFFICIENCY_PROBE_PROMPT,
    answer_format_hint,
)
from agentic_autorag.examiner.seeders import (
    emit_cross_doc_pair_seeds,
    emit_same_doc_pair_seeds,
    emit_single_chunk_seeds,
)

logger = logging.getLogger(__name__)

_RETRY_COOLDOWNS = (10, 30, 60)

# Hard cap on canonical_answer length — matches R7 in the prompt. Words
# (whitespace-split) are tokenizer-independent and align with how a reader
# perceives length.
MAX_CANONICAL_WORDS = 15

# Reused from the previous MCQ pipeline — strict regex bank that rejects
# question texts that proxy a source ("the document", "the study", ...).
SELF_CONTAINED_FILTERS = [
    re.compile(
        r'\b(documentation|paper|article|research|study|passage|text|excerpt)\b\s*"[^"]+"',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(the\s+)?(above|given|provided|following)\s+(documentation|passage|text|excerpt|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\baccording\s+to\s+(the\s+)?(document|documentation|paper|article|passage|text|report|PDF|filing|contract)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^based\s+on\s+(the\s+)?(given\s+|provided\s+|above\s+)?"
        r"(text|passage|information|content|material|excerpt|context|document)",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+the\s+(PDF|report|filing|contract|document)\b", re.IGNORECASE),
    re.compile(
        r"\b(in|within)\s+(this|that|these)\s+"
        r"(document|text|passage|report|study|article|paper|PDF|filing|contract|agreement|form)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+(the|this)\s+"
        r"(document|text|passage|report|study|provided|given|following|attached|example|section|excerpt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bas\s+(mentioned|discussed|stated|shown|noted|indicated|referenced|cited)\s+"
        r"(above|below|in\s+the\s+document|in\s+the\s+report|in\s+this\s+study)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(in|from)\s+(the\s+)?(following|preceding|previous|next|above|below)\s+"
        r"(section|paragraph|passage|example|excerpt|chapter|part|statement|clause|provision|text)\b",
        re.IGNORECASE,
    ),
    # 'the study', 'the research', 'the trial' etc. used as document proxies.
    re.compile(
        r"\bthe\s+(?:study'?s?|research'?s?|trial'?s?|experiment'?s?|analysis'?s?|survey'?s?|review'?s?|"
        r"findings?|results?|manuscript|investigators?|authors?)(?=[^\w]|$)",
        re.IGNORECASE,
    ),
]


@dataclass
class CompositionResult:
    """One per-seed outcome of a batched composition LLM call.

    ``rejection_explanation`` is the LLM's free-text reason for refusing
    to compose. Populated on ``linkable: false`` only. ``reason`` is an
    internal label set when the harness decides the result is not
    linkable for non-LLM reasons (parse errors, transient failures) —
    distinct from the LLM's own refusal text.

    ``preferred_type`` carries the type the orchestrator asked the LLM to
    generate. ``reasoning_type`` is the type the LLM actually produced
    (which may differ if the LLM fell back). ``preferred_type_used`` is
    used for per-type yield logging only — not persisted on the question.

    ``formula`` and ``formula_kind`` are populated for ``numeric``
    questions; the harness verifies the math against ``canonical_answer``.
    """

    seed: Seed
    linkable: bool
    preferred_type: str = "bridge"
    reasoning_type: str = "bridge"
    preferred_type_used: bool = True
    question: str = ""
    canonical_answer: str = ""
    answer_variants: list[str] = field(default_factory=list)
    source_span_A: str = ""
    source_span_B: str = ""
    formula: str | None = None
    formula_kind: str | None = None
    rejection_explanation: str = ""
    reason: str = ""


def self_containment_failure(question_text: str) -> tuple[int, str] | None:
    """Return (pattern_index, matched_snippet) for the first failing filter, or None."""
    for idx, pattern in enumerate(SELF_CONTAINED_FILTERS):
        m = pattern.search(question_text)
        if m:
            return idx, m.group(0)
    return None


@dataclass
class PreparedCorpus:
    """Result of one-time corpus preparation for exam generation.

    ``composition_results`` carries every per-seed outcome from the most
    recent ``generate_exam`` run, including LLM refusals — the orchestrator
    persists those alongside accepted candidates so the user can audit
    why the LLM declined to compose a question.
    """

    chunks: list[ChunkRecord]
    seeds: list[Seed] = field(default_factory=list)
    composition_results: list[CompositionResult] = field(default_factory=list)


class ExamAgent:
    """Generates open-ended typed 2-hop candidate questions from a corpus.

    The agent is stateless across runs — each call to ``generate_exam`` builds
    the embedding-pair index fresh. A user can also call ``prepare_corpus``
    once and reuse the resulting ``PreparedCorpus`` across multiple
    compositions (e.g. backfilling when too few candidates survive
    validation).

    Each seed is assigned a **preferred** question type, sampled from
    ``config.question_type_weights`` with a project-seeded RNG so the same
    corpus + project name yields the same type assignment across runs.
    """

    def __init__(
        self,
        config: ExaminerConfig,
        examiner_model: str,
        corpus_description: str = "",
        temperature: float | None = None,
        concurrency: int = 10,
        embed_callable: Callable[[list[str]], np.ndarray] | None = None,
        type_sampler_seed: int | str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.config = config
        self.examiner_model = examiner_model
        self.corpus_description = corpus_description or "General enterprise documents."
        # Default temperature comes from config; explicit kwarg overrides it.
        self.temperature = temperature if temperature is not None else config.composition_temperature
        self.concurrency = concurrency
        # Tests inject a deterministic stub; production lazily loads a
        # SentenceTransformer the first time prepare_corpus runs.
        self._embed_callable = embed_callable
        # Reproducible per-seed preferred-type sampler. Fed by the orchestrator
        # from project_name so the same corpus produces the same type
        # assignment across runs; tests pass an int directly.
        self._type_rng = random.Random(type_sampler_seed)
        self._reasoning_effort = _resolve_reasoning_effort(examiner_model, reasoning_effort)
        if self._reasoning_effort is not None:
            logger.info("ExamAgent using reasoning_effort=%s on %s", self._reasoning_effort, examiner_model)

    # ------------------------------------------------------------------
    # Corpus preparation
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: list[str], doc_ids: list[str]) -> list[ChunkRecord]:
        """Split documents into chunk-sized records for the embedding index.

        Each chunk is labelled with a heuristic ``SectionLabel`` so the
        downstream pairing step can drop structurally non-substantive chunks
        (citation lists, acknowledgments, author/affiliation blocks).
        """
        if len(documents) != len(doc_ids):
            raise ValueError(f"documents ({len(documents)}) and doc_ids ({len(doc_ids)}) must align")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.doc_section_word_size * 5,  # rough chars/word
            chunk_overlap=self.config.doc_section_word_size // 10 * 5,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )

        chunks: list[ChunkRecord] = []
        for doc_text, doc_id in zip(documents, doc_ids, strict=True):
            stripped = doc_text.strip()
            if not stripped:
                continue
            if len(stripped.split()) < self.config.min_doc_words:
                continue
            doc_pieces = [p for p in splitter.split_text(stripped) if p.strip()]
            if not doc_pieces:
                continue
            section_labels = classify_chunks_in_document(doc_pieces)
            for i, (piece, label) in enumerate(zip(doc_pieces, section_labels, strict=True)):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::chunk_{i}",
                        doc_id=doc_id,
                        text=piece,
                        section=label,
                    )
                )
        return chunks

    def prepare_corpus(
        self,
        documents: list[str],
        doc_ids: list[str],
        *,
        eligible_sections: frozenset[SectionLabel] | None = DEFAULT_ELIGIBLE_SECTIONS,
    ) -> PreparedCorpus:
        """Build chunks → mixed-origin seeds for the corpus.

        The total seed pool is sized to ``exam_size *
        pair_overgeneration_factor`` and split across single-chunk,
        same-doc, and cross-doc origins by ``ExaminerConfig.seed_mix``.
        Embedding is shared across the same-doc and cross-doc paths.
        """
        chunks = self.chunk_documents(documents, doc_ids)
        target_seed_count = max(
            1,
            int(self.config.exam_size * self.config.pair_overgeneration_factor),
        )
        mix = self.config.seed_mix
        n_single = int(round(target_seed_count * mix["single_chunk"]))
        n_same = int(round(target_seed_count * mix["same_doc_pair"]))
        n_cross = max(0, target_seed_count - n_single - n_same)

        # Generate same-doc seeds first, then redistribute any deficit to
        # cross-doc, then redistribute cross-doc deficit to single-chunk.
        # This keeps the total seed pool full when one origin is structurally
        # infeasible (e.g. HotpotQA: same-doc impossible, cross-doc gets the
        # extra share; legal/medical with no cross-doc overlap: cross-doc
        # short, single-chunk gets the extra share).
        seeds: list[Seed] = []
        same_doc_seeds: list[Seed] = []
        cross_doc_seeds: list[Seed] = []

        if n_same > 0 or n_cross > 0:
            embed_callable = self._embed_callable or make_pair_embedder(self.config.pair_embedding_model)
            if n_same > 0:
                same_doc_seeds = emit_same_doc_pair_seeds(
                    chunks,
                    embed_callable,
                    target_count=n_same,
                    cos_min=self.config.same_doc_pair_cosine_min,
                    cos_max=self.config.same_doc_pair_cosine_max,
                    eligible_sections=eligible_sections,
                    model_name=self.config.pair_embedding_model,
                )
                same_doc_deficit = n_same - len(same_doc_seeds)
                if same_doc_deficit > 0:
                    logger.info(
                        "DIAG Seed budget redistribution: same_doc short by %d → cross_doc target +%d",
                        same_doc_deficit,
                        same_doc_deficit,
                    )
                    n_cross += same_doc_deficit
            if n_cross > 0:
                cross_doc_seeds = emit_cross_doc_pair_seeds(
                    chunks,
                    embed_callable,
                    top_k_per_chunk=self.config.pair_top_k_per_chunk,
                    target_count=n_cross,
                    eligible_sections=eligible_sections,
                    model_name=self.config.pair_embedding_model,
                )

        cross_doc_deficit = n_cross - len(cross_doc_seeds)
        if cross_doc_deficit > 0:
            logger.info(
                "DIAG Seed budget redistribution: cross_doc short by %d → single_chunk target +%d",
                cross_doc_deficit,
                cross_doc_deficit,
            )
            n_single += cross_doc_deficit

        single_chunk_seeds: list[Seed] = []
        if n_single > 0:
            single_chunk_seeds = emit_single_chunk_seeds(
                chunks,
                target_count=n_single,
                eligible_sections=eligible_sections,
            )

        seeds.extend(single_chunk_seeds)
        seeds.extend(same_doc_seeds)
        seeds.extend(cross_doc_seeds)

        logger.info(
            "Prepared corpus: %d chunks, %d seeds (target=%d; single=%d, same_doc=%d, cross_doc=%d)",
            len(chunks),
            len(seeds),
            target_seed_count,
            len(single_chunk_seeds),
            len(same_doc_seeds),
            len(cross_doc_seeds),
        )
        return PreparedCorpus(chunks=chunks, seeds=seeds)

    # ------------------------------------------------------------------
    # Batched composition
    # ------------------------------------------------------------------

    def _sample_preferred_types(self, n: int) -> list[str]:
        """Draw n preferred question types from the configured weight map."""
        weights = self.config.question_type_weights
        # Filter to known types with positive weight, in canonical order so
        # the RNG sees a stable categorical distribution across runs.
        items = [(t, weights.get(t, 0.0)) for t in QUESTION_TYPES if weights.get(t, 0.0) > 0]
        if not items:
            return ["bridge"] * n
        labels = [t for t, _ in items]
        ws = [w for _, w in items]
        return self._type_rng.choices(labels, weights=ws, k=n)

    async def compose_multihop_batched(self, seeds: list[Seed]) -> list[CompositionResult]:
        """Run batched composition LLM calls over the seed list.

        Seeds are partitioned into batches of ``composition_batch_size``. Each
        seed in a batch carries an independently sampled **preferred** question
        type, surfaced in the prompt as a hint the LLM may follow or fall back
        from.

        Per-element parse failures don't poison the batch — malformed entries
        are logged and skipped.
        """
        if not seeds:
            return []

        preferred_types = self._sample_preferred_types(len(seeds))

        k = self.config.composition_batch_size
        batches: list[list[tuple[Seed, str]]] = [
            list(zip(seeds[i : i + k], preferred_types[i : i + k], strict=True)) for i in range(0, len(seeds), k)
        ]
        sem = asyncio.Semaphore(self.concurrency)

        results: list[CompositionResult] = []
        results_lock = asyncio.Lock()
        # DIAG per-batch wall-clock latency.
        batch_latencies: list[float] = []
        latency_lock = asyncio.Lock()

        async def _process_batch(batch: list[tuple[Seed, str]]) -> None:
            t0 = asyncio.get_event_loop().time()
            async with sem:
                batch_results = await self._compose_one_batch(batch)
            elapsed = asyncio.get_event_loop().time() - t0
            async with results_lock:
                results.extend(batch_results)
            async with latency_lock:
                batch_latencies.append(elapsed)

        with tqdm(total=len(batches), desc="Composing typed 2-hop questions", unit="batch") as pbar:

            async def _bounded(batch: list[tuple[Seed, str]]) -> None:
                await _process_batch(batch)
                pbar.update(1)

            await asyncio.gather(*[_bounded(b) for b in batches])

        # DIAG composition latency p50/p95.
        if batch_latencies:
            sorted_lat = sorted(batch_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[min(n - 1, int(n * 0.95))]
            mean_lat = sum(sorted_lat) / n
            logger.info(
                "DIAG Composition latency: n=%d batches, mean=%.1fs p50=%.1fs p95=%.1fs",
                n,
                mean_lat,
                p50,
                p95,
            )

        return results

    async def _compose_one_batch(self, batch: list[tuple[Seed, str]]) -> list[CompositionResult]:
        """Compose one batch with up to N retries on transient LLM errors."""
        for attempt, cooldown in enumerate((0, *_RETRY_COOLDOWNS), start=0):
            if cooldown:
                await asyncio.sleep(cooldown)
            try:
                raw = await self._call_composition_llm(batch)
                return self._parse_composition_batch(raw, batch)
            except Exception as exc:
                if not is_transient_llm_error(exc):
                    logger.info("Composition batch failed permanently: %s", format_llm_error(exc))
                    return [
                        CompositionResult(
                            seed=s,
                            preferred_type=pt,
                            linkable=False,
                            reason="composition_error",
                        )
                        for s, pt in batch
                    ]
                if attempt == len(_RETRY_COOLDOWNS):
                    logger.warning(
                        "Composition batch exhausted retries (%s)",
                        format_llm_error(exc),
                    )
                    return [
                        CompositionResult(
                            seed=s,
                            preferred_type=pt,
                            linkable=False,
                            reason="composition_transient",
                        )
                        for s, pt in batch
                    ]
        return []  # pragma: no cover

    async def _call_composition_llm(self, batch: list[tuple[Seed, str]]) -> str:
        seed_blocks = []
        for i, (seed, preferred_type) in enumerate(batch):
            # Surface the same per-type answer-shape hint that the eval-time
            # grader will use, so canonical_answer is produced in the shape
            # the downstream RAG pipeline is graded against.
            preferred_kind = "arithmetic" if preferred_type == "numeric" else None
            shape_hint = answer_format_hint(preferred_type, preferred_kind)
            block_lines = [
                f"Seed #{i}",
                f"  Origin: {seed.origin}",
                f"  Preferred reasoning type: {preferred_type}",
                f"  Expected canonical_answer shape: {shape_hint}",
                f"  chunk_A (doc_id={seed.chunk_a.doc_id}):\n{seed.chunk_a.text}",
            ]
            if seed.chunk_b is not None:
                block_lines.append(f"  chunk_B (doc_id={seed.chunk_b.doc_id}):\n{seed.chunk_b.text}")
            seed_blocks.append("\n".join(block_lines))
        user = COMPOSITION_BATCH_USER_PROMPT.format(
            domain_description=self.corpus_description,
            k=len(batch),
            seed_blocks="\n\n".join(seed_blocks),
        )
        # Use a cooler temperature when the batch is dominated by numeric
        # questions — reduces formula-mismatch rejections downstream.
        numeric_share = sum(1 for _, pt in batch if pt == "numeric") / len(batch)
        numeric_temp = self.config.composition_temperature_numeric
        temperature = numeric_temp if numeric_share >= 0.5 and numeric_temp is not None else self.temperature
        kwargs: dict = {
            "model": self.examiner_model,
            "messages": [
                {"role": "system", "content": COMPOSITION_BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "num_retries": 0,
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    def _parse_composition_batch(
        self,
        raw: str,
        batch: list[tuple[Seed, str]],
    ) -> list[CompositionResult]:
        """Parse a JSON array of K objects, one per (seed, preferred_type) pair.

        Schema for each entry:
          - refusal:  {"seed_id": i, "linkable": false, "explanation": "..."}
          - accepted: {"seed_id": i, "linkable": true,
                       "reasoning": "...",
                       "reasoning_type": "...",
                       "preferred_type_used": true|false,
                       "question": "...", "canonical_answer": "...",
                       "answer_variants": [...],
                       "formula": null | "...",
                       "formula_kind": null | "arithmetic",
                       "source_span_A": "...", "source_span_B": "..."}
        """
        text = _strip_markdown_fences(raw)
        items = _try_parse_json_array(text)
        if items is None:
            items = _extract_json_array(text)
        if items is None:
            logger.info("Composition batch parse failed; raw=%.200s", raw)
            return [
                CompositionResult(seed=s, preferred_type=pt, linkable=False, reason="parse_error") for s, pt in batch
            ]

        valid_types = set(QUESTION_TYPES)
        out: list[CompositionResult] = []
        for i, (seed, preferred_type) in enumerate(batch):
            entry: dict | None = None
            # Prefer entries that explicitly identify themselves by seed_id.
            for it in items:
                if isinstance(it, dict) and it.get("seed_id") == i:
                    entry = it
                    break
            if entry is None and i < len(items) and isinstance(items[i], dict):
                entry = items[i]

            if entry is None:
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="missing_in_batch",
                    )
                )
                continue

            linkable_raw = entry.get("linkable", False)
            if not isinstance(linkable_raw, bool):
                linkable_raw = bool(linkable_raw)

            if not linkable_raw:
                explanation = str(entry.get("explanation") or entry.get("reason") or "llm_marked_not_linkable")[:480]
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        rejection_explanation=explanation,
                    )
                )
                continue

            try:
                question = str(entry["question"]).strip()
                canonical = str(entry["canonical_answer"]).strip()
                span_a = str(entry["source_span_A"]).strip()
                span_b = str(entry.get("source_span_B", "") or "").strip()
            except (KeyError, TypeError) as exc:
                logger.info("Seed %d composition entry missing required fields: %s", i, exc)
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="missing_fields",
                    )
                )
                continue

            # R7: canonical answers must fit in ≤ 15 words. Long abstractive
            # answers (mostly definitional) hurt EM scoring and force the
            # judge into work it shouldn't need to do.
            if len(canonical.split()) > MAX_CANONICAL_WORDS:
                out.append(
                    CompositionResult(
                        seed=seed,
                        preferred_type=preferred_type,
                        linkable=False,
                        reason="answer_too_long",
                    )
                )
                continue

            # Normalise the LLM's reported type. Accept either field name to
            # handle older prompt versions still emitting ``question_type``.
            reported_type = str(entry.get("reasoning_type") or entry.get("question_type") or "").strip()
            if reported_type not in valid_types:
                reported_type = preferred_type
            preferred_used_raw = entry.get("preferred_type_used")
            if isinstance(preferred_used_raw, bool):
                preferred_type_used = preferred_used_raw
            else:
                preferred_type_used = reported_type == preferred_type

            variants_raw = entry.get("answer_variants", []) or []
            if isinstance(variants_raw, str):
                variants = [variants_raw]
            elif isinstance(variants_raw, list):
                variants = [str(v).strip() for v in variants_raw if isinstance(v, str) and v.strip()]
            else:
                variants = []

            formula_raw = entry.get("formula")
            formula = str(formula_raw).strip() if isinstance(formula_raw, str) and formula_raw.strip() else None
            formula_kind_raw = entry.get("formula_kind")
            formula_kind: str | None = None
            if isinstance(formula_kind_raw, str) and formula_kind_raw.strip():
                kind = formula_kind_raw.strip()
                if kind == "arithmetic":
                    formula_kind = kind
            if formula is None or formula_kind is None:
                formula = None
                formula_kind = None

            out.append(
                CompositionResult(
                    seed=seed,
                    preferred_type=preferred_type,
                    reasoning_type=reported_type,
                    preferred_type_used=preferred_type_used,
                    linkable=True,
                    question=question,
                    canonical_answer=canonical,
                    answer_variants=variants,
                    source_span_A=span_a,
                    source_span_B=span_b,
                    formula=formula,
                    formula_kind=formula_kind,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Verification probes (single-hop sufficiency)
    # ------------------------------------------------------------------

    async def verify_multi_hop_dependency(
        self,
        candidates: list[OpenEndedQuestion],
    ) -> list[OpenEndedQuestion]:
        """Reject multi-hop questions answerable from the first span alone.

        For each candidate with ``num_hops >= 2``, ask the validator
        (examiner) model to answer using only ``source_spans[0]`` as
        context. If it returns anything other than the ``INSUFFICIENT``
        sentinel AND the answer is sufficiently close to the canonical
        answer, the question is decomposable — reject. Single-hop
        candidates skip this gate; the discrimination filter handles
        closed-book-easy questions automatically.
        """
        if not candidates:
            return []

        sem = asyncio.Semaphore(self.concurrency)
        verdicts: list[bool] = [False] * len(candidates)

        async def _probe(idx: int, q: OpenEndedQuestion) -> None:
            if q.num_hops < 2:
                verdicts[idx] = False
                return
            async with sem:
                try:
                    raw = await _call_completion(
                        self.examiner_model,
                        SINGLE_HOP_SUFFICIENCY_PROBE_PROMPT.format(
                            context=q.source_spans[0],
                            question=q.question,
                        ),
                        reasoning_effort=self._reasoning_effort,
                    )
                except Exception as exc:
                    if is_transient_llm_error(exc):
                        verdicts[idx] = False  # conservative: keep when probe is unreliable
                        return
                    logger.info("single-hop probe permanent error for %s: %s", q.id, format_llm_error(exc))
                    verdicts[idx] = True  # treat as solvable single-hop → reject
                    return
            answer = (raw or "").strip()
            if not answer or answer.upper().startswith("INSUFFICIENT"):
                verdicts[idx] = False
                return
            # If the answer is extremely close to the canonical answer, the
            # question is decomposable.
            verdicts[idx] = _answer_close_enough(answer, q.gold_answers)

        with tqdm(total=len(candidates), desc="Multi-hop dependency probe", unit="q") as pbar:

            async def _bounded(idx: int, q: OpenEndedQuestion) -> None:
                await _probe(idx, q)
                pbar.update(1)

            await asyncio.gather(*[_bounded(i, q) for i, q in enumerate(candidates)])

        from collections import Counter

        kept = [q for q, decomposable in zip(candidates, verdicts, strict=True) if not decomposable]
        n_removed = len(candidates) - len(kept)
        logger.info("Multi-hop dependency probe: removed %d/%d decomposable", n_removed, len(candidates))
        # DIAG per-type and per-origin removal breakdown.
        attempts_by_type: Counter[str] = Counter()
        removed_by_type: Counter[str] = Counter()
        attempts_by_origin: Counter[str] = Counter()
        removed_by_origin: Counter[str] = Counter()
        # Re-derive origin via num_hops + is_multi_doc (Seed.origin isn't on
        # OpenEndedQuestion — only single-hop vs same-doc vs cross-doc here).
        for q, decomposable in zip(candidates, verdicts, strict=True):
            if q.num_hops < 2:
                continue
            attempts_by_type[q.reasoning_type] += 1
            origin_label = "cross_doc_pair" if q.is_multi_doc else "same_doc_pair"
            attempts_by_origin[origin_label] += 1
            if decomposable:
                removed_by_type[q.reasoning_type] += 1
                removed_by_origin[origin_label] += 1
        if attempts_by_type:
            type_breakdown = ", ".join(
                f"{t}={removed_by_type[t]}/{attempts_by_type[t]}" for t in sorted(attempts_by_type.keys())
            )
            logger.info("DIAG Multi-hop dependency probe by type: %s", type_breakdown)
        if attempts_by_origin:
            origin_breakdown = ", ".join(
                f"{o}={removed_by_origin[o]}/{attempts_by_origin[o]}" for o in sorted(attempts_by_origin.keys())
            )
            logger.info("DIAG Multi-hop dependency probe by origin: %s", origin_breakdown)
        return kept

    # ------------------------------------------------------------------
    # End-to-end driver
    # ------------------------------------------------------------------

    async def generate_exam(
        self,
        documents: list[str],
        doc_ids: list[str],
        *,
        eligible_sections: frozenset[SectionLabel] | None = DEFAULT_ELIGIBLE_SECTIONS,
    ) -> tuple[list[OpenEndedQuestion], PreparedCorpus]:
        """Convenience wrapper: prepare corpus → typed compose → single-hop probe.

        The returned questions still need to pass the oracle answerability gate
        in ``exam_validator`` and the 4-probe discrimination filter in
        ``orchestrator._generate_exam``. The corpus's ``composition_results``
        field carries every per-seed outcome — the orchestrator persists those
        alongside accepted candidates so the user can audit why the LLM
        declined to compose a question.
        """
        corpus = self.prepare_corpus(documents, doc_ids, eligible_sections=eligible_sections)
        if not corpus.seeds:
            return [], corpus

        composition_results = await self.compose_multihop_batched(corpus.seeds)
        corpus.composition_results = composition_results
        questions = self._compositions_to_questions(composition_results)

        if not questions:
            return [], corpus

        questions = await self.verify_multi_hop_dependency(questions)
        return questions, corpus

    def _compositions_to_questions(self, results: list[CompositionResult]) -> list[OpenEndedQuestion]:
        """Convert composition results into validated ``OpenEndedQuestion``s.

        Pipeline (in order):
          - linkable filter (LLM refusals + harness errors)
          - self-containment regex check
          - empty-span-B check on multi-hop seeds (LLM produced a single-hop
            answer for a 2-chunk seed → reject typed)
          - source-span verbatim check (each span must substring its chunk)
          - numeric formula verification (if reasoning_type == "numeric")

        Tracks rejections in a single ``Counter`` keyed by reason and a
        per-origin × reason matrix for diagnostic logging. Returns the
        questions that survive every gate.
        """
        from collections import Counter

        kept: list[OpenEndedQuestion] = []
        reasons: Counter[str] = Counter()
        # DIAG per-origin × reason matrix (Step 6 surfaces in the log).
        rejections_by_origin: dict[str, Counter[str]] = {}
        # DIAG per-origin survival counts (attempts → kept).
        origin_attempts: Counter[str] = Counter()
        origin_kept: Counter[str] = Counter()
        # DIAG sample up to 3 rejections per reason for human inspection.
        sample_rejections: dict[str, list[str]] = {}
        # Per-preferred-type {"attempts", "refused", "kept", "fallback"}.
        type_stats: dict[str, dict[str, int]] = {
            t: {"attempts": 0, "refused": 0, "kept": 0, "fallback": 0} for t in QUESTION_TYPES
        }

        def _reject(origin: str, reason: str, *, sample: str = "") -> None:
            reasons[reason] += 1
            rejections_by_origin.setdefault(origin, Counter())[reason] += 1
            if sample and len(sample_rejections.get(reason, [])) < 3:
                sample_rejections.setdefault(reason, []).append(sample)

        for i, r in enumerate(results, start=1):
            origin = r.seed.origin
            origin_attempts[origin] += 1
            stats = type_stats.setdefault(
                r.preferred_type,
                {"attempts": 0, "refused": 0, "kept": 0, "fallback": 0},
            )
            stats["attempts"] += 1
            if not r.linkable:
                stats["refused"] += 1
                # Harness errors carry r.reason; LLM refusals carry only
                # r.rejection_explanation (free-text) — bucket those as
                # "llm_refused".
                code = r.reason or ("llm_refused" if r.rejection_explanation else "unspecified")
                _reject(
                    origin,
                    code,
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.rejection_explanation[:200]}"
                    if code == "llm_refused"
                    else f"{r.seed.chunk_a.chunk_id}",
                )
                continue

            sc_fail = self_containment_failure(r.question)
            if sc_fail is not None:
                _reject(origin, "self_contained", sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}")
                logger.info("self-contained-fail: %r", sc_fail[1])
                continue

            # Empty span_A is an LLM error regardless of seed shape — every
            # accepted question must ground in chunk_A. Reject typed so the
            # funnel log shows the rate without falling through to pydantic.
            if not r.source_span_A.strip():
                _reject(
                    origin,
                    "empty_span_a",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}",
                )
                continue

            # The LLM produced a single-hop answer (empty source_span_B) for a
            # multi-hop seed. We don't auto-convert: the seed asked for a 2-hop
            # question, and quietly accepting a 1-hop answer lets the LLM dodge
            # R1. Reject typed so the funnel log shows the rate.
            if r.seed.chunk_b is not None and not r.source_span_B.strip():
                _reject(
                    origin,
                    "empty_span_b_for_multi_hop_seed",
                    sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}",
                )
                continue

            if r.source_span_A not in r.seed.chunk_a.text:
                _reject(origin, "span_missing_a", sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}")
                continue
            if r.seed.chunk_b is not None and r.source_span_B and r.source_span_B not in r.seed.chunk_b.text:
                _reject(origin, "span_missing_b", sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:160]}")
                continue

            if r.reasoning_type == "numeric":
                if not r.formula or not r.formula_kind:
                    _reject(
                        origin,
                        "formula_missing",
                        sample=f"{r.seed.chunk_a.chunk_id} :: {r.question[:120]} -> {r.canonical_answer}",
                    )
                    logger.info(
                        "numeric question missing formula: q=%r answer=%r",
                        r.question[:120],
                        r.canonical_answer,
                    )
                    continue
                if not verify_formula(r.formula, r.formula_kind, r.canonical_answer):
                    _reject(
                        origin,
                        "formula_mismatch",
                        sample=f"{r.seed.chunk_a.chunk_id} :: formula={r.formula!r} answer={r.canonical_answer!r}",
                    )
                    logger.info(
                        "formula mismatch: formula=%r kind=%s answer=%r",
                        r.formula,
                        r.formula_kind,
                        r.canonical_answer,
                    )
                    continue

            if r.seed.chunk_b is not None:
                source_chunk_ids = [r.seed.chunk_a.chunk_id, r.seed.chunk_b.chunk_id]
                source_doc_ids = [r.seed.chunk_a.doc_id, r.seed.chunk_b.doc_id]
                source_spans = [r.source_span_A, r.source_span_B]
                cluster_id = r.seed.chunk_b.cluster_id
            else:
                source_chunk_ids = [r.seed.chunk_a.chunk_id]
                source_doc_ids = [r.seed.chunk_a.doc_id]
                source_spans = [r.source_span_A]
                cluster_id = r.seed.chunk_a.cluster_id

            try:
                question = OpenEndedQuestion(
                    id=f"C{i:04d}",
                    question=r.question,
                    canonical_answer=r.canonical_answer,
                    answer_variants=r.answer_variants,
                    reasoning_type=r.reasoning_type,
                    source_chunk_ids=source_chunk_ids,
                    source_doc_ids=source_doc_ids,
                    source_spans=source_spans,
                    formula=r.formula,
                    formula_kind=r.formula_kind,
                    cluster_id=cluster_id,
                )
            except Exception as exc:  # noqa: BLE001
                _reject(origin, "pydantic_validation", sample=f"{r.seed.chunk_a.chunk_id} :: {exc}")
                logger.info("OpenEndedQuestion validation failed: %s", exc)
                continue
            kept.append(question)
            origin_kept[origin] += 1
            stats["kept"] += 1
            if not r.preferred_type_used:
                stats["fallback"] += 1

        n_total = len(results)
        n_kept = len(kept)
        n_rejected = n_total - n_kept
        logger.info(
            "Composition → questions: %d kept / %d total (%d rejected)",
            n_kept,
            n_total,
            n_rejected,
        )
        if reasons:
            breakdown = ", ".join(f"{code}={n}" for code, n in reasons.most_common())
            logger.info("Composition rejections by reason: %s", breakdown)
        # DIAG per-origin survival funnel.
        if origin_attempts:
            survival = ", ".join(
                f"{origin}={origin_kept[origin]}/{origin_attempts[origin]}" for origin in sorted(origin_attempts.keys())
            )
            logger.info("DIAG Composition survival by origin: %s", survival)
        # DIAG per-origin × reason matrix.
        if rejections_by_origin:
            for origin in sorted(rejections_by_origin.keys()):
                ctr = rejections_by_origin[origin]
                origin_breakdown = ", ".join(f"{code}={n}" for code, n in ctr.most_common())
                logger.info("DIAG Composition rejections [%s]: %s", origin, origin_breakdown)
        # DIAG sample rejections (up to 3 per reason) for human inspection.
        if sample_rejections:
            for reason, samples in sorted(sample_rejections.items()):
                for j, s in enumerate(samples, start=1):
                    logger.info("DIAG Reject sample [%s #%d]: %s", reason, j, s)
        type_lines = []
        for t in QUESTION_TYPES:
            stats = type_stats.get(t)
            if not stats or stats["attempts"] == 0:
                continue
            type_lines.append(
                f"{t}: {stats['kept']} kept / {stats['attempts']} attempts "
                f"(refused={stats['refused']}, fallback={stats['fallback']})"
            )
        if type_lines:
            logger.info("Per-preferred-type yield: %s", " | ".join(type_lines))
        # Per-actual-type counts of kept questions (after possible fallback).
        actual_counts: dict[str, int] = {}
        for q in kept:
            actual_counts[q.reasoning_type] = actual_counts.get(q.reasoning_type, 0) + 1
        if actual_counts:
            actual_line = ", ".join(f"{t}={actual_counts.get(t, 0)}" for t in QUESTION_TYPES if t in actual_counts)
            logger.info("Per-actual-type kept counts: %s", actual_line)
        return kept


# --- helpers ---------------------------------------------------------------


def _resolve_reasoning_effort(model: str, effort: str | None) -> str | None:
    """Return ``effort`` if the model supports reasoning, else None.

    Mirrors ``ReasoningAgent._resolve_reasoning_effort`` so callers can ask
    LiteLLM to pass ``reasoning_effort`` through without crashing on models
    that don't support it.
    """
    if not effort:
        return None
    try:
        supported = bool(litellm.supports_reasoning(model=model))
    except Exception:
        supported = True
    return effort if supported else None


async def _call_completion(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "num_retries": 0,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content or ""


def _answer_close_enough(pred: str, gold_answers: list[str]) -> bool:
    """Token-level F1 ≥ 0.7 against any gold answer.

    Used by the single-hop probe: \"can the model answer with chunk_A only?\"
    Requires real overlap, not just a single shared word.
    """
    from agentic_autorag.benchmark_eval.scoring import best_f1

    return best_f1(pred, gold_answers) >= 0.7


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _try_parse_json_array(text: str) -> list[dict] | None:
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return None


def _extract_json_array(text: str) -> list[dict] | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
                return None
    return None


__all__ = [
    "CompositionResult",
    "ExamAgent",
    "PreparedCorpus",
    "SELF_CONTAINED_FILTERS",
    "self_containment_failure",
]
