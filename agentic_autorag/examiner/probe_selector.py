"""Probe-based question selection for discriminating exam construction.

Candidate questions are evaluated against a small set of diverse pipeline
configurations (probes). Questions that produce mixed results across probes
are the most discriminating between strong and weak RAG setups.
"""

from __future__ import annotations

import json
import logging

from agentic_autorag.config.models import (
    OpenEndedQuestion,
    ProjectConfig,
    TrialConfig,
    _dim_max_value,
    _dim_min_value,
)
from agentic_autorag.examiner.evaluator import ExamResult

logger = logging.getLogger(__name__)

_MIN_KB_COVERAGE = 3  # need at least this many known models for KB ranking to be useful

_RANK_MODELS_PROMPT = """\
Given these {model_type} models used in RAG pipelines, rank them from weakest \
to strongest quality for RAG evaluation.

Return ONLY a JSON array of the exact model names in order from weakest to strongest. \
Do not include any other text.

Models:
{model_list}"""


async def rank_models_for_probes(
    model_names: list[str],
    model_type: str,
    knowledge_base: object | None,
    optimizer_model: str | None = None,
    reasoning_allowed: dict[str, bool] | None = None,
    reasoning_effort: str = "medium",
) -> list[str]:
    """Rank models from weakest to strongest for probe config generation.

    ``reasoning_allowed`` and ``reasoning_effort`` are only consulted for
    ``model_type == "llm"`` — embeddings and rerankers have no reasoning
    variants — and are forwarded to ``KnowledgeBase.rank_llms`` so the
    OFF/ON variant matching the project's
    ``SearchSpace.is_reasoning_allowed`` and the configured effort
    (``GeneratorSearchSpace.reasoning_effort``) is what's scored.

    Cascade:
    1. KB has data for >= _MIN_KB_COVERAGE models → rank by KB scores,
       interleave unknowns at median position.
    2. KB insufficient + optimizer_model available → single LLM call
       asking the optimizer to rank all models by quality.
    3. LLM fails or unavailable → use partial KB data + warn.
    """
    if len(model_names) <= 1:
        return list(model_names)

    # Step 1: Try KB ranking
    ranked_known, unknowns = _kb_rank(model_names, model_type, knowledge_base, reasoning_allowed, reasoning_effort)

    if not unknowns:
        return ranked_known

    if len(ranked_known) >= _MIN_KB_COVERAGE:
        # Sufficient KB data — interleave unknowns at median position
        return _interleave_at_median(ranked_known, unknowns)

    # Step 2: KB insufficient — try LLM fallback
    if optimizer_model:
        llm_ranked = await _llm_rank(model_names, model_type, optimizer_model)
        if llm_ranked:
            return llm_ranked

    # Step 3: Fallback — partial KB + unknowns at median + warn
    if ranked_known:
        logger.warning(
            "KB has only %d/%d %s models; probe ranking may be imprecise",
            len(ranked_known),
            len(model_names),
            model_type,
        )
        return _interleave_at_median(ranked_known, unknowns)

    logger.warning("No KB data for %s models; using original list order for probes", model_type)
    return list(model_names)


def _kb_rank(
    model_names: list[str],
    model_type: str,
    knowledge_base: object | None,
    reasoning_allowed: dict[str, bool] | None = None,
    reasoning_effort: str = "medium",
) -> tuple[list[str], list[str]]:
    """Rank using KnowledgeBase. Returns (ranked_known, unknowns)."""
    if knowledge_base is None:
        return [], list(model_names)

    method_name = {"llm": "rank_llms", "embedding": "rank_embeddings", "reranker": "rank_rerankers"}.get(model_type)
    if method_name is None:
        return [], list(model_names)

    rank_fn = getattr(knowledge_base, method_name, None)
    if rank_fn is None:
        return [], list(model_names)

    try:
        if model_type == "llm":
            return rank_fn(model_names, reasoning_allowed, reasoning_effort)
        return rank_fn(model_names)
    except Exception:
        logger.debug("KB ranking failed for %s models", model_type, exc_info=True)
        return [], list(model_names)


def _interleave_at_median(ranked_known: list[str], unknowns: list[str]) -> list[str]:
    """Insert unknown models at the median position of the known ranking."""
    if not ranked_known:
        return unknowns
    mid = len(ranked_known) // 2
    return ranked_known[:mid] + unknowns + ranked_known[mid:]


async def _llm_rank(
    model_names: list[str],
    model_type: str,
    optimizer_model: str,
) -> list[str] | None:
    """Ask the optimizer LLM to rank models. Returns None on failure."""
    try:
        from agentic_autorag.litellm_runtime import acompletion_with_cost

        prompt = _RANK_MODELS_PROMPT.format(
            model_type=model_type,
            model_list="\n".join(f"- {m}" for m in model_names),
        )
        response, _ = await acompletion_with_cost(
            cost_category="exam_generation",
            model=optimizer_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        ranked = json.loads(text)
        if not isinstance(ranked, list):
            return None

        # Validate all names are present
        name_set = set(model_names)
        ranked_clean = [m for m in ranked if m in name_set]
        # Add any missing models at the end
        seen = set(ranked_clean)
        for m in model_names:
            if m not in seen:
                ranked_clean.append(m)
        return ranked_clean

    except Exception:
        logger.debug("LLM ranking fallback failed for %s models", model_type, exc_info=True)
        return None


def select_probe_configs(
    config: ProjectConfig,
    ranked_llms: list[str] | None = None,
    ranked_embeds: list[str] | None = None,
    ranked_rerankers: list[str] | None = None,
) -> list[tuple[str, TrialConfig]]:
    """Build 4 ordinally-spread probe configurations spanning the search space.

    Probes are emitted **weakest-first** so the discrimination scorer can
    rank-correlate outcomes against probe strength directly. Both the LLM
    and embedding axes are spread evenly across their respective ranked
    lists at indices ``0``, ``n//4``, ``3n//4``, ``-1``. The reranker axis
    stays binary (off for T1/T2, best for T3/T4) — the retrieval-stack
    threshold at T2→T3 is intentional and preserves cross-tier reranker
    cache reuse. Chunk size and top_k step up monotonically:

      - **Tier1-weak**: weakest LLM, weakest embed, no reranker, small chunk, small top_k.
      - **Tier2-lower-mid**: lower-mid LLM, lower-mid embed, no reranker, mid chunk, mid top_k.
      - **Tier3-upper-mid**: upper-mid LLM, upper-mid embed, best reranker, mid-large chunk, large top_k.
      - **Tier4-strong**: strongest LLM, strongest embed, best reranker, max chunk, max top_k.

    A 4-point embedding gradient (rather than a binary weakest/strongest
    split) keeps Tier3 and Tier4 outcomes from saturating together when
    both already use strong-LLM + best-reranker — the embedding axis is
    where most of the remaining T3↔T4 separation comes from.

    The previous "Cross" probe (max retrieval + weak LLM) is gone — its
    diagnostic value was attribution between LLM and retrieval, but the
    discrimination scorer needs ordinal probes to rank-correlate against,
    and Cross sat off-axis. Use four ordered tiers instead.

    Chunk token sizes are capped at each embedding model's max_tokens
    (from ``config.embedding_token_limits``).

    Returns:
        List of ``(label, TrialConfig)`` tuples in weakest-first order.
        May be shorter than 4 if the search space is narrow (duplicates
        collapse on ``(structural_fingerprint, llm, reranker)``).
    """
    ss = config.search_space
    token_limits = config.embedding_token_limits

    llms = ranked_llms or ss.all_llm_models()
    embeds = ranked_embeds or ss.embedding.models
    rerankers = ranked_rerankers or ss.reranker.models

    chunk_min = int(_dim_min_value(ss.chunking.chunk_token_size))
    chunk_max = int(_dim_max_value(ss.chunking.chunk_token_size))
    top_k_min = int(_dim_min_value(ss.retrieval.top_k))
    top_k_max = int(_dim_max_value(ss.retrieval.top_k))
    top_k_lo_mid = (top_k_min + (top_k_min + top_k_max) // 2) // 2
    top_k_hi_mid = ((top_k_min + top_k_max) // 2 + top_k_max) // 2

    n_llms = len(llms)
    tier1_llm = llms[0]
    tier2_llm = llms[n_llms // 4] if n_llms >= 4 else llms[max(0, n_llms // 3)]
    tier3_llm = llms[3 * n_llms // 4] if n_llms >= 4 else llms[min(n_llms - 1, 2 * n_llms // 3)]
    tier4_llm = llms[-1]

    n_embeds = len(embeds)
    tier1_embed = embeds[0]
    tier2_embed = embeds[n_embeds // 4] if n_embeds >= 4 else embeds[max(0, n_embeds // 3)]
    tier3_embed = embeds[3 * n_embeds // 4] if n_embeds >= 4 else embeds[min(n_embeds - 1, 2 * n_embeds // 3)]
    tier4_embed = embeds[-1]

    best_reranker = next((r for r in reversed(rerankers) if r != "none"), "none")
    reranker_top_n_min = int(_dim_min_value(ss.reranker.top_n))
    reranker_top_n_max = int(_dim_max_value(ss.reranker.top_n))
    reranker_top_n_mid = (reranker_top_n_min + reranker_top_n_max) // 2

    def _cap_chunk(size: int, embed_model: str) -> int:
        limit = token_limits.get(embed_model)
        if limit is not None and size > limit:
            return limit
        return size

    def _overlap(chunk_token_size: int) -> int:
        return min(max(0, chunk_token_size // 10), chunk_token_size - 1)

    chunk_t1 = _cap_chunk(chunk_min, tier1_embed)
    chunk_t2 = _cap_chunk((chunk_min + (chunk_min + chunk_max) // 2) // 2, tier2_embed)
    chunk_t3 = _cap_chunk(((chunk_min + chunk_max) // 2 + chunk_max) // 2, tier3_embed)
    chunk_t4 = _cap_chunk(chunk_max, tier4_embed)

    def _short(model: str) -> str:
        return model.rsplit("/", 1)[-1]

    labelled_dicts: list[tuple[str, dict]] = [
        (
            f"Tier1-weak (llm={_short(tier1_llm)}, embed={_short(tier1_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t1,
                "chunk_token_overlap": _overlap(chunk_t1),
                "embedding_model": tier1_embed,
                "index_type": ss.retrieval.index_types[0],
                "top_k": top_k_min,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "generator_llm": tier1_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier2-lower-mid (llm={_short(tier2_llm)}, embed={_short(tier2_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t2,
                "chunk_token_overlap": _overlap(chunk_t2),
                "embedding_model": tier2_embed,
                "index_type": ss.retrieval.index_types[0],
                "top_k": top_k_lo_mid,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "generator_llm": tier2_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier3-upper-mid (llm={_short(tier3_llm)}, embed={_short(tier3_embed)}, reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t3,
                "chunk_token_overlap": _overlap(chunk_t3),
                "embedding_model": tier3_embed,
                "index_type": ss.retrieval.index_types[-1],
                "top_k": top_k_hi_mid,
                "reranker": best_reranker,
                "reranker_top_n": reranker_top_n_mid,
                "generator_llm": tier3_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier4-strong (llm={_short(tier4_llm)}, embed={_short(tier4_embed)}, reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[-1]
                if len(ss.chunking.strategies) > 1
                else ss.chunking.strategies[0],
                "chunk_token_size": chunk_t4,
                "chunk_token_overlap": _overlap(chunk_t4),
                "embedding_model": tier4_embed,
                "index_type": ss.retrieval.index_types[-1],
                "top_k": top_k_max,
                "reranker": best_reranker,
                "reranker_top_n": reranker_top_n_max,
                "generator_llm": tier4_llm,
                "temperature": 0.0,
            },
        ),
    ]

    seen: set[str] = set()
    probes: list[tuple[str, TrialConfig]] = []
    for label, d in labelled_dicts:
        try:
            tc = TrialConfig.model_validate(d)
        except Exception:
            continue
        key = tc.structural_fingerprint() + tc.generator_llm + tc.reranker
        if key not in seen:
            seen.add(key)
            probes.append((label, tc))

    if len(probes) <= 1:
        logger.warning(
            "Search space is narrow — all probe configs are identical. Probe-based selection will have no effect."
        )
    elif len(probes) < 3:
        logger.warning(
            "Only %d distinct probe tier(s) — Kendall-tau discrimination is noisy below 3 tiers.",
            len(probes),
        )

    logger.info("Generated %d unique probe configs", len(probes))
    return probes


# Curated allowlist of 4-probe outcome patterns and their target shares of
# the exam. Probes are emitted weakest-first (indices 0,1,2,3 = weak, lower-
# mid, upper-mid, strong). Weights sum to 1.0. Any pattern not listed here
# is excluded from the exam — it never enters regardless of inventory.
#
# Rationale:
#   (0,0,0,1) — only strongest probe solves; cleanest top-boundary discrimination.
#   (0,0,1,0) — upper-mid solves, strong fails; top-pair disagreement, valuable
#               when the two top probes trade blows on a corpus.
#   (0,0,0,0) — all probes fail; the optimizer may still build configs that
#               beat the probes, so reserve a slot.
#   (0,0,1,1) — top half solves; secondary top-boundary signal.
#   (0,1,1,1) — only weakest fails; mid/weak boundary, sanity-check signal.
#   (0,1,1,0) — middle solves; mid-band anomaly, kept at low weight.
#   (1,1,1,0) — anti-aligned at top; strong overthink or probe noise.
# Patterns with a 1-bit at probe-0 and 0-bits above (e.g., (1,0,0,0)) are
# overwhelmingly probe-noise and excluded.
PATTERN_WEIGHTS: dict[tuple[int, ...], float] = {
    (0, 0, 0, 1): 0.40,
    (0, 0, 1, 0): 0.30,
    (0, 0, 0, 0): 0.15,
    (0, 0, 1, 1): 0.10,
    (0, 1, 1, 1): 0.02,
    (0, 1, 1, 0): 0.02,
    (1, 1, 1, 0): 0.01,
}

_EXPECTED_PROBE_COUNT = len(next(iter(PATTERN_WEIGHTS)))


def collect_probe_outcomes(
    probe_results: list[ExamResult],
    questions: list[OpenEndedQuestion],
) -> tuple[dict[str, list[int]], set[str]]:
    """Build per-question binary correctness vectors across probe runs.

    Returns ``(outcomes, errored_ids)``:
      * ``outcomes`` — vector per question, ordered weakest→strongest.
        Missing entries (probe didn't evaluate a question due to error
        or content filter) are recorded as 0.
      * ``errored_ids`` — questions where at least one probe failed to
        produce a verdict. ``select_exam`` excludes these because a
        0-defaulted outcome can't be distinguished from a legitimate
        wrong answer; in particular it would corrupt the all-wrong
        bucket (whose intent is "questions a stronger config might
        still answer") with probe-noise items.
    """
    if not probe_results:
        return {q.id: [] for q in questions}, set()

    out: dict[str, list[int]] = {q.id: [] for q in questions}
    errored_ids: set[str] = set()
    for result in probe_results:
        evaluated = {qr.question_id for qr in result.question_results}
        result_map = {qr.question_id: int(qr.correct) for qr in result.question_results}
        for qid in out:
            if qid in evaluated:
                out[qid].append(result_map[qid])
            else:
                out[qid].append(0)
                errored_ids.add(qid)
    return out, errored_ids


def attach_probe_metadata(
    questions: list[OpenEndedQuestion],
    outcomes: dict[str, list[int]],
) -> list[OpenEndedQuestion]:
    """Return copies of the questions with ``probe_outcomes`` filled in."""
    updated: list[OpenEndedQuestion] = []
    for q in questions:
        updated.append(
            q.model_copy(update={"probe_outcomes": list(outcomes.get(q.id, []))}),
        )
    return updated


def _stratum_label(q: OpenEndedQuestion) -> str:
    """Derive the seed-origin stratum from question shape (logging only)."""
    if q.num_hops == 1:
        return "single_chunk"
    if q.is_multi_doc:
        return "cross_doc_pair"
    return "same_doc_pair"


def allocate_quotas(
    weights: dict[tuple[int, ...], float],
    inventory: dict[tuple[int, ...], int],
    exam_size: int,
) -> dict[tuple[int, ...], int]:
    """Iterated largest-remainder allocation with inventory caps.

    Each iteration distributes the remaining slots across patterns that
    still have inventory, proportional to their weights. When a pattern
    hits its inventory cap it's evicted from the pool and the remaining
    deficit cascades proportionally to the surviving patterns. Bounded
    by ``len(weights) + 1`` iterations: each iter either fills the exam
    or evicts at least one pattern.
    """
    remaining = {p: w for p, w in weights.items() if w > 0}
    quota: dict[tuple[int, ...], int] = {p: 0 for p in weights}
    slots = exam_size

    for _ in range(len(weights) + 1):
        if slots <= 0 or not remaining:
            break
        wsum = sum(remaining.values())
        if wsum <= 0:
            break

        ideals = {p: (w / wsum) * slots for p, w in remaining.items()}
        round_down = {p: int(v) for p, v in ideals.items()}
        used = sum(round_down.values())
        residuals = sorted(
            ((v - round_down[p], p) for p, v in ideals.items()),
            reverse=True,
        )
        for _, p in residuals[: slots - used]:
            round_down[p] += 1

        progress = 0
        for p, want in round_down.items():
            room = inventory.get(p, 0) - quota[p]
            take = min(want, room)
            quota[p] += take
            progress += take
            if take < want:
                remaining.pop(p, None)

        if progress == 0:
            break
        slots -= progress

    return quota


def select_exam(
    candidates: list[OpenEndedQuestion],
    outcomes: dict[str, list[int]],
    exam_size: int,
    errored_ids: set[str] | None = None,
) -> list[OpenEndedQuestion]:
    """Pick ``exam_size`` candidates by curated pattern allowlist.

    Each pattern in ``PATTERN_WEIGHTS`` gets a target slot count
    proportional to its weight. Under-supplied patterns cascade their
    deficit proportionally to the remaining allowlisted patterns via
    ``allocate_quotas``. Patterns not in the allowlist are excluded
    entirely. Within each pattern bucket candidates are sorted by id
    so selection is reproducible.

    Questions in ``errored_ids`` (where at least one probe failed to
    produce a verdict) are excluded entirely — their outcome vectors
    contain 0-defaulted slots that can't be distinguished from real
    wrong answers, and admitting them would corrupt the all-wrong
    bucket with probe-noise.

    Fallback: if the probe count doesn't match the allowlist's pattern
    length (e.g., a narrow search space yielded fewer probes), this
    returns the first ``exam_size`` candidates by id and logs a warning.
    """
    if not candidates:
        return []

    errored_ids = errored_ids or set()
    if errored_ids:
        n_total = len(candidates)
        candidates = [q for q in candidates if q.id not in errored_ids]
        logger.info(
            "Excluded %d/%d candidate(s) with probe-evaluation errors",
            n_total - len(candidates),
            n_total,
        )
        if not candidates:
            return []

    sample = next(
        (outcomes.get(q.id) for q in candidates if outcomes.get(q.id)),
        None,
    )
    if sample is None or len(sample) != _EXPECTED_PROBE_COUNT:
        logger.warning(
            "select_exam: probe count is %d (expected %d); falling back to id-order truncation",
            len(sample) if sample else 0,
            _EXPECTED_PROBE_COUNT,
        )
        return sorted(candidates, key=lambda q: q.id)[:exam_size]

    by_pat: dict[tuple[int, ...], list[OpenEndedQuestion]] = {p: [] for p in PATTERN_WEIGHTS}
    for q in candidates:
        pat = tuple(outcomes.get(q.id, []))
        if pat in by_pat:
            by_pat[pat].append(q)
    for items in by_pat.values():
        items.sort(key=lambda q: q.id)

    inventory = {p: len(items) for p, items in by_pat.items()}
    quota = allocate_quotas(PATTERN_WEIGHTS, inventory, exam_size)

    selected: list[OpenEndedQuestion] = []
    for p, n in quota.items():
        if n > 0:
            selected.extend(by_pat[p][:n])

    audit = [
        f"{''.join(str(b) for b in p)}: wanted={round(w * exam_size)} got={quota[p]} avail={inventory[p]}"
        for p, w in PATTERN_WEIGHTS.items()
    ]
    logger.info("Exam selection by pattern: %s", "; ".join(audit))

    if len(selected) < exam_size:
        logger.warning(
            "Exam under-filled: %d/%d. Probe set may be miscalibrated for this corpus — "
            "expand pair_overgeneration_factor or recheck probe tiers.",
            len(selected),
            exam_size,
        )

    return selected
