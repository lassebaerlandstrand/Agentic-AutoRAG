"""Probe-based question selection for discriminating exam construction.

When probe_selection is enabled, the orchestrator evaluates candidate questions
against a small set of diverse pipeline configurations (probes). Questions that
produce mixed results across probes (some correct, some wrong) are the most
discriminating — they actually differentiate between strong and weak RAG setups.

Public API:
  rank_models_for_probes   — rank models by quality (KB + LLM fallback)
  select_probe_configs     — build diverse probe TrialConfigs from a ProjectConfig
  collect_probe_outcomes   — per-question 4-bit correctness vectors across probes
  score_questions_by_discrimination  — compute per-question discrimination score
  attach_probe_metadata    — write probe_outcomes + discrimination_entropy onto questions
  select_exam              — greedy selection respecting cluster diversity
"""

from __future__ import annotations

import hashlib
import json
import logging
import random

import numpy as np

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
) -> list[str]:
    """Rank models from weakest to strongest for probe config generation.

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
    ranked_known, unknowns = _kb_rank(model_names, model_type, knowledge_base)

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
    rank-correlate outcomes against probe strength directly. Tiers are
    spaced evenly across the ranked LLM list at indices ``0``, ``n//4``,
    ``3n//4``, ``-1`` and paired with a monotonically improving retrieval
    stack (embed / reranker / chunk size / top_k all step up):

      - **Tier1-weak**: weakest LLM, weakest embed, no reranker, small chunk, small top_k.
      - **Tier2-lower-mid**: lower-mid LLM, weakest embed, no reranker, mid chunk, mid top_k.
      - **Tier3-upper-mid**: upper-mid LLM, strongest embed, best reranker, mid-large chunk, large top_k.
      - **Tier4-strong**: strongest LLM, strongest embed, best reranker, max chunk, max top_k.

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

    llms = ranked_llms or ss.llm_models.all_models()
    embeds = ranked_embeds or ss.embedding_models
    rerankers = ranked_rerankers or ss.reranker.models

    chunk_min = int(_dim_min_value(ss.chunking.chunk_token_size))
    chunk_max = int(_dim_max_value(ss.chunking.chunk_token_size))
    top_k_min = int(_dim_min_value(ss.top_k))
    top_k_max = int(_dim_max_value(ss.top_k))
    top_k_lo_mid = (top_k_min + (top_k_min + top_k_max) // 2) // 2
    top_k_hi_mid = ((top_k_min + top_k_max) // 2 + top_k_max) // 2

    n_llms = len(llms)
    tier1_llm = llms[0]
    tier2_llm = llms[n_llms // 4] if n_llms >= 4 else llms[max(0, n_llms // 3)]
    tier3_llm = llms[3 * n_llms // 4] if n_llms >= 4 else llms[min(n_llms - 1, 2 * n_llms // 3)]
    tier4_llm = llms[-1]
    weak_embed = embeds[0]
    strong_embed = embeds[-1]

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

    chunk_t1 = _cap_chunk(chunk_min, weak_embed)
    chunk_t2 = _cap_chunk((chunk_min + (chunk_min + chunk_max) // 2) // 2, weak_embed)
    chunk_t3 = _cap_chunk(((chunk_min + chunk_max) // 2 + chunk_max) // 2, strong_embed)
    chunk_t4 = _cap_chunk(chunk_max, strong_embed)

    def _short(model: str) -> str:
        return model.rsplit("/", 1)[-1]

    labelled_dicts: list[tuple[str, dict]] = [
        (
            f"Tier1-weak (llm={_short(tier1_llm)}, embed={_short(weak_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t1,
                "chunk_token_overlap": _overlap(chunk_t1),
                "embedding_model": weak_embed,
                "index_type": ss.index_types[0],
                "top_k": top_k_min,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "generator_llm": tier1_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier2-lower-mid (llm={_short(tier2_llm)}, embed={_short(weak_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t2,
                "chunk_token_overlap": _overlap(chunk_t2),
                "embedding_model": weak_embed,
                "index_type": ss.index_types[0],
                "top_k": top_k_lo_mid,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "generator_llm": tier2_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier3-upper-mid (llm={_short(tier3_llm)}, embed={_short(strong_embed)}, "
            f"reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": chunk_t3,
                "chunk_token_overlap": _overlap(chunk_t3),
                "embedding_model": strong_embed,
                "index_type": ss.index_types[-1],
                "top_k": top_k_hi_mid,
                "reranker": best_reranker,
                "reranker_top_n": reranker_top_n_mid,
                "generator_llm": tier3_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Tier4-strong (llm={_short(tier4_llm)}, embed={_short(strong_embed)}, reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[-1]
                if len(ss.chunking.strategies) > 1
                else ss.chunking.strategies[0],
                "chunk_token_size": chunk_t4,
                "chunk_token_overlap": _overlap(chunk_t4),
                "embedding_model": strong_embed,
                "index_type": ss.index_types[-1],
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


_ALL_WRONG_HARD_CAP_RATIO = 0.3


def collect_probe_outcomes(
    probe_results: list[ExamResult],
    questions: list[OpenEndedQuestion],
) -> dict[str, list[int]]:
    """Build per-question binary correctness vectors across probe runs.

    Order in the returned vector matches the order of ``probe_results``.
    Missing entries (probe that didn't evaluate a question due to an error
    or content filter) are recorded as 0 — same as ``score_questions_by_
    discrimination`` does — so callers can tell apart a "broken probe"
    from a "probe that answered incorrectly" only by also looking at
    has-error sets if needed.
    """
    if not probe_results:
        return {q.id: [] for q in questions}

    out: dict[str, list[int]] = {q.id: [] for q in questions}
    for result in probe_results:
        result_map = {qr.question_id: int(qr.correct) for qr in result.question_results}
        for qid in out:
            out[qid].append(result_map.get(qid, 0))
    return out


def attach_probe_metadata(
    questions: list[OpenEndedQuestion],
    outcomes: dict[str, list[int]],
    scores: dict[str, float],
) -> list[OpenEndedQuestion]:
    """Return copies of the questions with probe_outcomes + discrimination_entropy filled in."""
    updated: list[OpenEndedQuestion] = []
    for q in questions:
        updated.append(
            q.model_copy(
                update={
                    "probe_outcomes": list(outcomes.get(q.id, [])),
                    "discrimination_entropy": float(scores.get(q.id, 0.0)),
                }
            )
        )
    return updated


def score_questions_by_discrimination(
    probe_results: list[ExamResult],
    questions: list[OpenEndedQuestion],
) -> dict[str, float]:
    """Compute discrimination score for each question across probe results.

    The exam's purpose is to rank RAG configurations by quality, so a
    question is informative when its outcome correlates with probe
    strength: stronger probes solve it, weaker probes fail it. We score
    each question by Kendall's tau between the probe-strength ranking
    (probes are passed weakest-first) and the binary outcome vector,
    multiplied by ``(1 - mean_correctness)`` so harder splits score
    higher than easier splits at equal correlation.

    * **Aligned mixed** (positive tau, mixed outcomes): primary signal.
      Score in [0, 1] — items only the strong probes solve get top
      scores.
    * **Anti-aligned mixed** (negative tau): clipped to 0 so anomalies
      don't compete with informative items, but counted in DIAG so we
      can monitor probe-rank breakage.
    * **All correct** (mean=1): score = 0 (saturated, no signal).
    * **All wrong** (mean=0, no errors): synthetic interleave across
      the mixed ranking — keeps a small share of very-hard items in
      the exam, capped downstream by ``select_exam``.
    * **Error** (any probe returned no answer due to API / content
      filter): score = 0 (broken, not hard).

    Returns:
        dict mapping question_id → discrimination score.
    """
    if not probe_results:
        return {q.id: 0.0 for q in questions}

    question_ids = {q.id for q in questions}
    n_probes = len(probe_results)

    responses: dict[str, list[int]] = {qid: [] for qid in question_ids}
    has_error: set[str] = set()

    for result in probe_results:
        evaluated = {qr.question_id for qr in result.question_results}
        result_map = {qr.question_id: int(qr.correct) for qr in result.question_results}
        for qid in question_ids:
            if qid not in evaluated:
                has_error.add(qid)
                responses[qid].append(0)
            else:
                responses[qid].append(result_map[qid])

    mixed_scores: dict[str, float] = {}
    all_wrong_ids: list[str] = []
    n_anti_aligned = 0

    # Probe-strength rank: probes were emitted weakest-first by select_probe_configs,
    # so a strict ascending vector matches "stronger → more likely correct".
    strength_ranks = list(range(n_probes))

    for qid, binary_vec in responses.items():
        arr = np.array(binary_vec, dtype=np.float32)
        mean_val = float(arr.mean())

        if qid in has_error:
            mixed_scores[qid] = 0.0
            continue
        if mean_val == 0.0:
            all_wrong_ids.append(qid)
            continue
        if mean_val == 1.0:
            mixed_scores[qid] = 0.0
            continue

        tau = _kendall_tau_binary(strength_ranks, binary_vec)
        if tau < 0:
            n_anti_aligned += 1
        clipped = max(0.0, tau)
        mixed_scores[qid] = clipped * (1.0 - mean_val)

    if n_anti_aligned > 0:
        logger.info(
            "DIAG Discrimination anti-aligned items: %d (tau<0; weaker probes solved while stronger failed)",
            n_anti_aligned,
        )

    scores = dict(mixed_scores)
    if all_wrong_ids:
        sorted_mixed = sorted(
            [(s, qid) for qid, s in mixed_scores.items() if s > 0.0],
            reverse=True,
        )
        n_mixed = len(sorted_mixed)
        n_hard = len(all_wrong_ids)
        if n_mixed == 0:
            for qid in all_wrong_ids:
                scores[qid] = 0.01
        else:
            for i, qid in enumerate(all_wrong_ids):
                pos = int((i + 1) * n_mixed / (n_hard + 1))
                pos = min(pos, n_mixed - 1)
                scores[qid] = sorted_mixed[pos][0] - 1e-6

    return scores


def _kendall_tau_binary(strength_ranks: list[int], outcomes: list[int]) -> float:
    """Kendall's tau between two equal-length sequences, accepting binary outcomes.

    Returns a value in [-1, 1]. With 4 probes and a binary outcome vector,
    ties are common — we use tau-b (denominator includes ties), which
    keeps the score interpretable when several probes share the same
    outcome bit.
    """
    n = len(outcomes)
    if n < 2 or n != len(strength_ranks):
        return 0.0
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = strength_ranks[j] - strength_ranks[i]
            dy = outcomes[j] - outcomes[i]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    denom_x = total_pairs - ties_x
    denom_y = total_pairs - ties_y
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return (concordant - discordant) / ((denom_x * denom_y) ** 0.5)


def _stratum_label(q: OpenEndedQuestion) -> str:
    """Derive the seed-origin stratum from question shape (logging only)."""
    if q.num_hops == 1:
        return "single_chunk"
    if q.is_multi_doc:
        return "cross_doc_pair"
    return "same_doc_pair"


def select_exam(
    candidates: list[OpenEndedQuestion],
    scores: dict[str, float],
    exam_size: int,
    all_wrong_ids: set[str] | None = None,
) -> list[OpenEndedQuestion]:
    """Pick the most discriminating ``exam_size`` candidates, with cluster diversity.

    The exam's purpose is to differentiate RAG configurations, so selection
    is driven by raw discrimination score — no per-origin or per-type quota.
    Cluster diversity (proportional allocation across ``cluster_id``) is the
    only structural constraint, since questions sharing a cluster come from
    nearly-duplicate chunk content and probe identically by construction.

    Strategy:
    1. Group by ``cluster_id`` and allocate ``exam_size`` proportionally
       (largest-remainder rounding) so we don't burn all slots on one
       topical bucket.
    2. Within each cluster, take the highest-scoring candidates first.
    3. Backfill any unallocated slots from the global score-sorted tail.
    4. Cap "all wrong" questions at ~10% of exam_size so a few very-hard
       items are kept but they don't dominate.

    The final log line breaks down origin and reasoning_type for visibility,
    but neither shapes the selection.
    """
    if not candidates:
        return []

    exam_size = min(exam_size, len(candidates))

    clusters: dict[int, list[OpenEndedQuestion]] = {}
    for q in candidates:
        clusters.setdefault(q.cluster_id, []).append(q)

    cluster_ids = sorted(clusters.keys())
    cluster_sizes = np.array([len(clusters[c]) for c in cluster_ids], dtype=np.float64)
    total_weight = cluster_sizes.sum()

    selected: list[OpenEndedQuestion] = []
    used_ids: set[str] = set()
    if total_weight > 0:
        raw_alloc = cluster_sizes / total_weight * exam_size
        floor_alloc = np.floor(raw_alloc).astype(int)
        remainders = raw_alloc - floor_alloc
        deficit = exam_size - int(floor_alloc.sum())
        if deficit > 0:
            for i in np.argsort(-remainders)[:deficit]:
                floor_alloc[i] += 1
        for i, cid in enumerate(cluster_ids):
            floor_alloc[i] = min(int(floor_alloc[i]), len(clusters[cid]))

        for i, cid in enumerate(cluster_ids):
            cl_quota = int(floor_alloc[i])
            if cl_quota <= 0:
                continue
            sorted_qs = sorted(clusters[cid], key=lambda q: scores.get(q.id, 0.0), reverse=True)
            for q in sorted_qs[:cl_quota]:
                selected.append(q)
                used_ids.add(q.id)

    if len(selected) < exam_size:
        remaining = [q for q in candidates if q.id not in used_ids]
        remaining.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
        for q in remaining[: exam_size - len(selected)]:
            selected.append(q)
            used_ids.add(q.id)

    if all_wrong_ids:
        max_hard = max(1, int(exam_size * _ALL_WRONG_HARD_CAP_RATIO))
        hard_in_exam = [q for q in selected if q.id in all_wrong_ids]
        if len(hard_in_exam) > max_hard:
            seed_str = "|".join(sorted(q.id for q in selected))
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:16], 16)
            shuffled = list(hard_in_exam)
            random.Random(seed).shuffle(shuffled)
            drop = {q.id for q in shuffled[max_hard:]}
            selected = [q for q in selected if q.id not in drop]
            remaining = [q for q in candidates if q.id not in {s.id for s in selected} and q.id not in all_wrong_ids]
            remaining.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
            for q in remaining[: len(drop)]:
                selected.append(q)
            logger.info(
                "Capped all-wrong questions: kept %d / %d (cap=%.0f%%), replaced %d with mixed",
                max_hard,
                len(hard_in_exam),
                _ALL_WRONG_HARD_CAP_RATIO * 100,
                len(drop),
            )

    origin_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for q in selected:
        origin_counts[_stratum_label(q)] = origin_counts.get(_stratum_label(q), 0) + 1
        type_counts[q.reasoning_type] = type_counts.get(q.reasoning_type, 0) + 1
    origin_breakdown = ", ".join(f"{lab}={origin_counts[lab]}" for lab in sorted(origin_counts.keys()))
    type_breakdown = ", ".join(f"{t}={type_counts[t]}" for t in sorted(type_counts.keys()))
    logger.info(
        "Probe-based selection: %d/%d candidates selected (exam_size=%d; origins: %s; types: %s)",
        len(selected),
        len(candidates),
        exam_size,
        origin_breakdown,
        type_breakdown,
    )
    return selected
