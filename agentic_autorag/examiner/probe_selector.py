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

import json
import logging

import numpy as np

from agentic_autorag.config.models import MCQQuestion, ProjectConfig, TrialConfig
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
        import litellm

        prompt = _RANK_MODELS_PROMPT.format(
            model_type=model_type,
            model_list="\n".join(f"- {m}" for m in model_names),
        )
        response = await litellm.acompletion(
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
    """Build 2-4 diverse probe configurations from the search space extremes.

    Probes represent the extremes of the search space so that a question
    that is hard under all probes (or trivially easy for all) gets a low
    discrimination score, while a question that differentiates the configs
    gets a high score.

    When ranked model lists are provided (from ``rank_models_for_probes``),
    those are used to pick weak/strong extremes. Otherwise falls back to
    search space list ordering.

    Chunk token sizes are capped at each embedding model's max_tokens
    (from ``config.embedding_token_limits``) to avoid invalid configurations.

    Args:
        config: Full project configuration.
        ranked_llms: LLM models sorted weakest-first.
        ranked_embeds: Embedding models sorted weakest-first.
        ranked_rerankers: Reranker models sorted weakest-first.

    Returns:
        List of ``(label, TrialConfig)`` tuples. Labels describe the probe
        archetype (e.g. ``"Weak (weak LLM, weak embed, no reranker)"``).
        May be shorter than 4 if the search space is narrow.
    """
    ss = config.search_space
    token_limits = config.embedding_token_limits

    llms = ranked_llms or ss.llm_models
    embeds = ranked_embeds or ss.embedding_models
    rerankers = ranked_rerankers or ss.reranker.models

    chunk_min = int(ss.chunking.chunk_token_size.min)
    chunk_max = int(ss.chunking.chunk_token_size.max)
    top_k_values = sorted([int(ss.top_k.min), int(ss.top_k.max)])

    weak_llm = llms[0]
    strong_llm = llms[-1]
    weak_embed = embeds[0]
    strong_embed = embeds[-1]
    small_top_k = top_k_values[0]
    large_top_k = top_k_values[-1]

    # Best reranker = last non-"none" in the ranked list
    best_reranker = next((r for r in reversed(rerankers) if r != "none"), "none")
    reranker_top_n_min = int(ss.reranker.top_n.min)
    reranker_top_n_max = int(ss.reranker.top_n.max)

    def _cap_chunk(size: int, embed_model: str) -> int:
        """Cap chunk_token_size at the embedding model's max_tokens."""
        limit = token_limits.get(embed_model)
        if limit is not None and size > limit:
            return limit
        return size

    # Overlap must be < chunk_token_size; use 10% of chunk_token_size
    def _overlap(chunk_token_size: int) -> int:
        return min(max(0, chunk_token_size // 10), chunk_token_size - 1)

    # Compute chunk sizes per embedding model, respecting token limits
    small_chunk_weak = _cap_chunk(chunk_min, weak_embed)
    large_chunk_strong = _cap_chunk(chunk_max, strong_embed)
    mid_chunk_weak = _cap_chunk((chunk_min + chunk_max) // 2, weak_embed)
    large_chunk_cross = _cap_chunk(chunk_max, strong_embed)

    def _short(model: str) -> str:
        """Last path component of a model name for compact logging."""
        return model.rsplit("/", 1)[-1]

    labelled_dicts: list[tuple[str, dict]] = [
        (
            f"Weak (llm={_short(weak_llm)}, embed={_short(weak_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": small_chunk_weak,
                "chunk_token_overlap": _overlap(small_chunk_weak),
                "embedding_model": weak_embed,
                "index_type": ss.index_types[0],
                "top_k": small_top_k,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "llm_model": weak_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Strong (llm={_short(strong_llm)}, embed={_short(strong_embed)}, reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[-1]
                if len(ss.chunking.strategies) > 1
                else ss.chunking.strategies[0],
                "chunk_token_size": large_chunk_strong,
                "chunk_token_overlap": _overlap(large_chunk_strong),
                "embedding_model": strong_embed,
                "index_type": ss.index_types[-1],
                "top_k": large_top_k,
                "reranker": best_reranker,
                "reranker_top_n": reranker_top_n_max,
                "llm_model": strong_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Balanced (llm={_short(weak_llm)}, embed={_short(weak_embed)}, no reranker)",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": mid_chunk_weak,
                "chunk_token_overlap": _overlap(mid_chunk_weak),
                "embedding_model": weak_embed,
                "index_type": ss.index_types[0],
                "top_k": (small_top_k + large_top_k) // 2,
                "reranker": "none",
                "reranker_top_n": reranker_top_n_min,
                "llm_model": weak_llm,
                "temperature": 0.0,
            },
        ),
        (
            f"Cross (llm={_short(weak_llm)}, embed={_short(strong_embed)}, reranker={_short(best_reranker)})",
            {
                "chunking_strategy": ss.chunking.strategies[0],
                "chunk_token_size": large_chunk_cross,
                "chunk_token_overlap": _overlap(large_chunk_cross),
                "embedding_model": strong_embed,
                "index_type": ss.index_types[0],
                "top_k": large_top_k,
                "reranker": best_reranker,
                "reranker_top_n": reranker_top_n_max,
                "llm_model": weak_llm,
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
        key = tc.structural_fingerprint() + tc.llm_model + tc.reranker
        if key not in seen:
            seen.add(key)
            probes.append((label, tc))

    if len(probes) <= 1:
        logger.warning(
            "Search space is narrow — all probe configs are identical. Probe-based selection will have no effect."
        )

    logger.info("Generated %d unique probe configs", len(probes))
    return probes


_ALL_WRONG_HARD_CAP_RATIO = 0.1


def collect_probe_outcomes(
    probe_results: list[ExamResult],
    questions: list[MCQQuestion],
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
    questions: list[MCQQuestion],
    outcomes: dict[str, list[int]],
    scores: dict[str, float],
) -> list[MCQQuestion]:
    """Return copies of the questions with probe_outcomes + discrimination_entropy filled in."""
    updated: list[MCQQuestion] = []
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
    questions: list[MCQQuestion],
) -> dict[str, float]:
    """Compute discrimination score for each question across probe results.

    Questions are classified into three tiers:

    * **Mixed** (some probes correct, some wrong): score = variance of the
      binary response vector.  These are the proven discriminators.
    * **All correct** (every probe right): score = 0  (too easy).
    * **All wrong** (every probe wrong, no errors): assigned synthetic scores
      that interleave them evenly across the mixed-question ranking.  This
      ensures the final exam contains a few very-hard items that can
      differentiate configs beyond the strong probe.
    * **Error** (any probe returned no answer due to API / content-filter
      failures): score = 0  (broken, not hard).

    Returns:
        dict mapping question_id → discrimination score.
    """
    if not probe_results:
        return {q.id: 0.0 for q in questions}

    question_ids = {q.id for q in questions}

    # Build per-question binary response vectors and track errors
    responses: dict[str, list[int]] = {qid: [] for qid in question_ids}
    has_error: set[str] = set()

    for result in probe_results:
        evaluated = {qr.question_id for qr in result.question_results}
        result_map = {qr.question_id: int(qr.correct) for qr in result.question_results}
        for qid in question_ids:
            if qid not in evaluated:
                # Missing from results → probe error (content filter, timeout, etc.)
                has_error.add(qid)
                responses[qid].append(0)
            else:
                responses[qid].append(result_map[qid])

    # Score mixed questions by variance
    mixed_scores: dict[str, float] = {}
    all_wrong_ids: list[str] = []

    for qid, binary_vec in responses.items():
        arr = np.array(binary_vec, dtype=np.float32)
        mean_val = float(arr.mean())
        variance = float(np.var(arr))

        if qid in has_error:
            # Broken question — treat as zero discrimination
            mixed_scores[qid] = 0.0
        elif mean_val == 0.0:
            # All wrong — genuinely very hard, handle separately
            all_wrong_ids.append(qid)
        elif variance == 0.0:
            # All correct — too easy
            mixed_scores[qid] = 0.0
        else:
            mixed_scores[qid] = variance

    # Interleave all-wrong questions evenly across the mixed ranking
    scores = dict(mixed_scores)
    if all_wrong_ids:
        sorted_mixed = sorted(
            [(s, qid) for qid, s in mixed_scores.items() if s > 0.0],
            reverse=True,
        )
        n_mixed = len(sorted_mixed)
        n_hard = len(all_wrong_ids)

        if n_mixed == 0:
            # No mixed questions at all — give all-wrong a small positive score
            for qid in all_wrong_ids:
                scores[qid] = 0.01
        else:
            # Place at evenly spaced positions: pos_i = (i+1) * n_mixed / (n_hard+1)
            for i, qid in enumerate(all_wrong_ids):
                pos = int((i + 1) * n_mixed / (n_hard + 1))
                pos = min(pos, n_mixed - 1)
                scores[qid] = sorted_mixed[pos][0] - 1e-6

    return scores


def select_exam(
    candidates: list[MCQQuestion],
    scores: dict[str, float],
    exam_size: int,
    all_wrong_ids: set[str] | None = None,
) -> list[MCQQuestion]:
    """Greedy exam selection that maximises discrimination while preserving cluster diversity.

    Strategy:
    1. Compute proportional cluster allocations (largest remainder method).
    2. Fill each cluster quota with the highest-scoring candidates from that cluster.
    3. Fill any remaining slots globally from the highest-scoring unused candidates.
    4. Cap "all wrong" questions at ~10 % of ``exam_size`` to keep the exam
       representative without letting unsolvable items dominate.

    Args:
        candidates: All validated candidate questions.
        scores: Per-question discrimination score from score_questions_by_discrimination.
        exam_size: Target number of questions in the final exam.
        all_wrong_ids: Optional set of question IDs that ALL probes answered incorrectly.

    Returns:
        Selected questions, up to exam_size.
    """
    if not candidates:
        return []

    exam_size = min(exam_size, len(candidates))

    # Group candidates by cluster
    clusters: dict[int, list[MCQQuestion]] = {}
    for q in candidates:
        clusters.setdefault(q.cluster_id, []).append(q)

    cluster_ids = sorted(clusters.keys())
    cluster_sizes = np.array([len(clusters[c]) for c in cluster_ids], dtype=np.float64)

    # Proportional allocation
    total_weight = cluster_sizes.sum()
    if total_weight == 0:
        return candidates[:exam_size]

    raw_alloc = cluster_sizes / total_weight * exam_size
    floor_alloc = np.floor(raw_alloc).astype(int)
    remainders = raw_alloc - floor_alloc
    deficit = exam_size - floor_alloc.sum()

    # Distribute deficit to clusters with largest remainders
    if deficit > 0:
        remainder_order = np.argsort(-remainders)
        for i in range(int(deficit)):
            if i < len(remainder_order):
                floor_alloc[remainder_order[i]] += 1

    # Cap each allocation at cluster size
    for i, cid in enumerate(cluster_ids):
        floor_alloc[i] = min(floor_alloc[i], len(clusters[cid]))

    selected: list[MCQQuestion] = []
    used_ids: set[str] = set()

    # Fill per-cluster quotas
    for i, cid in enumerate(cluster_ids):
        quota = int(floor_alloc[i])
        if quota <= 0:
            continue
        sorted_qs = sorted(clusters[cid], key=lambda q: scores.get(q.id, 0.0), reverse=True)
        for q in sorted_qs[:quota]:
            selected.append(q)
            used_ids.add(q.id)

    # Global fill if quota total falls short of exam_size
    if len(selected) < exam_size:
        remaining = [q for q in candidates if q.id not in used_ids]
        remaining.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
        for q in remaining[: exam_size - len(selected)]:
            selected.append(q)

    # Cap "all wrong" questions to prevent unsolvable items from dominating
    if all_wrong_ids:
        max_hard = max(1, exam_size // 10)
        hard_in_exam = [q for q in selected if q.id in all_wrong_ids]
        if len(hard_in_exam) > max_hard:
            drop = set(q.id for q in hard_in_exam[max_hard:])
            selected = [q for q in selected if q.id not in drop]
            # Backfill dropped slots from unused mixed candidates
            remaining = [q for q in candidates if q.id not in {s.id for s in selected} and q.id not in all_wrong_ids]
            remaining.sort(key=lambda q: scores.get(q.id, 0.0), reverse=True)
            for q in remaining[: len(drop)]:
                selected.append(q)
            logger.info("Capped all-wrong questions: kept %d, replaced %d with mixed", max_hard, len(drop))

    logger.info(
        "Probe-based selection: %d/%d candidates selected (exam_size=%d)",
        len(selected),
        len(candidates),
        exam_size,
    )
    return selected
