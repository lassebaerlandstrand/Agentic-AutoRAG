"""Probe-based question selection for discriminating exam construction.

When n_probes > 0, the orchestrator evaluates candidate questions against
a small set of diverse pipeline configurations (probes). Questions that
produce mixed results across probes (some correct, some wrong) are the most
discriminating — they actually differentiate between strong and weak RAG setups.

Public API:
  select_probe_configs  — build diverse probe TrialConfigs from a ProjectConfig
  score_questions_by_discrimination  — compute per-question discrimination score
  select_exam  — greedy selection respecting cluster diversity
"""

from __future__ import annotations

import logging

import numpy as np

from agentic_autorag.config.models import MCQQuestion, ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult

logger = logging.getLogger(__name__)


def select_probe_configs(config: ProjectConfig) -> list[TrialConfig]:
    """Build 2-4 diverse probe configurations from the search space extremes.

    Probes represent the extremes of the search space so that a question
    that is hard under all probes (or trivially easy for all) gets a low
    discrimination score, while a question that differentiates the configs
    gets a high score.

    Args:
        config: Full project configuration.

    Returns:
        List of unique probe TrialConfigs. May be shorter than 4 if the
        search space is narrow (e.g., only one embedding model, one LLM).
    """
    ss = config.search_space
    llm_models = ss.llm_models
    embed_models = ss.embedding_models
    chunk_sizes = sorted([int(ss.chunking.chunk_size.min), int(ss.chunking.chunk_size.max)])
    top_k_values = sorted([int(ss.top_k.min), int(ss.top_k.max)])

    weak_llm = llm_models[0]
    strong_llm = llm_models[-1]
    weak_embed = embed_models[0]
    strong_embed = embed_models[-1]
    small_chunk = chunk_sizes[0]
    large_chunk = chunk_sizes[-1]
    small_top_k = top_k_values[0]
    large_top_k = top_k_values[-1]

    # Overlap must be < chunk_size; use 10% of chunk_size
    def _overlap(chunk_size: int) -> int:
        return min(max(0, chunk_size // 10), chunk_size - 1)

    probe_dicts = [
        # Weak probe: small chunks, low top_k, first LLM
        {
            "chunking_strategy": ss.chunking.strategies[0],
            "chunk_size": small_chunk,
            "chunk_overlap": _overlap(small_chunk),
            "embedding_model": weak_embed,
            "index_type": ss.index_types[0],
            "top_k": small_top_k,
            "llm_model": weak_llm,
            "temperature": 0.0,
        },
        # Strong probe: large chunks, high top_k, last LLM
        {
            "chunking_strategy": ss.chunking.strategies[-1]
            if len(ss.chunking.strategies) > 1
            else ss.chunking.strategies[0],
            "chunk_size": large_chunk,
            "chunk_overlap": _overlap(large_chunk),
            "embedding_model": strong_embed,
            "index_type": ss.index_types[-1],
            "top_k": large_top_k,
            "llm_model": strong_llm,
            "temperature": 0.0,
        },
        # Balanced probe: midpoint values, first LLM
        {
            "chunking_strategy": ss.chunking.strategies[0],
            "chunk_size": (small_chunk + large_chunk) // 2,
            "chunk_overlap": _overlap((small_chunk + large_chunk) // 2),
            "embedding_model": weak_embed,
            "index_type": ss.index_types[0],
            "top_k": (small_top_k + large_top_k) // 2,
            "llm_model": weak_llm,
            "temperature": 0.0,
        },
        # Cross probe: strong embed + weak LLM
        {
            "chunking_strategy": ss.chunking.strategies[0],
            "chunk_size": large_chunk,
            "chunk_overlap": _overlap(large_chunk),
            "embedding_model": strong_embed,
            "index_type": ss.index_types[0],
            "top_k": large_top_k,
            "llm_model": weak_llm,
            "temperature": 0.0,
        },
    ]

    seen: set[str] = set()
    probes: list[TrialConfig] = []
    for d in probe_dicts:
        try:
            tc = TrialConfig.model_validate(d)
        except Exception:
            continue
        key = tc.structural_fingerprint() + tc.llm_model
        if key not in seen:
            seen.add(key)
            probes.append(tc)

    if len(probes) <= 1:
        logger.warning(
            "Search space is narrow — all probe configs are identical. "
            "Consider setting n_probes=0 to skip probe-based selection."
        )

    logger.info("Generated %d unique probe configs", len(probes))
    return probes


def score_questions_by_discrimination(
    probe_results: list[ExamResult],
    questions: list[MCQQuestion],
) -> dict[str, float]:
    """Compute discrimination score for each question across probe results.

    A question is discriminating if some probes answer it correctly and others
    don't — i.e., it actually differentiates strong from weak RAG pipelines.

    Score = variance of binary correct/incorrect responses across probes.
    - All probes correct → variance = 0 (too easy, low score)
    - All probes wrong → variance = 0 (too hard, low score)
    - Mixed → variance > 0 (discriminating, high score)

    Args:
        probe_results: List of ExamResult, one per probe configuration.
        questions: Candidate questions (used to build the question_id index).

    Returns:
        dict mapping question_id → discrimination score.
    """
    if not probe_results:
        return {q.id: 0.0 for q in questions}

    question_ids = {q.id for q in questions}
    # Build per-question binary response vectors: shape (n_probes,)
    responses: dict[str, list[int]] = {qid: [] for qid in question_ids}

    for result in probe_results:
        result_map = {qr.question_id: int(qr.correct) for qr in result.question_results}
        for qid in question_ids:
            # If a question wasn't evaluated by this probe, treat as incorrect
            responses[qid].append(result_map.get(qid, 0))

    scores: dict[str, float] = {}
    for qid, binary_vec in responses.items():
        arr = np.array(binary_vec, dtype=np.float32)
        scores[qid] = float(np.var(arr))

    return scores


def select_exam(
    candidates: list[MCQQuestion],
    scores: dict[str, float],
    exam_size: int,
) -> list[MCQQuestion]:
    """Greedy exam selection that maximises discrimination while preserving cluster diversity.

    Strategy:
    1. Compute proportional cluster allocations (largest remainder method).
    2. Fill each cluster quota with the highest-scoring candidates from that cluster.
    3. Fill any remaining slots globally from the highest-scoring unused candidates.

    Args:
        candidates: All validated candidate questions.
        scores: Per-question discrimination score from score_questions_by_discrimination.
        exam_size: Target number of questions in the final exam.

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

    logger.info(
        "Probe-based selection: %d/%d candidates selected (exam_size=%d)",
        len(selected),
        len(candidates),
        exam_size,
    )
    return selected
