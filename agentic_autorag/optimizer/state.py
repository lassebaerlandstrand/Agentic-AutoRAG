"""Pure-function helpers that compute trial metrics and the optimizer state card.

These do not call an LLM. The Diagnoser and Proposer read the rendered output
of these functions in their prompts; both agents see the same grounded signal.
"""

from __future__ import annotations

from agentic_autorag.config.models import TrialConfig
from agentic_autorag.examiner.evaluator import _ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL, ExamResult
from agentic_autorag.optimizer.diagnosis import StateCard, TrialMetrics

_ERROR_SENTINELS = (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)

_EXPLOIT_SCORE_FLOOR = 0.5

_CONFIG_DIFF_FIELDS: tuple[str, ...] = (
    "index_type",
    "embedding_model",
    "chunking_strategy",
    "chunk_token_size",
    "chunk_token_overlap",
    "reranker",
    "reranker_top_n",
    "top_k",
    "hybrid_alpha",
    "llm_model",
    "reasoning",
    "query_expansion",
    "graph_query_mode",
    "graph_top_k",
)


def compute_trial_metrics(exam_result: ExamResult) -> TrialMetrics:
    """Compute the seven open-ended quality signals from a completed exam.

    Excludes questions that hit system-error sentinels (timeouts, API failures)
    from all rates — those are not diagnostic of the pipeline.
    """
    valid = [qr for qr in exam_result.question_results if qr.generated_response not in _ERROR_SENTINELS]
    n = len(valid)
    if n == 0:
        return TrialMetrics()

    n_complete = sum(1 for qr in valid if qr.retrieval_status == "both")
    n_only_a = sum(1 for qr in valid if qr.retrieval_status == "only_A")
    n_only_b = sum(1 for qr in valid if qr.retrieval_status == "only_B")
    n_miss = sum(1 for qr in valid if qr.retrieval_status == "neither")
    n_refused = sum(1 for qr in valid if qr.refused)
    n_correct = sum(1 for qr in valid if qr.correct)
    n_correct_given_complete = sum(1 for qr in valid if qr.correct and qr.retrieval_status == "both")

    return TrialMetrics(
        answer_accuracy=n_correct / n,
        retrieval_complete=n_complete / n,
        retrieval_partial_a_only=n_only_a / n,
        retrieval_partial_b_only=n_only_b / n,
        retrieval_miss=n_miss / n,
        refusal_rate=n_refused / n,
        answer_correct_given_complete_retrieval=(n_correct_given_complete / n_complete if n_complete else 0.0),
        n_valid=n,
    )


def build_state_card(
    *,
    trial_number: int,
    trials_remaining: int,
    current_score: float,
    history_records: list,
    max_trials: int,
    current_config: TrialConfig | None = None,
    current_top_failure_modes: list[str] | None = None,
) -> StateCard:
    """Mechanically summarise optimizer state. Used by both agents.

    ``phase`` is "explore" while there is budget remaining and we have not
    found a config above a loose floor; otherwise "exploit". The Proposer
    prompt treats this as guidance, not a hard constraint.

    ``current_config`` and ``current_top_failure_modes`` describe the
    just-completed trial and are appended as the last entry of
    ``trial_summaries`` so the agents see the full N-trial history.
    """
    sorted_hist = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))

    best_score = current_score
    best_trial: int | None = trial_number
    for rec in sorted_hist:
        s = float(getattr(rec, "score", 0.0))
        if s > best_score:
            best_score = s
            best_trial = int(getattr(rec, "trial_number", 0))

    prior_scores = [
        float(getattr(r, "score", 0.0)) for r in sorted_hist if getattr(r, "trial_number", 0) < trial_number
    ]
    last_delta = current_score - prior_scores[-1] if prior_scores else 0.0

    in_first_half = trial_number <= max(1, max_trials // 2)
    phase: str = "explore" if (in_first_half or best_score < _EXPLOIT_SCORE_FLOOR) else "exploit"

    summaries = _trial_summaries(sorted_hist)
    summaries.append(
        {
            "trial_number": trial_number,
            "score": float(current_score),
            "what_changed_from_prev": _config_diff_summary(
                getattr(sorted_hist[-1], "config", None) if sorted_hist else None,
                current_config,
            ),
            "top_failure_modes": list(current_top_failure_modes or []),
        }
    )

    return StateCard(
        trial_number=trial_number,
        trials_remaining=trials_remaining,
        best_score_so_far=best_score,
        best_trial_number=best_trial,
        last_trial_delta=last_delta,
        phase=phase,  # type: ignore[arg-type]
        trial_summaries=summaries,
    )


def _trial_summaries(ordered_records: list) -> list[dict]:
    """Per-trial: trial_number, score, what_changed_from_prev, top_failure_modes."""
    out: list[dict] = []
    for i, rec in enumerate(ordered_records):
        prev_cfg = getattr(ordered_records[i - 1], "config", None) if i else None
        cfg = getattr(rec, "config", None)
        diag = getattr(rec, "diagnosis", None)
        modes: list[str] = []
        if diag is not None:
            modes = [b.stage for b in getattr(diag, "bottlenecks", [])[:2]]
        out.append(
            {
                "trial_number": int(getattr(rec, "trial_number", 0)),
                "score": float(getattr(rec, "score", 0.0)),
                "what_changed_from_prev": _config_diff_summary(prev_cfg, cfg),
                "top_failure_modes": modes,
            }
        )
    return out


def _config_diff_summary(a: TrialConfig | None, b: TrialConfig | None) -> list[str]:
    """List of ``"field: old → new"`` strings; empty for first trial."""
    if a is None or b is None:
        return []
    out: list[str] = []
    for f in _CONFIG_DIFF_FIELDS:
        va = getattr(a, f, None)
        vb = getattr(b, f, None)
        va = getattr(va, "value", va)
        vb = getattr(vb, "value", vb)
        if va != vb:
            out.append(f"{f}: {va} → {vb}")
    return out
