"""Pure-function helpers that compute trial metrics and the optimizer state card.

These do not call an LLM. The Diagnoser and Proposer read the rendered output
of these functions in their prompts; both agents see the same grounded signal.
"""

from __future__ import annotations

from agentic_autorag.config.models import TrialConfig
from agentic_autorag.examiner.evaluator import _ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL, ExamResult
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.diagnosis import StateCard, TrialMetrics

_ERROR_SENTINELS = (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)

# Default polish-phase parameters. ``polish_fraction`` is the tail share of
# the trial budget eligible for cost reduction; ``polish_score_floor`` gates
# polish on actually having a working config; ``polish_score_tolerance`` is
# the score band around the leader the agent is expected to hold during polish.
_DEFAULT_POLISH_FRACTION = 0.3
_DEFAULT_POLISH_SCORE_FLOOR = 0.5
_DEFAULT_POLISH_SCORE_TOLERANCE = 0.05

_HV_DELTA_WINDOW = 3

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

    n_complete = sum(1 for qr in valid if qr.context_sufficient)
    n_partial = sum(1 for qr in valid if 0 < qr.retrieved_spans < qr.n_spans)
    n_miss = sum(1 for qr in valid if qr.retrieved_spans == 0)
    n_refused = sum(1 for qr in valid if qr.refused)
    n_correct = sum(1 for qr in valid if qr.correct)
    n_correct_given_complete = sum(1 for qr in valid if qr.correct and qr.context_sufficient)

    return TrialMetrics(
        answer_accuracy=n_correct / n,
        retrieval_complete=n_complete / n,
        retrieval_partial=n_partial / n,
        retrieval_miss=n_miss / n,
        refusal_rate=n_refused / n,
        answer_correct_given_complete_retrieval=(n_correct_given_complete / n_complete if n_complete else 0.0),
        n_valid=n,
        mean_llm_cost_per_query_usd=float(getattr(exam_result, "mean_llm_cost_per_query_usd", 0.0)),
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
    current_cost_usd: float = 0.0,
    polish_fraction: float = _DEFAULT_POLISH_FRACTION,
    polish_score_floor: float = _DEFAULT_POLISH_SCORE_FLOOR,
    polish_score_tolerance: float = _DEFAULT_POLISH_SCORE_TOLERANCE,
) -> StateCard:
    """Mechanically summarise optimizer state. Used by both agents.

    ``phase`` is computed by ``pareto.phase_label`` from a mechanical split
    of the trial budget plus a score-floor gate: polish only engages when
    a working config exists. The Pareto block (frontier, knee, nearest
    dominator, cheapest at score threshold) is computed by direct dominance
    over (score, cost) — no interpretive aggregation.

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

    phase = pareto.phase_label(
        trial_number=trial_number,
        max_trials=max_trials,
        best_score=best_score,
        polish_fraction=polish_fraction,
        polish_score_floor=polish_score_floor,
    )

    summaries = _trial_summaries(sorted_hist)
    summaries.append(
        {
            "trial_number": trial_number,
            "score": float(current_score),
            "cost_usd": float(current_cost_usd),
            "what_changed_from_prev": _config_diff_summary(
                getattr(sorted_hist[-1], "config", None) if sorted_hist else None,
                current_config,
            ),
            "top_failure_modes": list(current_top_failure_modes or []),
        }
    )

    pareto_view = _build_pareto_view(
        sorted_hist=sorted_hist,
        current_trial_number=trial_number,
        current_score=current_score,
        current_cost_usd=current_cost_usd,
        best_score=best_score,
        polish_score_tolerance=polish_score_tolerance,
    )

    return StateCard(
        trial_number=trial_number,
        trials_remaining=trials_remaining,
        best_score_so_far=best_score,
        best_trial_number=best_trial,
        last_trial_delta=last_delta,
        phase=phase,  # type: ignore[arg-type]
        trial_summaries=summaries,
        pareto_frontier=pareto_view["frontier"],
        hypervolume=pareto_view["hypervolume"],
        hypervolume_delta_last_3=pareto_view["hypervolume_delta_last_3"],
        knee_trial_number=pareto_view["knee_trial_number"],
        nearest_dominator_trial=pareto_view["nearest_dominator_trial"],
        current_trial_cost_usd=float(current_cost_usd),
        cheapest_at_score_threshold_usd=pareto_view["cheapest_at_score_threshold_usd"],
        cheapest_at_score_threshold_trial=pareto_view["cheapest_at_score_threshold_trial"],
    )


def _trial_summaries(ordered_records: list) -> list[dict]:
    """Per-trial: trial_number, score, cost, what_changed_from_prev, top_failure_modes."""
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
                "cost_usd": float(getattr(rec, "mean_llm_cost_per_query_usd", 0.0)),
                "what_changed_from_prev": _config_diff_summary(prev_cfg, cfg),
                "top_failure_modes": modes,
            }
        )
    return out


class _PareToRecord:
    """Lightweight record adapter for ``pareto.*`` helpers.

    The Pareto helpers only need ``trial_number``, ``score``, and
    ``mean_llm_cost_per_query_usd`` (Protocol). Used to include the
    current (in-flight) trial in frontier computations before it lands in
    ``HistoryLog`` — and to wrap heterogeneous history records so we can
    run frontier math without forcing every caller to pass real
    ``TrialRecord`` objects (e.g. tests pass ``SimpleNamespace``-likes).
    """

    __slots__ = ("trial_number", "score", "mean_llm_cost_per_query_usd", "_source")

    def __init__(self, trial_number: int, score: float, cost: float, source: object | None = None) -> None:
        self.trial_number = int(trial_number)
        self.score = float(score)
        self.mean_llm_cost_per_query_usd = float(cost)
        self._source = source


def _to_pareto_record(rec: object) -> _PareToRecord:
    return _PareToRecord(
        trial_number=int(getattr(rec, "trial_number", 0)),
        score=float(getattr(rec, "score", 0.0)),
        cost=float(getattr(rec, "mean_llm_cost_per_query_usd", 0.0)),
        source=rec,
    )


def _build_pareto_view(
    *,
    sorted_hist: list,
    current_trial_number: int,
    current_score: float,
    current_cost_usd: float,
    best_score: float,
    polish_score_tolerance: float,
) -> dict:
    """Compute frontier, HV, HV delta, knee, nearest dominator, and score-band-cheapest.

    The current (in-flight) trial is included as a synthetic record so the
    agent can see its position relative to the frontier on the same call.
    """
    all_records: list[_PareToRecord] = [_to_pareto_record(r) for r in sorted_hist]
    current_record = _PareToRecord(
        trial_number=current_trial_number,
        score=current_score,
        cost=current_cost_usd,
    )
    # Replace any prior record with the same trial_number (defensive — should
    # not happen in practice since the orchestrator builds the state card
    # before history.add).
    all_records = [r for r in all_records if r.trial_number != current_trial_number]
    all_records.append(current_record)

    frontier = pareto.compute_frontier(all_records)
    cost_values = [r.mean_llm_cost_per_query_usd for r in all_records]
    cost_ref = max(cost_values) if cost_values else 0.0
    if cost_ref <= 0.0:
        cost_ref = 1.0  # sentinel so HV is 0 when no cost data exists
    hv = pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))

    hv_history: list[float] = []
    for r in all_records:
        n = r.trial_number
        subset = [x for x in all_records if x.trial_number <= n]
        sub_frontier = pareto.compute_frontier(subset)
        hv_history.append(pareto.compute_hypervolume(sub_frontier, ref_point=(0.0, cost_ref)))
    hv_delta_last_3 = (
        hv_history[-1] - hv_history[-(_HV_DELTA_WINDOW + 1)]
        if len(hv_history) > _HV_DELTA_WINDOW
        else (hv_history[-1] - hv_history[0] if hv_history else 0.0)
    )

    knee = pareto.find_knee(frontier)
    knee_trial = knee.trial_number if knee is not None else None

    dominator = pareto.nearest_dominator(current_record, all_records)
    dominator_trial = dominator.trial_number if dominator is not None else None

    cheapest_band: _PareToRecord | None = None
    threshold = best_score - polish_score_tolerance
    for r in all_records:
        if r.score < threshold:
            continue
        if cheapest_band is None or r.mean_llm_cost_per_query_usd < cheapest_band.mean_llm_cost_per_query_usd:
            cheapest_band = r

    frontier_view: list[dict] = []
    for r in frontier:
        source = getattr(r, "_source", None)
        config = getattr(source, "config", None)
        frontier_view.append(
            {
                "trial_number": r.trial_number,
                "score": r.score,
                "cost_usd": r.mean_llm_cost_per_query_usd,
                "config_summary": _short_config_summary(config),
            }
        )

    return {
        "frontier": frontier_view,
        "hypervolume": hv,
        "hypervolume_delta_last_3": hv_delta_last_3,
        "knee_trial_number": knee_trial,
        "nearest_dominator_trial": dominator_trial,
        "cheapest_at_score_threshold_usd": (
            cheapest_band.mean_llm_cost_per_query_usd if cheapest_band is not None else None
        ),
        "cheapest_at_score_threshold_trial": (cheapest_band.trial_number if cheapest_band is not None else None),
    }


def _short_config_summary(config: TrialConfig | None) -> str:
    """One-line config summary used inside the frontier table rendering."""
    if config is None:
        return "(current trial)"
    reasoning_tag = " +reasoning" if getattr(config, "reasoning", False) else ""
    return (
        f"{config.embedding_model.split('/')[-1]} + {config.llm_model.split('/')[-1]}{reasoning_tag}, "
        f"top_k={config.top_k}, reranker={config.reranker}"
    )


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
