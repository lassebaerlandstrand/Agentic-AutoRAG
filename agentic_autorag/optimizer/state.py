"""Pure-function helpers that compute trial metrics and the optimizer state card.

These do not call an LLM. The Diagnoser and Proposer read the rendered output
of these functions in their prompts; both agents see the same grounded signal.
"""

from __future__ import annotations

import math

from agentic_autorag.config.models import OpenEndedQuestion, TrialConfig
from agentic_autorag.examiner._errors import ERROR_SENTINELS
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.diagnosis import (
    FailureAttribution,
    FrontierContext,
    LeverEffectDelta,
    StateCard,
    Strategy,
    TrialMetrics,
)

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
    valid = [qr for qr in exam_result.question_results if qr.generated_response not in ERROR_SENTINELS]
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
    polish_score_tolerance: float = pareto.DEFAULT_POLISH_SCORE_TOLERANCE,
    previous_strategy: Strategy | None = None,
    allow_early_exit: bool = True,
    min_trials_before_done: int = 4,
    min_frontier_size_for_done: int = 2,
    early_exit_hv_epsilon: float = 0.001,
) -> StateCard:
    """Mechanically summarise optimizer state. Used by both agents.

    The optimizer phase is owned by the agent via ``Strategy.stance``; this
    card just hands the agent the data (Pareto frontier, knee, hypervolume,
    cheapest-in-band) plus its own strategy history and the orchestrator-
    computed ``done_eligible`` gate. Pareto fields are arithmetic — dominance
    and the knee point are direct computations over (score, cost), not
    interpretive aggregates.

    ``current_config`` and ``current_top_failure_modes`` describe the
    just-completed trial and are appended as the last entry of
    ``trial_summaries`` so the agents see the full N-trial history.

    ``done_eligible`` is True only when ``allow_early_exit`` is True AND the
    minimum trial floor, frontier size, and HV-plateau conditions are all
    met — the agent may emit ``stance="done"`` only when this is True.
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

    strategy_history_summary = _strategy_history_summary(sorted_hist)
    revision_count_this_run = previous_strategy.revision_count if previous_strategy is not None else 0
    done_eligible, done_blocked_reason = _compute_done_eligibility(
        trial_number=trial_number,
        max_trials=max_trials,
        frontier_size=len(pareto_view["frontier"]),
        hypervolume_delta_last_3=pareto_view["hypervolume_delta_last_3"],
        allow_early_exit=allow_early_exit,
        min_trials_before_done=min_trials_before_done,
        min_frontier_size_for_done=min_frontier_size_for_done,
        early_exit_hv_epsilon=early_exit_hv_epsilon,
    )

    return StateCard(
        trial_number=trial_number,
        trials_remaining=trials_remaining,
        best_score_so_far=best_score,
        best_trial_number=best_trial,
        last_trial_delta=last_delta,
        trial_summaries=summaries,
        pareto_frontier=pareto_view["frontier"],
        hypervolume=pareto_view["hypervolume"],
        hypervolume_delta_last_3=pareto_view["hypervolume_delta_last_3"],
        knee_trial_number=pareto_view["knee_trial_number"],
        nearest_dominator_trial=pareto_view["nearest_dominator_trial"],
        current_trial_cost_usd=float(current_cost_usd),
        cheapest_at_score_threshold_usd=pareto_view["cheapest_at_score_threshold_usd"],
        cheapest_at_score_threshold_trial=pareto_view["cheapest_at_score_threshold_trial"],
        previous_strategy=previous_strategy,
        strategy_history_summary=strategy_history_summary,
        revision_count_this_run=revision_count_this_run,
        done_eligible=done_eligible,
        done_blocked_reason=done_blocked_reason,
    )


def _strategy_history_summary(sorted_hist: list) -> list[dict]:
    """Per-trial stance/intent/revision_count for the agent's own trajectory.

    Records that pre-date the structured Strategy hand-off (or whose meta is
    None — e.g. failure-recovery trials) are skipped silently so the
    rendered summary stays terse.
    """
    out: list[dict] = []
    for rec in sorted_hist:
        meta = getattr(rec, "meta", None)
        strategy = getattr(meta, "strategy", None) if meta is not None else None
        if strategy is None:
            continue
        out.append(
            {
                "trial_number": int(getattr(rec, "trial_number", 0)),
                "stance": strategy.stance,
                "intent": strategy.intent,
                "revision_count": int(strategy.revision_count),
            }
        )
    return out


def _compute_done_eligibility(
    *,
    trial_number: int,
    max_trials: int,
    frontier_size: int,
    hypervolume_delta_last_3: float,
    allow_early_exit: bool,
    min_trials_before_done: int,
    min_frontier_size_for_done: int,
    early_exit_hv_epsilon: float,
) -> tuple[bool, str | None]:
    """Return (eligible, reason_blocked) for the ``done`` stance.

    The trial-floor is the max of the configured ``min_trials_before_done``
    and ``ceil(max_trials * 0.4)`` — the latter prevents trivially-cheap
    early exits on long runs while still letting short runs honour the
    configured floor.
    """
    if not allow_early_exit:
        return False, "allow_early_exit=False in MetaConfig"
    floor = max(int(min_trials_before_done), math.ceil(max_trials * 0.4))
    if trial_number < floor:
        return False, f"trial {trial_number} below minimum trial floor for done ({floor})"
    if frontier_size < min_frontier_size_for_done:
        return False, (
            f"only {frontier_size} frontier member(s); need at least "
            f"{min_frontier_size_for_done} (an observed cost/score trade-off)"
        )
    if hypervolume_delta_last_3 > early_exit_hv_epsilon:
        return False, (
            f"hypervolume still expanding (Δ_last_3={hypervolume_delta_last_3:.4f} > ε={early_exit_hv_epsilon})"
        )
    return True, None


def build_frontier_context(
    *,
    history_records: list,
    current_trial_number: int,
    current_score: float,
    current_cost_usd: float,
    current_config: TrialConfig | None,
) -> FrontierContext:
    """Compute the current trial's position relative to the Pareto frontier.

    Returns ``is_on_frontier`` plus, when the trial is dominated, the nearest
    dominator's trial number, score, cost, and a config diff (``"field: a → b"``
    strings) so the diagnoser can reason about *which knobs* the dominator
    changed to beat this trial on (score, cost).
    """
    sorted_hist = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))
    others = [_to_pareto_record(r) for r in sorted_hist if int(getattr(r, "trial_number", 0)) != current_trial_number]
    current = _PareToRecord(
        trial_number=current_trial_number,
        score=current_score,
        cost=current_cost_usd,
    )
    all_records = [*others, current]

    is_on_frontier = not any(pareto.dominates(o, current) for o in others)
    dominator = pareto.nearest_dominator(current, all_records)
    if dominator is None:
        return FrontierContext(is_on_frontier=is_on_frontier)

    dominator_source = getattr(dominator, "_source", None)
    dominator_config = getattr(dominator_source, "config", None)
    # Diff direction: "current → dominator" — the values the current trial
    # would need to adopt to match the dominator. Empty list when both
    # configs are identical (e.g. dominator differs only in non-tracked
    # fields like timestamp) or either config is missing.
    diff = _config_diff_summary(current_config, dominator_config)
    return FrontierContext(
        is_on_frontier=is_on_frontier,
        nearest_dominator_trial=int(dominator.trial_number),
        nearest_dominator_score=float(dominator.score),
        nearest_dominator_cost_usd=float(dominator.mean_llm_cost_per_query_usd),
        nearest_dominator_config_diff=diff,
        score_gap_to_dominator=float(dominator.score) - float(current_score),
        cost_gap_to_dominator_usd=float(current_cost_usd) - float(dominator.mean_llm_cost_per_query_usd),
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
            attribution = getattr(diag, "failure_attribution", None)
            if attribution is not None:
                modes = _top_stages_from_attribution(attribution, n=2)
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


def _top_stages_from_attribution(attribution, n: int = 2) -> list[str]:
    """Top ``n`` stage names from a ``FailureAttribution``, descending; drops zeros."""
    pairs = [
        ("retrieval", float(getattr(attribution, "retrieval", 0.0))),
        ("ranking", float(getattr(attribution, "ranking", 0.0))),
        ("generation", float(getattr(attribution, "generation", 0.0))),
        ("composition", float(getattr(attribution, "composition", 0.0))),
    ]
    pairs.sort(key=lambda p: -p[1])
    return [name for name, frac in pairs[:n] if frac > 0.0]


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
                "config": _config_to_dict(config),
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


def _config_to_dict(config: TrialConfig | None) -> dict | None:
    """Full TrialConfig dump for frontier entries the proposer can anchor on.

    Returned alongside the one-line summary so the agent can perturb a
    specific frontier member's config rather than guess from the summary.
    """
    if config is None:
        return None
    return config.model_dump(mode="json")


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


def build_failure_attribution(question_results: list[QuestionResult]) -> FailureAttribution:
    """Compute the per-stage fraction of failures from per-question failure modes.

    Mapping (mechanical):
      - retrieval_miss → retrieval
      - retrieval_partial (correct or not) → retrieval
      - refused with miss/partial retrieval → retrieval
      - refused with complete retrieval → generation (model gave up despite evidence)
      - generation_wrong (complete retrieval, non-refusal wrong answer) → generation
      - ranking is not separately observable from QuestionResult alone — the
        retriever-vs-ranker split needs reranker_top_n knowledge — so it
        stays 0.0 here. The Diagnoser may re-attribute in its narrative.
      - composition (exam malformed) is not detectable mechanically; left at 0.0.

    System-error sentinels are excluded. Returns zeros when there are no
    failures. Sums to ~1.0 (drift only from floating-point rounding).
    """
    valid = [qr for qr in question_results if qr.generated_response not in ERROR_SENTINELS]
    failures = [qr for qr in valid if not qr.correct]
    n = len(failures)
    if n == 0:
        return FailureAttribution()

    retrieval = 0
    generation = 0
    for qr in failures:
        if qr.refused and qr.context_sufficient:
            generation += 1
        elif qr.refused or qr.retrieved_spans == 0 or qr.retrieved_spans < qr.n_spans:
            retrieval += 1
        else:
            generation += 1

    return FailureAttribution(
        retrieval=retrieval / n,
        ranking=0.0,
        generation=generation / n,
        composition=0.0,
    )


def build_failure_cross_tab(
    question_results: list[QuestionResult],
    exam_questions: list[OpenEndedQuestion],
) -> str:
    """Render a ``failure_mode × reasoning_type × n_spans-bucket`` cross-tab.

    Counts only failures (correctness=False) on valid questions. Cells with
    zero count are omitted. The output is one line per cell, plus a header,
    suitable for direct inclusion in a prompt. Universal across corpora — the
    only inputs are the per-question results and the question metadata.
    """
    by_id = {q.id: q for q in exam_questions}
    valid = [qr for qr in question_results if qr.generated_response not in ERROR_SENTINELS]
    failures = [qr for qr in valid if not qr.correct]
    if not failures:
        return "No failures this trial."

    counts: dict[tuple[str, str, str], int] = {}
    for qr in failures:
        mode = _failure_mode(qr)
        q = by_id.get(qr.question_id)
        rt = q.reasoning_type if q is not None else "unknown"
        bucket = _n_spans_bucket(qr.n_spans)
        key = (mode, rt, bucket)
        counts[key] = counts.get(key, 0) + 1

    lines = [f"failure_mode × reasoning_type × n_spans (total failures: {len(failures)})"]
    for (mode, rt, bucket), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {mode:18s} × {rt:12s} × {bucket:8s} : {n}")
    return "\n".join(lines)


def _failure_mode(qr: QuestionResult) -> str:
    """Open-ended failure-mode categorisation. Mirrors reasoning_agent._failure_mode."""
    if qr.refused:
        return "refused"
    if qr.retrieved_spans == 0:
        return "retrieval_miss"
    if qr.retrieved_spans < qr.n_spans:
        return "retrieval_partial"
    if not qr.correct:
        return "generation_wrong"
    return "retrieval_complete"


def _n_spans_bucket(n: int) -> str:
    if n <= 1:
        return "n=1"
    if n == 2:
        return "n=2"
    return "n>=3"


def compute_lever_effect_deltas(
    *,
    history_records: list,
    current_config: TrialConfig | None,
    current_metrics: TrialMetrics | None,
    current_cost_usd: float,
    anchor_trial: int | None,
) -> list[LeverEffectDelta]:
    """For each lever that differs between ``anchor_trial``'s config and the
    current trial's config, compute the delta on score / acc_given_complete /
    retrieval_complete / cost_usd.

    When ``anchor_trial`` is None or the anchor record is missing, falls back
    to the most-recent prior trial. Returns an empty list when there is no
    prior trial OR no current metrics OR no current config.
    """
    if current_config is None or current_metrics is None:
        return []
    sorted_hist = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))
    if not sorted_hist:
        return []

    anchor = None
    if anchor_trial is not None:
        anchor = next((r for r in sorted_hist if int(getattr(r, "trial_number", 0)) == int(anchor_trial)), None)
    if anchor is None:
        anchor = sorted_hist[-1]

    anchor_config = getattr(anchor, "config", None)
    anchor_metrics = getattr(anchor, "trial_metrics", None)
    if anchor_config is None or anchor_metrics is None:
        return []

    changes = _config_diff_summary(anchor_config, current_config)
    if not changes:
        return []

    score_delta = float(current_metrics.answer_accuracy) - float(anchor_metrics.answer_accuracy)
    acc_delta = float(current_metrics.answer_correct_given_complete_retrieval) - float(
        anchor_metrics.answer_correct_given_complete_retrieval
    )
    rcomp_delta = float(current_metrics.retrieval_complete) - float(anchor_metrics.retrieval_complete)
    cost_delta = float(current_cost_usd) - float(getattr(anchor, "mean_llm_cost_per_query_usd", 0.0))

    return [
        LeverEffectDelta(
            change=change,
            score_delta=score_delta,
            acc_given_complete_delta=acc_delta,
            retrieval_complete_delta=rcomp_delta,
            cost_delta_usd=cost_delta,
        )
        for change in changes
    ]
