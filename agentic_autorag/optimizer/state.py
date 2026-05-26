"""Pure-function helpers that compute trial metrics and the optimizer state card.

These do not call an LLM. The Diagnoser and Proposer read the rendered output
of these functions in their prompts; both agents see the same grounded signal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_autorag.config.models import OpenEndedQuestion, TrialConfig
from agentic_autorag.examiner._errors import ERROR_SENTINELS
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.diagnosis import (
    BundleEffectDelta,
    FrontierContext,
    StateCard,
    Strategy,
    TrialMetrics,
)

_HV_DELTA_WINDOW_DEFAULT = 3

CONFIG_LEVER_FIELDS: tuple[str, ...] = (
    "chunking_strategy",
    "chunk_token_size",
    "chunk_token_overlap",
    "embedding_model",
    "index_type",
    "top_k",
    "hybrid_alpha",
    "bm25_vector_fusion",
    "long_context_reorder",
    "passage_compressor",
    "reranker",
    "reranker_top_n",
    "query_expansion",
    "compressor_llm",
    "expander_llm",
    "generator_llm",
    "temperature",
    "reasoning",
    "graph_query_mode",
    "graph_top_k",
)


class FailureAttribution(BaseModel):
    """Mechanical fraction of this trial's failures attributable to each
    pipeline stage. Rendered into the state cards as reference signal only —
    the LLM does not re-emit it. Sums to ~1.0."""

    retrieval: float = Field(default=0.0, ge=0.0, le=1.0)
    ranking: float = Field(default=0.0, ge=0.0, le=1.0)
    generation: float = Field(default=0.0, ge=0.0, le=1.0)
    composition: float = Field(default=0.0, ge=0.0, le=1.0)


def compute_trial_metrics(exam_result: ExamResult) -> TrialMetrics:
    """Compute the seven open-ended quality signals. Excludes questions that
    hit system-error sentinels (timeouts, API failures) from all rates."""
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


_COVERAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("generators", "generator_llm"),
    ("embeddings", "embedding_model"),
    ("rerankers", "reranker"),
)


def build_state_card(
    *,
    trial_number: int,
    trials_remaining: int,
    current_score: float,
    history_records: list,
    current_config: TrialConfig | None = None,
    current_top_failure_modes: list[str] | None = None,
    current_cost_usd: float = 0.0,
    current_retrieval_complete: float = 0.0,
    cost_aware: bool = True,
    previous_strategy: Strategy | None = None,
    hv_delta_window: int = _HV_DELTA_WINDOW_DEFAULT,
    search_space_sizes: dict[str, int] | None = None,
) -> StateCard:
    """Mechanically summarise optimizer state. Hands the agent best-score +
    trial summaries + Pareto frontier (cost-aware only) + previous strategy
    carry-over. Phase ownership is on the agent via ``Strategy.stance``."""
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
            "retrieval_complete": float(current_retrieval_complete),
            "what_changed_from_prev": _config_diff_summary(
                getattr(sorted_hist[-1], "config", None) if sorted_hist else None,
                current_config,
            ),
            "top_failure_modes": list(current_top_failure_modes or []),
        }
    )

    if cost_aware:
        pareto_view = _build_pareto_view(
            sorted_hist=sorted_hist,
            current_trial_number=trial_number,
            current_score=current_score,
            current_cost_usd=current_cost_usd,
            hv_delta_window=hv_delta_window,
        )
    else:
        pareto_view = _empty_pareto_view()

    stance_history = _extract_stance_history(sorted_hist) if cost_aware else []
    trials_since_best = max(0, trial_number - best_trial) if best_trial is not None else 0
    coverage = _compute_coverage(sorted_hist, current_config, search_space_sizes or {})

    return StateCard(
        cost_aware=cost_aware,
        trial_number=trial_number,
        trials_remaining=trials_remaining,
        best_score_so_far=best_score,
        best_trial_number=best_trial,
        last_trial_delta=last_delta,
        trials_since_best_score=trials_since_best,
        coverage=coverage,
        trial_summaries=summaries,
        pareto_frontier=pareto_view["frontier"],
        hypervolume=pareto_view["hypervolume"],
        hypervolume_delta_last_3=pareto_view["hypervolume_delta_last_3"],
        current_trial_cost_usd=float(current_cost_usd) if cost_aware else 0.0,
        previous_strategy=previous_strategy,
        stance_history=stance_history,
    )


def _compute_coverage(
    sorted_hist: list,
    current_config: TrialConfig | None,
    sizes: dict[str, int],
) -> list[dict]:
    """Distinct-values-tried-vs-total for each surveyed lever.

    Caller supplies ``{config_field: search_space_size}``; we count distinct
    values across history + current trial. Empty list when sizes is empty
    (caller didn't survey — score-only callers may opt out).
    """
    if not sizes:
        return []
    out: list[dict] = []
    for label, field in _COVERAGE_FIELDS:
        total = int(sizes.get(field, 0))
        if total <= 0:
            continue
        seen: set = set()
        for rec in sorted_hist:
            cfg = getattr(rec, "config", None)
            if cfg is not None:
                seen.add(getattr(cfg, field, None))
        if current_config is not None:
            seen.add(getattr(current_config, field, None))
        seen.discard(None)
        out.append({"label": label, "tried": len(seen), "total": total})
    return out


def _extract_stance_history(sorted_hist: list) -> list[tuple[int, str]]:
    """``(trial_number, stance)`` for every prior trial with a declared
    stance. Records without meta/strategy/stance are skipped (initial trial,
    failure-recovery rows in score-only mode)."""
    out: list[tuple[int, str]] = []
    for rec in sorted_hist:
        meta = getattr(rec, "meta", None)
        strategy = getattr(meta, "strategy", None) if meta is not None else None
        stance = getattr(strategy, "stance", None) if strategy is not None else None
        if stance is None:
            continue
        out.append((int(getattr(rec, "trial_number", 0)), str(stance)))
    return out


def _empty_pareto_view() -> dict:
    """Return a Pareto-view dict with all fields at the score-only-mode defaults."""
    return {
        "frontier": [],
        "hypervolume": 0.0,
        "hypervolume_delta_last_3": 0.0,
    }


def build_frontier_context(
    *,
    history_records: list,
    current_trial_number: int,
    current_score: float,
    current_cost_usd: float,
    current_config: TrialConfig | None,
) -> FrontierContext:
    """Compute the current trial's position relative to the Pareto frontier.
    Returns ``is_on_frontier`` plus, when dominated, the nearest dominator's
    trial / score / cost and a config diff so the diagnoser can reason about
    which knobs the dominator changed."""
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
    """Per-trial: ``trial_number, score, cost, what_changed_from_prev,
    top_failure_modes``. ``top_failure_modes`` is computed from each record's
    QuestionResults; the Diagnoser does not restate it."""
    out: list[dict] = []
    for i, rec in enumerate(ordered_records):
        prev_cfg = getattr(ordered_records[i - 1], "config", None) if i else None
        cfg = getattr(rec, "config", None)
        qrs = getattr(rec, "question_results", None) or []
        modes = _top_failure_modes(qrs, n=2) if qrs else []
        out.append(
            {
                "trial_number": int(getattr(rec, "trial_number", 0)),
                "score": float(getattr(rec, "score", 0.0)),
                "cost_usd": float(getattr(rec, "mean_llm_cost_per_query_usd", 0.0)),
                "retrieval_complete": float(
                    getattr(getattr(rec, "trial_metrics", None), "retrieval_complete", 0.0)
                ),
                "what_changed_from_prev": _config_diff_summary(prev_cfg, cfg),
                "top_failure_modes": modes,
            }
        )
    return out


def _top_failure_modes(question_results: list[QuestionResult], n: int = 2) -> list[str]:
    """Top ``n`` pipeline-stage labels for this trial's failures. Mapping
    mirrors ``build_failure_attribution``."""
    valid = [qr for qr in question_results if qr.generated_response not in ERROR_SENTINELS]
    failures = [qr for qr in valid if not qr.correct]
    counts = {"retrieval": 0, "generation": 0}
    for qr in failures:
        if qr.refused and qr.context_sufficient:
            counts["generation"] += 1
        elif qr.refused or qr.retrieved_spans == 0 or qr.retrieved_spans < qr.n_spans:
            counts["retrieval"] += 1
        else:
            counts["generation"] += 1
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])
    return [name for name, c in pairs[:n] if c > 0]


def _top_stages_from_attribution(attribution: FailureAttribution, n: int = 2) -> list[str]:
    """Top ``n`` stage names from a ``FailureAttribution``, descending. For
    callers that already have one in hand (no re-walk over QuestionResults)."""
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
    hv_delta_window: int = _HV_DELTA_WINDOW_DEFAULT,
) -> dict:
    """Compute frontier + HV + HV delta for the cost-aware Pareto block. The
    current (in-flight) trial is included as a synthetic record so the agent
    sees its position relative to the frontier in the same call."""
    all_records: list[_PareToRecord] = [_to_pareto_record(r) for r in sorted_hist]
    current_record = _PareToRecord(
        trial_number=current_trial_number,
        score=current_score,
        cost=current_cost_usd,
    )
    all_records = [r for r in all_records if r.trial_number != current_trial_number]
    all_records.append(current_record)

    frontier = pareto.compute_frontier(all_records)
    cost_values = [r.mean_llm_cost_per_query_usd for r in all_records]
    cost_ref = max(cost_values) if cost_values else 0.0
    if cost_ref <= 0.0:
        cost_ref = 1.0
    hv = pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))

    hv_history: list[float] = []
    for r in all_records:
        n = r.trial_number
        subset = [x for x in all_records if x.trial_number <= n]
        sub_frontier = pareto.compute_frontier(subset)
        hv_history.append(pareto.compute_hypervolume(sub_frontier, ref_point=(0.0, cost_ref)))
    hv_delta_last_3 = hv_history[-1] - hv_history[-(hv_delta_window + 1)] if len(hv_history) > hv_delta_window else 0.0

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
    }


def _short_config_summary(config: TrialConfig | None) -> str:
    """One-line config summary used inside the frontier table rendering."""
    if config is None:
        return "(current trial)"
    reasoning_tag = " +reasoning" if getattr(config, "reasoning", False) else ""
    stage_llms = [config.generator_llm, config.compressor_llm, config.expander_llm]
    active = [m for m in stage_llms if m is not None]
    if active and all(m == active[0] for m in active):
        llm_str = active[0].split("/")[-1]
    else:
        llm_str = "gen=" + config.generator_llm.split("/")[-1]
        if config.compressor_llm and config.compressor_llm != config.generator_llm:
            llm_str += "/comp=" + config.compressor_llm.split("/")[-1]
        if config.expander_llm and config.expander_llm != config.generator_llm:
            llm_str += "/exp=" + config.expander_llm.split("/")[-1]
    return (
        f"{config.embedding_model.split('/')[-1]} + {llm_str}{reasoning_tag}, "
        f"top_k={config.top_k}, reranker={config.reranker}"
    )


def _config_to_dict(config: TrialConfig | None) -> dict | None:
    """Full TrialConfig dump for frontier entries the proposer can anchor on."""
    if config is None:
        return None
    return config.model_dump(mode="json")


def _config_diff_summary(a: TrialConfig | None, b: TrialConfig | None) -> list[str]:
    """List of ``"field: old → new"`` strings; empty for first trial."""
    if a is None or b is None:
        return []
    out: list[str] = []
    for f in CONFIG_LEVER_FIELDS:
        va = getattr(a, f, None)
        vb = getattr(b, f, None)
        va = getattr(va, "value", va)
        vb = getattr(vb, "value", vb)
        if va != vb:
            out.append(f"{f}: {va} → {vb}")
    return out


def build_failure_attribution(question_results: list[QuestionResult]) -> FailureAttribution:
    """Compute the per-stage fraction of failures from per-question modes.

    Refusal-with-complete-retrieval and generation_wrong → ``generation``;
    everything else with a retrieval shortfall → ``retrieval``. Ranking and
    composition can't be observed mechanically and stay at 0.0; the Diagnoser
    may re-attribute in its narrative. System-error sentinels excluded.
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


def compute_bundle_effect(
    *,
    history_records: list,
    current_config: TrialConfig | None,
    current_metrics: TrialMetrics | None,
    current_cost_usd: float,
    anchor_trial: int | None,
) -> BundleEffectDelta | None:
    """Bundled effect of all lever changes between ``anchor_trial`` and the
    current trial. Deltas can't be attributed to any individual lever from
    observation alone. ``None`` when no prior trial exists, no metrics are
    available, or no levers differ. Falls back to the most-recent prior trial
    when the anchor is missing."""
    if current_config is None or current_metrics is None:
        return None
    sorted_hist = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))
    if not sorted_hist:
        return None

    anchor = None
    if anchor_trial is not None:
        anchor = next((r for r in sorted_hist if int(getattr(r, "trial_number", 0)) == int(anchor_trial)), None)
    if anchor is None:
        anchor = sorted_hist[-1]

    anchor_config = getattr(anchor, "config", None)
    anchor_metrics = getattr(anchor, "trial_metrics", None)
    if anchor_config is None or anchor_metrics is None:
        return None

    changes = _config_diff_summary(anchor_config, current_config)
    if not changes:
        return None

    return BundleEffectDelta(
        changes=changes,
        score_delta=float(current_metrics.answer_accuracy) - float(anchor_metrics.answer_accuracy),
        acc_given_complete_delta=(
            float(current_metrics.answer_correct_given_complete_retrieval)
            - float(anchor_metrics.answer_correct_given_complete_retrieval)
        ),
        retrieval_complete_delta=float(current_metrics.retrieval_complete) - float(anchor_metrics.retrieval_complete),
        cost_delta_usd=float(current_cost_usd) - float(getattr(anchor, "mean_llm_cost_per_query_usd", 0.0)),
    )
