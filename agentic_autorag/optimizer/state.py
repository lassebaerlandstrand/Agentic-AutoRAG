"""Pure-function helpers that compute diagnostic state from exam results and history.

These deliberately do NOT call an LLM. They replace what the current Diagnoser has
to re-derive from free text every trial with mechanical arithmetic on the fields
already stored in ``QuestionResult``. An LLM's job shrinks to interpretation and
selection, not re-discovery.

Imported by the ReasoningAgent before each LLM call; the results are rendered into
the prompt so both Diagnoser and Proposer see the same grounded signal.
"""

from __future__ import annotations

from agentic_autorag.config.models import TrialConfig
from agentic_autorag.examiner.evaluator import _ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL, ExamResult
from agentic_autorag.optimizer.diagnosis import (
    METRIC_POLARITY,
    HypothesisCheck,
    MoveType,
    ProposalMeta,
    Stage,
    StageMetrics,
    StateCard,
)

_ERROR_SENTINELS = (_ERROR_SENTINEL, _PERMANENT_ERROR_SENTINEL)

# A confirmed hypothesis must predict an improvement AND the observed improvement
# must be at least this fraction of the predicted improvement. 0.5 tolerates
# noisy directional wins while rejecting predictions whose magnitudes are wildly
# off. See TECHNICAL_DESIGN.md for the rationale.
_HYPOTHESIS_TOLERANCE_RATIO = 0.5

# Move-type state-machine thresholds. These encode the taxonomy in the design
# doc — tuned rather than derived, and worth flagging in the paper as
# hyperparameters of the optimizer itself.
_REVERT_REGRESSION_THRESHOLD = -0.05  # score drop ≤ this → REVERT
_PIVOT_AFTER_NON_IMPROVEMENTS = 2  # N non-improvements in a row → PIVOT
_COMPOUND_MIN_TRIALS_REMAINING = 2  # "late in budget" cutoff for COMPOUND
_COMPOUND_MIN_CONFIRMED = 2  # need ≥ this many confirmed interventions to COMPOUND


def compute_stage_metrics(exam_result: ExamResult, reranker_top_n: int) -> StageMetrics:
    """Compute stage-decomposed metrics from a completed exam.

    Excludes questions that hit system-error sentinels (timeouts, API failures) from
    all rates — those are not diagnostic of the pipeline, only of external flakiness.
    ``reranker_top_n`` gates whether the gold chunk landed in the window the LLM
    actually saw; pass the trial's runtime value.
    """
    valid = [qr for qr in exam_result.question_results if qr.generated_response not in _ERROR_SENTINELS]
    n = len(valid)
    if n == 0:
        return StageMetrics()

    n_sufficient = sum(1 for qr in valid if qr.context_sufficient)
    retrieval_success = n_sufficient / n

    sufficient = [qr for qr in valid if qr.context_sufficient and qr.source_fact_rank > 0]
    ranking_quality = sum(1.0 / qr.source_fact_rank for qr in sufficient) / len(sufficient) if sufficient else 0.0

    in_window = sum(1 for qr in valid if 0 < qr.source_fact_rank <= reranker_top_n)
    gold_in_reranker_window = in_window / n

    gen_correct = sum(1 for qr in valid if qr.context_sufficient and qr.correct)
    generation_given_context = gen_correct / n_sufficient if n_sufficient else 0.0

    return StageMetrics(
        retrieval_success=retrieval_success,
        ranking_quality=ranking_quality,
        gold_in_reranker_window=gold_in_reranker_window,
        generation_given_context=generation_given_context,
        n_eligible_for_generation=n_sufficient,
    )


def check_prior_hypothesis(
    prev_meta: ProposalMeta | None,
    prev_metrics: StageMetrics | None,
    current_metrics: StageMetrics,
) -> HypothesisCheck:
    """Arithmetically verify whether the previous Proposer's hypothesis came true.

    Returns a ``HypothesisCheck`` with verdict ``confirmed`` / ``falsified`` /
    ``inconclusive`` / ``n/a``. No LLM involvement — this is a truth signal.

    Rules:
    - If there was no prior hypothesis (first trial, or empty meta), verdict is n/a.
    - If the target_metric can't be read from StageMetrics, verdict is inconclusive.
    - A hypothesis is judged against the metric's *polarity* (higher-is-better vs
      lower-is-better; currently all tracked metrics are higher-is-better).
      The Proposer's ``expected_delta`` multiplied by polarity gives the
      expected *improvement*; a non-positive expected improvement means the
      Proposer predicted a regression or no-op, which counts as a falsified
      hypothesis. Confirmed iff the observed improvement is at least
      ``_HYPOTHESIS_TOLERANCE_RATIO`` of the predicted improvement.
    """
    if prev_meta is None or not prev_meta.target_metric or prev_metrics is None:
        return HypothesisCheck()

    prev_val = getattr(prev_metrics, prev_meta.target_metric, None)
    curr_val = getattr(current_metrics, prev_meta.target_metric, None)
    if prev_val is None or curr_val is None or not isinstance(prev_val, (int, float)):
        return HypothesisCheck(
            prior_hypothesis=prev_meta.hypothesis,
            target_metric=prev_meta.target_metric,
            expected_delta=prev_meta.expected_delta,
            verdict="inconclusive",
        )

    polarity = METRIC_POLARITY.get(prev_meta.target_metric, +1)
    observed = float(curr_val) - float(prev_val)
    expected = prev_meta.expected_delta
    expected_improvement = expected * polarity
    observed_improvement = observed * polarity

    if expected == 0.0:
        verdict = "inconclusive"
    elif expected_improvement <= 0.0:
        # Proposer's prediction doesn't describe an improvement once polarity is
        # applied — predicting a regression or no-op counts as falsified.
        verdict = "falsified"
    elif observed_improvement >= _HYPOTHESIS_TOLERANCE_RATIO * expected_improvement and observed_improvement > 0:
        verdict = "confirmed"
    else:
        verdict = "falsified"

    return HypothesisCheck(
        prior_hypothesis=prev_meta.hypothesis,
        target_metric=prev_meta.target_metric,
        expected_delta=expected,
        observed_delta=observed,
        verdict=verdict,
    )


def build_state_card(
    trial_number: int,
    trials_remaining: int,
    current_metrics: StageMetrics,
    current_score: float,
    history_records: list,  # list[TrialRecord]; typed loosely to avoid circular import
) -> StateCard:
    """Mechanically summarise optimizer state from history.

    ``history_records`` is expected to contain the just-completed trial plus any
    earlier ones; the function reads ``record.score``, ``record.diagnosis``, and
    ``record.meta``. When any field is missing (e.g. initial trial), sensible
    zero-defaults are used.
    """
    bottleneck = current_metrics.bottleneck()
    prev_bottleneck = prior_bottleneck(history_records)
    bottleneck_stable = prev_bottleneck is not None and prev_bottleneck == bottleneck

    best_score_so_far = current_score
    best_trial_number = trial_number
    last_trial_delta = 0.0
    consecutive_non_improvements = 0

    sorted_hist = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))
    for rec in sorted_hist:
        score = getattr(rec, "score", 0.0)
        tnum = getattr(rec, "trial_number", 0)
        if score > best_score_so_far:
            best_score_so_far = score
            best_trial_number = tnum

    # last delta = current vs. immediately previous (if any)
    prior_scores = [getattr(r, "score", 0.0) for r in sorted_hist if getattr(r, "trial_number", 0) < trial_number]
    if prior_scores:
        last_trial_delta = current_score - prior_scores[-1]

    # consecutive_non_improvements counts trials that didn't raise the running best,
    # including the current one. A trial strictly greater than the running best resets
    # the counter to 0; otherwise the counter increments.
    best_seen_so_far = -float("inf")
    sequence = [getattr(rec, "score", 0.0) for rec in sorted_hist] + [current_score]
    for score in sequence:
        if score > best_seen_so_far:
            best_seen_so_far = score
            consecutive_non_improvements = 0
        else:
            consecutive_non_improvements += 1

    interventions_tried = _collect_interventions(sorted_hist)
    top_trials = _top_trials(sorted_hist, k=3)

    return StateCard(
        trial_number=trial_number,
        trials_remaining=trials_remaining,
        best_score_so_far=best_score_so_far,
        best_trial_number=best_trial_number,
        last_trial_delta=last_trial_delta,
        consecutive_non_improvements=consecutive_non_improvements,
        current_bottleneck=bottleneck,
        bottleneck_stable=bottleneck_stable,
        interventions_tried=interventions_tried,
        top_trials=top_trials,
        suggested_move_type=suggest_move_type(
            bottleneck=bottleneck,
            bottleneck_stable=bottleneck_stable,
            consecutive_non_improvements=consecutive_non_improvements,
            last_trial_delta=last_trial_delta,
            trials_remaining=trials_remaining,
            interventions_tried=interventions_tried,
        ),
    )


def suggest_move_type(
    bottleneck: Stage,
    bottleneck_stable: bool,
    consecutive_non_improvements: int,
    last_trial_delta: float,
    trials_remaining: int,
    interventions_tried: list[tuple[str, str, str, str]],
) -> MoveType:
    """Recommend a move type based on optimizer state. Proposer may override.

    The recommendation encodes the state-driven taxonomy from the plan:
      - REVERT when a bad regression just happened.
      - PIVOT after 2+ non-improvements, or after every tried intervention for the
        current bottleneck was *falsified* (pending/inconclusive don't count).
      - COMPOUND late in the budget when we have confirmed interventions to combine.
      - REFINE when we're trending up with a stable bottleneck.
      - PROBE otherwise (the default when there's a bottleneck to investigate).
    """
    confirmed_count = sum(1 for *_, verdict in interventions_tried if verdict == "confirmed")

    if last_trial_delta <= _REVERT_REGRESSION_THRESHOLD:
        return MoveType.REVERT

    if consecutive_non_improvements >= _PIVOT_AFTER_NON_IMPROVEMENTS or _all_bottleneck_interventions_failed(
        bottleneck, interventions_tried
    ):
        return MoveType.PIVOT

    if trials_remaining <= _COMPOUND_MIN_TRIALS_REMAINING and confirmed_count >= _COMPOUND_MIN_CONFIRMED:
        return MoveType.COMPOUND

    if bottleneck_stable and last_trial_delta > 0:
        return MoveType.REFINE

    return MoveType.PROBE


def stage_metrics_from_config_and_result(exam_result: ExamResult, config: TrialConfig) -> StageMetrics:
    """Convenience wrapper — picks the correct ``reranker_top_n`` off the config."""
    return compute_stage_metrics(exam_result, reranker_top_n=config.reranker_top_n)


def prior_bottleneck(history_records: list) -> Stage | None:
    """The most-recently diagnosed bottleneck in ``history_records``, or None."""
    for rec in sorted(history_records, key=lambda r: getattr(r, "trial_number", 0), reverse=True):
        diagnosis = getattr(rec, "diagnosis", None)
        if diagnosis is None:
            continue
        # diagnosis.bottleneck is an enum; it may be None on the very first trial
        bot = getattr(diagnosis, "bottleneck", None)
        if bot is not None:
            return Stage(bot)
    return None


def _collect_interventions(history_records: list) -> list[tuple[str, str, str, str]]:
    """Walk history and collect (lever, value_from, value_to, verdict) tuples.

    ``TrialRecord.meta`` stored on trial N describes the Proposer's plan for
    trial N+1 (the orchestrator attaches ``proposal_meta`` after trial N
    evaluates and before trial N+1 starts). So the intervention this meta
    describes is realised in trial N+1's config, not trial N's. ``verdict``
    comes from trial N+1's ``hypothesis_check`` — that's the diagnosis
    that verified whether the predicted change held up.

    Records whose forward-pointing intervention hasn't been executed yet
    (the last trial in history) are skipped: they have no ``value_to`` to
    report yet.
    """
    out: list[tuple[str, str, str, str]] = []
    ordered = sorted(history_records, key=lambda r: getattr(r, "trial_number", 0))
    for idx, rec in enumerate(ordered):
        meta = getattr(rec, "meta", None)
        if meta is None or not meta.primary_lever:
            continue
        if idx + 1 >= len(ordered):
            continue  # Intervention hasn't been realised in a subsequent trial yet.
        next_rec = ordered[idx + 1]
        value_from = _lever_value_str(getattr(rec, "config", None), meta.primary_lever)
        value_to = _lever_value_str(getattr(next_rec, "config", None), meta.primary_lever)
        verdict = "pending"
        nxt_diag = getattr(next_rec, "diagnosis", None)
        if nxt_diag is not None and nxt_diag.hypothesis_check:
            verdict = nxt_diag.hypothesis_check.verdict
        out.append((meta.primary_lever, value_from, value_to, verdict))
    return out


def _lever_value_str(config, lever: str) -> str:
    """Stringify a config lever value, unwrapping enums to their .value."""
    if config is None or not lever:
        return ""
    raw = getattr(config, lever, None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw))


def _top_trials(history_records: list, k: int) -> list[dict]:
    """Return the top-k trial summaries by composite score as serialisable dicts."""
    ordered = sorted(history_records, key=lambda r: getattr(r, "score", 0.0), reverse=True)
    out = []
    for rec in ordered[:k]:
        config = getattr(rec, "config", None)
        summary = {
            "trial_number": getattr(rec, "trial_number", 0),
            "score": float(getattr(rec, "score", 0.0)),
            "mcq_accuracy": float(getattr(rec, "mcq_accuracy", 0.0)),
        }
        if config is not None:
            summary["primary_levers"] = {
                "index_type": getattr(config.index_type, "value", str(config.index_type)),
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
                "reranker": config.reranker,
                "chunk_token_size": config.chunk_token_size,
                "top_k": config.top_k,
                "reasoning": config.reasoning,
            }
        out.append(summary)
    return out


def _all_bottleneck_interventions_failed(
    bottleneck: Stage, interventions_tried: list[tuple[str, str, str, str]]
) -> bool:
    """True when every resolved intervention for this bottleneck was falsified.

    ``pending`` and ``inconclusive`` don't count as failures — only explicit
    ``falsified`` verdicts do. If no intervention has resolved yet, returns False
    so the optimizer stays in PROBE / REFINE mode rather than prematurely
    escalating to PIVOT.
    """
    relevant_levers = PRIMARY_LEVERS_BY_STAGE[bottleneck]
    resolved = [
        verdict
        for lever, _from, _to, verdict in interventions_tried
        if lever in relevant_levers and verdict in {"confirmed", "falsified"}
    ]
    if not resolved:
        return False
    return all(v == "falsified" for v in resolved)


# Primary levers associated with each stage.  Referenced by the move-type validators
# so the Proposer can be told "the retrieval bottleneck is addressed by these levers".
PRIMARY_LEVERS_BY_STAGE: dict[Stage, frozenset[str]] = {
    Stage.RETRIEVAL: frozenset({"embedding_model", "chunking_strategy", "chunk_token_size", "index_type"}),
    Stage.RANKING: frozenset({"reranker", "top_k", "hybrid_alpha"}),
    Stage.GENERATION: frozenset({"llm_model", "reasoning"}),
}

# The full set of "primary" parameters across all stages. Move-type validators
# count changes against this set: secondary params (top_n fine-tuning, overlap,
# temperature) are free to co-move.
PRIMARY_LEVERS: frozenset[str] = frozenset().union(*PRIMARY_LEVERS_BY_STAGE.values())

# Non-primary "tuning" parameters — changes to these don't count toward the
# per-move-type lever cap. Kept explicit so the validator's behaviour is auditable.
SECONDARY_LEVERS: frozenset[str] = frozenset(
    {"chunk_token_overlap", "reranker_top_n", "temperature", "query_expansion", "graph_query_mode", "graph_top_k"}
)

# Small-step thresholds for REFINE moves — a REFINE may change a primary lever only
# if the delta stays within the bound for that lever. Discrete params (model names,
# index_type) are not allowed to change in REFINE at all.
REFINE_SMALL_STEPS: dict[str, float] = {
    "chunk_token_size": 0.25,  # ±25% relative
    "top_k": 3,  # ±3 absolute
    "hybrid_alpha": 0.15,  # ±0.15 absolute
}
