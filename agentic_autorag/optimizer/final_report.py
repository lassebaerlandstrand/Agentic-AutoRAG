"""LLM-written, plain-language summary of a completed optimization run.

The structured artifacts (``frontier_report.md``, ``recommended.yaml``,
``history.jsonl``) are precise but dense. This module hands the optimizer model
the run's trajectory — per-trial scores, costs, retrieval metrics, and the
agent's own diagnosis and rationale for each change — and, in cost-aware mode,
asks it to *pick* the recommended config from the Pareto frontier and justify
it; in score-only mode the recommendation is the highest-scoring trial and the
model only narrates it. The orchestrator calls this once at end-of-run and
writes the result to ``optimization_summary.md``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.cost_ledger import CostLedger
from agentic_autorag.litellm_runtime import acompletion_with_cost
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.history import TrialRecord

logger = logging.getLogger(__name__)

# Cost bucket the report's own LLM call is credited to (see CostLedger).
COST_CATEGORY = "final_report"

# Attempts allowed for the cost-aware report to name a valid frontier trial.
_MAX_RECOMMEND_ATTEMPTS = 2

# First ``recommended_trial: <n>`` line the cost-aware model emits ahead of its report.
_RECOMMEND_LINE = re.compile(r"recommended_trial:\s*(\d+)", re.IGNORECASE)

_REPORT_SECTIONS = """## Summary
One short paragraph: what was tuned, how many configs were tried, and the headline result \
(the recommended config's trial number, score, and cost per query).

## What the search found
The trajectory in plain language: which changes moved the score or cost, and what the \
recurring bottleneck was. Ground this in the per-trial diagnoses and rationales — do not \
invent findings.

## Recommendation
{recommendation_guidance}

## What to try next
2-4 concrete, specific suggestions grounded in the observed bottlenecks (e.g. widen a \
search-space range, add a reranker, raise top_k). No generic advice."""

_RULES = """Rules:
- Ground every claim in the provided data. Never invent numbers, configs, or findings.
- Refer to configurations by their trial number.
- Be specific and concise — no filler, no restating the input verbatim, no apologies."""

_SYSTEM_PROMPT_SCORE_ONLY = (
    """You are the reporting voice of Agentic AutoRAG, a tool that tunes Retrieval-Augmented \
Generation pipelines by running an LLM reasoning loop over many trial configurations. You are \
given a digest of one completed run: the per-trial trajectory (scores, costs, retrieval \
metrics, and the agent's own diagnosis and rationale for each change), the Pareto frontier, the \
recommended configuration, and the exam the configs were graded on. This run optimized exam \
score only — cost was not a target, so the recommended config is simply the highest-scoring \
trial.

Write a concise markdown report (roughly 300-500 words) for the technical user who ran the \
optimization and wants to understand what happened and what to do next. Use exactly these \
sections:

"""
    + _REPORT_SECTIONS.format(
        recommendation_guidance="The recommended config (the highest-scoring trial) and what makes it work."
    )
    + "\n\n"
    + _RULES
    + "\n- Output only the markdown report, starting with the `## Summary` heading. Do not wrap it "
    "in a code fence."
)

_SYSTEM_PROMPT_COST_AWARE = (
    """You are the reporting voice of Agentic AutoRAG, a tool that tunes Retrieval-Augmented \
Generation pipelines by running an LLM reasoning loop over many trial configurations. You are \
given a digest of one completed run: the per-trial trajectory (scores, costs, retrieval \
metrics, and the agent's own diagnosis and rationale for each change), the Pareto frontier of \
non-dominated configs (score vs. cost per query, with each config shown), and the exam the \
configs were graded on.

This run optimized two objectives: exam score (higher is better) and cost per query (lower is \
better). Your job is to recommend the single config the user should ship — the best \
capable-and-cheap one. Choose from the Pareto frontier. A higher score is worth a higher cost \
only up to a point: prefer the config where capability is high and paying more buys little extra \
score. Do NOT recommend a cheap config whose score is too low to be useful, and do NOT default \
to the most expensive top scorer when a much cheaper frontier config is nearly as good. Reason \
about the actual frontier shape and the corpus.

Output format — start your reply with exactly this line and nothing before it:
recommended_trial: <a trial number from the Pareto frontier>

then a blank line, then the markdown report. Use exactly these sections:

"""
    + _REPORT_SECTIONS.format(
        recommendation_guidance=(
            "The config you chose and why — name the score/cost tradeoff against the alternatives "
            "(the most expensive top scorer and the cheaper frontier points)."
        )
    )
    + "\n\n"
    + _RULES
    + "\n- `recommended_trial` MUST be one of the frontier trial numbers listed in the digest."
    "\n- After the `recommended_trial:` line and a blank line, output only the markdown report, "
    "starting with `## Summary`. Do not wrap it in a code fence."
)


async def generate_final_report(
    *,
    model: str,
    records: list[TrialRecord],
    fallback_trial: int,
    exam: list[OpenEndedQuestion],
    ledger: CostLedger,
    cost_aware: bool,
    include_graph: bool,
    corpus_description: str,
) -> tuple[int, str]:
    """Return ``(recommended_trial, markdown_body)`` for the completed run.

    ``records`` must be non-empty — the caller only invokes this once at least
    one trial has succeeded. In cost-aware mode the model picks the recommended
    trial from the Pareto frontier; the pick is validated against the frontier,
    retried once, and falls back to ``fallback_trial`` (the max-score trial) on
    failure. In score-only mode there is nothing to choose: ``fallback_trial``
    is the recommendation and the model only narrates it. The LLM spend is
    credited to the ``final_report`` bucket of the active cost ledger.
    """
    frontier = pareto.compute_frontier(list(records))
    frontier_trials = {r.trial_number for r in frontier}

    if not cost_aware:
        context = _build_context(
            records=records,
            recommended_trial=fallback_trial,
            exam=exam,
            ledger=ledger,
            cost_aware=cost_aware,
            include_graph=include_graph,
            corpus_description=corpus_description,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_SCORE_ONLY},
            {"role": "user", "content": context},
        ]
        response, _ = await acompletion_with_cost(cost_category=COST_CATEGORY, model=model, messages=messages)
        return fallback_trial, _strip_code_fence(response.choices[0].message.content or "")

    context = _build_context(
        records=records,
        recommended_trial=None,
        exam=exam,
        ledger=ledger,
        cost_aware=cost_aware,
        include_graph=include_graph,
        corpus_description=corpus_description,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_COST_AWARE},
        {"role": "user", "content": context},
    ]
    raw = ""
    for attempt in range(_MAX_RECOMMEND_ATTEMPTS):
        response, _ = await acompletion_with_cost(cost_category=COST_CATEGORY, model=model, messages=messages)
        raw = response.choices[0].message.content or ""
        trial, body = _parse_recommendation(raw)
        if trial is not None and trial in frontier_trials:
            return trial, body
        if attempt < _MAX_RECOMMEND_ATTEMPTS - 1:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your `recommended_trial` was missing or not on the Pareto frontier. "
                        f"Choose one of these frontier trial numbers: {sorted(frontier_trials)}. "
                        "Begin your reply with `recommended_trial: <n>`, then a blank line, then the report."
                    ),
                }
            )

    logger.warning(
        "Final report did not name a valid frontier trial after %d attempt(s); "
        "falling back to max-score trial %d.",
        _MAX_RECOMMEND_ATTEMPTS,
        fallback_trial,
    )
    _, body = _parse_recommendation(raw)
    return fallback_trial, body


def _parse_recommendation(raw: str) -> tuple[int | None, str]:
    """Split the cost-aware model output into ``(recommended_trial, body)``.

    The trial is read from the first ``recommended_trial: <n>`` line, which is
    then stripped from the body. Returns ``(None, body)`` when no such line is
    present so the caller can retry / fall back.
    """
    match = _RECOMMEND_LINE.search(raw)
    trial = int(match.group(1)) if match else None
    if match:
        raw = raw[: match.start()] + raw[match.end() :]
    return trial, _strip_code_fence(raw)


def _build_context(
    *,
    records: list[TrialRecord],
    recommended_trial: int | None,
    exam: list[OpenEndedQuestion],
    ledger: CostLedger,
    cost_aware: bool,
    include_graph: bool,
    corpus_description: str,
) -> str:
    """Assemble the plain-text digest of the run that the model summarizes.

    ``recommended_trial`` is the pre-selected pick to narrate (score-only mode);
    ``None`` in cost-aware mode, where the model picks from the frontier itself.
    """
    frontier = pareto.compute_frontier(list(records))
    max_score = max(records, key=lambda r: r.score)
    by_trial = {r.trial_number: r for r in records}

    lines: list[str] = [
        f"Corpus: {corpus_description}",
        f"Cost-aware mode: {'on' if cost_aware else 'off'}",
        f"Trials completed: {len(records)}",
        "",
        "## Trial trajectory",
    ]
    for r in records:
        lines.extend(_trial_lines(r))

    lines += ["", "## Pareto frontier (non-dominated configs)"]
    for r in sorted(frontier, key=lambda x: x.score):
        tags = []
        if recommended_trial is not None and r.trial_number == recommended_trial:
            tags.append("recommended")
        if r.trial_number == max_score.trial_number:
            tags.append("max score")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"- trial {r.trial_number}: score={r.score:.3f} cost=${r.mean_llm_cost_per_query_usd:.4f}/q{tag_str}"
        )
        for key, value in r.config.to_prompt_dump(include_graph=include_graph).items():
            lines.append(f"    {key}: {value}")

    if recommended_trial is not None:
        rec = by_trial.get(recommended_trial)
        lines += ["", "## Recommended configuration"]
        if rec is not None:
            lines.append(
                f"Trial {rec.trial_number} "
                f"(score={rec.score:.3f}, cost=${rec.mean_llm_cost_per_query_usd:.4f}/q):"
            )
            for key, value in rec.config.to_prompt_dump(include_graph=include_graph).items():
                lines.append(f"  {key}: {value}")

    lines += ["", "## Exam (the questions configs were graded on)", f"Total questions: {len(exam)}"]
    for rtype, count in sorted(Counter(q.reasoning_type for q in exam).items()):
        lines.append(f"  {rtype}: {count}")

    lines += ["", "## Cost breakdown (USD)"]
    lines.extend(_cost_lines(ledger))
    return "\n".join(lines)


def _trial_lines(record: TrialRecord) -> list[str]:
    """One trial's header, metrics, change rationale, and diagnosis for the digest."""
    out = [
        f"### Trial {record.trial_number}: score={record.score:.3f} cost=${record.mean_llm_cost_per_query_usd:.4f}/q"
    ]
    metrics = record.trial_metrics
    if metrics is not None:
        out.append(
            f"  accuracy={metrics.answer_accuracy:.2f} "
            f"retrieval(complete/partial/miss)="
            f"{metrics.retrieval_complete:.2f}/{metrics.retrieval_partial:.2f}/{metrics.retrieval_miss:.2f} "
            f"refusal={metrics.refusal_rate:.2f}"
        )
    if record.meta is not None and record.meta.rationale:
        out.append(f"  Change: {record.meta.rationale}")
    if record.diagnosis is not None:
        if record.diagnosis.narrative:
            out.append(f"  Diagnosis: {record.diagnosis.narrative}")
        if record.diagnosis.confirmed_findings:
            out.append(f"  Findings: {'; '.join(record.diagnosis.confirmed_findings)}")
    return out


def _cost_lines(ledger: CostLedger) -> list[str]:
    """Per-bucket and total LLM spend for the digest (the report's own call is
    not yet recorded when this runs, so it is excluded by construction)."""
    if not ledger.buckets:
        return ["  (no cost recorded)"]
    out = [f"  {name}: ${bucket.usd:.4f} ({bucket.n_calls} call(s))" for name, bucket in sorted(ledger.buckets.items())]
    out.append(f"  total: ${ledger.total_usd():.4f}")
    return out


def _strip_code_fence(text: str) -> str:
    """Drop a surrounding ``` fence if the model wrapped the whole report in one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.splitlines()[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()
