"""LLM-written, plain-language summary of a completed optimization run.

The structured artifacts (``frontier_report.md``, ``recommended.yaml``,
``history.jsonl``) are precise but dense. This module hands the optimizer
model the run's trajectory — per-trial scores, costs, retrieval metrics, and
the agent's own diagnosis and rationale for each change — and asks it to write
a short report: what the search found, why the recommended config was chosen,
and what to try next. The orchestrator calls it once at end-of-run and writes
the result to ``optimization_summary.md``.
"""

from __future__ import annotations

from collections import Counter

from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.cost_ledger import CostLedger
from agentic_autorag.litellm_runtime import acompletion_with_cost
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.history import TrialRecord

# Cost bucket the report's own LLM call is credited to (see CostLedger).
COST_CATEGORY = "final_report"

_SYSTEM_PROMPT = """You are the reporting voice of Agentic AutoRAG, a tool that tunes \
Retrieval-Augmented Generation pipelines by running an LLM reasoning loop over many trial \
configurations. You are given a digest of one completed run: the per-trial trajectory \
(scores, costs, retrieval metrics, and the agent's own diagnosis and rationale for each \
change), the Pareto frontier, the recommended configuration, and the exam the configs were \
graded on.

Write a concise markdown report (roughly 300-500 words) for the technical user who ran the \
optimization and wants to understand what happened and what to do next. Use exactly these \
sections:

## Summary
One short paragraph: what was tuned, how many configs were tried, and the headline result \
(the recommended config's trial number, score, and cost per query).

## What the search found
The trajectory in plain language: which changes moved the score or cost, and what the \
recurring bottleneck was. Ground this in the per-trial diagnoses and rationales — do not \
invent findings.

## Recommendation
The recommended config and why it was selected (state the selection policy). If cost-aware \
mode is on and cheaper frontier configs exist, note the score/cost tradeoff.

## What to try next
2-4 concrete, specific suggestions grounded in the observed bottlenecks (e.g. widen a \
search-space range, add a reranker, raise top_k). No generic advice.

Rules:
- Ground every claim in the provided data. Never invent numbers, configs, or findings.
- Refer to configurations by their trial number.
- Be specific and concise — no filler, no restating the input verbatim, no apologies.
- Output only the markdown report, starting with the `## Summary` heading. Do not wrap it \
in a code fence."""


async def generate_final_report(
    *,
    model: str,
    records: list[TrialRecord],
    recommended: TrialRecord | None,
    objective: pareto.SelectionPolicy,
    exam: list[OpenEndedQuestion],
    ledger: CostLedger,
    cost_aware: bool,
    include_graph: bool,
    corpus_description: str,
) -> str:
    """Return the model's markdown report body (without a title header).

    ``records`` must be non-empty — the caller only invokes this once at least
    one trial has succeeded. The LLM spend is credited to the ``final_report``
    bucket of the active cost ledger.
    """
    context = _build_context(
        records=records,
        recommended=recommended,
        objective=objective,
        exam=exam,
        ledger=ledger,
        cost_aware=cost_aware,
        include_graph=include_graph,
        corpus_description=corpus_description,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    response, _ = await acompletion_with_cost(cost_category=COST_CATEGORY, model=model, messages=messages)
    return _strip_code_fence(response.choices[0].message.content or "")


def _build_context(
    *,
    records: list[TrialRecord],
    recommended: TrialRecord | None,
    objective: pareto.SelectionPolicy,
    exam: list[OpenEndedQuestion],
    ledger: CostLedger,
    cost_aware: bool,
    include_graph: bool,
    corpus_description: str,
) -> str:
    """Assemble the plain-text digest of the run that the model summarizes."""
    frontier = pareto.compute_frontier(list(records))
    knee = pareto.find_knee(frontier)
    max_score = max(records, key=lambda r: r.score)

    lines: list[str] = [
        f"Corpus: {corpus_description}",
        f"Selection policy: {objective.kind} — {objective.describe()}",
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
        if recommended is not None and r.trial_number == recommended.trial_number:
            tags.append("recommended")
        if knee is not None and r.trial_number == knee.trial_number:
            tags.append("knee")
        if r.trial_number == max_score.trial_number:
            tags.append("max score")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"- trial {r.trial_number}: score={r.score:.3f} cost=${r.mean_llm_cost_per_query_usd:.4f}/q{tag_str}"
        )

    lines += ["", "## Recommended configuration"]
    if recommended is not None:
        lines.append(
            f"Trial {recommended.trial_number} "
            f"(score={recommended.score:.3f}, cost=${recommended.mean_llm_cost_per_query_usd:.4f}/q):"
        )
        for key, value in recommended.config.to_prompt_dump(include_graph=include_graph).items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("None — no frontier member satisfied the selection policy.")

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
