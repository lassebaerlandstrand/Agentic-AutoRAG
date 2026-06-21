"""Pure markdown renderers for the Pareto-frontier sections of the run report.

No LLM calls and no file I/O. Each function renders one block — the frontier
table, the accuracy-vs-cost chart, tradeoff bullets, per-member configs, a
score-only trials leaderboard, or the recommended config. The ``final_report``
module stitches these into a single ``optimization_summary.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_autorag.config.models import TrialConfig
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.history import TrialRecord

# ASCII chart dimensions. Width fits within an 80-col terminal once axis
# labels are added; height is short enough that a sparse frontier still
# reads cleanly without empty bands dominating the figure.
_CHART_WIDTH = 50
_CHART_HEIGHT = 12


@dataclass
class FrontierMember:
    record: TrialRecord
    is_max_accuracy: bool
    is_recommended: bool


def build_members(records: list[TrialRecord], *, recommended_trial: int | None) -> list[FrontierMember]:
    """Compute the Pareto frontier and tag the max-accuracy + recommended members.

    Returns members sorted by accuracy ascending; empty when there are no
    non-dominated configs (i.e. no records).
    """
    frontier = pareto.compute_frontier(records)
    if not frontier:
        return []
    max_score_record = max(frontier, key=lambda r: r.answer_accuracy)
    return [
        FrontierMember(
            record=r,
            is_max_accuracy=(r.trial_number == max_score_record.trial_number),
            is_recommended=(r.trial_number == recommended_trial),
        )
        for r in sorted(frontier, key=lambda r: r.answer_accuracy)
    ]


def frontier_hypervolume(records: list[TrialRecord], frontier: list[TrialRecord]) -> float:
    """Dominated hypervolume of ``frontier`` against a cost reference drawn from
    every recorded trial (the same convention used for the saved artifacts)."""
    cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
    cost_ref = pareto.cost_reference(cost_values)
    return pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))


def render_run_stats(records: list[TrialRecord], frontier: list[TrialRecord], hv: float) -> str:
    """One-line run header: trial count, frontier size, hypervolume."""
    return f"{len(records)} trial(s) · {len(frontier)} non-dominated config(s) · hypervolume {hv:.4f}"


def render_frontier_table(members: list[FrontierMember]) -> str:
    lines = [
        "## Pareto frontier",
        "",
        "| Trial | Accuracy | Cost / query | Notes |",
        "|------:|---------:|-------------:|-------|",
    ]
    for m in members:
        notes = []
        if m.is_max_accuracy:
            notes.append("max accuracy")
        if m.is_recommended:
            notes.append("**recommended**")
        notes_str = ", ".join(notes) if notes else ""
        lines.append(
            f"| {m.record.trial_number} | "
            f"{m.record.answer_accuracy:.3f} | "
            f"${m.record.mean_llm_cost_per_query_usd:.4f} | "
            f"{notes_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_trials_leaderboard(records: list[TrialRecord], *, recommended_trial: int | None) -> str:
    """Score-only leaderboard: every trial ranked by accuracy.

    Cost is shown for information only — it is not an optimization objective in
    this mode, so there is no Pareto framing, chart, or tradeoff section.
    """
    best = max(records, key=lambda r: r.answer_accuracy) if records else None
    lines = [
        "## Trials (by accuracy)",
        "",
        "| Trial | Accuracy | Cost / query (info) | Notes |",
        "|------:|---------:|--------------------:|-------|",
    ]
    for r in sorted(records, key=lambda r: r.answer_accuracy, reverse=True):
        notes = []
        if best is not None and r.trial_number == best.trial_number:
            notes.append("best accuracy")
        if r.trial_number == recommended_trial:
            notes.append("**recommended**")
        notes_str = ", ".join(notes) if notes else ""
        lines.append(
            f"| {r.trial_number} | {r.answer_accuracy:.3f} | ${r.mean_llm_cost_per_query_usd:.4f} | {notes_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_frontier_chart(members: list[FrontierMember]) -> str:
    """Simple ASCII scatter of the frontier in (cost, accuracy) space.

    Single-point and zero-range frontiers fall back to one-liners — there's
    no useful chart for those, and a degenerate grid would be more confusing
    than helpful.
    """
    lines = ["### Accuracy vs cost", "", "```"]
    if len(members) < 2:
        lines.extend(["(too few frontier members for a chart)", "```", ""])
        return "\n".join(lines)

    scores = [m.record.answer_accuracy for m in members]
    costs = [m.record.mean_llm_cost_per_query_usd for m in members]
    score_min, score_max = min(scores), max(scores)
    cost_min, cost_max = min(costs), max(costs)
    score_range = score_max - score_min
    cost_range = cost_max - cost_min
    if score_range <= 1e-9 or cost_range <= 1e-9:
        lines.extend(["(degenerate frontier — accuracy or cost range is zero)", "```", ""])
        return "\n".join(lines)

    grid = [[" "] * _CHART_WIDTH for _ in range(_CHART_HEIGHT)]
    for m in members:
        x = int((m.record.mean_llm_cost_per_query_usd - cost_min) / cost_range * (_CHART_WIDTH - 1))
        y_norm = (m.record.answer_accuracy - score_min) / score_range
        y = (_CHART_HEIGHT - 1) - int(y_norm * (_CHART_HEIGHT - 1))
        marker = "★" if m.is_recommended else "*"
        grid[y][x] = marker

    for i, row in enumerate(grid):
        row_str = "".join(row)
        if i == 0:
            lines.append(f"accuracy {score_max:.3f} |{row_str}")
        elif i == _CHART_HEIGHT - 1:
            lines.append(f"accuracy {score_min:.3f} |{row_str}")
        else:
            lines.append("               |" + row_str)
    lines.append("               +" + ("-" * _CHART_WIDTH))
    lines.append(f"              cost ${cost_min:.4f}/q" + " " * (_CHART_WIDTH - 22) + f"${cost_max:.4f}/q")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_tradeoffs(members: list[FrontierMember]) -> str:
    """One bullet per frontier member describing its tradeoff vs. the max-accuracy config."""
    lines = ["### Tradeoffs", ""]
    if len(members) < 2:
        lines.append("(only one frontier member — no tradeoff to describe)")
        lines.append("")
        return "\n".join(lines)

    leader = next((m for m in members if m.is_max_accuracy), members[-1])
    leader_score = leader.record.answer_accuracy
    leader_cost = leader.record.mean_llm_cost_per_query_usd
    for m in members:
        rec = m.record
        if m.is_max_accuracy:
            lines.append(
                f"- **trial {rec.trial_number}** (max accuracy): accuracy={rec.answer_accuracy:.3f}, "
                f"cost=${rec.mean_llm_cost_per_query_usd:.4f}/q. The accuracy leader."
            )
            continue
        score_delta_pct = (rec.answer_accuracy - leader_score) / leader_score * 100.0 if leader_score > 0 else 0.0
        cost_delta_pct = (
            (rec.mean_llm_cost_per_query_usd - leader_cost) / leader_cost * 100.0 if leader_cost > 0 else 0.0
        )
        lines.append(
            f"- **trial {rec.trial_number}**: "
            f"{score_delta_pct:+.1f}% accuracy, {cost_delta_pct:+.1f}% cost vs. trial "
            f"{leader.record.trial_number}."
        )
    lines.append("")
    return "\n".join(lines)


def render_full_configs(members: list[FrontierMember], *, include_graph: bool) -> str:
    """Per-frontier-member compact YAML rendering.

    Mirrors the per-trial YAML emitted to ``frontier/`` so a reader scanning
    the report sees the configs without opening every file.
    """
    lines = ["### Per-frontier-member configs", ""]
    for m in members:
        cfg = m.record.config
        tags = []
        if m.is_recommended:
            tags.append("recommended")
        if m.is_max_accuracy:
            tags.append("max accuracy")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"#### Trial {m.record.trial_number}{tag_str}\n")
        lines.append("```yaml")
        lines.extend(_compact_config_yaml_lines(cfg, include_graph=include_graph))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_recommended_config(record: TrialRecord, *, include_graph: bool) -> str:
    """Full YAML of the recommended config (score-only mode, where there is no
    Pareto frontier to enumerate)."""
    lines = ["## Recommended config", "", "```yaml"]
    lines.extend(_compact_config_yaml_lines(record.config, include_graph=include_graph))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _compact_config_yaml_lines(cfg: TrialConfig, *, include_graph: bool) -> list[str]:
    """Render a TrialConfig as YAML lines suitable for embedding in markdown."""
    payload = cfg.to_prompt_dump(include_graph=include_graph)
    return [f"{k}: {_yaml_scalar(v)}" for k, v in payload.items()]


def _yaml_scalar(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    return str(v)
