"""Render the final Pareto frontier as a human-readable markdown report.

Pure functions over already-computed frontier records; no LLM calls.
The orchestrator calls ``render_report`` at the end of a run and writes
the result to ``frontier_report.md`` in the output directory.
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
class _FrontierMember:
    record: TrialRecord
    is_knee: bool
    is_max_score: bool
    is_recommended: bool


def render_report(
    *,
    records: list[TrialRecord],
    policy: pareto.SelectionPolicy,
    recommended_trial: int | None,
    include_graph: bool,
) -> str:
    """Return a markdown report describing the Pareto frontier and the recommended pick.

    ``recommended_trial`` is the trial number resolved by ``policy``. May be
    ``None`` when no frontier member satisfies the policy (e.g. cheapest_above
    with an unmet score threshold).
    """
    if not records:
        return "# Pareto Frontier Report\n\nNo trials completed.\n"

    frontier = pareto.compute_frontier(records)
    if not frontier:
        return "# Pareto Frontier Report\n\nNo non-dominated trials found.\n"

    knee_record = pareto.find_knee(frontier)
    knee_trial = knee_record.trial_number if knee_record else None
    max_score_record = max(frontier, key=lambda r: r.score)

    members = [
        _FrontierMember(
            record=r,
            is_knee=(r.trial_number == knee_trial),
            is_max_score=(r.trial_number == max_score_record.trial_number),
            is_recommended=(r.trial_number == recommended_trial),
        )
        for r in sorted(frontier, key=lambda r: r.score)
    ]

    cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
    cost_ref = max(cost_values) if cost_values else 1.0
    if cost_ref <= 0.0:
        cost_ref = 1.0
    hv = pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))

    sections: list[str] = []
    sections.append("# Pareto Frontier Report\n")
    sections.append(_render_summary(records, frontier, hv, policy, recommended_trial))
    sections.append(_render_table(members))
    sections.append(_render_chart(members))
    sections.append(_render_tradeoffs(members))
    sections.append(_render_full_configs(members, include_graph=include_graph))
    return "\n".join(sections).rstrip() + "\n"


def _render_summary(
    records: list[TrialRecord],
    frontier: list[TrialRecord],
    hv: float,
    policy: pareto.SelectionPolicy,
    recommended_trial: int | None,
) -> str:
    rec_line = (
        f"**Recommended trial**: #{recommended_trial} (`recommended.yaml`)"
        if recommended_trial is not None
        else "**Recommended trial**: none — no frontier member satisfies the policy."
    )
    return (
        f"**Run summary**: {len(records)} trial(s), "
        f"{len(frontier)} non-dominated config(s), hypervolume = {hv:.4f}.\n\n"
        f"**Selection policy**: `{policy.kind}` — {policy.describe()}\n\n"
        f"{rec_line}\n"
    )


def _render_table(members: list[_FrontierMember]) -> str:
    lines = [
        "## Frontier",
        "",
        "| Trial | Score | Cost / query | Notes |",
        "|------:|------:|-------------:|-------|",
    ]
    for m in members:
        notes = []
        if m.is_knee:
            notes.append("knee (best score per dollar)")
        if m.is_max_score:
            notes.append("max score")
        if m.is_recommended:
            notes.append("**recommended**")
        notes_str = ", ".join(notes) if notes else ""
        lines.append(
            f"| {m.record.trial_number} | "
            f"{m.record.score:.3f} | "
            f"${m.record.mean_llm_cost_per_query_usd:.4f} | "
            f"{notes_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_chart(members: list[_FrontierMember]) -> str:
    """Simple ASCII scatter of the frontier in (cost, score) space.

    Single-point and zero-range frontiers fall back to one-liners — there's
    no useful chart for those, and a degenerate grid would be more confusing
    than helpful.
    """
    lines = ["## Score vs cost", "", "```"]
    if len(members) < 2:
        lines.extend(["(too few frontier members for a chart)", "```", ""])
        return "\n".join(lines)

    scores = [m.record.score for m in members]
    costs = [m.record.mean_llm_cost_per_query_usd for m in members]
    score_min, score_max = min(scores), max(scores)
    cost_min, cost_max = min(costs), max(costs)
    score_range = score_max - score_min
    cost_range = cost_max - cost_min
    if score_range <= 1e-9 or cost_range <= 1e-9:
        lines.extend(["(degenerate frontier — score or cost range is zero)", "```", ""])
        return "\n".join(lines)

    grid = [[" "] * _CHART_WIDTH for _ in range(_CHART_HEIGHT)]
    for m in members:
        x = int((m.record.mean_llm_cost_per_query_usd - cost_min) / cost_range * (_CHART_WIDTH - 1))
        y_norm = (m.record.score - score_min) / score_range
        y = (_CHART_HEIGHT - 1) - int(y_norm * (_CHART_HEIGHT - 1))
        marker = "★" if m.is_recommended else "*"
        grid[y][x] = marker

    for i, row in enumerate(grid):
        row_str = "".join(row)
        if i == 0:
            lines.append(f"score {score_max:.3f} |{row_str}")
        elif i == _CHART_HEIGHT - 1:
            lines.append(f"score {score_min:.3f} |{row_str}")
        else:
            lines.append("            |" + row_str)
    lines.append("            +" + ("-" * _CHART_WIDTH))
    lines.append(f"           cost ${cost_min:.4f}/q" + " " * (_CHART_WIDTH - 22) + f"${cost_max:.4f}/q")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_tradeoffs(members: list[_FrontierMember]) -> str:
    """One bullet per frontier member describing its tradeoff vs. the max-score config."""
    lines = ["## Tradeoffs", ""]
    if len(members) < 2:
        lines.append("(only one frontier member — no tradeoff to describe)")
        lines.append("")
        return "\n".join(lines)

    leader = next((m for m in members if m.is_max_score), members[-1])
    leader_score = leader.record.score
    leader_cost = leader.record.mean_llm_cost_per_query_usd
    for m in members:
        rec = m.record
        if m.is_max_score:
            lines.append(
                f"- **trial {rec.trial_number}** (max score): score={rec.score:.3f}, "
                f"cost=${rec.mean_llm_cost_per_query_usd:.4f}/q. The score leader."
            )
            continue
        score_delta_pct = (rec.score - leader_score) / leader_score * 100.0 if leader_score > 0 else 0.0
        cost_delta_pct = (
            (rec.mean_llm_cost_per_query_usd - leader_cost) / leader_cost * 100.0 if leader_cost > 0 else 0.0
        )
        knee_tag = " (knee)" if m.is_knee else ""
        lines.append(
            f"- **trial {rec.trial_number}**{knee_tag}: "
            f"{score_delta_pct:+.1f}% score, {cost_delta_pct:+.1f}% cost vs. trial "
            f"{leader.record.trial_number}."
        )
    lines.append("")
    return "\n".join(lines)


def _render_full_configs(members: list[_FrontierMember], *, include_graph: bool) -> str:
    """Per-frontier-member compact YAML rendering.

    Mirrors the per-trial YAML emitted to ``frontier/`` so a reader scanning
    the report sees the configs without opening every file.
    """
    lines = ["## Per-frontier-member configs", ""]
    for m in members:
        cfg = m.record.config
        tags = []
        if m.is_recommended:
            tags.append("recommended")
        if m.is_knee:
            tags.append("knee")
        if m.is_max_score:
            tags.append("max score")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"### Trial {m.record.trial_number}{tag_str}\n")
        lines.append("```yaml")
        lines.extend(_compact_config_yaml_lines(cfg, include_graph=include_graph))
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
