"""Classify AA model entries as reasoning ON / OFF / unclassified.

LiteLLM models reasoning as a runtime parameter (`reasoning_effort=medium`),
but Artificial Analysis uses separate slugs:

  Pattern A — base = OFF, sibling `-reasoning` = ON
              e.g. claude-4-5-haiku + claude-4-5-haiku-reasoning
  Pattern B — base = ON, sibling `-non-reasoning*` = OFF
              e.g. glm-4-7 + glm-4-7-non-reasoning, gpt-5-1 + gpt-5-1-non-reasoning
  Pattern C — base = OFF, sibling `-thinking` = ON
              e.g. claude-4-5-sonnet + claude-4-5-sonnet-thinking
  Pattern D — base = OFF, siblings `-reasoning-{low,medium,high}` = ON at effort
              e.g. nova-2-0-lite + nova-2-0-lite-reasoning(-medium|-low)
  Pattern E — base = ON (highest effort), siblings `-{low,medium,high}` = ON at effort
              e.g. gpt-oss-120b + gpt-oss-120b-low, gpt-5-4-mini + -medium + -non-reasoning
  Pattern F — base only, no siblings: cannot classify mode

The classifier returns "on" / "off" / None for any entry, and a helper
`select_pair` returns the (off_entry, on_entry) tuple to display, preferring
the medium-effort variant for the ON row to match LiteLLM's default
`reasoning_effort=medium`.
"""

from __future__ import annotations

# Variant types that mean "reasoning is ON" for that AA entry.
_REASONING_ON_VARIANTS = frozenset(
    {
        "reasoning",
        "thinking",
        "adaptive",
        "low",
        "medium",
        "high",
        "reasoning-low",
        "reasoning-medium",
        "reasoning-high",
    }
)

# Variant types that mean "reasoning is OFF".
_REASONING_OFF_VARIANTS = frozenset(
    {
        "non-reasoning",
        "non-reasoning-low-effort",
    }
)

# Subset of ON variants that explicitly mark "reasoning turned on" (vs the
# base). When a sibling has one of these, the base is the OFF default.
_EXPLICIT_REASONING_ON_VARIANTS = frozenset(
    {
        "reasoning",
        "thinking",
        "adaptive",
        "reasoning-low",
        "reasoning-medium",
        "reasoning-high",
    }
)

# ON variants that mark a different effort *within* an always-on model
# family (e.g. gpt-oss-120b + gpt-oss-120b-low — both reasoning, just at
# different effort levels). Their presence doesn't imply the base is OFF.
_EFFORT_LEVEL_VARIANTS = frozenset({"low", "medium", "high"})

# When picking the ON entry for display, prefer entries closest to LiteLLM's
# `reasoning_effort=medium` default. Lower score = better match for "medium".
# The base entry (no variant_type) sits between explicit medium-style variants
# and high/low effort variants — for families where the base is the high-effort
# default (e.g. gpt-oss-120b), the base is closer to medium than `-low`.
_ON_EFFORT_SCORE: dict[str | None, int] = {
    "reasoning-medium": 0,
    "medium": 0,
    "reasoning": 1,
    "thinking": 1,
    "adaptive": 1,
    None: 2,  # base entry
    "high": 3,
    "reasoning-high": 3,
    "low": 4,
    "reasoning-low": 4,
}


def classify(entry: dict, siblings: list[dict] | None = None) -> str | None:
    """Classify a single AA entry as ``"on"``, ``"off"``, or ``None``.

    For variant entries (those with ``variant_type``) the classification is
    intrinsic. For the base entry (no ``variant_type``) the classification
    depends on its siblings — if any sibling explicitly marks the OPPOSITE
    mode, the base is the inferred mode.
    """
    variant_type = (entry.get("variant_type") or "").strip()
    if variant_type:
        if variant_type in _REASONING_OFF_VARIANTS:
            return "off"
        if variant_type in _REASONING_ON_VARIANTS:
            return "on"
        return None

    if siblings is None:
        return None

    sibling_types = {(s.get("variant_type") or "").strip() for s in siblings}
    # An explicit OFF sibling means the base is the ON default.
    if any(t in _REASONING_OFF_VARIANTS for t in sibling_types):
        return "on"
    # An explicit ON sibling (`-reasoning`, `-thinking`, `-adaptive`,
    # `-reasoning-medium`, ...) means the base is the OFF default.
    if any(t in _EXPLICIT_REASONING_ON_VARIANTS for t in sibling_types):
        return "off"
    # Only effort-level siblings (`-low`, `-medium`, `-high`): the family
    # is reasoning-only and the base is the highest-effort entry.
    if any(t in _EFFORT_LEVEL_VARIANTS for t in sibling_types):
        return "on"
    return None


def select_pair(
    base_entry: dict,
    sibling_entries: list[dict],
) -> tuple[dict | None, dict | None]:
    """Return ``(off_entry, on_entry)`` for the model rooted at ``base_entry``.

    ``off_entry`` is whichever entry — base or variant — has reasoning OFF.
    ``on_entry`` is the ON entry preferred by ``_ON_PREFERENCE`` (best match
    for LiteLLM's ``reasoning_effort=medium`` default). Either may be None
    when the corresponding mode has no benchmark on AA.
    """
    base_mode = classify(base_entry, sibling_entries)

    off_entry = base_entry if base_mode == "off" else next((s for s in sibling_entries if classify(s) == "off"), None)

    on_candidates: list[dict] = []
    if base_mode == "on":
        on_candidates.append(base_entry)
    on_candidates.extend(s for s in sibling_entries if classify(s) == "on")

    on_entry = _pick_on(on_candidates)
    return off_entry, on_entry


def _pick_on(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    def effort_score(entry: dict) -> int:
        vt = (entry.get("variant_type") or "").strip() or None
        return _ON_EFFORT_SCORE.get(vt, 5)

    return min(candidates, key=effort_score)
