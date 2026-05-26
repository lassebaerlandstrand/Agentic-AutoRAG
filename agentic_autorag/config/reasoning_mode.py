"""Classify AA model entries as reasoning ON / OFF / unclassified.

LiteLLM models reasoning as a runtime parameter (``reasoning_effort=medium``),
but Artificial Analysis uses separate slugs (``-reasoning``, ``-thinking``,
``-non-reasoning``, effort suffixes like ``-low/-medium/-high``). This
module maps AA slugs to ``"on" | "off" | None`` and pairs them up for
display.
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

# When picking the ON entry, prefer entries closest to the target reasoning
# effort. Lower score = better match. The base entry (no variant_type) is
# treated as approximately high effort because most reasoning-model families
# publish their headline benchmarks under the highest-effort setting (the
# canonical AA pattern); thus base ranks just below "high" for a "high" target
# and several steps away from "low".
_ON_EFFORT_SCORES: dict[str, dict[str | None, int]] = {
    "medium": {
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
    },
    "high": {
        "reasoning-high": 0,
        "high": 0,
        None: 1,  # base usually IS the highest-effort default
        "reasoning": 2,
        "thinking": 2,
        "adaptive": 2,
        "reasoning-medium": 3,
        "medium": 3,
        "low": 4,
        "reasoning-low": 4,
    },
    "low": {
        "reasoning-low": 0,
        "low": 0,
        "reasoning-medium": 1,
        "medium": 1,
        "reasoning": 2,
        "thinking": 2,
        "adaptive": 2,
        None: 3,
        "high": 4,
        "reasoning-high": 4,
    },
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
    reasoning_effort: str = "medium",
) -> tuple[dict | None, dict | None]:
    """Return ``(off_entry, on_entry)`` for the model rooted at ``base_entry``.

    ``off_entry`` is whichever entry — base or variant — has reasoning OFF.
    ``on_entry`` is the ON entry whose effort variant best matches
    ``reasoning_effort`` (one of ``"low"``, ``"medium"``, ``"high"`` — pass
    the project's ``GeneratorSearchSpace.reasoning_effort``). Either may be
    None when the corresponding mode has no benchmark on AA. Unknown
    effort strings fall back to ``"medium"``.
    """
    base_mode = classify(base_entry, sibling_entries)

    off_entry = base_entry if base_mode == "off" else next((s for s in sibling_entries if classify(s) == "off"), None)

    on_candidates: list[dict] = []
    if base_mode == "on":
        on_candidates.append(base_entry)
    on_candidates.extend(s for s in sibling_entries if classify(s) == "on")

    on_entry = _pick_on(on_candidates, reasoning_effort)
    return off_entry, on_entry


def _pick_on(candidates: list[dict], reasoning_effort: str = "medium") -> dict | None:
    if not candidates:
        return None

    score_map = _ON_EFFORT_SCORES.get(reasoning_effort, _ON_EFFORT_SCORES["medium"])

    def effort_score(entry: dict) -> int:
        vt = (entry.get("variant_type") or "").strip() or None
        return score_map.get(vt, 5)

    return min(candidates, key=effort_score)
