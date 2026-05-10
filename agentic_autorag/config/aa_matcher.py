"""Match Artificial Analysis (AA) slugs to LiteLLM model IDs.

Both naming conventions encode the same model family but differ in:
  - Provider/region prefixes:    `bedrock/global.anthropic.` vs none on AA
  - Doubled maker tokens:        `qwen.qwen3-32b` (Bedrock) vs `qwen3-32b` (AA)
  - Suffix conventions:          `-instruct`, `-it`, `-chat`, `-1:0`, `-20251001`
  - Token-order swaps:           `llama3-3-70b-instruct` vs `llama-3-3-instruct-70b`
  - Family-version glueing:      `llama3` vs `llama-3`, `qwen3` vs `qwen-3`

The matcher canonicalises both sides and compares as token multisets, then ranks
candidates by a priority tier so the most precise match wins.
"""

from __future__ import annotations

import re
from collections import Counter

INTERIOR_NOISE_TOKENS = frozenset({"instruct", "it", "0"})
TERMINAL_NOISE_TOKENS = frozenset({"chat", "base"})
MODALITY_TOKENS = frozenset({"vl", "vision", "omni", "audio", "multimodal", "mm"})

# Suffixes AA uses to mark inference-mode variants of a base model. A slug
# ending with one of these is deprioritised at the same match tier so the
# base wins ties (e.g. a Bedrock id with no `-reasoning` should map to the
# base AA slug, not its `-reasoning` sibling).
VARIANT_SUFFIXES = (
    "-non-reasoning-low-effort",
    "-non-reasoning",
    "-reasoning",
    "-thinking",
    "-adaptive",
    "-low",
    "-medium",
    "-high",
)

_BEDROCK_REGIONS = ("us", "eu", "apac", "global", "jp", "au")
# Bedrock prefixes the model with either a cross-region group (`us.`, `global.`)
# or a specific zone (`us-east-1.`, `eu-north-1.`, `ap-northeast-1.`). Both
# forms are noise for matching.
_BEDROCK_REGION_RE = re.compile(r"^(?:" + "|".join(_BEDROCK_REGIONS) + r"|[a-z]{2}(?:-[a-z]+)+-\d+)\.")
_FAMILY_VERSION_RE = re.compile(r"^([a-z]+)(\d.*)$")
_DATE_SUFFIX_RE = re.compile(r"[-@:]\d{6,}")
_VERSION_SUFFIX_RE = re.compile(r"-v?\d+:\d+$")
_TRAILING_COLON_RE = re.compile(r":\d+$")
# Anchored prefix segment ending in `/`. Applied iteratively so that nested
# routes like `bedrock/us-east-1/1-month-commitment/...` collapse fully.
_PROVIDER_PREFIX_RE = re.compile(r"^[a-z0-9_\-]+/")

SUBSET_MIN_OVERLAP = 3


def normalize(name: str) -> str:
    """Reduce a model identifier to a canonical hyphen-separated form.

    Strips provider/region prefixes, date and `:N` version suffixes, replaces
    `.`/`_` with `-`, splits glued family-version tokens (`llama3` → `llama-3`),
    and collapses adjacent duplicate tokens (`qwen-qwen-3` → `qwen-3`).
    """
    s = name.lower()
    while True:
        new_s, n = _PROVIDER_PREFIX_RE.subn("", s, count=1)
        if n == 0:
            break
        s = new_s
    s = _BEDROCK_REGION_RE.sub("", s)
    s = _DATE_SUFFIX_RE.sub("", s)
    s = _VERSION_SUFFIX_RE.sub("", s)
    s = _TRAILING_COLON_RE.sub("", s)
    s = s.replace(".", "-").replace("_", "-").replace("/", "-")

    parts: list[str] = []
    for p in s.split("-"):
        if not p:
            continue
        m = _FAMILY_VERSION_RE.match(p)
        if m:
            parts.extend([m.group(1), m.group(2)])
        else:
            parts.append(p)

    deduped: list[str] = []
    for p in parts:
        # Collapse only adjacent maker-style duplicates (qwen-qwen, mistral-mistral).
        # Numeric repeats like '3-3' encode meaningful version structure and stay.
        if deduped and deduped[-1] == p and p.isalpha():
            continue
        deduped.append(p)
    return "-".join(deduped)


def _strip_terminal_noise(parts: list[str]) -> list[str]:
    while parts and parts[-1] in TERMINAL_NOISE_TOKENS:
        parts = parts[:-1]
    return parts


def tokens(norm: str) -> Counter[str]:
    """Multiset of tokens after dropping interior + terminal noise."""
    parts = [t for t in norm.split("-") if t]
    parts = _strip_terminal_noise(parts)
    return Counter(t for t in parts if t not in INTERIOR_NOISE_TOKENS)


def _is_submultiset(a: Counter[str], b: Counter[str]) -> bool:
    return all(b.get(k, 0) >= n for k, n in a.items())


def _msize(c: Counter[str]) -> int:
    return sum(c.values())


def match_priority(
    litellm_id: str,
    aa_slug: str,
    *,
    norm_litellm: str | None = None,
    norm_aa: str | None = None,
    tokens_litellm: Counter[str] | None = None,
    tokens_aa: Counter[str] | None = None,
) -> int | None:
    """Return matching priority tier (higher = better) or None.

    Tiers:
      3 — exact normalised equality, OR LiteLLM normalised ends with `-` + AA.
      2 — token multiset equality after noise drop.
      1 — token multiset subset (≥ SUBSET_MIN_OVERLAP overlap) with modality
          safety: skip when AA introduces a `vl`/`vision`/`omni`/... token absent
          on the LiteLLM side.
    """
    norm_l = norm_litellm if norm_litellm is not None else normalize(litellm_id)
    norm_s = norm_aa if norm_aa is not None else normalize(aa_slug)

    if norm_s and (norm_l == norm_s or norm_l.endswith("-" + norm_s)):
        return 3

    t_l = tokens_litellm if tokens_litellm is not None else tokens(norm_l)
    t_s = tokens_aa if tokens_aa is not None else tokens(norm_s)
    if not t_l or not t_s:
        return None

    if t_l == t_s and _msize(t_l) >= 2:
        return 2

    smaller, larger = (t_l, t_s) if _msize(t_l) <= _msize(t_s) else (t_s, t_l)
    if _msize(smaller) >= SUBSET_MIN_OVERLAP and _is_submultiset(smaller, larger):
        if larger is t_s:
            extra = larger - smaller
            if any(tok in MODALITY_TOKENS for tok in extra):
                return None
        return 1

    return None


def _has_variant_suffix(slug: str) -> bool:
    return any(slug.endswith(s) for s in VARIANT_SUFFIXES)


def find_best_aa_slug(litellm_id: str, aa_slugs: list[str]) -> str | None:
    """Return the AA slug that best matches ``litellm_id`` (or None).

    Ranking key: (priority desc, non-variant first, AA token-count desc, slug
    asc). Higher priority dominates; at the same priority, base slugs win
    over `-reasoning`/`-non-reasoning`/etc. siblings — so a Bedrock id with
    no variant suffix in the name maps to the base, not its variant.
    """
    norm_l = normalize(litellm_id)
    t_l = tokens(norm_l)
    if not t_l:
        return None

    best_slug: str | None = None
    best_key: tuple[int, int, int, str] | None = None
    for slug in aa_slugs:
        norm_s = normalize(slug)
        t_s = tokens(norm_s)
        prio = match_priority(
            litellm_id,
            slug,
            norm_litellm=norm_l,
            tokens_litellm=t_l,
            norm_aa=norm_s,
            tokens_aa=t_s,
        )
        if prio is None:
            continue
        spec = _msize(t_s)
        variant_flag = 1 if _has_variant_suffix(slug) else 0
        key = (-prio, variant_flag, -spec, slug)
        if best_key is None or key < best_key:
            best_key = key
            best_slug = slug
    return best_slug


def build_aa_to_litellm_mapping(aa_slugs: list[str], litellm_keys: list[str]) -> dict[str, list[str]]:
    """Assign each LiteLLM key to its single best AA slug, then group by slug.

    Iteration order over `litellm_keys` is preserved within each AA slug's list,
    so output is deterministic for a given input order.
    """
    mapping: dict[str, list[str]] = {slug: [] for slug in aa_slugs}
    for key in litellm_keys:
        slug = find_best_aa_slug(key, aa_slugs)
        if slug is not None:
            mapping[slug].append(key)
    return mapping
