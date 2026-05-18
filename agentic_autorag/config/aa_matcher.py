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

INTERIOR_NOISE_TOKENS = frozenset({"instruct", "it", "0", "latest"})
TERMINAL_NOISE_TOKENS = frozenset({"chat", "base"})
MODALITY_TOKENS = frozenset({"vl", "vision", "omni", "audio", "multimodal", "mm", "image", "live", "video"})

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

# Path segments that may precede the model name in a LiteLLM id. Includes
# every entry of ``litellm.models_by_provider`` and the upstream-provider /
# org names seen as second-level segments (Replicate's `anthropic/`, HF-style
# `meta-llama/`, fireworks' `accounts/fireworks/models/`). Limiting stripping
# to known prefixes avoids the FLUX bug where `fal_ai/fal-ai/flux-pro/v1.1`
# had its `flux-pro/` chewed off as if it were a provider, leaving only
# `v-1-1` as tokens.
_KNOWN_PREFIXES = frozenset(
    {
        "accounts",
        "ai21",
        "aiml",
        "aisingapore",
        "alibaba",
        "allenai",
        "aleph_alpha",
        "amazon",
        "amazon_nova",
        "anthropic",
        "anyscale",
        "assemblyai",
        "aws_polly",
        "azure",
        "azure_ai",
        "azure_anthropic",
        "azure_text",
        "baai",
        "baidu",
        "baseten",
        "bedrock",
        "bedrock_mantle",
        "black_forest_labs",
        "cerebras",
        "chatgpt",
        "clarifai",
        "cloudflare",
        "codellama",
        "codestral",
        "cohere",
        "cohere_chat",
        "cometapi",
        "dashscope",
        "databricks",
        "datarobot",
        "deepgram",
        "deepinfra",
        "deepseek",
        "elevenlabs",
        "fal_ai",
        "featherless_ai",
        "fireworks",
        "fireworks_ai",
        "friendliai",
        "galadriel",
        "gemini",
        "gigachat",
        "github_copilot",
        "gmi",
        "google",
        "gradient_ai",
        "groq",
        "gryphe",
        "heroku",
        "huggingface",
        "hyperbolic",
        "ibm",
        "ibm_granite",
        "infinity",
        "jina_ai",
        "kwaipilot",
        "lambda_ai",
        "lemonade",
        "llamagate",
        "maritalk",
        "meta",
        "meta_llama",
        "microsoft",
        "minimax",
        "minimaxai",
        "mistral",
        "mistralai",
        "models",
        "moonshot",
        "moonshotai",
        "morph",
        "nebius",
        "nlp_cloud",
        "nousresearch",
        "novita",
        "nscale",
        "nvidia",
        "nvidia_nim",
        "oci",
        "ollama",
        "ollama_chat",
        "openai",
        "openrouter",
        "ovhcloud",
        "palm",
        "perplexity",
        "petals",
        "publicai",
        "qwen",
        "recraft",
        "replicate",
        "runwayml",
        "sambanova",
        "snowflake",
        "stability",
        "text_completion_codestral",
        "text_completion_openai",
        "together_ai",
        "togethercomputer",
        "v0",
        "vercel_ai_gateway",
        "vertex_ai",
        "volcengine",
        "voyage",
        "wandb",
        "watsonx",
        "wizardlm",
        "xai",
        "zai",
    }
)

SUBSET_MIN_OVERLAP = 3
ANCHOR_MIN_LEN = 4

# Bedrock path segments that aren't providers but still need stripping:
# AWS regions (`us-east-1`, `eu-central-1`, `us-gov-east-1`, `*`) and
# commitment tiers (`1-month-commitment`, `6-month-commitment`).
_BEDROCK_PATH_RE = re.compile(r"^(?:[a-z]{2}(?:-[a-z]+)+-\d+|\*|\d+-month-commitment)$")


def _strip_known_prefixes(s: str) -> str:
    """Iteratively strip ``<known-provider>/`` from the head of ``s``.

    A segment is recognised when its underscore-normalised form appears in
    ``_KNOWN_PREFIXES``, or it matches an AWS region / commitment-tier
    pattern. Model-name segments like ``flux-pro/`` are left intact.
    """
    while True:
        idx = s.find("/")
        if idx == -1:
            return s
        segment = s[:idx]
        if segment.replace("-", "_") in _KNOWN_PREFIXES or _BEDROCK_PATH_RE.fullmatch(segment):
            s = s[idx + 1 :]
        else:
            return s


def normalize(name: str) -> str:
    """Reduce a model identifier to a canonical hyphen-separated form.

    Strips provider/region prefixes, date and `:N` version suffixes, replaces
    `.`/`_` with `-`, splits glued family-version tokens (`llama3` → `llama-3`),
    and collapses adjacent duplicate tokens (`qwen-qwen-3` → `qwen-3`).
    """
    s = _strip_known_prefixes(name.lower())
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
        extra = larger - smaller
        if any(tok in MODALITY_TOKENS for tok in extra):
            return None
        # Require an alphabetic anchor of length >= ANCHOR_MIN_LEN in the
        # common tokens. Stops `gpt-5-pro` from claiming AA `gpt-5-4-pro`
        # (overlap is only `{gpt, 5, pro}`, none ≥ 4 chars alphabetic) and
        # rejects the FLUX↔Llama collision when its `flux`/`pro` tokens
        # survive the normalisation fix.
        if not any(tok.isalpha() and len(tok) >= ANCHOR_MIN_LEN for tok in smaller):
            return None
        return 1

    return None


def _has_variant_suffix(slug: str) -> bool:
    return any(slug.endswith(s) for s in VARIANT_SUFFIXES)


# Polarity buckets for variant suffixes. `-non-reasoning*` is "off"; everything
# else in VARIANT_SUFFIXES (`-reasoning`, `-thinking`, `-adaptive`, effort
# levels) is "on"; absence is the base (None).
_OFF_SUFFIXES = ("-non-reasoning-low-effort", "-non-reasoning")
_ON_SUFFIXES = ("-reasoning", "-thinking", "-adaptive", "-low", "-medium", "-high")


def _mode_polarity(slug: str) -> str | None:
    """Classify a slug's variant suffix as ``"on"``, ``"off"``, or ``None`` (base)."""
    if any(slug.endswith(s) for s in _OFF_SUFFIXES):
        return "off"
    if any(slug.endswith(s) for s in _ON_SUFFIXES):
        return "on"
    return None


def _polarity_mismatch(litellm_polarity: str | None, candidate_polarity: str | None) -> int:
    """Rank how compatible a LiteLLM polarity is with an AA candidate's polarity.

    0: identical (both base, both on, or both off) — best.
    1: one side is base and the other a variant — partial match.
    2: opposite polarity (on vs off) — never the right answer when a base
       or same-polarity variant is also available.
    """
    if litellm_polarity == candidate_polarity:
        return 0
    if litellm_polarity is None or candidate_polarity is None:
        return 1
    return 2


def find_best_aa_slug(litellm_id: str, aa_slugs: list[str]) -> str | None:
    """Return the AA slug that best matches ``litellm_id`` (or None).

    Ranking key: (priority desc, polarity-match, token-distance asc,
    aa-larger flag, slug asc). At a tier the polarity check disambiguates
    `-reasoning` vs `-non-reasoning` siblings; token-distance (symmetric
    difference between AA and LiteLLM token multisets) picks the closest
    fit; and when distances tie, the AA candidate that is a *subset* of
    LiteLLM (i.e. LiteLLM is the more specific id, AA names the base
    model) beats one that adds tokens beyond LiteLLM (different / more
    specific AA model).

    Concretely this makes `vertex_ai/gemini-3-pro-preview` route to AA
    `gemini-3-pro` (the base, status-extended) rather than AA
    `gemini-3-1-pro-preview` (a distinct 3.1 model that ties on distance
    but is the wrong family).
    """
    norm_l = normalize(litellm_id)
    t_l = tokens(norm_l)
    if not t_l:
        return None

    litellm_polarity = _mode_polarity(norm_l)
    size_l = _msize(t_l)

    best_slug: str | None = None
    best_key: tuple[int, int, int, int, str] | None = None
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
        mismatch = _polarity_mismatch(litellm_polarity, _mode_polarity(slug))
        size_s = _msize(t_s)
        token_distance = abs(size_s - size_l)
        aa_larger = 1 if size_s > size_l else 0
        key = (-prio, mismatch, token_distance, aa_larger, slug)
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
