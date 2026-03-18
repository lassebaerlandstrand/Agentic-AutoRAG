"""Build the knowledge base YAML files from external data sources.

Sources:
  - Artificial Analysis API  → LLM benchmarks + throughput
  - LiteLLM model_cost dict  → LLM pricing + context limits (preferred over AA pricing)
  - MTEB benchmark cache     → Embedding model benchmarks

Usage:
  uv run python scripts/build_knowledge_base.py
  uv run python scripts/build_knowledge_base.py --llm-only
  uv run python scripts/build_knowledge_base.py --embedding-only
  uv run python scripts/build_knowledge_base.py --output-dir knowledge_base/

Requires:
  ARTIFICIAL_ANALYSIS_API_KEY environment variable (for LLM knowledge base).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import re
import sys
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_BENCHMARKS = ["mmlu_pro", "gpqa", "ifbench", "artificial_analysis_intelligence_index"]
EMBEDDING_TASKS = ["Retrieval", "STS", "Reranking"]
AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# Ordered longest-first so greedy matching works correctly.
VARIANT_SUFFIXES = [
    "-non-reasoning-low-effort",
    "-non-reasoning",
    "-reasoning",
    "-adaptive",
    "-thinking",
    "-low",
    "-medium",
    "-high",
]


def _normalize(name: str) -> str:
    """Reduce a model name to a canonical form for matching across naming conventions."""
    s = name.lower()
    # Strip provider prefix first so the region.provider. pattern becomes visible
    s = re.sub(r"^[a-z0-9_\-]+/", "", s)
    # Strip Bedrock cross-region + provider prefix (us.anthropic., eu.amazon., etc.)
    s = re.sub(r"^(us|eu|apac|global|jp|au)\.[a-z0-9]+\.", "", s)
    # Strip date-like suffixes: -20241022, @20251001
    s = re.sub(r"[-@:]\d{6,}", "", s)
    # Strip version suffixes: -v1:0 (with 'v') or -1:0 (bare, e.g. gpt-oss-20b-1:0)
    s = re.sub(r"-v?\d+:\d+$", "", s)
    # Strip any remaining bare :N suffix
    s = re.sub(r":\d+$", "", s)
    s = s.replace(".", "-").replace("_", "-")
    return s


def _sig_tokens(norm: str) -> frozenset[str]:
    """Token set of a normalised name, filtering the '0' minor-version token.

    This allows 'nova-2-lite' and 'nova-2-0-lite' to match (same tokens after
    dropping '0') without creating false positives like 'flash' ⊆ 'flash-lite'.
    """
    return frozenset(t for t in norm.split("-") if t != "0")


def _strip_variant_suffixes(slug: str) -> tuple[str, str] | tuple[None, None]:
    """Recursively strip known AA variant suffixes from a slug.

    Returns ``(base_slug, variant_type)`` where *variant_type* is a ``-``
    joined string of all stripped suffixes (e.g. ``"reasoning-low"`` for
    ``nova-2-0-lite-reasoning-low``).  Returns ``(None, None)`` when no
    suffix could be removed.
    """
    stripped_parts: list[str] = []
    current = slug
    while True:
        matched = False
        for suffix in VARIANT_SUFFIXES:
            if current.endswith(suffix):
                # suffix includes the leading '-', strip it for the type label
                stripped_parts.append(suffix.lstrip("-"))
                current = current[: -len(suffix)]
                matched = True
                break
        if not matched:
            break
    if not stripped_parts:
        return None, None
    # Parts were collected inner→outer; reverse for natural reading order
    variant_type = "-".join(reversed(stripped_parts))
    return current, variant_type


def _detect_variants(
    mapping: dict[str, list[str]],
    all_aa_slugs: set[str],
) -> dict[str, tuple[str, str]]:
    """Detect AA variant slugs and link them to their base.

    Only considers slugs that got zero LiteLLM matches in the two-pass
    algorithm (protects real models like ``o3-mini-high``).

    Returns a dict of ``{variant_slug: (base_slug, variant_type)}``.
    """
    matched_slugs = {slug for slug, ids in mapping.items() if ids}
    variants: dict[str, tuple[str, str]] = {}

    for slug in mapping:
        if slug in matched_slugs:
            continue
        base, vtype = _strip_variant_suffixes(slug)
        if base is not None and base in all_aa_slugs:
            variants[slug] = (base, vtype)

    return variants


def _build_name_mapping(aa_slugs: list[str], litellm_keys: list[str]) -> dict[str, list[str]]:
    """Map each AA slug to matching LiteLLM model keys.

    Uses two passes:
    1. Exact or suffix match on normalised names (precise).
    2. Token-set equality after filtering the '0' minor-version token. This handles
       word-order differences ('claude-haiku-4-5' == 'claude-4-5-haiku') and minor
       versioning gaps ('nova-2-lite' == 'nova-2-0-lite') without false positives
       like 'gemini-flash' matching 'gemini-flash-lite'.
    """
    norm_to_litellm: dict[str, list[str]] = {}
    for key in litellm_keys:
        norm = _normalize(key)
        norm_to_litellm.setdefault(norm, []).append(key)

    mapping: dict[str, list[str]] = {}
    for slug in aa_slugs:
        norm_slug = _normalize(slug)
        sig_slug = _sig_tokens(norm_slug)
        matches: list[str] = []
        seen: set[str] = set()
        for norm_key, originals in norm_to_litellm.items():
            exact = norm_slug == norm_key or norm_key.endswith(norm_slug)
            token_match = len(sig_slug) >= 2 and sig_slug == _sig_tokens(norm_key)
            if exact or token_match:
                for orig in originals:
                    if orig not in seen:
                        matches.append(orig)
                        seen.add(orig)
        mapping[slug] = matches

    return mapping


def _fetch_aa_models(api_key: str) -> list[dict]:
    logger.info("Fetching models from Artificial Analysis API…")
    resp = requests.get(AA_API_URL, headers={"x-api-key": api_key, "Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    logger.info("  Retrieved %d models from AA", len(models))
    return models


def _load_litellm_data() -> tuple[dict[str, dict], list[str]]:
    """Return (model_cost_dict, all_valid_litellm_ids).

    all_valid_litellm_ids combines:
    - ``litellm.model_cost`` keys (have pricing data)
    - ``litellm.models_by_provider`` entries (all supported IDs, including those
      like ``vertex_ai/gemini-2.5-flash`` that are not in model_cost)
    """
    import litellm  # noqa: PLC0415

    costs: dict[str, dict] = litellm.model_cost  # type: ignore[attr-defined]
    all_ids: set[str] = set(costs.keys())

    for provider, models in litellm.models_by_provider.items():
        for model_name in models:
            # Avoid double-prefixing entries that already carry '{provider}/'
            full_id = model_name if model_name.startswith(f"{provider}/") else f"{provider}/{model_name}"
            all_ids.add(full_id)

    logger.info(
        "  Loaded %d LiteLLM IDs (%d from model_cost, %d from provider listings)",
        len(all_ids),
        len(costs),
        len(all_ids) - len(costs),
    )
    return costs, list(all_ids)


def _get_litellm_pricing(litellm_id: str, costs: dict[str, dict]) -> dict | None:
    entry = costs.get(litellm_id)
    if entry is None:
        return None
    input_cpt = entry.get("input_cost_per_token")
    output_cpt = entry.get("output_cost_per_token")
    if input_cpt is None or output_cpt is None:
        return None
    return {
        "input_per_1m_tokens": round(input_cpt * 1_000_000, 4),
        "output_per_1m_tokens": round(output_cpt * 1_000_000, 4),
        "max_input_tokens": entry.get("max_input_tokens"),
        "max_output_tokens": entry.get("max_output_tokens"),
    }


def build_llm_knowledge_base(output_dir: Path, api_key: str) -> None:
    """Fetch AA + LiteLLM data and write knowledge_base/llms.yaml."""
    aa_models = _fetch_aa_models(api_key)
    litellm_costs, all_litellm_ids = _load_litellm_data()

    aa_slugs = [m["slug"] for m in aa_models]
    mapping = _build_name_mapping(aa_slugs, all_litellm_ids)

    matched = sum(1 for v in mapping.values() if v)
    logger.info("  Name mapping: %d/%d AA models matched to LiteLLM keys", matched, len(aa_models))
    unmatched = [s for s, v in mapping.items() if not v]
    if unmatched:
        logger.warning("  Unmatched AA slugs (%d): %s", len(unmatched), unmatched[:20])

    # Pass 3: detect variant slugs and link to their base
    all_aa_slugs = set(aa_slugs)
    variants = _detect_variants(mapping, all_aa_slugs)
    logger.info("  Variant detection: %d variant slugs linked to base models", len(variants))

    models_out: dict[str, dict] = {}
    for aa in aa_models:
        slug = aa["slug"]
        litellm_ids = mapping.get(slug, [])

        evals = aa.get("evaluations") or {}
        benchmarks: dict[str, float | None] = {}
        for b in LLM_BENCHMARKS:
            val = evals.get(b)
            benchmarks[b] = round(val, 4) if val is not None else None

        perf: dict[str, float | None] = {
            "median_output_tokens_per_second": aa.get("median_output_tokens_per_second"),
            "median_time_to_first_token_seconds": aa.get("median_time_to_first_token_seconds"),
        }

        # Prefer LiteLLM pricing over AA pricing
        pricing: dict | None = None
        for lid in litellm_ids:
            pricing = _get_litellm_pricing(lid, litellm_costs)
            if pricing:
                break

        if pricing is None:
            aa_pricing = aa.get("pricing") or {}
            input_1m = aa_pricing.get("price_1m_input_tokens")
            output_1m = aa_pricing.get("price_1m_output_tokens")
            if input_1m is not None:
                pricing = {
                    "input_per_1m_tokens": input_1m,
                    "output_per_1m_tokens": output_1m,
                    "max_input_tokens": None,
                    "max_output_tokens": None,
                    "source": "artificial_analysis",
                }

        creator = aa.get("model_creator") or {}
        entry: dict = {
            "name": aa["name"],
            "slug": slug,
            "creator": creator.get("name", ""),
            "release_date": aa.get("release_date"),
            "litellm_ids": litellm_ids,
            "benchmarks": benchmarks,
            "performance": perf,
            "pricing": pricing,
        }

        if slug in variants:
            base_slug, variant_type = variants[slug]
            entry["base_slug"] = base_slug
            entry["variant_type"] = variant_type

        models_out[slug] = entry

    output = {
        "_metadata": {
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "aa_model_count": len(aa_models),
            "litellm_key_count": len(all_litellm_ids),
            "matched_count": matched,
        },
        "models": models_out,
    }

    out_path = output_dir / "llms.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    logger.info("Wrote %s (%d models)", out_path, len(models_out))


def build_embedding_knowledge_base(output_dir: Path) -> None:
    """Fetch MTEB results and write knowledge_base/embeddings.yaml."""
    import mteb  # noqa: PLC0415

    logger.info("Loading MTEB benchmark results…")
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    cache = mteb.ResultCache()
    logger.info("  Downloading results from remote (this may take a few minutes)…")
    cache.download_from_remote()
    results = cache.load_results(tasks=benchmark)
    df = results.get_benchmark_result()

    logger.info("  Loaded results for %d models", len(df))

    models_out: dict[str, dict] = {}
    for _, row in df.iterrows():
        raw_model = str(row.get("Model", ""))
        url_match = re.search(r"\(https://huggingface\.co/([^)]+)\)", raw_model)
        hf_id = url_match.group(1) if url_match else re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_model).strip()

        if not hf_id:
            continue

        def _to_float(val: object) -> float | None:
            try:
                f = float(val) if val is not None else None  # type: ignore[arg-type]
                return None if f is not None and math.isnan(f) else f
            except (ValueError, TypeError):
                return None

        params_b = _to_float(row.get("Number of Parameters (B)"))
        memory_mb = _to_float(row.get("Memory Usage (MB)"))
        dim_raw = _to_float(row.get("Embedding Dimensions"))
        dimensions = int(dim_raw) if dim_raw is not None and not math.isnan(dim_raw) else None
        tok_raw = _to_float(row.get("Max Tokens"))
        max_tokens = int(tok_raw) if tok_raw is not None and not math.isnan(tok_raw) else None

        scores: dict[str, float | None] = {}
        for task in EMBEDDING_TASKS:
            val = _to_float(row.get(task))
            scores[task.lower()] = round(val, 4) if val is not None else None

        models_out[hf_id] = {
            "hf_id": hf_id,
            "parameters_billions": params_b,
            "memory_usage_mb": memory_mb,
            "embedding_dimensions": dimensions,
            "max_tokens": max_tokens,
            "scores": scores,
        }

    output = {
        "_metadata": {
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "benchmark": "MTEB(eng, v2)",
            "model_count": len(models_out),
            "tasks_included": EMBEDDING_TASKS,
        },
        "models": models_out,
    }

    out_path = output_dir / "embeddings.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    logger.info("Wrote %s (%d models)", out_path, len(models_out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Agentic AutoRAG knowledge base YAML files.")
    parser.add_argument("--output-dir", default="knowledge_base", help="Directory to write YAML files")
    parser.add_argument("--llm-only", action="store_true", help="Only build llms.yaml")
    parser.add_argument("--embedding-only", action="store_true", help="Only build embeddings.yaml")
    parser.add_argument("--aa-api-key", default=None, help="Artificial Analysis API key (overrides env var)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_llm = not args.embedding_only
    build_embed = not args.llm_only

    if build_llm:
        api_key = args.aa_api_key or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        if not api_key:
            logger.error(
                "ARTIFICIAL_ANALYSIS_API_KEY not set. Set the env var or use --aa-api-key. Skipping LLM knowledge base."
            )
            if args.llm_only:
                sys.exit(1)
        else:
            try:
                build_llm_knowledge_base(output_dir, api_key)
            except Exception as e:
                logger.error("Failed to build LLM knowledge base: %s", e)
                if args.llm_only:
                    raise

    if build_embed:
        try:
            build_embedding_knowledge_base(output_dir)
        except Exception as e:
            logger.error("Failed to build embedding knowledge base: %s", e)
            if args.embedding_only:
                raise

    logger.info(
        "Done. Static files (rerankers.yaml, parameter_descriptions.yaml) are hand-authored — no rebuild needed."
    )


if __name__ == "__main__":
    main()
