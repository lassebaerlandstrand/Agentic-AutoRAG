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
  uv run python scripts/build_knowledge_base.py --refresh-aa-cache  # force re-fetch
  uv run python scripts/build_knowledge_base.py --use-cache-only    # skip API entirely

The AA API response is cached at ``<output-dir>/_aa_response_cache.json``.
Default behaviour: reuse the cache silently if present; only re-fetch when
``--refresh-aa-cache`` is given.

Requires:
  ARTIFICIAL_ANALYSIS_API_KEY environment variable (only when the cache is
  absent or ``--refresh-aa-cache`` is used).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import re
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from agentic_autorag.config.aa_matcher import VARIANT_SUFFIXES, build_aa_to_litellm_mapping

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_BENCHMARKS = ["mmlu_pro", "gpqa", "ifbench", "artificial_analysis_intelligence_index"]
EMBEDDING_TASKS = ["Retrieval", "STS", "Reranking"]
AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
AA_CACHE_FILENAME = "_aa_response_cache.json"


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
    aa_slugs: list[str],
    all_aa_slugs: set[str],
) -> dict[str, tuple[str, str]]:
    """Detect AA variant slugs and link them to their base.

    Runs for every AA slug, not only the unmatched ones — so a variant that
    happened to find LiteLLM matches (e.g. ``grok-4-1-fast-reasoning``) still
    gets ``base_slug`` linkage that ``reasoning_mode.select_pair`` needs.
    The ``base in all_aa_slugs`` guard prevents false positives like
    ``o3-mini-high`` when AA has no ``o3-mini`` base entry.

    Returns a dict of ``{variant_slug: (base_slug, variant_type)}``.
    """
    variants: dict[str, tuple[str, str]] = {}
    for slug in aa_slugs:
        base, vtype = _strip_variant_suffixes(slug)
        if base is not None and base in all_aa_slugs:
            variants[slug] = (base, vtype)
    return variants


def _build_name_mapping(aa_slugs: list[str], litellm_keys: list[str]) -> dict[str, list[str]]:
    """Map each AA slug to matching LiteLLM model keys.

    Delegates to :func:`agentic_autorag.config.aa_matcher.build_aa_to_litellm_mapping`,
    which assigns each LiteLLM key to its single best AA slug using a multi-tier
    priority (exact ≫ token-multiset equality ≫ subset with modality safety).
    """
    return build_aa_to_litellm_mapping(aa_slugs, litellm_keys)


def _fetch_aa_models_remote(api_key: str) -> list[dict]:
    logger.info("Fetching models from Artificial Analysis API…")
    resp = requests.get(AA_API_URL, headers={"x-api-key": api_key, "Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    logger.info("  Retrieved %d models from AA", len(models))
    return models


def _write_aa_cache(cache_path: Path, models: list[dict]) -> None:
    payload = {
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "data": models,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_aa_cache(cache_path: Path) -> tuple[list[dict], str] | None:
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("data", []), payload.get("fetched_at", "unknown")


def _load_aa_models(api_key: str | None, cache_path: Path, *, refresh: bool, cache_only: bool) -> list[dict]:
    """Return AA models, preferring the on-disk cache.

    Order:
      1. ``cache_only=True``  → use cache, fail if absent.
      2. ``refresh=True``     → re-fetch and overwrite cache.
      3. cache present        → use silently, log age.
      4. cache absent         → fetch and write cache.
    """
    cached = _read_aa_cache(cache_path)

    if cache_only:
        if cached is None:
            raise FileNotFoundError(f"--use-cache-only set but no AA cache at {cache_path}")
        models, fetched_at = cached
        logger.info("  Using AA cache (%d models, fetched %s)", len(models), fetched_at)
        return models

    if not refresh and cached is not None:
        models, fetched_at = cached
        logger.info(
            "  Using AA cache (%d models, fetched %s) — pass --refresh-aa-cache to re-fetch", len(models), fetched_at
        )
        return models

    if not api_key:
        raise RuntimeError("ARTIFICIAL_ANALYSIS_API_KEY required to fetch AA data (cache miss or --refresh-aa-cache)")

    models = _fetch_aa_models_remote(api_key)
    _write_aa_cache(cache_path, models)
    logger.info("  Wrote AA cache to %s", cache_path)
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


# Reasonable bounds for hosted-inference prices in USD per 1M tokens.
# Outside this range the entry is unit-corrupted (wandb stores per-1M in the
# per-token field → $15K/M) or it's a self-hosted/local model with $0/M (not
# a useful proxy for hosted inference). Both are rejected so the next
# candidate ID can supply a sane price.
_MIN_PLAUSIBLE_INPUT_PER_1M = 0.001
_MAX_PLAUSIBLE_INPUT_PER_1M = 1000.0

# Route preference when multiple LiteLLM ids back the same AA model. Lower
# wins. Picks first-party vendor pricing over 3rd-party reseller pricing so
# `bedrock/google.gemma-3-27b-it` shows AWS's $0.23/$0.38, not DeepInfra's
# $0.09/$0.16. The bare-id case (no provider prefix) usually denotes the
# canonical AWS Bedrock id (e.g. ``qwen.qwen3-32b-v1:0``).
_DIRECT_VENDOR_PROVIDERS = frozenset({"anthropic", "openai", "gemini", "vertex_ai"})
_BEDROCK_AZURE_PROVIDERS = frozenset(
    {"bedrock", "azure", "azure_ai", "azure_anthropic", "azure_text", "bedrock_mantle", "amazon_nova"}
)
_MANAGED_CLOUD_PROVIDERS = frozenset(
    {"replicate", "openrouter", "perplexity", "github_copilot", "databricks", "watsonx", "snowflake", "wandb"}
)


def _route_priority(litellm_id: str) -> int:
    if "/" not in litellm_id:
        return 0
    provider = litellm_id.split("/", 1)[0]
    if provider in _DIRECT_VENDOR_PROVIDERS:
        return 0
    if provider in _BEDROCK_AZURE_PROVIDERS:
        return 1
    if provider in _MANAGED_CLOUD_PROVIDERS:
        return 2
    return 3


def _get_litellm_pricing(litellm_id: str, costs: dict[str, dict]) -> dict | None:
    entry = costs.get(litellm_id)
    if entry is None and "/" in litellm_id:
        # Bedrock / Azure ids like `bedrock/qwen.qwen3-32b-v1:0` aren't in
        # ``model_cost`` directly — LiteLLM keys them by the inner model id
        # (`qwen.qwen3-32b-v1:0`). Strip one leading prefix segment.
        entry = costs.get(litellm_id.split("/", 1)[1])
    if entry is None:
        return None
    input_cpt = entry.get("input_cost_per_token")
    output_cpt = entry.get("output_cost_per_token")
    if input_cpt is None or output_cpt is None:
        return None
    input_per_1m = round(input_cpt * 1_000_000, 4)
    output_per_1m = round(output_cpt * 1_000_000, 4)
    if not (_MIN_PLAUSIBLE_INPUT_PER_1M <= input_per_1m <= _MAX_PLAUSIBLE_INPUT_PER_1M):
        return None
    if not (0.0 <= output_per_1m <= _MAX_PLAUSIBLE_INPUT_PER_1M):
        return None
    return {
        "input_per_1m_tokens": input_per_1m,
        "output_per_1m_tokens": output_per_1m,
        "max_input_tokens": entry.get("max_input_tokens"),
        "max_output_tokens": entry.get("max_output_tokens"),
    }


def build_llm_knowledge_base(
    output_dir: Path,
    api_key: str | None,
    *,
    refresh_aa_cache: bool = False,
    use_cache_only: bool = False,
) -> None:
    """Fetch AA + LiteLLM data and write knowledge_base/llms.yaml."""
    cache_path = output_dir / AA_CACHE_FILENAME
    aa_models = _load_aa_models(api_key, cache_path, refresh=refresh_aa_cache, cache_only=use_cache_only)
    litellm_costs, all_litellm_ids = _load_litellm_data()

    aa_slugs = [m["slug"] for m in aa_models]
    mapping = _build_name_mapping(aa_slugs, all_litellm_ids)

    matched = sum(1 for v in mapping.values() if v)
    logger.info("  Name mapping: %d/%d AA models matched to LiteLLM keys", matched, len(aa_models))
    unmatched = [s for s, v in mapping.items() if not v]
    if unmatched:
        logger.warning("  Unmatched AA slugs (%d): %s", len(unmatched), unmatched[:20])

    all_aa_slugs = set(aa_slugs)
    variants = _detect_variants(aa_slugs, all_aa_slugs)
    logger.info("  Variant detection: %d slugs linked to base models", len(variants))

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

        # Prefer LiteLLM pricing over AA pricing. Iterate ids in route-priority
        # order so the bedrock/azure/anthropic id wins over a 3rd-party
        # reseller id that happened to come first in the matcher's output.
        pricing: dict | None = None
        for lid in sorted(litellm_ids, key=_route_priority):
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

    # Second pass: variant entries with no LiteLLM ids of their own share the
    # base's price — the user calls the same LiteLLM route for both modes
    # (e.g. `bedrock/qwen.qwen3-32b-v1:0`) so the per-token rate is identical.
    # AA's mode-specific pricing only applies to vendors that charge a
    # reasoning premium, which the LiteLLM source-of-truth does not encode.
    for entry in models_out.values():
        base_slug = entry.get("base_slug")
        if not base_slug or entry.get("litellm_ids"):
            continue
        base_entry = models_out.get(base_slug)
        if base_entry and base_entry.get("pricing"):
            entry["pricing"] = dict(base_entry["pricing"])

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
    parser.add_argument(
        "--refresh-aa-cache",
        action="store_true",
        help="Force re-fetch of the AA API response and overwrite the on-disk cache.",
    )
    parser.add_argument(
        "--use-cache-only",
        action="store_true",
        help="Use the on-disk AA cache only; never hit the API (fails if cache is absent).",
    )
    args = parser.parse_args()

    if args.refresh_aa_cache and args.use_cache_only:
        parser.error("--refresh-aa-cache and --use-cache-only are mutually exclusive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_llm = not args.embedding_only
    build_embed = not args.llm_only

    if build_llm:
        api_key = args.aa_api_key or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        try:
            build_llm_knowledge_base(
                output_dir,
                api_key,
                refresh_aa_cache=args.refresh_aa_cache,
                use_cache_only=args.use_cache_only,
            )
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
