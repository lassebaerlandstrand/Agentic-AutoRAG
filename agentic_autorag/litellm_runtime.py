"""LiteLLM runtime configuration and model-alias resolution."""

from __future__ import annotations

import logging
import os
from typing import Any

import litellm

from agentic_autorag.cost_ledger import get_active_ledger

logger = logging.getLogger(__name__)

DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS = 300.0

# Process-wide alias map. Empty = pure passthrough (every model name is sent
# to LiteLLM verbatim). Populated by ``install_model_aliases``.
_MODEL_ALIASES: dict[str, str | dict[str, Any]] = {}


def configure_litellm_runtime(model_aliases: dict[str, Any] | None = None) -> None:
    """Configure LiteLLM logging worker timeout for long-running workloads.

    Also enables ``drop_params`` so provider-specific parameters (e.g. OpenAI's
    ``seed``) are silently dropped for models that don't accept them, instead
    of raising UnsupportedParamsError. The whole point of LiteLLM here is
    cross-provider portability; strict per-param enforcement fights that.

    When ``model_aliases`` is provided, installs the mapping so that subsequent
    ``acompletion_with_cost`` calls resolve alias names to their real targets.
    See ``install_model_aliases`` for the supported value shapes.
    """
    os.environ.setdefault(
        "LOGGING_WORKER_MAX_TIME_PER_COROUTINE",
        str(DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS),
    )
    litellm.drop_params = True
    # Silence LiteLLM's stderr "Provider List" / "Give Feedback" prints. They
    # fire whenever LiteLLM's catalog can't fully identify a model name (e.g.
    # custom Azure deployment names) and add noise without surfacing actionable
    # information — the real errors still raise normally.
    litellm.suppress_debug_info = True
    if model_aliases is not None:
        install_model_aliases(model_aliases)


def install_model_aliases(aliases: dict[str, Any]) -> None:
    """Install (or replace) the process-wide model alias map.

    Each entry maps a short name used in configs and search spaces to the real
    LiteLLM model identifier Agentic AutoRAG should call. Two value shapes are
    supported:

    - **Simple** (``alias: "provider/deployment"``): the alias resolves to a
      single LiteLLM model string.

    - **Extended** (``alias: {"model": "...", "api_base": "...", ...}``): the
      alias resolves to a model string plus extra kwargs (``api_base``,
      ``api_key``, ``api_version``, ...) merged into every call. Useful for
      custom OpenAI-compatible endpoints (vLLM, self-hosted, LiteLLM Proxy)
      where the env var setup wouldn't otherwise route correctly.

    Also calls ``litellm.register_model`` so cost lookups and capability
    detection on the resolved target inherit the alias key's catalog entry
    (when one exists). Aliases whose key is unknown to LiteLLM are left
    unregistered — those calls return zero cost and skip reasoning-aware
    request shaping.
    """
    global _MODEL_ALIASES
    _MODEL_ALIASES = dict(aliases)
    _register_alias_costs()


def _register_alias_costs() -> None:
    for alias, target in _MODEL_ALIASES.items():
        target_model = target if isinstance(target, str) else target.get("model")
        if not target_model or target_model in litellm.model_cost:
            continue
        cost_data = _find_cost_data_for_alias(alias)
        if cost_data is None:
            continue
        entry = dict(cost_data)
        if "/" in target_model:
            entry["litellm_provider"] = target_model.split("/", 1)[0]
        litellm.register_model({target_model: entry})


def _find_cost_data_for_alias(alias: str) -> dict | None:
    if alias in litellm.model_cost:
        return litellm.model_cost[alias]
    if "/" in alias:
        bare = alias.split("/", 1)[1]
        if bare in litellm.model_cost:
            return litellm.model_cost[bare]
    return None


def resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    """Resolve an alias to ``(target_model, extra_kwargs)``.

    Returns ``(model, {})`` unchanged when the name isn't aliased.
    Call-site kwargs always win over the alias's extra kwargs.
    """
    target = _MODEL_ALIASES.get(model)
    if target is None:
        return model, {}
    if isinstance(target, str):
        return target, {}
    extra = dict(target)
    resolved = extra.pop("model")
    return resolved, extra


def _extract_cache_tokens(usage_obj: Any) -> tuple[int, int]:
    """Return ``(cache_read_input_tokens, cache_creation_input_tokens)`` from a LiteLLM usage object.

    LiteLLM normalizes Anthropic-style fields onto the top-level usage object
    (``cache_creation_input_tokens`` / ``cache_read_input_tokens``) and also
    mirrors the read count under ``prompt_tokens_details.cached_tokens`` for
    OpenAI-shape compatibility. OpenAI's implicit prompt cache only exposes
    the latter. We prefer the top-level Anthropic field when present and fall
    back to the OpenAI-shape field for cache reads.
    """
    if usage_obj is None:
        return 0, 0
    cache_creation = int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
    if cache_read == 0:
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            cache_read = int(getattr(details, "cached_tokens", 0) or 0)
    return cache_read, cache_creation


async def acompletion_with_cost(
    *,
    cost_category: str | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, float | int]]:
    """``litellm.acompletion`` wrapper that also returns USD cost and token counts.

    Returns ``(response, {"usd": float, "prompt_tokens": int, "completion_tokens": int,
    "cache_read_input_tokens": int, "cache_creation_input_tokens": int})``. Cost
    falls back to 0.0 when LiteLLM has no pricing for the model (local/self-hosted,
    or an alias whose key is unknown to LiteLLM) or the cost call raises — token
    counts come from the response usage block when available.

    ``litellm.completion_cost`` already accounts for cached/cache-creation tokens
    when present, so ``usd`` is the correctly-discounted billable amount. The
    cache token counts are returned for transparency.

    Resolves the ``model`` kwarg through ``install_model_aliases`` before
    calling LiteLLM, so call sites can use friendly names from the config's
    ``model_aliases`` section.

    When ``cost_category`` is set and a ledger is active (see
    ``agentic_autorag.cost_ledger.set_active_ledger``), credits the call to
    that bucket so the orchestrator can print a per-category breakdown at the
    end of a run.

    Emits a DEBUG log per call with model, tokens, and USD so a user running
    with ``--verbose`` can audit every billable call in ``run.log``.
    """
    original_model = kwargs.pop("model")
    resolved_model, alias_extras = resolve_model(original_model)
    call_kwargs = {**alias_extras, **kwargs, "model": resolved_model}
    response = await litellm.acompletion(**call_kwargs)
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0
    cache_read, cache_creation = _extract_cache_tokens(usage_obj)
    try:
        usd = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        usd = 0.0
        logger.debug("completion_cost failed for model=%s", resolved_model, exc_info=True)
    logger.debug(
        "LLM call: model=%s (alias=%s) category=%s prompt_tokens=%d completion_tokens=%d "
        "cache_read=%d cache_creation=%d cost=$%.6f",
        resolved_model,
        original_model if original_model != resolved_model else "-",
        cost_category or "-",
        prompt_tokens,
        completion_tokens,
        cache_read,
        cache_creation,
        usd,
    )
    if cost_category is not None:
        ledger = get_active_ledger()
        if ledger is not None:
            ledger.record(
                cost_category,
                usd,
                prompt_tokens,
                completion_tokens,
                cache_read,
                cache_creation,
            )
    return response, {
        "usd": usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }
