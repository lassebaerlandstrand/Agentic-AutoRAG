"""LiteLLM runtime configuration helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import litellm

from agentic_autorag.cost_ledger import get_active_ledger

logger = logging.getLogger(__name__)

DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS = 300.0


def configure_litellm_runtime() -> None:
    """Configure LiteLLM logging worker timeout for long-running workloads.

    Also enables ``drop_params`` so provider-specific parameters (e.g. OpenAI's
    ``seed``) are silently dropped for models that don't accept them, instead
    of raising UnsupportedParamsError. The whole point of LiteLLM here is
    cross-provider portability; strict per-param enforcement fights that.
    """
    os.environ.setdefault(
        "LOGGING_WORKER_MAX_TIME_PER_COROUTINE",
        str(DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS),
    )
    litellm.drop_params = True


async def acompletion_with_cost(
    *,
    cost_category: str | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, float | int]]:
    """``litellm.acompletion`` wrapper that also returns USD cost and token counts.

    Returns ``(response, {"usd": float, "prompt_tokens": int, "completion_tokens": int})``.
    Cost falls back to 0.0 when LiteLLM has no pricing for the model
    (local/self-hosted) or the cost call raises — token counts come from the
    response usage block when available.

    When ``cost_category`` is set and a ledger is active (see
    ``agentic_autorag.cost_ledger.set_active_ledger``), credits the call to
    that bucket so the orchestrator can print a per-category breakdown at the
    end of a run.

    Emits a DEBUG log per call with model, tokens, and USD so a user running
    with ``--verbose`` can audit every billable call in ``run.log``.
    """
    response = await litellm.acompletion(**kwargs)
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0
    try:
        usd = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        usd = 0.0
        logger.debug("completion_cost failed for model=%s", kwargs.get("model"), exc_info=True)
    logger.debug(
        "LLM call: model=%s category=%s prompt_tokens=%d completion_tokens=%d cost=$%.6f",
        kwargs.get("model"),
        cost_category or "-",
        prompt_tokens,
        completion_tokens,
        usd,
    )
    if cost_category is not None:
        ledger = get_active_ledger()
        if ledger is not None:
            ledger.record(cost_category, usd, prompt_tokens, completion_tokens)
    return response, {
        "usd": usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
