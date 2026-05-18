"""Tests for upfront LLM endpoint verification."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from agentic_autorag.optimizer.verify_models import (
    EndpointVerificationError,
    assert_all_ok,
    verify_llm_endpoints,
)


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "cache" / "llm_verification.json"


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_all_ok_writes_cache(mock_litellm, cache_path) -> None:
    mock_litellm.acompletion = AsyncMock(return_value=object())
    results = await verify_llm_endpoints(["a/x", "b/y"], cache_path=cache_path)
    assert all(r.ok for r in results)
    assert mock_litellm.acompletion.await_count == 2
    cache = json.loads(cache_path.read_text())
    assert set(cache.keys()) == {"a/x", "b/y"}
    assert all(cache[m]["ok"] is True for m in cache)


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_cache_hit_skips_ping(mock_litellm, cache_path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).isoformat()
    cache_path.write_text(json.dumps({"cached/model": {"ok": True, "error": "", "checked_at": now}}))
    mock_litellm.acompletion = AsyncMock(return_value=object())

    results = await verify_llm_endpoints(["cached/model"], cache_path=cache_path)

    assert mock_litellm.acompletion.await_count == 0
    assert results[0].from_cache is True
    assert results[0].ok is True


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_stale_cache_triggers_ping(mock_litellm, cache_path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale = (datetime.now(tz=UTC) - timedelta(days=60)).isoformat()
    cache_path.write_text(json.dumps({"stale/model": {"ok": True, "error": "", "checked_at": stale}}))
    mock_litellm.acompletion = AsyncMock(return_value=object())

    results = await verify_llm_endpoints(["stale/model"], cache_path=cache_path, ttl_days=30)

    assert mock_litellm.acompletion.await_count == 1
    assert results[0].from_cache is False


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_force_bypasses_cache(mock_litellm, cache_path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).isoformat()
    cache_path.write_text(json.dumps({"fresh/model": {"ok": True, "error": "", "checked_at": now}}))
    mock_litellm.acompletion = AsyncMock(return_value=object())

    results = await verify_llm_endpoints(["fresh/model"], cache_path=cache_path, force=True)

    assert mock_litellm.acompletion.await_count == 1
    assert results[0].from_cache is False


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_max_tokens_reached_counts_as_reachable(mock_litellm, cache_path) -> None:
    """Reasoning models can blow their output budget on hidden thinking tokens
    and surface as 'Could not finish the message because max_tokens or model
    output limit was reached'. That error proves the endpoint IS reachable —
    the verifier must NOT flag it as a failure."""

    async def truncate(**_kwargs):
        raise RuntimeError(
            "AzureException BadRequestError - Could not finish the message "
            "because max_tokens or model output limit was reached. "
            "Please try again with higher max_tokens."
        )

    mock_litellm.acompletion = truncate
    results = await verify_llm_endpoints(["azure/o4-mini"], cache_path=cache_path)
    assert results[0].ok is True
    # Cache persists the PASS so reruns don't re-ping a known-reachable model.
    cache = json.loads(cache_path.read_text())
    assert cache["azure/o4-mini"]["ok"] is True


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_failure_recorded_and_not_cached(mock_litellm, cache_path) -> None:
    """Failures get reported and trigger EndpointVerificationError, but are
    NOT cached — credentials get fixed all the time and forcing the user to
    bust the cache after every fix is a footgun."""

    async def fail(**_kwargs):
        raise RuntimeError("Operation not allowed")

    mock_litellm.acompletion = fail
    results = await verify_llm_endpoints(["broken/model"], cache_path=cache_path)

    assert results[0].ok is False
    assert "Operation not allowed" in results[0].error
    with pytest.raises(EndpointVerificationError) as exc:
        assert_all_ok(results)
    assert "broken/model" in str(exc.value)
    assert "Operation not allowed" in str(exc.value)
    # Cache must NOT persist the failure.
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        assert "broken/model" not in cache, "Failures must not be cached"


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_stale_cached_failure_is_re_pinged(mock_litellm, cache_path) -> None:
    """A previously-cached failure (from an older code version, or after
    credentials were rotated) must NOT be trusted — the model is re-pinged
    so the user doesn't have to manually invalidate the cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).isoformat()
    cache_path.write_text(
        json.dumps({"recovered/model": {"ok": False, "error": "old failure", "checked_at": now}})
    )
    mock_litellm.acompletion = AsyncMock(return_value=object())

    results = await verify_llm_endpoints(["recovered/model"], cache_path=cache_path)

    assert mock_litellm.acompletion.await_count == 1
    assert results[0].ok is True
    assert results[0].from_cache is False
    # Now-passing model is cached as PASS.
    cache = json.loads(cache_path.read_text())
    assert cache["recovered/model"]["ok"] is True


@patch("agentic_autorag.optimizer.verify_models.litellm")
async def test_logs_one_line_per_model(mock_litellm, cache_path, caplog) -> None:
    mock_litellm.acompletion = AsyncMock(return_value=object())
    with caplog.at_level(logging.INFO, logger="agentic_autorag.optimizer.verify_models"):
        await verify_llm_endpoints(["a/x", "b/y"], cache_path=cache_path)
    messages = [r.getMessage() for r in caplog.records]
    assert any("Verifying search space" in m for m in messages)
    assert any("OK  a/x" in m for m in messages)
    assert any("OK  b/y" in m for m in messages)
    assert any("Search space verified" in m for m in messages)
