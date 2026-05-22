"""Tests for the per-category LLM cost ledger.

Covers the ledger's bookkeeping plus the wiring inside ``acompletion_with_cost``
that credits the active ledger when a ``cost_category`` is supplied.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.cost_ledger import (
    CostLedger,
    get_active_ledger,
    reset_active_ledger,
    set_active_ledger,
)
from agentic_autorag.litellm_runtime import acompletion_with_cost


def _mock_response(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    *,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cached_tokens_details: int | None = None,
) -> MagicMock:
    """Build a response mock with explicit cache fields.

    ``MagicMock`` auto-creates attributes on access, so leaving cache fields
    unset would have ``int(mock.cache_read_input_tokens)`` return 1 instead of
    raising — silently inflating ledger counts. Setting them explicitly keeps
    the test honest. ``cached_tokens_details`` opts into the OpenAI-shape
    ``prompt_tokens_details.cached_tokens`` field; ``None`` (default) leaves
    that path inactive.
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.cache_read_input_tokens = cache_read_input_tokens
    response.usage.cache_creation_input_tokens = cache_creation_input_tokens
    if cached_tokens_details is None:
        response.usage.prompt_tokens_details = None
    else:
        details = MagicMock()
        details.cached_tokens = cached_tokens_details
        response.usage.prompt_tokens_details = details
    return response


class TestCostLedger:
    def test_record_creates_bucket_on_first_call(self) -> None:
        ledger = CostLedger()
        ledger.record("rag_eval", usd=0.5, prompt_tokens=10, completion_tokens=5)
        assert "rag_eval" in ledger.buckets
        bucket = ledger.buckets["rag_eval"]
        assert bucket.usd == 0.5
        assert bucket.prompt_tokens == 10
        assert bucket.completion_tokens == 5
        assert bucket.n_calls == 1

    def test_record_accumulates_into_existing_bucket(self) -> None:
        ledger = CostLedger()
        ledger.record("judge", usd=0.1, prompt_tokens=5, completion_tokens=2)
        ledger.record("judge", usd=0.2, prompt_tokens=8, completion_tokens=3)
        bucket = ledger.buckets["judge"]
        assert bucket.usd == pytest.approx(0.3)
        assert bucket.prompt_tokens == 13
        assert bucket.completion_tokens == 5
        assert bucket.n_calls == 2

    def test_total_usd_sums_across_buckets(self) -> None:
        ledger = CostLedger()
        ledger.record("rag_eval", usd=1.0, prompt_tokens=0, completion_tokens=0)
        ledger.record("judge", usd=0.25, prompt_tokens=0, completion_tokens=0)
        ledger.record("agent_proposal", usd=0.5, prompt_tokens=0, completion_tokens=0)
        assert ledger.total_usd() == pytest.approx(1.75)

    def test_to_dict_round_trips_buckets(self) -> None:
        ledger = CostLedger()
        ledger.record("rag_eval", usd=0.5, prompt_tokens=10, completion_tokens=5)
        payload = ledger.to_dict()
        assert payload["total_usd"] == pytest.approx(0.5)
        assert payload["buckets"]["rag_eval"]["n_calls"] == 1
        assert payload["buckets"]["rag_eval"]["usd"] == pytest.approx(0.5)

    def test_record_accumulates_cache_tokens(self) -> None:
        ledger = CostLedger()
        ledger.record(
            "exam_generation",
            usd=0.5,
            prompt_tokens=1000,
            completion_tokens=200,
            cache_read_input_tokens=600,
            cache_creation_input_tokens=400,
        )
        ledger.record(
            "exam_generation",
            usd=0.3,
            prompt_tokens=800,
            completion_tokens=100,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=0,
        )
        bucket = ledger.buckets["exam_generation"]
        assert bucket.cache_read_input_tokens == 900
        assert bucket.cache_creation_input_tokens == 400

    def test_record_cache_tokens_default_to_zero(self) -> None:
        ledger = CostLedger()
        ledger.record("judge", usd=0.1, prompt_tokens=10, completion_tokens=5)
        bucket = ledger.buckets["judge"]
        assert bucket.cache_read_input_tokens == 0
        assert bucket.cache_creation_input_tokens == 0


class TestActiveLedger:
    def test_get_returns_none_when_no_ledger_set(self) -> None:
        assert get_active_ledger() is None

    def test_set_and_reset_roundtrip(self) -> None:
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            assert get_active_ledger() is ledger
        finally:
            reset_active_ledger(token)
        assert get_active_ledger() is None


class TestAcompletionWithCostLedger:
    async def test_credits_active_ledger_when_category_provided(self) -> None:
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            with (
                patch(
                    "agentic_autorag.litellm_runtime.litellm.acompletion",
                    new=AsyncMock(return_value=_mock_response(prompt_tokens=12, completion_tokens=4)),
                ),
                patch(
                    "agentic_autorag.litellm_runtime.litellm.completion_cost",
                    return_value=0.0007,
                ),
            ):
                _, cost = await acompletion_with_cost(
                    cost_category="rag_eval",
                    model="test/model",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            reset_active_ledger(token)

        assert cost["usd"] == pytest.approx(0.0007)
        bucket = ledger.buckets["rag_eval"]
        assert bucket.usd == pytest.approx(0.0007)
        assert bucket.prompt_tokens == 12
        assert bucket.completion_tokens == 4
        assert bucket.n_calls == 1

    async def test_no_credit_when_no_category(self) -> None:
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            with (
                patch(
                    "agentic_autorag.litellm_runtime.litellm.acompletion",
                    new=AsyncMock(return_value=_mock_response()),
                ),
                patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.001),
            ):
                await acompletion_with_cost(
                    model="test/model",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            reset_active_ledger(token)
        assert ledger.buckets == {}

    async def test_no_credit_when_no_active_ledger(self) -> None:
        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new=AsyncMock(return_value=_mock_response()),
            ),
            patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.001),
        ):
            _, cost = await acompletion_with_cost(
                cost_category="rag_eval",
                model="test/model",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert cost["usd"] == pytest.approx(0.001)
        # No assertion on a ledger because there's no active ledger; just
        # confirms the call doesn't raise when no ledger is installed.

    async def test_credits_anthropic_top_level_cache_fields(self) -> None:
        """Anthropic populates ``usage.cache_read_input_tokens`` / ``cache_creation_input_tokens`` directly."""
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            with (
                patch(
                    "agentic_autorag.litellm_runtime.litellm.acompletion",
                    new=AsyncMock(return_value=_mock_response(
                        prompt_tokens=2000,
                        completion_tokens=300,
                        cache_read_input_tokens=1500,
                        cache_creation_input_tokens=200,
                    )),
                ),
                patch(
                    "agentic_autorag.litellm_runtime.litellm.completion_cost",
                    return_value=0.002,
                ),
            ):
                _, cost = await acompletion_with_cost(
                    cost_category="exam_generation",
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            reset_active_ledger(token)
        assert cost["cache_read_input_tokens"] == 1500
        assert cost["cache_creation_input_tokens"] == 200
        bucket = ledger.buckets["exam_generation"]
        assert bucket.cache_read_input_tokens == 1500
        assert bucket.cache_creation_input_tokens == 200

    async def test_credits_openai_implicit_cache_via_prompt_tokens_details(self) -> None:
        """OpenAI exposes its implicit cache only through ``prompt_tokens_details.cached_tokens``."""
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            with (
                patch(
                    "agentic_autorag.litellm_runtime.litellm.acompletion",
                    new=AsyncMock(return_value=_mock_response(
                        prompt_tokens=3000,
                        completion_tokens=100,
                        cached_tokens_details=2200,
                    )),
                ),
                patch(
                    "agentic_autorag.litellm_runtime.litellm.completion_cost",
                    return_value=0.0011,
                ),
            ):
                _, cost = await acompletion_with_cost(
                    cost_category="judge",
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            reset_active_ledger(token)
        assert cost["cache_read_input_tokens"] == 2200
        assert cost["cache_creation_input_tokens"] == 0
        bucket = ledger.buckets["judge"]
        assert bucket.cache_read_input_tokens == 2200
        assert bucket.cache_creation_input_tokens == 0

    async def test_zero_cache_fields_when_provider_silent(self) -> None:
        """When usage has no cache fields, ledger sees zeros (no MagicMock-truthy leak)."""
        ledger = CostLedger()
        token = set_active_ledger(ledger)
        try:
            with (
                patch(
                    "agentic_autorag.litellm_runtime.litellm.acompletion",
                    new=AsyncMock(return_value=_mock_response(prompt_tokens=10, completion_tokens=5)),
                ),
                patch(
                    "agentic_autorag.litellm_runtime.litellm.completion_cost",
                    return_value=0.0001,
                ),
            ):
                _, cost = await acompletion_with_cost(
                    cost_category="rag_eval",
                    model="test/model",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            reset_active_ledger(token)
        assert cost["cache_read_input_tokens"] == 0
        assert cost["cache_creation_input_tokens"] == 0
        bucket = ledger.buckets["rag_eval"]
        assert bucket.cache_read_input_tokens == 0
        assert bucket.cache_creation_input_tokens == 0
