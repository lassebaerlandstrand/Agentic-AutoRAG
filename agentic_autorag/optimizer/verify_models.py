"""Upfront LLM endpoint verification.

Pings every LLM in the active search space with a 1-token request before any
trial runs. Caches results to ``~/.cache/agentic-autorag/llm_verification.json``
keyed by model id so repeat runs skip re-pinging. Fails early with a clear
per-model error list when any endpoint is unreachable.

The cache stores ``{model_id: {ok, error, checked_at}}`` and is invalidated
after ``ttl_days`` (default 30) or when ``force`` is True.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import litellm

from agentic_autorag.litellm_runtime import resolve_model

logger = logging.getLogger(__name__)

_VERIFY_TIMEOUT_SECONDS = 20.0
_DEFAULT_TTL_DAYS = 30
_FORCE_VERIFY_ENV = "AGENTIC_AUTORAG_FORCE_VERIFY"
# Reasoning models (Azure o4-mini, gpt-5.x-nano, etc.) spend their output
# budget on hidden thinking tokens before producing visible text. max_tokens=1
# is too tight — Azure raises BadRequestError "Could not finish the message".
# 16 leaves enough headroom for any model to either answer or fail with a
# meaningful auth error, while staying ~free per call.
_VERIFY_MAX_TOKENS = 16
# Provider error fragments that prove the endpoint IS reachable — the call
# made it through auth/routing and only failed at output-budget time. Treat
# these as PASS so reasoning models don't get false-positive-flagged.
_REACHABILITY_PROOF_FRAGMENTS: tuple[str, ...] = (
    "max_tokens or model output limit was reached",
    "max_tokens reached",
    "output limit was reached",
    "max output tokens",
)


class VerificationResult(NamedTuple):
    model: str
    ok: bool
    error: str
    from_cache: bool


def _default_cache_path() -> Path:
    return Path.home() / ".cache" / "agentic-autorag" / "llm_verification.json"


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("verify_models: cache at %s is unreadable; re-verifying", path)
        return {}


def _save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _is_fresh(entry: dict, ttl_days: int) -> bool:
    iso = entry.get("checked_at")
    if not iso:
        return False
    try:
        checked = datetime.fromisoformat(iso)
    except ValueError:
        return False
    return datetime.now(tz=UTC) - checked < timedelta(days=ttl_days)


async def _ping(model: str) -> tuple[bool, str]:
    """Send a tiny completion to ``model``. Return (ok, error_summary).

    Errors whose body proves the request reached the model (e.g. "max_tokens
    reached" on a reasoning model that burned its budget on hidden thinking
    tokens) are treated as PASS — the endpoint is verifiable even if the
    visible output is empty.
    """
    target, extras = resolve_model(model)
    try:
        await asyncio.wait_for(
            litellm.acompletion(
                model=target,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=_VERIFY_MAX_TOKENS,
                **extras,
            ),
            timeout=_VERIFY_TIMEOUT_SECONDS,
        )
        return True, ""
    except Exception as e:  # noqa: BLE001 — any failure is a verification failure
        msg = f"{type(e).__name__}: {e}"
        # Reasoning-model truncation errors prove the endpoint is reachable.
        if any(frag in msg for frag in _REACHABILITY_PROOF_FRAGMENTS):
            return True, ""
        # Trim very long provider error bodies so cache entries stay readable.
        return False, msg[:240]


async def verify_llm_endpoints(
    models: list[str],
    *,
    cache_path: Path | None = None,
    force: bool = False,
    ttl_days: int = _DEFAULT_TTL_DAYS,
    logger_: logging.Logger | None = None,
) -> list[VerificationResult]:
    """Verify every model in ``models``. Cache the outcomes.

    Concurrent pings via ``asyncio.gather`` — each ping has a 20s timeout. The
    cache is keyed by the alias name (the value the search space holds), so
    aliased and non-aliased calls share the same cache entry as long as the
    alias-to-target map is stable.
    """
    log = logger_ or logger
    cache_path = cache_path or _default_cache_path()
    env_force = os.getenv(_FORCE_VERIFY_ENV, "").strip().lower() in ("1", "true", "yes")
    force = force or env_force

    deduped = sorted(set(models))
    cache = _load_cache(cache_path)

    needs_ping: list[str] = []
    results: dict[str, VerificationResult] = {}
    for m in deduped:
        entry = cache.get(m)
        # Only PASS results are cacheable — re-ping anything that was cached
        # as a failure (covers pre-fix cache files that still hold stale
        # failures from before this branch landed).
        if (
            not force
            and isinstance(entry, dict)
            and entry.get("ok") is True
            and _is_fresh(entry, ttl_days)
        ):
            results[m] = VerificationResult(model=m, ok=True, error="", from_cache=True)
        else:
            needs_ping.append(m)

    if needs_ping:
        log.info("Verifying search space: %d LLM endpoint(s) (%d cached)", len(needs_ping), len(results))
        ping_outcomes = await asyncio.gather(*[_ping(m) for m in needs_ping])
        now_iso = datetime.now(tz=UTC).isoformat()
        cache_dirty = False
        for model, (ok, err) in zip(needs_ping, ping_outcomes, strict=True):
            results[model] = VerificationResult(model=model, ok=ok, error=err, from_cache=False)
            # Only cache successes. Failures are often transient (rotated
            # credentials, region access changes) — re-pinging them every run
            # is cheap and avoids the user having to manually invalidate the
            # cache after fixing their env.
            if ok:
                cache[model] = {"ok": True, "error": "", "checked_at": now_iso}
                cache_dirty = True
        if cache_dirty:
            _save_cache(cache_path, cache)
    else:
        log.info("Verifying search space: %d LLM endpoint(s) (all cached)", len(deduped))

    ordered = [results[m] for m in deduped]
    for r in ordered:
        suffix = " (cached)" if r.from_cache else ""
        if r.ok:
            log.info("  OK  %s%s", r.model, suffix)
        else:
            log.error("  FAIL %s%s — %s", r.model, suffix, r.error)

    n_new = sum(1 for r in ordered if not r.from_cache)
    n_cached = len(ordered) - n_new
    n_failed = sum(1 for r in ordered if not r.ok)
    if n_failed == 0:
        log.info("Search space verified (%d new, %d cached).", n_new, n_cached)
    return ordered


class EndpointVerificationError(RuntimeError):
    """Raised when one or more LLMs in the search space failed verification."""

    def __init__(self, failures: list[VerificationResult]) -> None:
        lines = ["One or more LLM endpoints failed pre-run verification:"]
        for r in failures:
            lines.append(f"  - {r.model}: {r.error}")
        lines.append("")
        lines.append(
            "Remove the failing models from the relevant stage pool "
            "(search_space.generator.models / .passage_compressor.models / "
            ".query_expansion.models), or re-run "
            "with --force-verify (or AGENTIC_AUTORAG_FORCE_VERIFY=1) once the "
            "endpoint is fixed."
        )
        self.failures = failures
        super().__init__("\n".join(lines))


def assert_all_ok(results: list[VerificationResult]) -> None:
    """Raise EndpointVerificationError if any result is a failure."""
    failures = [r for r in results if not r.ok]
    if failures:
        raise EndpointVerificationError(failures)


def invalidate(model: str, cache_path: Path | None = None) -> None:
    """Drop a single model from the cache so the next run re-pings it.

    Useful when a previously-cached PASS turns out to be stale (provider
    changed). Not currently called automatically; exposed for ops use.
    """
    cache_path = cache_path or _default_cache_path()
    cache = _load_cache(cache_path)
    if cache.pop(model, None) is not None:
        _save_cache(cache_path, cache)


def time_now_seconds() -> float:
    """Wall-clock helper, kept here so tests can monkeypatch without touching
    the stdlib ``time`` module globally."""
    return time.time()
