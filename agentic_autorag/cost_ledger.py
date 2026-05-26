"""Per-category LLM cost ledger.

Each LLM call site routes through ``acompletion_with_cost(cost_category=...)``
and credits the active ledger by category. The orchestrator owns the ledger
lifecycle and prints a breakdown at the end of a run.

The ``rag_eval`` bucket double-tracks the same calls captured in
``ExamResult.total_llm_cost_usd`` (used for the Pareto frontier); the ledger
just lifts those numbers out of the per-trial path so the run total also
covers exam generation, judging, agent proposals, and graph build.
"""

from __future__ import annotations

import contextvars
from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass
class CostBucket:
    usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cache token sub-totals. ``prompt_tokens`` already includes these — LiteLLM
    # normalizes cache_creation + cache_read into ``usage.prompt_tokens`` for
    # Anthropic, and OpenAI's prompt_tokens always include cached_tokens. They
    # exist here for transparency, not as additive components of ``usd``.
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # Embedding-model input tokens credited to this bucket. Counted via the
    # embedder's own tokenizer at index-build time; the ``embedding_build``
    # bucket is the canonical place for these. LLM buckets leave this at 0.
    embedding_input_tokens: int = 0
    n_calls: int = 0


@dataclass
class CostLedger:
    """Aggregates LLM cost into named buckets across the lifetime of a run."""

    buckets: dict[str, CostBucket] = field(default_factory=dict)

    def record(
        self,
        category: str,
        usd: float,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        embedding_input_tokens: int = 0,
    ) -> None:
        bucket = self.buckets.setdefault(category, CostBucket())
        bucket.usd += float(usd)
        bucket.prompt_tokens += int(prompt_tokens)
        bucket.completion_tokens += int(completion_tokens)
        bucket.cache_read_input_tokens += int(cache_read_input_tokens)
        bucket.cache_creation_input_tokens += int(cache_creation_input_tokens)
        bucket.embedding_input_tokens += int(embedding_input_tokens)
        bucket.n_calls += 1

    def snapshot(self) -> dict[str, CostBucket]:
        """Return a deep copy of buckets for per-trial delta computation.

        Used by the orchestrator to capture the ledger state at trial start
        and end; the diff is written to ``trial_cost_ledger.jsonl``.
        """
        return deepcopy(self.buckets)

    def delta_since(self, before: dict[str, CostBucket]) -> dict[str, dict[str, float | int]]:
        """Return per-bucket field deltas since the given snapshot.

        Buckets that existed in ``before`` but were not touched since the
        snapshot produce an all-zero delta entry; brand-new buckets get a
        full delta. Used to write per-trial lines to ``trial_cost_ledger.jsonl``.
        """
        names = set(self.buckets) | set(before)
        out: dict[str, dict[str, float | int]] = {}
        for name in names:
            after = self.buckets.get(name, CostBucket())
            base = before.get(name, CostBucket())
            out[name] = {
                "usd": after.usd - base.usd,
                "prompt_tokens": after.prompt_tokens - base.prompt_tokens,
                "completion_tokens": after.completion_tokens - base.completion_tokens,
                "cache_read_input_tokens": after.cache_read_input_tokens - base.cache_read_input_tokens,
                "cache_creation_input_tokens": after.cache_creation_input_tokens - base.cache_creation_input_tokens,
                "embedding_input_tokens": after.embedding_input_tokens - base.embedding_input_tokens,
                "n_calls": after.n_calls - base.n_calls,
            }
        return out

    def total_usd(self) -> float:
        return sum(b.usd for b in self.buckets.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "total_usd": self.total_usd(),
            "buckets": {name: asdict(bucket) for name, bucket in self.buckets.items()},
        }


_current_ledger: contextvars.ContextVar[CostLedger | None] = contextvars.ContextVar(
    "agentic_autorag_cost_ledger",
    default=None,
)


def set_active_ledger(ledger: CostLedger | None) -> contextvars.Token:
    """Install ``ledger`` as the active ledger; returns a token for ``reset_active_ledger``."""
    return _current_ledger.set(ledger)


def reset_active_ledger(token: contextvars.Token) -> None:
    _current_ledger.reset(token)


def get_active_ledger() -> CostLedger | None:
    return _current_ledger.get()
