"""Per-category LLM cost ledger.

Each LLM call site routes through ``acompletion_with_cost(cost_category=...)``
and credits the active ledger by category. The orchestrator owns the ledger
lifecycle and prints a breakdown at the end of a run.

The ``rag_eval`` bucket double-tracks the same calls captured in
``ExamResult.total_llm_cost_usd`` (used for the Pareto frontier); the ledger
just lifts those numbers out of the per-trial path so the framework total
also covers exam generation, judging, agent proposals, and graph build.
"""

from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, field


@dataclass
class CostBucket:
    usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0


@dataclass
class CostLedger:
    """Aggregates LLM cost into named buckets across the lifetime of a run."""

    buckets: dict[str, CostBucket] = field(default_factory=dict)

    def record(self, category: str, usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        bucket = self.buckets.setdefault(category, CostBucket())
        bucket.usd += float(usd)
        bucket.prompt_tokens += int(prompt_tokens)
        bucket.completion_tokens += int(completion_tokens)
        bucket.n_calls += 1

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
