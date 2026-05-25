"""Deterministic structural check for composed questions.

The composer cites chunks by their position in the neighborhood. We
verify that every cited position is in range; everything else
(load-bearing, indirection, span integrity, answerability,
decomposability) is enforced by the existing LLM-grade downstream
validators (span verifier, oracle gate, decomposability gate).
Brittle upstream heuristics (regex indirection, token-overlap
load-bearing) tend to produce false positives that hurt more than they
help — leave the LLM-grade gates to do their job.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuralCheckResult:
    ok: bool
    reason: str = ""


def check_selected_chunk_ids(selected_chunk_ids: list[int], neighborhood_size: int) -> StructuralCheckResult:
    """Verify every cited position is a valid index into the neighborhood.

    Rejects:
      - empty selection (the composer must cite at least one chunk)
      - any index outside ``[0, neighborhood_size)``
      - duplicate indices (the composer cited the same chunk twice)
    """
    if not selected_chunk_ids:
        return StructuralCheckResult(ok=False, reason="empty_selected_chunk_ids")
    if len(set(selected_chunk_ids)) != len(selected_chunk_ids):
        return StructuralCheckResult(ok=False, reason="duplicate_selected_chunk_ids")
    for idx in selected_chunk_ids:
        if not (0 <= idx < neighborhood_size):
            return StructuralCheckResult(ok=False, reason="uncited_chunk")
    return StructuralCheckResult(ok=True)
