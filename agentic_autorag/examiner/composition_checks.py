"""Deterministic structural check for composed questions.

The composer cites chunks by their position in the neighborhood and
attaches a verbatim span per citation. We verify that every cited
position is in range and that no two citations point at the same span
text — intra-chunk multi-hop (same chunk_id, distinct non-overlapping
spans) is allowed; parallel restatement (same span text, regardless of
chunk) is not. Everything else (load-bearing, indirection, span
integrity, answerability, decomposability) is enforced by the existing
LLM-grade downstream validators (span verifier, oracle gate,
decomposability gate). Brittle upstream heuristics (regex indirection,
token-overlap load-bearing) tend to produce false positives that hurt
more than they help — leave the LLM-grade gates to do their job.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuralCheckResult:
    ok: bool
    reason: str = ""


def _normalize_span(span: str) -> str:
    return " ".join(span.strip().lower().split())


def check_selected_chunk_ids(
    selected_chunk_ids: list[int],
    source_spans: list[str],
    neighborhood_size: int,
) -> StructuralCheckResult:
    """Verify the citation list is structurally valid.

    Rejects:
      - empty selection (the composer must cite at least one span)
      - misaligned ``selected_chunk_ids`` and ``source_spans`` lengths
      - any chunk index outside ``[0, neighborhood_size)``
      - two citations with the same normalized span text (parallel
        restatement is not multi-hop, whether across chunks or within
        the same chunk)

    Duplicate ``chunk_id`` entries are allowed when each points at a
    distinct span — this enables intra-chunk multi-hop (e.g., two
    non-adjacent sentences inside the same chunk).
    """
    if not selected_chunk_ids:
        return StructuralCheckResult(ok=False, reason="empty_selected_chunk_ids")
    if len(selected_chunk_ids) != len(source_spans):
        return StructuralCheckResult(ok=False, reason="spans_misaligned")
    for idx in selected_chunk_ids:
        if not (0 <= idx < neighborhood_size):
            return StructuralCheckResult(ok=False, reason="uncited_chunk")
    normalized = {_normalize_span(s) for s in source_spans}
    if len(normalized) != len(source_spans):
        return StructuralCheckResult(ok=False, reason="duplicate_selected_spans")
    return StructuralCheckResult(ok=True)
