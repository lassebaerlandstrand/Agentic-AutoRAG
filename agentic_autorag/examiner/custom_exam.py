"""Load a pre-built exam from JSON, the one door for every exam producer.

The optimizer runs against a ``list[OpenEndedQuestion]`` regardless of where it
came from — the corpus self-exam generator, the benchmark validation-exam builder, or a
user's hand-written questions. This module is the runtime loader for the latter
two: it parses a JSON file into ``OpenEndedQuestion`` records and never drops a
question. Any grounding tier (spans / doc-ids / bare) is accepted; malformed
records raise so a broken exam fails loud rather than silently shrinking.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.config.models import OpenEndedQuestion


def _parse_record(record: dict) -> OpenEndedQuestion:
    """Parse one record as an ``OpenEndedQuestion``, falling back to a
    ``BenchmarkQAPair`` (doc-level tier-B) when it lacks the exam fields.

    Raises ``ValueError`` with both errors when neither shape validates.
    """
    try:
        return OpenEndedQuestion.model_validate(record)
    except ValidationError as exam_err:
        try:
            return BenchmarkQAPair.model_validate(record).to_open_ended()
        except (ValidationError, ValueError) as qa_err:
            raise ValueError(
                f"record is neither a valid OpenEndedQuestion nor BenchmarkQAPair.\n"
                f"  as OpenEndedQuestion: {exam_err}\n"
                f"  as BenchmarkQAPair:   {qa_err}"
            ) from qa_err


def load_custom_exam(path: Path) -> list[OpenEndedQuestion]:
    """Load a custom exam JSON file into ``OpenEndedQuestion`` records.

    The file is a JSON list of records; each is parsed as an
    ``OpenEndedQuestion`` or, failing that, a ``BenchmarkQAPair`` converted via
    ``to_open_ended`` (doc-level gold, no spans). The loader **never drops** a
    question — ``len(loaded) == len(records)`` always holds — so a real user's
    partly-grounded or bare exam keeps every item. Duplicate ids and malformed
    or non-list JSON raise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"custom exam file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"custom exam file {path} must contain a JSON list, got {type(raw).__name__}")
    if not raw:
        raise ValueError(f"custom exam file {path} is empty")

    exam: list[OpenEndedQuestion] = []
    seen_ids: set[str] = set()
    for i, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"custom exam record {i} in {path} is not an object")
        try:
            question = _parse_record(record)
        except ValueError as exc:
            raise ValueError(f"custom exam record {i} in {path}: {exc}") from exc
        if question.id in seen_ids:
            raise ValueError(f"custom exam record {i} in {path}: duplicate question id {question.id!r}")
        seen_ids.add(question.id)
        exam.append(question)

    if len(exam) != len(raw):  # invariant: the loader never drops
        raise AssertionError(f"loader dropped questions: {len(raw)} in, {len(exam)} out")
    return exam
