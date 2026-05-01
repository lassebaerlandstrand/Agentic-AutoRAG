"""Convert our cached open-ended exam to AutoRAG's ``qa.parquet`` schema.

The conversion is a pure transform (no LLM calls). For each ``OpenEndedQuestion``:

- ``qid`` ← ``OpenEndedQuestion.id``
- ``query`` ← the question text verbatim. Open-ended questions are designed
  to be self-contained, so no inlined options are needed.
- ``retrieval_gt`` ← ``[[stem(d) for d in source_doc_ids]]`` — list-of-lists
  per AutoRAG's schema (top-level list is per-question, inner list holds the
  gold doc set; matches our corpus parquet's ``doc_id`` = ``f.stem``).
- ``generation_gt`` ← ``[canonical_answer, *answer_variants]`` — AutoRAG's
  generation metrics accept multiple gold strings and pick the best match.
- ``metadata`` ← dict with ``last_modified_datetime`` (AutoRAG schema requirement).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agentic_autorag.config.models import OpenEndedQuestion


def export_mcq_exam_to_parquet(exam_json_path: Path, output_path: Path) -> int:
    """Write ``qa.parquet`` from a cached ``exam.json``. Returns row count.

    The function name preserves backward compatibility with existing baseline
    drivers; the underlying exam is now open-ended free-text.
    """
    import pandas as pd

    raw = json.loads(exam_json_path.read_text(encoding="utf-8"))
    questions = [OpenEndedQuestion.model_validate(q) for q in raw]
    now = datetime.now()

    rows: list[dict] = []
    for q in questions:
        # Use stem so retrieval_gt aligns with corpus.parquet's doc_id (also stems).
        doc_stems = [Path(doc_id).stem for doc_id in q.source_doc_ids]
        gold_strings = [q.canonical_answer, *q.answer_variants]
        rows.append(
            {
                "qid": q.id,
                "query": q.question,
                "retrieval_gt": [doc_stems],
                "generation_gt": gold_strings,
                "metadata": {"last_modified_datetime": now},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False)
    return len(rows)
