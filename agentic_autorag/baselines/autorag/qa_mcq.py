"""Convert our cached MCQ exam to AutoRAG's ``qa.parquet`` schema.

The conversion is a pure transform (no LLM calls). For each ``MCQQuestion``:

- ``qid`` ← ``MCQQuestion.id``
- ``query`` ← question text + the four options inlined. AutoRAG's
  ``prompt_maker.fstring`` only supports ``{query}`` and ``{retrieved_contents}``
  placeholders, so the options have to live inside ``query`` itself for the
  generator to see them.
- ``retrieval_gt`` ← ``[[stem(d) for d in source_doc_ids]]`` — list-of-lists
  per AutoRAG's schema (top-level list is per-question, inner list holds the
  gold doc set; matches our corpus parquet's ``doc_id`` = ``f.stem``).
- ``generation_gt`` ← ``[option_text(correct_answer)]`` — text of the correct
  option; the registered ``mcq_accuracy`` metric scores via normalized
  substring match.
- ``metadata`` ← dict with ``last_modified_datetime`` (AutoRAG schema requirement).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agentic_autorag.config.models import MCQQuestion


def _format_query_with_options(question: str, options: dict[str, str]) -> str:
    """Embed the four MCQ options inline so AutoRAG's generator sees them."""
    options_block = "\n".join(f"{k}. {options[k]}" for k in sorted(options))
    return f"{question}\n\nOptions:\n{options_block}"


def export_mcq_exam_to_parquet(exam_json_path: Path, output_path: Path) -> int:
    """Write ``qa.parquet`` from a cached ``exam.json``. Returns row count."""
    import pandas as pd

    raw = json.loads(exam_json_path.read_text(encoding="utf-8"))
    questions = [MCQQuestion.model_validate(q) for q in raw]
    now = datetime.now()

    rows: list[dict] = []
    for q in questions:
        # Use stem so retrieval_gt aligns with corpus.parquet's doc_id (also stems).
        doc_stems = [Path(doc_id).stem for doc_id in q.source_doc_ids]
        gold_text = q.options[q.correct_answer]
        rows.append(
            {
                "qid": q.id,
                "query": _format_query_with_options(q.question, q.options),
                "retrieval_gt": [doc_stems],
                "generation_gt": [gold_text],
                "metadata": {"last_modified_datetime": now},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False)
    return len(rows)
