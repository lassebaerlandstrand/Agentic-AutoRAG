"""Export a markdown/txt corpus directory to AutoRAG's ``corpus.parquet``.

AutoRAG's loader expects ``doc_id`` (string), ``contents`` (string), and
``metadata`` (Python dict containing ``last_modified_datetime`` as a
``datetime.datetime`` instance — not a JSON string). We use ``f.stem`` as
``doc_id`` so it matches ``supporting_doc_ids`` in our benchmark ``qa.json``
and our MCQ ``source_doc_ids``, keeping retrieval-recall metrics consistent
across baselines.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_DIRECT_READ_EXTENSIONS = {".md", ".txt"}
_SKIP_FILENAMES = {"metadata.json"}


def export_corpus_to_parquet(corpus_path: Path, output_path: Path) -> int:
    """Write every ``.md``/``.txt`` file under ``corpus_path`` to a parquet.

    Returns the number of documents written.
    """
    import pandas as pd

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

    rows: list[dict] = []
    for path in sorted(corpus_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name in _SKIP_FILENAMES:
            continue
        if path.suffix.lower() not in _DIRECT_READ_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(corpus_path))
        # AutoRAG requires last_modified_datetime to be a datetime instance.
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        rows.append(
            {
                "doc_id": path.stem,
                "contents": text,
                "metadata": {"last_modified_datetime": mtime, "path": rel},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False)
    return len(rows)
