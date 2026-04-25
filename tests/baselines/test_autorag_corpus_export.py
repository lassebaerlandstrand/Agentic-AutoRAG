"""Test ``corpus_export.export_corpus_to_parquet``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from agentic_autorag.baselines.autorag.corpus_export import export_corpus_to_parquet


def test_doc_id_is_stem_no_extension(tmp_path: Path) -> None:
    """``doc_id`` matches ``f.stem`` so ``retrieval_gt`` aligns across runs."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("# Alpha\nAlpha body.")
    (corpus / "beta.txt").write_text("Beta body.")

    out = tmp_path / "corpus.parquet"
    n = export_corpus_to_parquet(corpus, out)
    assert n == 2

    df = pd.read_parquet(out)
    doc_ids = sorted(df["doc_id"].tolist())
    assert doc_ids == ["alpha", "beta"]
    contents = dict(zip(df["doc_id"], df["contents"], strict=True))
    assert "Alpha body." in contents["alpha"]
    assert "Beta body." in contents["beta"]


def test_skips_metadata_and_hidden(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "real.txt").write_text("real")
    (corpus / "metadata.json").write_text("{}")
    (corpus / ".hidden").write_text("h")
    (corpus / "ignored.pdf").write_text("not md/txt")

    out = tmp_path / "corpus.parquet"
    n = export_corpus_to_parquet(corpus, out)
    assert n == 1
    df = pd.read_parquet(out)
    assert df["doc_id"].tolist() == ["real"]


def test_metadata_is_dict_with_required_fields(tmp_path: Path) -> None:
    """AutoRAG's loader requires ``metadata`` as a dict with a ``last_modified_datetime`` key."""
    corpus = tmp_path / "corpus"
    (corpus / "subdir").mkdir(parents=True)
    (corpus / "subdir" / "doc.md").write_text("Body.")

    out = tmp_path / "corpus.parquet"
    n = export_corpus_to_parquet(corpus, out)
    assert n == 1
    df = pd.read_parquet(out)
    metadata = df["metadata"].iloc[0]
    # pyarrow round-trips a struct column as a dict (not a JSON string).
    assert isinstance(metadata, dict)
    assert metadata["path"] == "subdir/doc.md"
    assert isinstance(metadata["last_modified_datetime"], datetime)


def test_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_corpus_to_parquet(tmp_path / "nope", tmp_path / "out.parquet")
