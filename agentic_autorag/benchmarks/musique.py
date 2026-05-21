"""MuSiQue-Ans adapter.

Materialises a MuSiQue-Ans validation sample into a corpus directory plus a
``qa.json`` held-out file, ready to feed into ``agentic-autorag optimize`` and
subsequently ``agentic-autorag benchmark-evaluate``.

Each MuSiQue row bundles ~20 paragraphs (same shape as HotpotQA-distractor).
We pool paragraphs across the sampled rows into one shared corpus, dedup by
title, and map the per-paragraph ``is_supporting`` flag to ``supporting_doc_ids``
for retrieval metrics.

Only ``answerable`` rows are kept; the unanswerable contrast set (MuSiQue-Full)
is out of scope.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from agentic_autorag.benchmarks.base import slugify
from agentic_autorag.benchmarks.schema import BenchmarkManifest, BenchmarkQAPair

logger = logging.getLogger(__name__)

_HF_REPO = "dgslibisey/MuSiQue"


class MuSiQueAdapter:
    """Download + convert MuSiQue-Ans for Agentic AutoRAG."""

    name = "musique"
    adapter_version = "v1"

    def prepare(
        self,
        output_dir: Path,
        split: str = "validation",
        sample_size: int | None = 2000,
        seed: int = 42,
        hf_revision: str | None = None,
    ) -> BenchmarkManifest:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        output_dir = Path(output_dir)
        corpus_dir = output_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        resolved_rev = hf_revision
        if resolved_rev is None:
            try:
                resolved_rev = HfApi().dataset_info(_HF_REPO).sha
                logger.warning(
                    "hf_revision not pinned; resolved to %s. Pin this for published results.",
                    resolved_rev,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Could not resolve hf_revision; continuing unpinned.")

        load_kwargs = {"split": split}
        if resolved_rev:
            load_kwargs["revision"] = resolved_rev
        logger.info("Loading MuSiQue split=%s from %s ...", split, _HF_REPO)
        ds = load_dataset(_HF_REPO, **load_kwargs)

        # MuSiQue-Ans = only answerable rows. Filter before sampling so the
        # requested sample_size reflects usable rows, not pre-filter draws.
        rows = [
            r
            for r in tqdm(ds, desc="MuSiQue: filtering answerable rows", unit="row", total=len(ds))
            if r.get("answerable")
        ]
        if sample_size is not None:
            if sample_size > len(rows):
                raise ValueError(
                    f"sample_size={sample_size} exceeds available answerable rows "
                    f"({len(rows)}) in MuSiQue split={split!r}. Lower sample_size or "
                    f"pass None to use all rows."
                )
            if sample_size < len(rows):
                rng = random.Random(seed)
                rows = rng.sample(sorted(rows, key=lambda r: r["id"]), sample_size)
        logger.info("Building corpus index over %d sampled rows ...", len(rows))

        used_slugs: set[str] = set()
        title_to_slug: dict[str, str] = {}
        # Per-title paragraph text (first-seen wins; duplicates logged if different).
        title_to_text: dict[str, str] = {}
        qa_pairs: list[BenchmarkQAPair] = []

        for row in tqdm(rows, desc="MuSiQue: dedup + build qa", unit="row"):
            supporting_titles: list[str] = []
            for para in row["paragraphs"]:
                title = para["title"]
                paragraph = (para["paragraph_text"] or "").strip()
                if not paragraph:
                    continue
                if title not in title_to_slug:
                    title_to_slug[title] = slugify(title, used=used_slugs)
                if title not in title_to_text:
                    title_to_text[title] = paragraph
                elif title_to_text[title] != paragraph:
                    logger.debug(
                        "Title %r appears with different paragraph text; keeping first",
                        title,
                    )
                if para.get("is_supporting") and title not in supporting_titles:
                    supporting_titles.append(title)

            supporting_doc_ids = [title_to_slug[t] for t in supporting_titles if t in title_to_slug]
            gold_answers = [row["answer"], *list(row.get("answer_aliases") or [])]

            qa_pairs.append(
                BenchmarkQAPair(
                    id=row["id"],
                    question=row["question"],
                    gold_answers=gold_answers,
                    supporting_doc_ids=supporting_doc_ids,
                    metadata={
                        "n_hops": len(row.get("question_decomposition") or []),
                    },
                )
            )

        total_chars = 0
        total_words = 0
        for title, text in tqdm(
            title_to_text.items(),
            desc="MuSiQue: writing corpus files",
            unit="file",
            total=len(title_to_text),
        ):
            slug = title_to_slug[title]
            doc_path = corpus_dir / f"{slug}.md"
            body = f"# {title}\n\n{text}\n"
            doc_path.write_text(body, encoding="utf-8")
            total_chars += len(body)
            total_words += len(body.split())

        qa_path = output_dir / "qa.json"
        qa_path.write_text(
            json.dumps([qa.model_dump(mode="json") for qa in qa_pairs], indent=2),
            encoding="utf-8",
        )

        doc_count = len(title_to_text)
        manifest = BenchmarkManifest(
            name=self.name,
            split=split,
            sample_size=len(rows),
            seed=seed,
            adapter_version=self.adapter_version,
            hf_revision=resolved_rev,
            corpus_doc_count=doc_count,
            corpus_total_chars=total_chars,
            corpus_total_words=total_words,
            corpus_avg_words_per_doc=(total_words / doc_count) if doc_count else 0.0,
        )
        (output_dir / "metadata.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "MuSiQue prepared: %d questions, %d corpus docs → %s",
            len(qa_pairs),
            len(title_to_text),
            output_dir,
        )
        return manifest
