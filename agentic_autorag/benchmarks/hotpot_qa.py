"""HotpotQA (distractor) adapter.

Materialises a HotpotQA-distractor validation sample into a corpus directory
plus a ``qa.json`` held-out file, ready to feed into ``agentic-autorag
optimize`` and subsequently ``agentic-autorag benchmark-evaluate``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from agentic_autorag.benchmarks.base import slugify
from agentic_autorag.benchmarks.schema import BenchmarkManifest, BenchmarkQAPair

logger = logging.getLogger(__name__)

_HF_REPO = "hotpotqa/hotpot_qa"
_CONFIG = "distractor"


class HotpotQAAdapter:
    """Download + convert HotpotQA-distractor for Agentic AutoRAG."""

    name = "hotpot_qa"
    adapter_version = "v1"

    def prepare(
        self,
        output_dir: Path,
        split: str = "validation",
        sample_size: int | None = 500,
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
        ds = load_dataset(_HF_REPO, _CONFIG, **load_kwargs)

        rows = list(ds)
        if sample_size is not None and sample_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(sorted(rows, key=lambda r: r["id"]), sample_size)

        used_slugs: set[str] = set()
        title_to_slug: dict[str, str] = {}
        # Per-title paragraph text (first-seen wins; duplicates logged if different).
        title_to_text: dict[str, str] = {}
        qa_pairs: list[BenchmarkQAPair] = []

        for row in rows:
            ctx_titles: list[str] = row["context"]["title"]
            ctx_sentences: list[list[str]] = row["context"]["sentences"]
            for title, sentences in zip(ctx_titles, ctx_sentences, strict=True):
                if title not in title_to_slug:
                    title_to_slug[title] = slugify(title, used=used_slugs)
                paragraph = "".join(sentences).strip()
                if not paragraph:
                    continue
                if title not in title_to_text:
                    title_to_text[title] = paragraph
                elif title_to_text[title] != paragraph:
                    logger.debug(
                        "Title %r appears twice with different paragraph text; keeping first",
                        title,
                    )

            supporting_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
            supporting_doc_ids = [title_to_slug[t] for t in supporting_titles if t in title_to_slug]

            qa_pairs.append(
                BenchmarkQAPair(
                    id=row["id"],
                    question=row["question"],
                    gold_answers=[row["answer"]],
                    supporting_doc_ids=supporting_doc_ids,
                    metadata={"level": row.get("level", ""), "type": row.get("type", "")},
                )
            )

        for title, text in title_to_text.items():
            slug = title_to_slug[title]
            doc_path = corpus_dir / f"{slug}.md"
            doc_path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")

        qa_path = output_dir / "qa.json"
        qa_path.write_text(
            json.dumps([qa.model_dump(mode="json") for qa in qa_pairs], indent=2),
            encoding="utf-8",
        )

        manifest = BenchmarkManifest(
            name=self.name,
            split=split,
            sample_size=len(rows),
            seed=seed,
            adapter_version=self.adapter_version,
            hf_revision=resolved_rev,
            corpus_doc_count=len(title_to_text),
        )
        (output_dir / "metadata.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "HotpotQA prepared: %d questions, %d corpus docs → %s",
            len(qa_pairs),
            len(title_to_text),
            output_dir,
        )
        return manifest
