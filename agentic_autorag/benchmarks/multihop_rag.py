"""MultiHop-RAG adapter — materialises the news corpus + multi-hop QA into
a prepared corpus directory plus ``qa.json``. Article identity is by title;
collisions resolved by sha-1-suffix ``slugify``."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from agentic_autorag.benchmarks.base import slugify
from agentic_autorag.benchmarks.schema import BenchmarkManifest, BenchmarkQAPair

logger = logging.getLogger(__name__)

_HF_REPO = "yixuantt/MultiHopRAG"
# yixuantt/MultiHopRAG ships two HF *configs* (not splits): ``corpus`` holds the
# news articles, ``MultiHopRAG`` the multi-hop QA. Each config's data lives in
# the ``train`` split. We treat the public ``split`` parameter as the QA config
# selector so the adapter signature stays uniform with hotpot/musique.
_CORPUS_CONFIG = "corpus"
_QA_CONFIG = "MultiHopRAG"


class MultiHopRAGAdapter:
    """Download + convert MultiHop-RAG for Agentic AutoRAG."""

    name = "multihop_rag"
    adapter_version = "v1"

    def prepare(
        self,
        output_dir: Path,
        split: str = _QA_CONFIG,
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

        common_kwargs = {"revision": resolved_rev} if resolved_rev else {}

        logger.info("Loading MultiHop-RAG corpus config=%s ...", _CORPUS_CONFIG)
        corpus_ds = load_dataset(_HF_REPO, _CORPUS_CONFIG, split="train", **common_kwargs)

        used_slugs: set[str] = set()
        title_to_slug: dict[str, str] = {}
        title_to_text: dict[str, str] = {}

        for row in tqdm(corpus_ds, desc="MultiHop-RAG: indexing corpus", unit="article", total=len(corpus_ds)):
            title = (row.get("title") or "").strip()
            body = (row.get("body") or "").strip()
            if not title or not body:
                continue
            if title not in title_to_slug:
                title_to_slug[title] = slugify(title, used=used_slugs)
            if title not in title_to_text:
                title_to_text[title] = body

        qa_config = split
        logger.info("Loading MultiHop-RAG QA config=%s ...", qa_config)
        qa_ds = load_dataset(_HF_REPO, qa_config, split="train", **common_kwargs)
        rows = list(qa_ds)
        if sample_size is not None:
            if sample_size > len(rows):
                raise ValueError(
                    f"sample_size={sample_size} exceeds available rows ({len(rows)}) "
                    f"in MultiHop-RAG split={split!r}. Lower sample_size or pass None."
                )
            if sample_size < len(rows):
                rng = random.Random(seed)
                rows = rng.sample(sorted(rows, key=lambda r: r.get("query", "")), sample_size)

        qa_pairs: list[BenchmarkQAPair] = []
        for i, row in enumerate(tqdm(rows, desc="MultiHop-RAG: building qa", unit="row")):
            query = (row.get("query") or "").strip()
            answer = row.get("answer")
            if not query or answer is None:
                continue
            supporting_titles: list[str] = []
            for ev in row.get("evidence_list") or []:
                ev_title = (ev.get("title") or "").strip()
                if ev_title and ev_title in title_to_slug and ev_title not in supporting_titles:
                    supporting_titles.append(ev_title)
            supporting_doc_ids = [title_to_slug[t] for t in supporting_titles]
            qa_pairs.append(
                BenchmarkQAPair(
                    id=str(row.get("id") or f"mhrag_{i}"),
                    question=query,
                    gold_answers=[str(answer)],
                    supporting_doc_ids=supporting_doc_ids,
                    metadata={
                        "question_type": row.get("question_type"),
                    },
                )
            )

        total_chars = 0
        total_words = 0
        for title, text in tqdm(
            title_to_text.items(),
            desc="MultiHop-RAG: writing corpus files",
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
            "MultiHop-RAG prepared: %d questions, %d corpus docs → %s",
            len(qa_pairs),
            doc_count,
            output_dir,
        )
        return manifest
