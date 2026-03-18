"""Download a BEIR corpus subset for testing.

All open BEIR datasets store their corpus as `corpus.jsonl.gz` on HuggingFace.
This script downloads that file via `hf_hub_download` (cached after the first run)
and streams through it, writing each document as a plain .txt file that 
the pipeline reads directly without a parser.

A metadata.json sidecar is written for human reference; it is skipped by ingestion.

Usage:
    uv run python scripts/download_beir_corpus.py
    uv run python scripts/download_beir_corpus.py --dataset BeIR/scifact
    uv run python scripts/download_beir_corpus.py --dataset BeIR/fiqa --max-docs 500
    uv run python scripts/download_beir_corpus.py --max-docs 100          # quick smoke test

Open BEIR datasets (all follow the same corpus.jsonl.gz format):
    BeIR/nq            2.68 M Wikipedia passages  (~large download)
    BeIR/hotpotqa      5.23 M Wikipedia paragraphs (~large download)
    BeIR/fever         5.42 M Wikipedia passages  (~large download)
    BeIR/climate-fever 5.42 M Wikipedia passages  (~large download)
    BeIR/scifact       5.18 K scientific abstracts (tiny, fast)
    BeIR/nfcorpus      3.6  K medical documents   (tiny, fast)
    BeIR/arguana       8.67 K debate arguments     (small, fast)
    BeIR/scidocs       25   K scientific abstracts (small)
    BeIR/fiqa          57   K financial posts      (medium)
    BeIR/msmarco       8.84 M web passages         (~large download)

Note: BeIR/trec-news requires a HuggingFace login (gated dataset).
"""

import argparse
import gzip
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_DATASET = "BeIR/nq"
DEFAULT_OUTPUT_DIR = Path("data/corpus/nq")
DEFAULT_MAX_DOCS = 1000


def download_corpus(dataset: str, output_dir: Path, max_docs: int) -> list[dict]:
    """Download corpus.jsonl.gz from a BEIR HuggingFace dataset and write .txt files.

    Args:
        dataset: HuggingFace dataset ID, e.g. "BeIR/scifact".
        output_dir: Directory to save .txt files and metadata.json.
        max_docs: Maximum number of documents to extract.

    Returns:
        List of metadata dicts (one per document).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading corpus.jsonl.gz from {dataset} …")
    local_gz = hf_hub_download(
        repo_id=dataset,
        filename="corpus.jsonl.gz",
        repo_type="dataset",
    )
    print(f"Cached at: {local_gz}")

    metadata: list[dict] = []
    print(f"Extracting up to {max_docs} documents …")

    with gzip.open(local_gz, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break

            doc = json.loads(line)
            doc_id: str = doc["_id"]
            title: str = doc.get("title") or ""
            text: str = doc.get("text") or ""

            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
                content_parts.append("")
            content_parts.append(text)
            content = "\n".join(content_parts)

            safe_id = doc_id.replace("/", "_")
            filepath = output_dir / f"{safe_id}.txt"
            filepath.write_text(content, encoding="utf-8")

            metadata.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text_preview": text[:120],
                    "filename": filepath.name,
                }
            )

            if (i + 1) % 100 == 0 or (i + 1) == max_docs:
                print(f"  {i + 1}/{max_docs} documents saved …")

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(metadata)} documents saved to {output_dir}/")
    print(f"Metadata written to {meta_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a BEIR corpus subset for Agentic AutoRAG testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace BEIR dataset ID (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save .txt files and metadata (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help=f"Number of documents to extract (default: {DEFAULT_MAX_DOCS})",
    )
    args = parser.parse_args()

    download_corpus(
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
