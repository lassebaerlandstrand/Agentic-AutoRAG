"""Download a development corpus of ArXiv PDFs for testing.

Fetches ~50 papers across 5 categories, saving PDFs and a metadata.json
sidecar to the output directory. Respects ArXiv rate limits.

Usage:
    uv run python scripts/download_arxiv_corpus.py
    uv run python scripts/download_arxiv_corpus.py --output-dir data/corpus/my-arxiv
    uv run python scripts/download_arxiv_corpus.py --max-per-category 2  # quick smoke test
"""

import argparse
import json
import time
from pathlib import Path

import arxiv

DEFAULT_CATEGORIES: dict[str, int] = {
    "cs.CL": 10,  # Computational Linguistics / NLP
    "cs.CV": 10,  # Computer Vision
    "cs.LG": 10,  # Machine Learning
    "stat.ML": 10,  # Statistics — Machine Learning
    "physics.optics": 10,  # Optics (out-of-domain control)
}


def download_corpus(
    output_dir: Path,
    categories: dict[str, int],
    max_per_category: int | None = None,
) -> list[dict]:
    """Download PDFs from ArXiv and return a metadata list.

    Args:
        output_dir: Directory to save PDFs and metadata.json.
        categories: Mapping of ArXiv category to desired paper count.
        max_per_category: If set, cap downloads per category (useful for testing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

    metadata: list[dict] = []
    for cat, count in categories.items():
        effective_count = min(count, max_per_category) if max_per_category else count
        print(f"Downloading {effective_count} papers from {cat}...")

        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=effective_count,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in client.results(search):
            safe_id = result.get_short_id().replace("/", "_")
            filename = f"{safe_id}.pdf"
            filepath = output_dir / filename

            if not filepath.exists():
                result.download_pdf(dirpath=str(output_dir), filename=filename)
                time.sleep(1)  # respect rate limits
            else:
                print(f"  (skipped, already exists) {filename}")

            metadata.append(
                {
                    "id": result.get_short_id(),
                    "title": result.title,
                    "category": cat,
                    "abstract": result.summary,
                    "filename": filename,
                }
            )
            print(f"  + {result.title[:80]}")

    # Write metadata sidecar
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(metadata)} papers saved to {output_dir}/")
    print(f"Metadata written to {meta_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ArXiv development corpus for Agentic AutoRAG testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/corpus/arxiv"),
        help="Directory to save PDFs and metadata (default: data/corpus/arxiv)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Cap downloads per category (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    download_corpus(
        output_dir=args.output_dir,
        categories=DEFAULT_CATEGORIES,
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
