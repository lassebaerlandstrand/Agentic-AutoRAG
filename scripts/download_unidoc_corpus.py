"""Download a UniDoc-Bench corpus subset for testing.

UniDoc-Bench (Salesforce, arXiv 2510.03663) contains 70 000+ real-world enterprise PDF
pages across 8 domains. It is the most realistic publicly available RAG corpus, with
complex layouts, tables, figures, and mixed modalities — matching what companies actually
feed into RAG pipelines.

This script downloads two types of documents for the selected domains:
  - PDFs  : full multi-page documents from domain tar.gz archives (exercises Docling)
  - Images: individual page PNGs stored directly in the repo (exercises Docling OCR)

Usage:
    uv run python scripts/download_unidoc_corpus.py
    uv run python scripts/download_unidoc_corpus.py --max-pdfs 4 --max-images 4   # smoke test
    uv run python scripts/download_unidoc_corpus.py --max-pdfs 50 --max-images 50
    uv run python scripts/download_unidoc_corpus.py --output-dir data/corpus/unidoc-large --max-pdfs 200
"""

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

DOMAINS: list[str] = ["healthcare", "legal"]

DOMAIN_ARCHIVE_MAP: dict[str, str] = {
    "healthcare": "healthcare_pdfs.tar.gz",  # 1,100 PDFs · 1.19 GB
    "legal": "legal_pdfs.tar.gz",  #   911 PDFs · 541  MB
    "education": "education_pdfs.tar.gz",  #   812 PDFs · 817  MB
    "crm": "crm_pdfs.tar.gz",  #   776 PDFs · 705  MB
    "energy": "energy_pdfs.tar.gz",  #   766 PDFs · 708  MB
    "construction": "construction_pdfs.tar.gz",  #   736 PDFs · 844  MB
    "commerce_manufacturing": "commerce_manufacturing_pdfs.tar.gz",  #   719 PDFs · 505  MB
    "finance": "finance_pdfs.tar.gz",  #   621 PDFs · 361  MB
}

REPO_ID = "Salesforce/UniDoc-Bench"

DEFAULT_OUTPUT_DIR = Path("data/corpus/unidoc")
DEFAULT_MAX_PDFS = 50
DEFAULT_MAX_IMAGES = 10


def _download_pdfs(
    domain: str,
    output_dir: Path,
    limit: int,
    metadata: list[dict],
) -> None:
    archive = DOMAIN_ARCHIVE_MAP[domain]
    print(f"\n[PDF] {domain} — downloading {archive} (cached after first run) …")
    local_path = hf_hub_download(repo_id=REPO_ID, filename=archive, repo_type="dataset")

    extracted = 0
    with tarfile.open(local_path, "r:gz") as tar:
        for member in tar:
            if extracted >= limit:
                break
            raw_stem = Path(member.name).name
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            if raw_stem.startswith("._"):  # macOS AppleDouble resource fork — skip
                continue

            dest_name = f"{domain}_{raw_stem}"
            dest_path = output_dir / dest_name

            if dest_path.exists():
                print(f"  (skip, exists) {dest_name}")
            else:
                f = tar.extractfile(member)
                if f is None:
                    continue
                dest_path.write_bytes(f.read())

            metadata.append({"domain": domain, "type": "pdf", "filename": dest_name})
            extracted += 1

            if extracted % 10 == 0 or extracted == limit:
                print(f"  {extracted}/{limit} PDFs …")

    print(f"  Done: {extracted} PDFs from {domain}")


def _download_images(
    domain: str,
    output_dir: Path,
    limit: int,
    metadata: list[dict],
) -> None:
    print(f"\n[IMG] {domain} — listing image documents …")
    fs = HfFileSystem()
    prefix = f"datasets/{REPO_ID}/images/{domain}"

    # Each sub-directory is one document (named by its 7-digit ID).
    doc_dirs = fs.ls(prefix, detail=False)

    downloaded = 0
    for doc_dir in doc_dirs:
        if downloaded >= limit:
            break

        doc_id = Path(doc_dir).name
        # Take only the first page of each document.
        img_hf_path = f"{doc_dir}/{doc_id}_page_0001.png"
        img_repo_path = img_hf_path.removeprefix(f"datasets/{REPO_ID}/")

        dest_name = f"{domain}_{doc_id}_page_0001.png"
        dest_path = output_dir / dest_name

        if dest_path.exists():
            print(f"  (skip, exists) {dest_name}")
        else:
            local = hf_hub_download(repo_id=REPO_ID, filename=img_repo_path, repo_type="dataset")
            dest_path.write_bytes(Path(local).read_bytes())

        metadata.append({"domain": domain, "type": "image", "filename": dest_name})
        downloaded += 1

        if downloaded % 10 == 0 or downloaded == limit:
            print(f"  {downloaded}/{limit} images …")

    print(f"  Done: {downloaded} images from {domain}")


def download_corpus(
    output_dir: Path,
    max_pdfs: int,
    max_images: int,
) -> None:
    unknown = [d for d in DOMAINS if d not in DOMAIN_ARCHIVE_MAP]
    if unknown:
        raise ValueError(f"Unknown domain(s) in DOMAINS constant: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_domain_pdfs = max_pdfs // len(DOMAINS)
    per_domain_images = max_images // len(DOMAINS)

    print(f"Domains  : {DOMAINS}")
    print(f"PDFs     : {max_pdfs} total ({per_domain_pdfs} per domain)")
    print(f"Images   : {max_images} total ({per_domain_images} per domain)")
    print(f"Output   : {output_dir}")

    metadata: list[dict] = []

    for domain in DOMAINS:
        if per_domain_pdfs > 0:
            _download_pdfs(domain, output_dir, per_domain_pdfs, metadata)
        if per_domain_images > 0:
            _download_images(domain, output_dir, per_domain_images, metadata)

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    pdfs = sum(1 for m in metadata if m["type"] == "pdf")
    imgs = sum(1 for m in metadata if m["type"] == "image")
    print(f"\nDone. {pdfs} PDFs + {imgs} images saved to {output_dir}/")
    print(f"Metadata written to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a UniDoc-Bench corpus subset for Agentic AutoRAG testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=DEFAULT_MAX_PDFS,
        help=f"Total PDFs to download across all domains (default: {DEFAULT_MAX_PDFS})",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f"Total page images to download across all domains (default: {DEFAULT_MAX_IMAGES})",
    )
    args = parser.parse_args()
    download_corpus(
        output_dir=args.output_dir,
        max_pdfs=args.max_pdfs,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
