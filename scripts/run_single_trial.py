"""Run a single RAG trial end-to-end.

Usage:
    python scripts/run_single_trial.py --config configs/starter.yaml

This script:
1. Parses the corpus (using the configured parser)
2. Chunks and embeds documents
3. Builds a LanceDB index
4. Generates an MCQ exam from the chunks
5. Constructs a RAG pipeline with a hardcoded config
6. Evaluates the pipeline against the exam
7. Prints the score and failed questions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    ExaminerConfig,
    IndexType,
    RuntimeConfig,
    StructuralConfig,
)
from agentic_autorag.engine.index_builder import IndexBuilder
from agentic_autorag.engine.parsers import BYPASS_EXTENSIONS, build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner.evaluator import MCQEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single RAG trial end-to-end.",
    )
    parser.add_argument(
        "--config",
        default="configs/starter.yaml",
        help="Path to the YAML search-space config (default: configs/starter.yaml)",
    )
    parser.add_argument(
        "--exam-size",
        type=int,
        default=10,
        help="Number of MCQ questions to generate (default: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5)",
    )
    return parser.parse_args()


def discover_and_parse(corpus_path: Path, parser_name: str) -> list[str]:
    """Recursively discover files in *corpus_path* and parse them to text."""
    parser = build_parser(parser_name)
    supported = parser.supported_extensions()

    documents: list[str] = []
    files = sorted(corpus_path.rglob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()

        # Bypass extensions are ingested directly
        if ext in (BYPASS_EXTENSIONS - {".json"}):
            logger.info("Reading directly: %s", file_path.name)
            documents.append(file_path.read_text(encoding="utf-8", errors="replace"))
            continue

        # Skip metadata and unsupported files
        if ext == ".json" or ext not in supported:
            continue

        logger.info("Parsing: %s", file_path.name)
        try:
            text = parser.parse(file_path)
            if text.strip():
                documents.append(text)
        except Exception:
            logger.warning("Failed to parse %s, skipping", file_path.name, exc_info=True)

    return documents


async def main() -> None:
    args = parse_args()
    space = load_config(args.config)

    # --- 1. Parse corpus ---
    corpus_path = Path(space.meta.corpus_path)
    if not corpus_path.exists():
        logger.error("Corpus path does not exist: %s", corpus_path)
        sys.exit(1)

    parser_name = space.structural.parsers[0]
    logger.info("Using parser: %s", parser_name)
    documents = discover_and_parse(corpus_path, parser_name)
    if not documents:
        logger.error("No documents found in %s", corpus_path)
        sys.exit(1)
    logger.info("Parsed %d documents", len(documents))

    # --- 2. Build index ---
    structural = StructuralConfig(
        parser=parser_name,
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
    )
    builder = IndexBuilder()
    logger.info("Building index (chunk_size=%d, overlap=%d)...", structural.chunk_size, structural.chunk_overlap)
    rag_index = await builder.build(documents, structural)
    logger.info("Index built: %d chunks", len(rag_index.chunks))

    # --- 3. Generate exam ---
    examiner_config = ExaminerConfig(
        exam_size=args.exam_size,
        mcq_options_count=space.examiner.mcq_options_count,
    )
    exam_agent = ExamAgent(
        config=examiner_config,
        examiner_model=space.agent.examiner_model,
    )
    chunk_ids = [f"chunk_{i}" for i in range(len(rag_index.chunks))]
    logger.info("Generating %d-question exam...", args.exam_size)
    exam = await exam_agent.generate_exam(rag_index.chunks, chunk_ids, rag_index.embeddings)
    logger.info("Generated %d questions", len(exam))

    if not exam:
        logger.error("No exam questions were generated. Exiting.")
        sys.exit(1)

    # --- 4. Construct RAG pipeline ---
    runtime = RuntimeConfig(
        top_k=args.top_k,
        reranker="none",
        llm_model="ollama/llama3.2",
        temperature=0.0,
    )
    embedder = builder._get_embedder(structural.embedding_model)
    pipeline = RAGPipeline(
        vector_store=rag_index.vector_store,
        graph_store=None,
        config=runtime,
        embedder=embedder,
        index_type=structural.index_type,
    )

    # --- 5. Evaluate ---
    evaluator = MCQEvaluator()
    logger.info("Running evaluation (top_k=%d)...", args.top_k)
    result = evaluator.evaluate(pipeline, exam)
    # evaluate is async
    result = await result

    # --- 6. Print results ---
    print(f"\n{'=' * 60}")
    print(f"  Score: {result.score:.1%}  ({result.n_correct}/{result.n_total} correct)")
    print(f"{'=' * 60}")

    failed = result.failed_questions()
    if failed:
        print(f"\n  Failed questions ({len(failed)}):\n")
        for i, qr in enumerate(failed, 1):
            # Find the original question for display
            original = next((q for q in exam if q.id == qr.question_id), None)
            q_text = original.question if original else "(question text unavailable)"
            print(f"  {i}. {q_text}")
            print(f"     Correct: {qr.correct_answer}  |  Selected: {qr.selected_answer}")
            ctx_preview = qr.retrieved_context[:200].replace("\n", " ")
            print(f"     Context: {ctx_preview}...")
            print()
    else:
        print("\n  All questions answered correctly!\n")


if __name__ == "__main__":
    asyncio.run(main())
