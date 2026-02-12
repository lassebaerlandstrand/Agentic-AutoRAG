"""Test the reasoning agent in isolation.

Usage:
    python scripts/run_agent_once.py --config configs/starter.yaml

This script:
1. Loads a saved ExamResult from a JSON file (or generates a mock one)
2. Asks the agent to diagnose failures and propose the next config
3. Prints the error trace and proposed config
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    IndexType,
    RuntimeConfig,
    StructuralConfig,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the reasoning agent in isolation.",
    )
    parser.add_argument(
        "--config",
        default="configs/starter.yaml",
        help="Path to the YAML search-space config (default: configs/starter.yaml)",
    )
    parser.add_argument(
        "--exam-result",
        default=None,
        help="Path to a saved ExamResult JSON file. If omitted, a mock result is generated.",
    )
    parser.add_argument(
        "--history",
        default="./experiments/history.jsonl",
        help="Path to the JSONL history file (default: ./experiments/history.jsonl)",
    )
    return parser.parse_args()


def _build_mock_exam_result() -> ExamResult:
    """Create a mock ExamResult with some fake failures for testing."""
    results = [
        QuestionResult(
            question_id=f"q{i}",
            correct=i < 4,
            selected_answer="B" if i < 4 else "A",
            correct_answer="B",
            retrieved_context=f"Context for question {i}: this is some retrieved text about topic {i}.",
            generated_response="B" if i < 4 else "A",
        )
        for i in range(10)
    ]
    n_correct = sum(1 for r in results if r.correct)
    return ExamResult(
        score=n_correct / len(results),
        n_correct=n_correct,
        n_total=len(results),
        question_results=results,
    )


def _build_default_config() -> TrialConfig:
    """Build a simple default TrialConfig for the initial trial."""
    return TrialConfig(
        structural=StructuralConfig(
            parser="pymupdf4llm",
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type=IndexType.VECTOR_ONLY,
        ),
        runtime=RuntimeConfig(
            top_k=5,
            reranker="none",
            llm_model="ollama/llama3.2",
            temperature=0.0,
        ),
    )


async def main() -> None:
    args = parse_args()
    space = load_config(args.config)

    history = HistoryLog(path=args.history)
    agent = ReasoningAgent(
        agent_model=space.agent.optimizer_model,
        search_space=space,
        history=history,
    )

    # --- 1. Propose initial config ---
    logger.info("Proposing initial configuration...")
    corpus_desc = space.meta.corpus_description or "No corpus description provided."
    try:
        initial_config = await agent.propose_initial(corpus_desc)
    except RuntimeError as e:
        logger.error("Failed to get initial config: %s", e)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  INITIAL CONFIG")
    print(f"{'=' * 60}")
    print(initial_config.model_dump_json(indent=2))

    # --- 2. Load or mock exam result ---
    if args.exam_result:
        logger.info("Loading exam result from %s", args.exam_result)
        with open(args.exam_result, encoding="utf-8") as f:
            exam_result = ExamResult.model_validate_json(f.read())
    else:
        logger.info("Using mock exam result (pass --exam-result to use real data)")
        exam_result = _build_mock_exam_result()

    print(f"\n  Exam score: {exam_result.score:.1%} ({exam_result.n_correct}/{exam_result.n_total})")
    print(f"  Failed questions: {len(exam_result.failed_questions())}")

    # --- 3. Add initial trial to history ---
    history.add(TrialRecord(
        trial_number=len(history.records) + 1,
        config=initial_config,
        score=exam_result.score,
        error_trace="(initial trial)",
        question_results=exam_result.question_results,
    ))

    # --- 4. Diagnose + propose next config ---
    logger.info("Analyzing failures and proposing next config...")
    try:
        error_trace, next_config = await agent.analyze_and_propose(exam_result, initial_config)
    except RuntimeError as e:
        logger.error("Failed to get next config: %s", e)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  ERROR TRACE")
    print(f"{'=' * 60}")
    print(error_trace)

    print(f"\n{'=' * 60}")
    print("  PROPOSED NEXT CONFIG")
    print(f"{'=' * 60}")
    print(next_config.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
