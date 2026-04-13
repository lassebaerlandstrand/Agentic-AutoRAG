"""Evaluate RAG probe configurations against UniDoc-Bench open-ended QA pairs.

Tests whether different RAG configurations produce meaningfully different scores
on an external benchmark (UniDoc-Bench), complementing the internal MCQ exam.
Reuses the probe system from exam generation to build diverse configs from
search space extremes, then evaluates each against the UniDoc QA pairs using
LLM-as-judge scoring.

Usage:
    uv run python scripts/evaluate_unidoc_bench.py --config configs/full.yaml
    uv run python scripts/evaluate_unidoc_bench.py --config configs/full.yaml --max-questions 10
    uv run python scripts/evaluate_unidoc_bench.py --config configs/full.yaml --judge-model "gemini/gemini-2.5-flash"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import litellm
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import ProjectConfig, TrialConfig
from agentic_autorag.engine.index_builder import IndexBuilder, RAGIndex
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner._errors import format_llm_error, is_permanent_llm_error
from agentic_autorag.examiner.probe_selector import rank_models_for_probes, select_probe_configs

logger = logging.getLogger(__name__)

UNIDOC_REPO = "Salesforce/UniDoc-Bench"

# Must match DOMAINS in scripts/download_unidoc_corpus.py.
DOMAINS: list[str] = ["healthcare"]

# Files that are skipped during corpus loading (same as orchestrator).
_SKIP_FILENAMES = {"metadata.json"}
_DIRECT_READ_EXTENSIONS = {".md", ".txt"}

QA_ANSWER_PROMPT = """\
Answer the following question based ONLY on the provided context.
Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

# LLM-as-judge prompt following RAGAS answer_correctness pattern:
# - Includes brief reasoning before the verdict (chain-of-thought)
# - Explicit rubric for what 0 vs 1 means
# - Only sees question + reference + generated (NOT retrieved context)
#
# References:
#   RAGAS answer_correctness: factual similarity via claim extraction
#   Hugging Face LLM-judge cookbook: structured evaluation + reasoning
#   arXiv 2408.09235: reference-guided verdict for free-form QA
JUDGE_PROMPT = """\
You are an impartial judge evaluating a RAG system's answer against a reference.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Determine if the generated answer correctly answers the question with accurate facts.

Score 1 if: the key claims in the generated answer are factually consistent with \
the reference. The answer does NOT need to cover every detail or case in the reference \
— partial but correct answers are acceptable.

Score 0 if: the generated answer contradicts the reference, makes false claims, \
or completely fails to address the question.

Provide your evaluation in exactly this format:
Evaluation: <one sentence reasoning>
Score: <0 or 1>"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class QAPair:
    """A single UniDoc-Bench question-answer pair."""

    __slots__ = ("question", "answer", "question_type", "answer_type", "domain", "doc_ids")

    def __init__(
        self,
        question: str,
        answer: str,
        question_type: str,
        answer_type: str,
        domain: str,
        doc_ids: list[str],
    ) -> None:
        self.question = question
        self.answer = answer
        self.question_type = question_type
        self.answer_type = answer_type
        self.domain = domain
        self.doc_ids = doc_ids


class QAResult:
    """Result of evaluating a single QA pair.

    Timing fields:
        retrieval_s:  wall-clock seconds for retrieval (query embedding + vector
                      search + optional reranking).
        generation_s: wall-clock seconds for the pipeline LLM to produce the answer.
        judge_s:      wall-clock seconds for the judge LLM to score the answer.

    Debug fields:
        retrieved_context: concatenated text from retrieved chunks (for inspection).
        judge_reasoning:   raw judge LLM response including evaluation reasoning.
    """

    __slots__ = (
        "question",
        "reference_answer",
        "generated_answer",
        "score",
        "source_files",
        "retrieval_s",
        "generation_s",
        "judge_s",
        "retrieved_context",
        "judge_reasoning",
    )

    def __init__(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        score: int,
        source_files: list[str] | None = None,
        retrieval_s: float = 0.0,
        generation_s: float = 0.0,
        judge_s: float = 0.0,
        retrieved_context: str = "",
        judge_reasoning: str = "",
    ) -> None:
        self.question = question
        self.reference_answer = reference_answer
        self.generated_answer = generated_answer
        self.score = score
        self.source_files = source_files or []
        self.retrieval_s = retrieval_s
        self.generation_s = generation_s
        self.judge_s = judge_s
        self.retrieved_context = retrieved_context
        self.judge_reasoning = judge_reasoning

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "generated_answer": self.generated_answer,
            "score": self.score,
            "source_files": self.source_files,
            "retrieval_s": self.retrieval_s,
            "generation_s": self.generation_s,
            "judge_s": self.judge_s,
            "retrieved_context": self.retrieved_context,
            "judge_reasoning": self.judge_reasoning,
        }


class ProbeResult:
    """Aggregated result for a single probe configuration."""

    __slots__ = ("label", "config", "score", "n_correct", "n_total", "question_results")

    def __init__(
        self,
        label: str,
        config: TrialConfig,
        score: float,
        n_correct: int,
        n_total: int,
        question_results: list[QAResult],
    ) -> None:
        self.label = label
        self.config = config
        self.score = score
        self.n_correct = n_correct
        self.n_total = n_total
        self.question_results = question_results

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "config": self.config.model_dump(),
            "score": self.score,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "question_results": [qr.to_dict() for qr in self.question_results],
        }


# ---------------------------------------------------------------------------
# QA pair loading and filtering
# ---------------------------------------------------------------------------


def _extract_doc_ids(image_paths: list[str]) -> set[str]:
    """Extract unique document IDs from UniDoc image paths.

    Path format: images/{domain}/{doc_id}/{doc_id}_page_XXXX.png
    """
    ids: set[str] = set()
    for path in image_paths:
        parts = path.split("/")
        # images / domain / doc_id / filename
        if len(parts) >= 3:
            ids.add(parts[2])
    return ids


def _scan_corpus_doc_ids(corpus_dir: Path) -> tuple[set[str], set[str]]:
    """Scan the corpus directory and extract document IDs from filenames.

    Returns (all_doc_ids, pdf_doc_ids). PDF-backed documents have the full
    multi-page content parsed by Docling; image-only documents only have
    page 1 and are missing later pages where answers typically reside.
    """
    all_ids: set[str] = set()
    pdf_ids: set[str] = set()
    for file_path in corpus_dir.iterdir():
        if not file_path.is_file():
            continue
        # Match 7-digit doc IDs in filenames (e.g., healthcare_0355944.pdf).
        # Can't use \b because _ is a word character.
        matches = re.findall(r"(\d{7})", file_path.stem)
        for did in matches:
            all_ids.add(did)
            if file_path.suffix.lower() == ".pdf":
                pdf_ids.add(did)
    return all_ids, pdf_ids


def load_qa_pairs(corpus_dir: Path, max_questions: int | None = None) -> list[QAPair]:
    """Load UniDoc-Bench QA pairs and filter to PDF-backed documents.

    Image-only documents only have page 1 in our corpus, but ground truth
    answers are on later pages. PDF documents are fully parsed by Docling
    (including OCR of figures and table extraction), so all answer types
    are answerable.
    """
    all_doc_ids, pdf_doc_ids = _scan_corpus_doc_ids(corpus_dir)
    if not all_doc_ids:
        raise RuntimeError(
            f"No document IDs found in corpus directory: {corpus_dir}. Run scripts/download_unidoc_corpus.py first."
        )
    n_img = len(all_doc_ids) - len(pdf_doc_ids)
    print(f"Found {len(all_doc_ids)} document IDs in corpus ({len(pdf_doc_ids)} PDFs, {n_img} image-only)")

    qa_pairs: list[QAPair] = []
    skipped_no_match = 0
    skipped_no_pdf = 0
    for domain in DOMAINS:
        print(f"Loading UniDoc-Bench QA pairs for domain: {domain}")
        dataset = load_dataset(UNIDOC_REPO, split=domain)

        for row in dataset:
            qa_doc_ids = _extract_doc_ids(row["longdoc_image_paths"])
            matching_ids = qa_doc_ids & all_doc_ids
            if not matching_ids:
                skipped_no_match += 1
                continue
            # Require at least one matching doc to be a PDF — image-only
            # docs only have page 1, missing the pages with the answer.
            matching_pdfs = matching_ids & pdf_doc_ids
            if not matching_pdfs:
                skipped_no_pdf += 1
                continue

            qa_pairs.append(
                QAPair(
                    question=row["question"],
                    answer=row["answer"],
                    question_type=row["question_type"],
                    answer_type=row["answer_type"],
                    domain=row["domain"],
                    doc_ids=sorted(matching_pdfs),
                )
            )

    print(f"Filtered to {len(qa_pairs)} QA pairs")
    print(f"  Skipped: {skipped_no_match} no corpus match, {skipped_no_pdf} image-only")

    if max_questions is not None and len(qa_pairs) > max_questions:
        qa_pairs = qa_pairs[:max_questions]
        print(f"Truncated to {max_questions} QA pairs")

    return qa_pairs


# ---------------------------------------------------------------------------
# Corpus parsing (same pattern as orchestrator)
# ---------------------------------------------------------------------------


def _corpus_cache_key(corpus_dir: Path, config: ProjectConfig) -> str:
    """Compute the same cache key as the orchestrator for corpus reuse."""
    parsing = config.parsing
    file_signatures: list[tuple[str, int, int]] = []
    for file_path in sorted(corpus_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.name in _SKIP_FILENAMES:
            continue
        stat = file_path.stat()
        rel = str(file_path.relative_to(corpus_dir))
        file_signatures.append((rel, stat.st_mtime_ns, stat.st_size))

    key_data = json.dumps(
        {
            "parser": parsing.parser,
            "ocr": parsing.ocr,
            "table_structure": parsing.table_structure,
            "files": file_signatures,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def parse_corpus(corpus_dir: Path, config: ProjectConfig, output_dir: Path) -> list[str]:
    """Parse all documents in the corpus directory to text.

    Checks for an existing orchestrator corpus cache first to avoid re-parsing.
    """
    cache_key = _corpus_cache_key(corpus_dir, config)
    cache_path = output_dir / ".cache" / f"corpus_{cache_key}.json"

    if cache_path.exists():
        print(f"Loading cached parsed corpus from {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    parser = build_parser(
        config.parsing.parser,
        ocr=config.parsing.ocr,
        table_structure=config.parsing.table_structure,
    )

    eligible: list[Path] = []
    for file_path in sorted(corpus_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.name in _SKIP_FILENAMES:
            continue
        eligible.append(file_path)

    documents: list[str] = []
    skipped = 0
    failed = 0
    for file_path in tqdm(eligible, desc="Parsing files", unit="file"):
        suffix = file_path.suffix.lower()
        try:
            if suffix in _DIRECT_READ_EXTENSIONS:
                text = file_path.read_text(encoding="utf-8")
            elif suffix in parser.supported_extensions():
                text = parser.parse(file_path)
            else:
                skipped += 1
                continue

            text = text.strip()
            if text:
                documents.append(text)
        except Exception:
            failed += 1
            logger.warning("Failed to parse %s, skipping", file_path, exc_info=True)

    if skipped:
        print(f"Skipped {skipped} unsupported file(s)")
    if failed:
        print(f"Failed to parse {failed} file(s)")

    # Cache for future runs
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write corpus cache", exc_info=True)

    return documents


# ---------------------------------------------------------------------------
# LLM-as-judge evaluation
# ---------------------------------------------------------------------------


def _parse_judge_score(response: str | None) -> int:
    """Extract binary score from judge LLM response.

    Expects format: "Evaluation: ... Score: 0" or "Score: 1".
    Falls back to scanning for bare 0/1 digits.
    """
    if not response:
        return 0
    text = response.strip()

    # Primary: look for "Score: 0" or "Score: 1"
    match = re.search(r"Score:\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: bare digit at start or end
    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0

    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return 0


async def _evaluate_single(
    pipeline: RAGPipeline,
    qa: QAPair,
    judge_model: str,
    sem: asyncio.Semaphore,
) -> QAResult:
    """Evaluate a single QA pair: retrieve, generate, judge."""
    async with sem:
        try:
            # Retrieve
            t0 = time.monotonic()
            retrieval_result = await pipeline.retrieve(qa.question)
            retrieval_s = time.monotonic() - t0

            context = "\n".join(doc.text for doc in retrieval_result.documents)

            # Generate answer
            prompt = QA_ANSWER_PROMPT.format(context=context, question=qa.question)
            t0 = time.monotonic()
            generated_answer = await pipeline.generate(prompt) or ""
            generation_s = time.monotonic() - t0

            # Judge — separate LLM call, not the pipeline LLM
            judge_prompt = JUDGE_PROMPT.format(
                question=qa.question,
                reference=qa.answer,
                generated=generated_answer,
            )
            t0 = time.monotonic()
            judge_response = await litellm.acompletion(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=1.0,
                num_retries=0,
            )
            judge_s = time.monotonic() - t0

            judge_raw = judge_response.choices[0].message.content or ""
            score = _parse_judge_score(judge_raw)

            source_files = [f"{qa.domain}_{did}.pdf" for did in qa.doc_ids]

            return QAResult(
                question=qa.question,
                reference_answer=qa.answer,
                generated_answer=generated_answer,
                score=score,
                source_files=source_files,
                retrieval_s=retrieval_s,
                generation_s=generation_s,
                judge_s=judge_s,
                retrieved_context=context,
                judge_reasoning=judge_raw,
            )
        except Exception as exc:
            error_summary = format_llm_error(exc)
            permanent = is_permanent_llm_error(exc)
            tqdm.write(f"  ERROR | {error_summary}")
            logger.debug("QA evaluation failed", exc_info=True)
            source_files = [f"{qa.domain}_{did}.pdf" for did in qa.doc_ids]
            return QAResult(
                question=qa.question,
                reference_answer=qa.answer,
                generated_answer=f"ERROR: {error_summary}" if permanent else "ERROR",
                source_files=source_files,
                score=0,
            )


async def evaluate_probe(
    pipeline: RAGPipeline,
    qa_pairs: list[QAPair],
    judge_model: str,
    concurrency: int,
    label: str,
) -> list[QAResult]:
    """Evaluate all QA pairs against a single probe pipeline."""
    sem = asyncio.Semaphore(concurrency)

    with tqdm(total=len(qa_pairs), desc=f"  {label}", unit="q") as pbar:

        async def _bounded(qa: QAPair) -> QAResult:
            qr = await _evaluate_single(pipeline, qa, judge_model, sem)
            if not qr.score:
                reason = qr.judge_reasoning.replace("\n", " ")[:120] if qr.judge_reasoning else "no response"
                tqdm.write(f"  MISS | {qa.question[:70]}...")
                tqdm.write(f"         Judge: {reason}")
            pbar.update(1)
            return qr

        results = list(await asyncio.gather(*[_bounded(qa) for qa in qa_pairs]))

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


async def run_evaluation(
    config: ProjectConfig,
    corpus_dir: Path,
    qa_pairs: list[QAPair],
    judge_model: str,
    concurrency: int,
    output_dir: Path,
) -> list[ProbeResult]:
    """Run probe evaluation against UniDoc QA pairs."""
    # Parse corpus
    print(f"\nParsing corpus from {corpus_dir}")
    documents = parse_corpus(corpus_dir, config, output_dir)
    print(f"Parsed {len(documents)} document(s)")
    if not documents:
        raise RuntimeError(f"No documents parsed from {corpus_dir}")

    # Rank models for probe generation — same as orchestrator (lines 633-641).
    # Uses KnowledgeBase if available, falls back to LLM ranking.
    print("\nRanking models for probe generation")
    ss = config.search_space
    knowledge_base = None
    try:
        from agentic_autorag.config.knowledge_base import KnowledgeBase

        knowledge_base = KnowledgeBase()
    except Exception:
        logger.info("KnowledgeBase not available, using LLM fallback for model ranking")

    optimizer_model = config.agent.optimizer_model
    ranked_llms = await rank_models_for_probes(ss.llm_models, "llm", knowledge_base, optimizer_model)
    ranked_embeds = await rank_models_for_probes(ss.embedding_models, "embedding", knowledge_base, optimizer_model)
    ranked_rerankers = await rank_models_for_probes(ss.reranker.models, "reranker", knowledge_base, optimizer_model)

    print(f"  LLMs (weak→strong):     {[m.rsplit('/', 1)[-1] for m in ranked_llms]}")
    print(f"  Embeddings (weak→strong): {[m.rsplit('/', 1)[-1] for m in ranked_embeds]}")
    print(f"  Rerankers (weak→strong):  {[m.rsplit('/', 1)[-1] for m in ranked_rerankers]}")

    # Generate probe configs using ranked model lists
    print("\nGenerating probe configurations from search space")
    labelled_probes = select_probe_configs(
        config,
        ranked_llms=ranked_llms,
        ranked_embeds=ranked_embeds,
        ranked_rerankers=ranked_rerankers,
    )
    # The probe selector hardcodes temperature=0.0 for deterministic MCQ scoring,
    # but some models (o-series) reject 0.0 and the search space may mandate a
    # different value.  Override to use the search space minimum.
    ss_temp = ss.temperature.min
    for _, tc in labelled_probes:
        tc.temperature = ss_temp
    print(f"Generated {len(labelled_probes)} probe config(s) (temperature={ss_temp})")

    # Build indices and evaluate
    db_path = output_dir / "lancedb_bench"
    index_builder = IndexBuilder(db_path=db_path)
    index_cache: dict[str, RAGIndex] = {}

    probe_results: list[ProbeResult] = []

    for i, (probe_label, probe_config) in enumerate(labelled_probes):
        print(f"\n{'=' * 60}")
        print(f"Probe {i + 1}/{len(labelled_probes)} — {probe_label}")
        print(f"  chunk={probe_config.chunk_token_size} embed={probe_config.embedding_model}")
        print(f"  top_k={probe_config.top_k} reranker={probe_config.reranker}")
        print(f"  llm={probe_config.llm_model}")
        print(f"{'=' * 60}")

        try:
            # Build or reuse index (same as orchestrator lines 799-808)
            structural = probe_config.to_structural()
            fp = structural.fingerprint()
            if fp in index_cache:
                index = index_cache[fp]
                print(f"  Reusing cached index {fp}")
            else:
                print(f"  Building index {fp}")
                index = await index_builder.build(
                    documents,
                    structural,
                    embedding_token_limits=config.embedding_token_limits,
                )
                index_cache[fp] = index
                print(f"  Index built: {len(index.chunks)} chunks")

            # Construct pipeline (same as orchestrator lines 810-825)
            embedder = index_builder.get_embedder(probe_config.embedding_model)
            cross_encoder = (
                index_builder.get_cross_encoder(probe_config.reranker)
                if probe_config.reranker and probe_config.reranker != "none"
                else None
            )
            pipeline = RAGPipeline(
                vector_store=index.vector_store,
                graph_store=None,
                config=probe_config.to_runtime(
                    reasoning_effort=config.search_space.reasoning_effort,
                ),
                embedder=embedder,
                index_type=probe_config.index_type,
                cross_encoder=cross_encoder,
            )

            # Evaluate
            results = await evaluate_probe(
                pipeline, qa_pairs, judge_model, concurrency, probe_label.split("(")[0].strip()
            )

            n_correct = sum(r.score for r in results)
            n_total = len(results)
            score = n_correct / n_total if n_total else 0.0

            probe_results.append(
                ProbeResult(
                    label=probe_label,
                    config=probe_config,
                    score=score,
                    n_correct=n_correct,
                    n_total=n_total,
                    question_results=results,
                )
            )

            print(f"\n  Result: {n_correct}/{n_total} correct ({score:.1%})")

        except Exception:
            logger.exception("Probe %d (%s) failed; skipping", i + 1, probe_label)
            print("\n  FAILED — see log for details")

    return probe_results


def _compute_discrimination(probe_results: list[ProbeResult]) -> dict[str, float]:
    """Compute per-question discrimination scores across probes.

    Questions that produce mixed results (correct on some probes, wrong on others)
    are the most discriminating. Returns variance of the binary response vector.
    """
    if len(probe_results) < 2:
        return {}

    # Collect all questions
    all_questions: set[str] = set()
    for pr in probe_results:
        for qr in pr.question_results:
            all_questions.add(qr.question)

    scores: dict[str, float] = {}
    for q in all_questions:
        responses: list[int] = []
        for pr in probe_results:
            for qr in pr.question_results:
                if qr.question == q:
                    responses.append(qr.score)
                    break
        if len(responses) < 2:
            scores[q] = 0.0
            continue
        mean = sum(responses) / len(responses)
        variance = sum((r - mean) ** 2 for r in responses) / len(responses)
        scores[q] = variance

    return scores


def print_summary(probe_results: list[ProbeResult]) -> None:
    """Print a summary table of probe results."""
    if not probe_results:
        print("\nNo probe results to report.")
        return

    print(f"\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 72}")

    # Header
    label_width = max(len(pr.label.split("(")[0].strip()) for pr in probe_results)
    label_width = max(label_width, 10)
    print(f"{'Probe':<{label_width}} | {'Score':>7} | {'Correct':>9} | {'Total':>5}")
    print(f"{'-' * label_width}-+-{'-' * 7}-+-{'-' * 9}-+-{'-' * 5}")

    for pr in probe_results:
        short_label = pr.label.split("(")[0].strip()
        print(f"{short_label:<{label_width}} | {pr.score:>6.1%} | {pr.n_correct:>4}/{pr.n_total:<4} | {pr.n_total:>5}")

    # Discrimination analysis
    discrimination = _compute_discrimination(probe_results)
    if discrimination:
        n_discriminating = sum(1 for v in discrimination.values() if v > 0)
        mean_disc = sum(discrimination.values()) / len(discrimination)
        max_disc = max(discrimination.values())
        print(f"\nDiscrimination analysis ({len(discrimination)} questions):")
        print(f"  Discriminating questions: {n_discriminating}/{len(discrimination)}")
        print(f"  Mean discrimination:      {mean_disc:.4f}")
        print(f"  Max discrimination:       {max_disc:.4f}")

    # Score spread
    scores = [pr.score for pr in probe_results]
    spread = max(scores) - min(scores)
    print(f"\nScore spread: {spread:.1%} (max={max(scores):.1%}, min={min(scores):.1%})")
    if spread < 0.05:
        print("  WARNING: Low score spread — the benchmark may not discriminate well between configs")
    elif spread > 0.15:
        print("  Good score spread — the benchmark differentiates configurations")


def save_results(probe_results: list[ProbeResult], output_path: Path) -> None:
    """Save detailed results to JSON."""
    discrimination = _compute_discrimination(probe_results)

    output = {
        "probes": [pr.to_dict() for pr in probe_results],
        "discrimination": {q[:100]: v for q, v in sorted(discrimination.items(), key=lambda x: -x[1])},
        "summary": {
            "n_probes": len(probe_results),
            "n_questions": probe_results[0].n_total if probe_results else 0,
            "scores": {pr.label.split("(")[0].strip(): pr.score for pr in probe_results},
            "score_spread": max(pr.score for pr in probe_results) - min(pr.score for pr in probe_results)
            if probe_results
            else 0.0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG probe configurations against UniDoc-Bench QA pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to project config YAML")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of QA pairs (default: all)")
    parser.add_argument("--judge-model", type=str, default=None, help="Override judge model (default: examiner_model)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Silence noisy third-party loggers (same approach as graph_store.py / cli.py).
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load config
    config = load_config(args.config)
    corpus_dir = Path(config.meta.corpus_path)
    output_dir = args.output_dir or Path(config.meta.output_dir)
    judge_model = args.judge_model or config.agent.examiner_model
    concurrency = args.concurrency or config.agent.concurrency

    print(f"Config:       {args.config}")
    print(f"Corpus:       {corpus_dir}")
    print(f"Judge model:  {judge_model}")
    print(f"Concurrency:  {concurrency}")

    # Load and filter QA pairs
    print(f"\nLoading UniDoc-Bench QA pairs (domains: {DOMAINS})")
    qa_pairs = load_qa_pairs(corpus_dir, max_questions=args.max_questions)
    if not qa_pairs:
        print("No matching QA pairs found. Ensure the corpus has been downloaded.")
        return

    # Print QA pair breakdown
    by_type: dict[str, int] = {}
    for qa in qa_pairs:
        by_type[qa.question_type] = by_type.get(qa.question_type, 0) + 1
    print(f"Question type breakdown: {by_type}")

    # Run evaluation
    probe_results = asyncio.run(run_evaluation(config, corpus_dir, qa_pairs, judge_model, concurrency, output_dir))

    # Report
    print_summary(probe_results)
    if probe_results:
        save_results(probe_results, output_dir / "unidoc_bench_results.json")


if __name__ == "__main__":
    main()
