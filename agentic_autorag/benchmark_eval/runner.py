"""Glue between CLI, config loader, index builder, pipeline, and FreeFormEvaluator.

Given a project config (corpus path, graph/vllm settings) + a trial config (flat
RAG parameters selected by an optimizer) + a prepared ``qa.json`` + an output
path, builds the RAG pipeline and writes a ``BenchmarkResult`` JSON.

For fastest iteration run this in the same ``meta.output_dir`` as the preceding
``optimize`` run so the ingredient cache (chunks + embeddings + graph) is reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import yaml

from agentic_autorag.benchmark_eval.evaluator import FreeFormEvaluator, is_error_sentinel
from agentic_autorag.benchmark_eval.models import BenchmarkResult, QAResult
from agentic_autorag.benchmark_eval.scoring import retrieval_metrics
from agentic_autorag.benchmarks import load_qa
from agentic_autorag.benchmarks.schema import BenchmarkManifest
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import GRAPH_INDEX_TYPES, ParsingConfig, ProjectConfig, TrialConfig
from agentic_autorag.engine._io import DIRECT_READ_EXTENSIONS, SKIP_FILENAMES
from agentic_autorag.engine.index_builder import IndexBuilder, IngredientCache
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.engine.vllm_server import VLLMServerManager

logger = logging.getLogger(__name__)
# User-facing progress. The CLI configures this logger with a bare %(message)s
# formatter and propagate=False (same pattern orchestrator uses for `optimize`).
run_logger = logging.getLogger("agentic_autorag.run")


def _config_hash(cfg) -> str:
    data = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _iter_corpus_files(corpus_path: Path):
    """Yield the same files, in the same order, as orchestrator._corpus_cache_key."""
    for f in sorted(corpus_path.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.name in SKIP_FILENAMES:
            continue
        yield f


def _load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    """Read every .md / .txt file under ``corpus_path`` into (filename, text) pairs.

    Fails loudly on unsupported files: this runner cannot spin up Docling, and
    silently skipping PDFs / DOCX would produce subtly wrong metrics (cached
    doc_indices from a fuller ``optimize`` run would no longer align).
    """
    supported: list[Path] = []
    unsupported: list[Path] = []
    for f in _iter_corpus_files(corpus_path):
        if f.suffix.lower() in DIRECT_READ_EXTENSIONS:
            supported.append(f)
        else:
            unsupported.append(f)
    if unsupported:
        sample = ", ".join(p.name for p in unsupported[:3])
        raise RuntimeError(
            f"benchmark-evaluate only supports .md/.txt corpora; found {len(unsupported)} "
            f"unsupported file(s) under {corpus_path} (e.g. {sample}). Convert to markdown "
            "or run optimize + benchmark-evaluate against a benchmark-prepared corpus."
        )
    if not supported:
        raise RuntimeError(f"No .md/.txt files found under {corpus_path}")

    filenames: list[str] = []
    texts: list[str] = []
    for f in supported:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # doc_id labels match what `benchmark-prepare` writes into
        # supporting_doc_ids (slug, no extension) so Recall@k / MRR align.
        filenames.append(f.stem)
        texts.append(text)
    return filenames, texts


def _corpus_hash(corpus_path: Path, parsing: ParsingConfig) -> str:
    """Deterministic corpus key. Matches orchestrator._corpus_cache_key exactly
    so the ingredient cache written by ``optimize`` is reused by this runner."""
    sigs: list[tuple[str, int, int]] = []
    for f in _iter_corpus_files(corpus_path):
        stat = f.stat()
        sigs.append((str(f.relative_to(corpus_path)), stat.st_mtime_ns, stat.st_size))
    key = json.dumps(
        {
            "parser": parsing.parser,
            "ocr": parsing.ocr,
            "table_structure": parsing.table_structure,
            "files": sigs,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(key).hexdigest()[:16]


def _aggregate(
    per_question: list[QAResult],
    supporting_present: bool,
    judge_enabled: bool,
) -> dict:
    """Compute aggregate metrics, excluding error-sentinel rows.

    Cost and token totals sum across *every* question (errors included), since
    LLM calls that errored after sending a request still incur cost.
    """
    valid = [r for r in per_question if not is_error_sentinel(r)]
    n_valid = len(valid)

    total_cost_usd = sum(r.llm_cost_usd for r in per_question)
    total_prompt_tokens = sum(r.prompt_tokens for r in per_question)
    total_completion_tokens = sum(r.completion_tokens for r in per_question)

    if not n_valid:
        return {
            "n_valid": 0,
            "em": 0.0,
            "f1": 0.0,
            "llm_judge_accuracy": None,
            "n_judge_invalid": 0,
            "recall_at_1": None,
            "recall_at_2": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "avg_retrieval_s": 0.0,
            "avg_generation_s": 0.0,
            "total_cost_usd": total_cost_usd,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }

    em = sum(r.em for r in valid) / n_valid
    f1 = sum(r.f1 for r in valid) / n_valid
    avg_retrieval_s = sum(r.retrieval_s for r in valid) / n_valid
    avg_generation_s = sum(r.generation_s for r in valid) / n_valid

    judged = [r for r in valid if r.judge is not None]
    n_judge_invalid = sum(1 for r in valid if r.judge is None) if judge_enabled else 0
    judge_acc: float | None = None
    if judge_enabled and judged:
        judge_acc = sum(r.judge for r in judged) / len(judged)

    recalls: dict[int, float] | None = None
    mrr: float | None = None
    if supporting_present:
        recall_sums = {1: 0.0, 2: 0.0, 5: 0.0, 10: 0.0}
        mrr_sum = 0.0
        n_with_gold = 0
        for r in valid:
            if not r.supporting_doc_ids:
                continue
            n_with_gold += 1
            r_recalls, first_rank = retrieval_metrics(r.retrieved_doc_ids, r.supporting_doc_ids)
            for k in recall_sums:
                recall_sums[k] += r_recalls[k]
            mrr_sum += 1.0 / first_rank if first_rank else 0.0
        if n_with_gold:
            recalls = {k: v / n_with_gold for k, v in recall_sums.items()}
            mrr = mrr_sum / n_with_gold

    return {
        "n_valid": n_valid,
        "em": em,
        "f1": f1,
        "llm_judge_accuracy": judge_acc,
        "n_judge_invalid": n_judge_invalid,
        "recall_at_1": recalls[1] if recalls else None,
        "recall_at_2": recalls[2] if recalls else None,
        "recall_at_5": recalls[5] if recalls else None,
        "recall_at_10": recalls[10] if recalls else None,
        "mrr": mrr,
        "avg_retrieval_s": avg_retrieval_s,
        "avg_generation_s": avg_generation_s,
        "total_cost_usd": total_cost_usd,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


async def run(
    project_config_path: str | Path,
    trial_config_path: str | Path,
    qa_path: str | Path,
    output_path: str | Path,
    judge_model: str | None = None,
    concurrency: int = 10,
    limit: int | None = None,
) -> BenchmarkResult:
    """Build pipeline from config pair, evaluate QA, write JSON, return result."""
    project: ProjectConfig = load_config(str(project_config_path))
    trial_data = yaml.safe_load(Path(trial_config_path).read_text(encoding="utf-8"))
    trial: TrialConfig = TrialConfig(**trial_data)

    if trial.index_type in GRAPH_INDEX_TYPES:
        raise RuntimeError(
            f"trial index_type {trial.index_type.value!r} requires a graph store. "
            "benchmark-evaluate does not build graphs in v1 — re-run with a "
            "vector_only or hybrid_bm25_vector trial config, or extend the runner."
        )

    qa_pairs = load_qa(Path(qa_path))
    if limit is not None:
        qa_pairs = qa_pairs[:limit]
    supporting_present = any(qa.supporting_doc_ids for qa in qa_pairs)

    manifest_path = Path(qa_path).parent / "metadata.json"
    if manifest_path.exists():
        manifest = BenchmarkManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    else:
        manifest = BenchmarkManifest(
            name="unknown",
            split="unknown",
            sample_size=len(qa_pairs),
            seed=0,
            adapter_version="unknown",
        )

    output_dir = Path(project.meta.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(project.meta.corpus_path)

    filenames, texts = _load_corpus(corpus_path)
    corpus_hash = _corpus_hash(corpus_path, project.parsing)

    reranker_desc = trial.reranker if trial.reranker and trial.reranker != "none" else "none"
    run_logger.info("=" * 60)
    run_logger.info("Benchmark: %s (%s split, %d questions)", manifest.name, manifest.split, len(qa_pairs))
    run_logger.info("Corpus:    %s (%d docs)", corpus_path, len(filenames))
    run_logger.info(
        "Config:    index=%s chunk=%d/%d embed=%s top_k=%d reranker=%s (top_n=%d)",
        trial.index_type.value,
        trial.chunk_token_size,
        trial.chunk_token_overlap,
        trial.embedding_model,
        trial.top_k,
        reranker_desc,
        trial.reranker_top_n,
    )
    stage_llms = [trial.generator_llm]
    if trial.compressor_llm:
        stage_llms.append(trial.compressor_llm)
    if trial.expander_llm:
        stage_llms.append(trial.expander_llm)
    run_logger.info(
        "Generator LLM: %s (temp=%.2f, reasoning=%s)", trial.generator_llm, trial.temperature, trial.reasoning
    )
    if trial.compressor_llm:
        run_logger.info("Compressor LLM: %s", trial.compressor_llm)
    if trial.expander_llm:
        run_logger.info("Expander LLM: %s", trial.expander_llm)
    run_logger.info("Judge:     %s", judge_model or "disabled")
    run_logger.info("Concurrency: %d", concurrency)
    run_logger.info("=" * 60)

    ingredient_cache = IngredientCache(
        cache_dir=output_dir / ".cache" / "ingredients",
        max_bytes=int(project.meta.cache_max_gb * 1024**3),
    )
    index_builder = IndexBuilder(cache=ingredient_cache)

    # Only start vLLM when the *selected* trial actually needs it; instantiating
    # VLLMServerManager validates the vllm binary exists on PATH.
    vllm_manager: VLLMServerManager | None = None
    if any(m.startswith("hosted_vllm/") for m in stage_llms):
        vllm_manager = VLLMServerManager(project.vllm, output_dir)

    t0 = time.monotonic()
    structural = trial.to_structural()
    emb_fp = structural.embeddings_fingerprint(corpus_hash)
    if ingredient_cache.has_embeddings(emb_fp):
        cache_state = "cache hit"
    elif ingredient_cache.has_chunks(structural.chunks_fingerprint(corpus_hash)):
        cache_state = "chunks cached, re-embedding"
    else:
        cache_state = "building from scratch"
    run_logger.info("Building index (%s)...", cache_state)
    index = await index_builder.build(texts, structural, corpus_hash=corpus_hash, doc_ids=filenames)
    run_logger.info("Index ready in %.2fs (%d chunks, %s)", time.monotonic() - t0, len(index.chunks), cache_state)
    run_logger.info("Evaluating %d questions...", len(qa_pairs))

    if vllm_manager:
        for m in stage_llms:
            if m.startswith("hosted_vllm/"):
                await vllm_manager.ensure_model(m)

    embedder = index_builder.get_embedder(trial.embedding_model)
    cross_encoder = (
        index_builder.get_cross_encoder(trial.reranker) if trial.reranker and trial.reranker != "none" else None
    )
    pipeline = RAGPipeline(
        vector_store=index.vector_store,
        graph_store=None,
        config=trial.to_runtime(reasoning_effort=project.search_space.reasoning_effort),
        embedder=embedder,
        index_type=trial.index_type,
        cross_encoder=cross_encoder,
    )

    evaluator = FreeFormEvaluator(concurrency=concurrency, judge_model=judge_model)
    per_question = await evaluator.evaluate(pipeline, qa_pairs)

    agg = _aggregate(
        per_question,
        supporting_present=supporting_present,
        judge_enabled=judge_model is not None,
    )

    result = BenchmarkResult(
        benchmark=manifest.name,
        n_total=len(qa_pairs),
        n_valid=agg["n_valid"],
        n_judge_invalid=agg["n_judge_invalid"],
        em=agg["em"],
        f1=agg["f1"],
        llm_judge_accuracy=agg["llm_judge_accuracy"],
        recall_at_1=agg["recall_at_1"],
        recall_at_2=agg["recall_at_2"],
        recall_at_5=agg["recall_at_5"],
        recall_at_10=agg["recall_at_10"],
        mrr=agg["mrr"],
        avg_retrieval_s=agg["avg_retrieval_s"],
        avg_generation_s=agg["avg_generation_s"],
        total_cost_usd=agg["total_cost_usd"],
        total_prompt_tokens=agg["total_prompt_tokens"],
        total_completion_tokens=agg["total_completion_tokens"],
        per_question=per_question,
        judge_model=judge_model,
        trial_config_hash=_config_hash(trial),
        project_config_hash=_config_hash(project),
        corpus_hash=corpus_hash,
        benchmark_manifest=manifest,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if vllm_manager:
        await vllm_manager.shutdown()

    run_logger.info("=" * 60)
    run_logger.info("Results (%d/%d valid questions)", result.n_valid, result.n_total)
    run_logger.info("  EM:           %.3f", result.em)
    run_logger.info("  F1:           %.3f", result.f1)
    if judge_model is not None and result.llm_judge_accuracy is None:
        run_logger.warning(
            "  Judge acc:    FAILED — all %d judge calls returned no YES/NO verdict. "
            "Check that %r is a valid model and credentials are configured (re-run with --verbose for details).",
            result.n_valid,
            judge_model,
        )
    elif result.llm_judge_accuracy is not None:
        judge_invalid = f" ({result.n_judge_invalid} parse failures)" if result.n_judge_invalid else ""
        run_logger.info("  Judge acc:    %.3f%s", result.llm_judge_accuracy, judge_invalid)
    if result.recall_at_1 is not None:
        run_logger.info(
            "  Recall@1/2/5/10: %.3f / %.3f / %.3f / %.3f",
            result.recall_at_1,
            result.recall_at_2 or 0.0,
            result.recall_at_5 or 0.0,
            result.recall_at_10 or 0.0,
        )
        run_logger.info("  MRR:          %.3f", result.mrr or 0.0)
    run_logger.info(
        "  Avg latency:  retrieve %.2fs / generate %.2fs",
        result.avg_retrieval_s,
        result.avg_generation_s,
    )
    run_logger.info(
        "  LLM cost:     $%.4f (prompt=%d, completion=%d tokens)",
        result.total_cost_usd,
        result.total_prompt_tokens,
        result.total_completion_tokens,
    )
    run_logger.info("Wrote %s", output_path)
    run_logger.info("=" * 60)
    return result
