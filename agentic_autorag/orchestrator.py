"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import MCQQuestion, ProjectConfig, StructuralConfig, TrialConfig
from agentic_autorag.engine.graph_store import LightRAGStore
from agentic_autorag.engine.index_builder import IndexBuilder, RAGIndex
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.engine.vllm_server import VLLMServerManager
from agentic_autorag.examiner.clustering import allocate_largest_remainder
from agentic_autorag.examiner.evaluator import ExamResult, MCQEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent
from agentic_autorag.examiner.exam_validator import run_validation_pipeline
from agentic_autorag.examiner.probe_selector import (
    rank_models_for_probes,
    score_questions_by_discrimination,
    select_exam,
    select_probe_configs,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent
from agentic_autorag.registry import IndexRegistry

logger = logging.getLogger(__name__)

# Files that are skipped during corpus loading.
_SKIP_FILENAMES = {"metadata.json"}
_DIRECT_READ_EXTENSIONS = {".md", ".txt"}

# Provider prefix → list of alternative auth methods.
# Each inner list is a set of env vars that together satisfy auth.
# The provider passes if ANY one alternative is fully present.
# The first alternative is shown to the user in error messages.
_PROVIDER_ENV_VARS: dict[str, list[list[str]]] = {
    "gemini": [["GEMINI_API_KEY"]],
    "openai": [["OPENAI_API_KEY"]],
    "anthropic": [["ANTHROPIC_API_KEY"]],
    "cohere": [["COHERE_API_KEY"]],
    "mistral": [["MISTRAL_API_KEY"]],
    "vertex_ai": [
        ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
    ],
    "bedrock": [
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        ["AWS_PROFILE", "AWS_REGION_NAME"],
        ["AWS_REGION_NAME"],
    ],
    "azure": [["AZURE_API_KEY", "AZURE_API_BASE"]],
    "azure_ai": [["AZURE_AI_API_KEY", "AZURE_AI_API_BASE"]],
}


def _check_api_keys(config: ProjectConfig) -> None:
    """Check that required API keys / env vars are set for all configured models.

    Each provider can have multiple alternative auth methods (e.g., Bedrock
    supports explicit keys, named profiles, or IAM roles). The check passes
    if ANY alternative is fully satisfied.

    Raises EnvironmentError with a clear message listing what's missing.
    """
    missing: list[tuple[str, list[str]]] = []

    models_to_check: list[str] = []
    models_to_check.extend(config.search_space.llm_models)
    models_to_check.append(config.agent.optimizer_model)
    models_to_check.append(config.agent.examiner_model)
    if config.graph is not None:
        models_to_check.append(config.graph.extraction_model)

    checked_prefixes: set[str] = set()

    for model_str in models_to_check:
        if "/" not in model_str:
            continue
        provider_prefix = model_str.split("/")[0]

        if provider_prefix in ("ollama", "sentence-transformers", "hosted_vllm"):
            continue
        if provider_prefix in checked_prefixes:
            continue
        if provider_prefix not in _PROVIDER_ENV_VARS:
            continue

        checked_prefixes.add(provider_prefix)
        auth_alternatives = _PROVIDER_ENV_VARS[provider_prefix]

        provider_ok = any(all(os.getenv(var) for var in required_set) for required_set in auth_alternatives)

        if not provider_ok:
            primary_set = auth_alternatives[0]
            missing_vars = [v for v in primary_set if not os.getenv(v)]
            missing.append((model_str, missing_vars))

    if missing:
        lines = ["Missing environment variables for configured models:"]
        for model_str, vars_list in missing:
            vars_str = ", ".join(vars_list)
            lines.append(f"  {model_str:<45} →  set {vars_str}")
        lines.append("")
        lines.append("See .env.example for all supported providers and auth methods.")
        raise OSError("\n".join(lines))


class Orchestrator:
    """Main optimization loop that ties all components together."""

    def __init__(self, config_path: str, debug_prompts: bool = False) -> None:
        self.config: ProjectConfig = load_config(config_path)
        _check_api_keys(self.config)
        meta = self.config.meta

        self.output_dir = Path(meta.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger(self.output_dir)

        self.history = HistoryLog(path=str(self.output_dir / "history.jsonl"))
        self.history.clear()

        try:
            self.knowledge_base: KnowledgeBase | None = KnowledgeBase()
        except Exception as e:
            logger.warning("Could not load knowledge base: %s. Agent will run without model context.", e)
            self.knowledge_base = None

        # Populate embedding token limits from KB for cross-field validation
        if self.knowledge_base:
            embed_models = self.knowledge_base._embeddings.get("models", {})
            for name in self.config.search_space.embedding_models:
                entry = embed_models.get(name)
                if entry and entry.get("max_tokens"):
                    self.config.embedding_token_limits[name] = int(entry["max_tokens"])

        self.agent = ReasoningAgent(
            agent_model=self.config.agent.optimizer_model,
            config=self.config,
            history=self.history,
            debug_prompts=debug_prompts,
            knowledge_base=self.knowledge_base,
        )
        self.evaluator = MCQEvaluator(
            concurrency=self.config.agent.concurrency,
            retrieval_quality_alpha=self.config.examiner.retrieval_quality_alpha,
        )

        parsing = self.config.parsing
        self.parser = build_parser(
            parsing.parser,
            ocr=parsing.ocr,
            table_structure=parsing.table_structure,
        )

        self.index_builder = IndexBuilder(
            db_path=str(self.output_dir / "lancedb"),
        )
        self.registry = IndexRegistry(str(self.output_dir / "indices")) if meta.index_registry else None

        # Graph store — only created when the config has a graph section
        self.graph_store: LightRAGStore | None = None
        if self.config.graph is not None:
            self.graph_store = LightRAGStore(
                working_dir=self.output_dir / "lightrag",
                build_config=self.config.graph,
            )

        # vLLM server — auto-managed when hosted_vllm/ models are in the search space
        has_vllm_models = any(m.startswith("hosted_vllm/") for m in self.config.search_space.llm_models)
        self.vllm_manager: VLLMServerManager | None = None
        if has_vllm_models:
            self.vllm_manager = VLLMServerManager(self.config.vllm, self.output_dir)

    @staticmethod
    def _truncate_list(items: list[str], limit: int = 5) -> str:
        """Join items with commas, adding '... +N more' when truncated."""
        if len(items) <= limit:
            return ", ".join(items)
        shown = ", ".join(items[:limit])
        return f"{shown} (+{len(items) - limit} more)"

    def _log_config_overview(self) -> None:
        """Log a summary of the project config and search space at startup."""
        meta = self.config.meta
        ss = self.config.search_space
        examiner = self.config.examiner
        agent = self.config.agent

        self.logger.info(
            "Project: %s | max_trials=%d | exam_size=%d",
            meta.project_name,
            meta.max_trials,
            examiner.exam_size,
        )
        self.logger.info("Optimizer model: %s", agent.optimizer_model)
        self.logger.info("Examiner model: %s", agent.examiner_model)
        self.logger.info(
            "Search space: %d LLM(s), %d embedding(s), %d reranker(s), %d index type(s)",
            len(ss.llm_models),
            len(ss.embedding_models),
            len(ss.reranker.models),
            len(ss.index_types),
        )
        self.logger.info("  LLMs: %s", self._truncate_list(ss.llm_models))
        self.logger.info("  Embeddings: %s", self._truncate_list(ss.embedding_models))
        self.logger.info("  Rerankers: %s", self._truncate_list(ss.reranker.models))
        self.logger.info("  Index types: %s", self._truncate_list([it.value for it in ss.index_types]))
        self.logger.info(
            "  Chunking: %s | size %d-%d | overlap %d-%d",
            self._truncate_list(ss.chunking.strategies),
            ss.chunking.chunk_token_size.min,
            ss.chunking.chunk_token_size.max,
            ss.chunking.chunk_token_overlap.min,
            ss.chunking.chunk_token_overlap.max,
        )

    async def run(self) -> TrialRecord:
        """Run the full optimization loop and return the best trial."""
        t_start = time.monotonic()
        meta = self.config.meta
        self._log_config_overview()

        # 1. Parse corpus
        self.logger.info("Loading corpus from %s", meta.corpus_path)
        t0 = time.monotonic()
        documents = self._load_and_parse_corpus()
        self.logger.info("Loaded %d document(s) in %.2fs", len(documents), time.monotonic() - t0)
        if not documents:
            raise RuntimeError(f"No documents found in {meta.corpus_path}")

        # 2. Build graph index (once, if graph is configured)
        if self.graph_store is not None:
            self.logger.info("Initialising LightRAG graph store")
            t0 = time.monotonic()
            await self.graph_store.initialize()
            if not self.graph_store.is_built():
                self.logger.info("Building LightRAG knowledge graph (this runs once and is cached)")
                await self.graph_store.build(documents)
                self.logger.info("Graph build complete in %.2fs", time.monotonic() - t0)
            else:
                self.logger.info("Loaded existing LightRAG graph in %.2fs", time.monotonic() - t0)

        # 3. Generate exam (or load from cache)
        self.logger.info("Generating/loading MCQ exam")
        t0 = time.monotonic()
        exam, from_cache = await self._generate_exam(
            documents,
            knowledge_base=self.knowledge_base,
            optimizer_model=self.config.agent.optimizer_model,
        )
        self._save_exam(exam)
        if from_cache:
            self.logger.info("Loaded %d questions in %.2fs", len(exam), time.monotonic() - t0)
        else:
            self.logger.info("Generated %d questions in %.2fs", len(exam), time.monotonic() - t0)
        self.logger.info("Saved exam to %s", self.output_dir / "exam.json")

        # 4. Agent proposes initial config
        self.logger.info("Agent proposing initial configuration")
        t0 = time.monotonic()
        current_config = await self.agent.propose_initial(
            corpus_description=meta.corpus_description,
        )
        self.logger.info("Initial config received in %.2fs", time.monotonic() - t0)
        self._log_config_summary("Initial config", current_config)

        # 5. Optimization loop
        best: TrialRecord | None = None
        pipeline: RAGPipeline | None = None
        for trial_num in range(1, meta.max_trials + 1):
            trial_start = time.monotonic()
            self.logger.info("%s", "=" * 60)
            self.logger.info("TRIAL %d/%d", trial_num, meta.max_trials)
            self.logger.info("%s", "=" * 60)

            # a. Build or load index
            fingerprint = current_config.structural_fingerprint()
            index_elapsed = 0.0
            index_source = "build"

            try:
                t0 = time.monotonic()
                loaded_from_cache = False
                if self.registry and self.registry.has(fingerprint):
                    try:
                        index = RAGIndex.load(self.registry.get(fingerprint))
                        index_source = "cache"
                        loaded_from_cache = True
                        self.logger.info("Loaded cached index %s", fingerprint)
                    except Exception:
                        self.logger.warning(
                            "Cached index %s is corrupted; rebuilding",
                            fingerprint,
                        )

                if not loaded_from_cache:
                    self.logger.info(
                        "Building index %s (embed=%s, chunk=%d, strategy=%s)",
                        fingerprint,
                        current_config.embedding_model,
                        current_config.chunk_token_size,
                        current_config.chunking_strategy,
                    )
                    index = await self.index_builder.build(
                        documents,
                        current_config.to_structural(),
                        embedding_token_limits=self.config.embedding_token_limits,
                    )
                    self.logger.info("Index built: %d chunks", len(index.chunks))
                    if self.registry:
                        staging = self.output_dir / ".index_staging" / fingerprint
                        if staging.exists():
                            shutil.rmtree(staging)
                        index.save(staging)
                        self.registry.register(fingerprint, staging, current_config.to_structural())
                        shutil.rmtree(staging)
                        self.logger.info("Registered index %s in cache", fingerprint)

                # Attach the graph store (already initialised at startup) regardless
                # of whether the vector index was cached or freshly built.
                index.graph_store = self.graph_store
                index_elapsed = time.monotonic() - t0
            except Exception:
                self.logger.exception("Index build/load failed for trial %d; skipping trial", trial_num)
                continue

            # b. Ensure vLLM is serving the right model (no-op if unchanged)
            if self.vllm_manager and current_config.llm_model.startswith("hosted_vllm/"):
                await self.vllm_manager.ensure_model(current_config.llm_model)

            # c. Construct pipeline
            if pipeline is not None:
                pipeline = None
            embedder = self.index_builder.get_embedder(current_config.embedding_model)
            cross_encoder = (
                self.index_builder.get_cross_encoder(current_config.reranker)
                if current_config.reranker and current_config.reranker != "none"
                else None
            )
            pipeline = RAGPipeline(
                vector_store=index.vector_store,
                graph_store=index.graph_store,
                config=current_config.to_runtime(
                    reasoning_effort=self.config.search_space.reasoning_effort,
                ),
                embedder=embedder,
                index_type=current_config.index_type,
                cross_encoder=cross_encoder,
            )

            # c. Evaluate
            self.logger.info("Evaluating %d questions", len(exam))
            t0 = time.monotonic()
            result: ExamResult = await self.evaluator.evaluate(pipeline, exam)
            score_elapsed = time.monotonic() - t0
            self.logger.info(
                "Score %.3f (mcq=%.3f, mrr=%.3f) (%d/%d) in %.2fs",
                result.score,
                result.mcq_accuracy,
                result.mean_retrieval_quality,
                result.n_correct,
                result.n_total,
                score_elapsed,
            )

            # d. Record trial
            error_trace = ""
            record = TrialRecord(
                trial_number=trial_num,
                config=current_config,
                score=result.score,
                error_trace=error_trace,
                question_results=result.question_results,
                mcq_accuracy=result.mcq_accuracy,
                mean_retrieval_quality=result.mean_retrieval_quality,
            )
            self.history.add(record)
            if best is None or result.score > best.score:
                best = record

            reasoning_elapsed = 0.0

            # e. Last trial — no need to propose next
            if trial_num == meta.max_trials:
                trial_elapsed = time.monotonic() - trial_start
                self.logger.info(
                    "Trial %d timings | index %.2fs (%s) | eval %.2fs | agent %.2fs | total %.2fs",
                    trial_num,
                    index_elapsed,
                    index_source,
                    score_elapsed,
                    reasoning_elapsed,
                    trial_elapsed,
                )
                break

            # f. Agent analyzes failures and proposes next config
            self.logger.info("Agent diagnosing and proposing next config")
            t0 = time.monotonic()
            error_trace, next_config = await self._propose_next_config_with_retries(result, current_config)
            reasoning_elapsed = time.monotonic() - t0
            record.error_trace = error_trace
            if error_trace:
                self.logger.debug("Error trace for trial %d:\n%s", trial_num, error_trace)
            self._log_config_diff(current_config, next_config)
            current_config = next_config

            trial_elapsed = time.monotonic() - trial_start
            self.logger.info(
                "Trial %d timings | index %.2fs (%s) | eval %.2fs | agent %.2fs | total %.2fs",
                trial_num,
                index_elapsed,
                index_source,
                score_elapsed,
                reasoning_elapsed,
                trial_elapsed,
            )

        # 6. Summary
        elapsed = time.monotonic() - t_start
        best = self.history.get_best()
        self._save_best_config(best)
        self.logger.info("Optimization complete in %.2fs", elapsed)
        if best:
            self.logger.info("Best score %.3f", best.score)
        else:
            self.logger.info("No successful trials completed")

        if self.graph_store is not None:
            await self.graph_store.close()
        if self.vllm_manager is not None:
            await self.vllm_manager.shutdown()

        return best

    def _corpus_cache_key(self) -> str:
        """Compute a deterministic cache key for the current corpus + parser."""
        corpus_path = Path(self.config.meta.corpus_path)
        parsing = self.config.parsing

        file_signatures: list[tuple[str, int, int]] = []
        for file_path in sorted(corpus_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            if file_path.name in _SKIP_FILENAMES:
                continue
            stat = file_path.stat()
            rel = str(file_path.relative_to(corpus_path))
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

    def _corpus_cache_path(self) -> Path:
        """Return the path to the corpus cache file."""
        cache_dir = self.output_dir / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"corpus_{self._corpus_cache_key()}.json"

    def _exam_cache_key(self) -> str:
        """Compute a deterministic cache key for the initial (pre-IRT) exam.

        Incorporates the corpus key so that corpus or parser changes invalidate the cache,
        and the examiner settings so that exam_size or cluster changes also invalidate it.
        """
        examiner = self.config.examiner
        key_data = json.dumps(
            {
                "corpus_key": self._corpus_cache_key(),
                "exam_size": examiner.exam_size,
                "initial_candidate_multiplier": examiner.initial_candidate_multiplier,
                "detect_parametric_leaks": examiner.detect_parametric_leaks,
                "source_fact_threshold": examiner.source_fact_threshold,
                "source_fact_substring_fallback": examiner.source_fact_substring_fallback,
                "source_fact_min_length": examiner.source_fact_min_length,
                "source_fact_window_chunk_size": examiner.source_fact_window_chunk_size,
                "source_fact_window_chunk_overlap": examiner.source_fact_window_chunk_overlap,
                "min_doc_words": examiner.min_doc_words,
                "parametric_leak_trials": examiner.parametric_leak_trials,
                "probe_selection": examiner.probe_selection,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _candidates_cache_key(self) -> str:
        """Compute a deterministic cache key for generated candidates.

        Includes corpus identity plus generation-only settings. Validation
        thresholds are intentionally excluded so validation can be re-run
        against the same generated candidates.
        """
        examiner = self.config.examiner
        key_data = json.dumps(
            {
                "corpus_key": self._corpus_cache_key(),
                "exam_size": examiner.exam_size,
                "initial_candidate_multiplier": examiner.initial_candidate_multiplier,
                "examiner_model": self.config.agent.examiner_model,
                "embedding_model": examiner.embedding_model,
                "min_doc_words": examiner.min_doc_words,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _load_and_parse_corpus(self) -> list[str]:
        """Recursively discover files in corpus_path and parse to text.

        Results are cached as JSON keyed by (parser, file paths + mtimes).
        """
        corpus_path = Path(self.config.meta.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

        cache_path = self._corpus_cache_path()
        if cache_path.exists():
            self.logger.info("Loading cached parsed corpus from %s", cache_path.name)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        # Collect eligible files first so we can show a progress bar.
        eligible: list[Path] = []
        for file_path in sorted(corpus_path.rglob("*")):
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
        for file_path in tqdm(eligible, desc="   Parsing files", unit="file"):
            suffix = file_path.suffix.lower()
            try:
                if suffix in _DIRECT_READ_EXTENSIONS:
                    text = file_path.read_text(encoding="utf-8")
                elif suffix in self.parser.supported_extensions():
                    text = self.parser.parse(file_path)
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
            self.logger.info("Skipped %d unsupported file(s)", skipped)
        if failed:
            self.logger.warning("Failed to parse %d file(s)", failed)

        try:
            cache_path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Cached parsed corpus to %s", cache_path.name)
        except Exception:
            self.logger.warning("Failed to write corpus cache", exc_info=True)

        return documents

    async def _generate_exam(
        self,
        documents: list[str],
        knowledge_base: KnowledgeBase | None = None,
        optimizer_model: str | None = None,
    ) -> tuple[list[MCQQuestion], bool]:
        """Generate and validate the frozen MCQ exam from the corpus.

        Uses an adaptive generation loop: generates an initial wave of
        candidates, validates them, and if fewer than ``exam_size`` survive,
        generates backfill waves targeting under-represented clusters until
        the target is reached or ``max_backfill_rounds`` is exhausted.

        Returns:
            (exam, from_cache) — the frozen exam and whether it was loaded from cache.
        """
        exam_path = self.output_dir / "exam.json"
        candidates_path = self.output_dir / "candidates.json"

        if exam_path.exists():
            self.logger.info("Loading existing exam from %s", exam_path.name)
            try:
                raw = json.loads(exam_path.read_text(encoding="utf-8"))
                exam = [MCQQuestion.model_validate(q) for q in raw]
                return exam, True
            except Exception:
                self.logger.warning("Existing exam file is invalid; regenerating", exc_info=True)

        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        doc_map = dict(zip(doc_ids, documents, strict=False))
        examiner = self.config.examiner
        exam_size = examiner.exam_size

        embedder = self.index_builder.get_embedder(examiner.embedding_model)

        exam_agent = ExamAgent(
            config=examiner,
            examiner_model=self.config.agent.examiner_model,
            embedding_model=embedder,
            corpus_description=self.config.meta.corpus_description,
            concurrency=self.config.agent.concurrency,
        )

        # Build weak retrieval index for difficulty filtering (if probes enabled)
        weak_index_chunks: list[str] | None = None
        weak_index_embeddings = None
        weak_index_embedder = None
        weak_index: RAGIndex | None = None
        weak_structural: StructuralConfig | None = None
        if examiner.probe_selection:
            ss = self.config.search_space
            ranked_llms = await rank_models_for_probes(ss.llm_models, "llm", knowledge_base, optimizer_model)
            ranked_embeds = await rank_models_for_probes(
                ss.embedding_models, "embedding", knowledge_base, optimizer_model
            )
            ranked_rerankers = await rank_models_for_probes(
                ss.reranker.models, "reranker", knowledge_base, optimizer_model
            )

            weak_embed = ranked_embeds[0]
            weak_chunk = int(ss.chunking.chunk_token_size.min)
            limit = self.config.embedding_token_limits.get(weak_embed)
            if limit and weak_chunk > limit:
                weak_chunk = limit
            weak_structural = StructuralConfig(
                chunking_strategy=ss.chunking.strategies[0],
                chunk_token_size=weak_chunk,
                chunk_token_overlap=max(0, weak_chunk // 10),
                embedding_model=weak_embed,
                index_type=ss.index_types[0],
            )
            self.logger.info("Building weak retrieval index (for difficulty filtering)...")
            weak_index = await self.index_builder.build(
                documents, weak_structural, embedding_token_limits=self.config.embedding_token_limits
            )
            weak_index_chunks = weak_index.chunks
            weak_index_embeddings = weak_index.embeddings
            weak_index_embedder = self.index_builder.get_embedder(weak_embed)

        # Load cached candidates or generate new ones
        all_candidates: list[MCQQuestion] | None = None
        if candidates_path.exists():
            try:
                raw_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
                all_candidates = [MCQQuestion.model_validate(q) for q in raw_candidates]
                self.logger.info("Loaded %d cached candidates from %s", len(all_candidates), candidates_path.name)
            except Exception:
                self.logger.warning("Cached candidates file is invalid; regenerating", exc_info=True)

        if all_candidates is None:
            # One-time corpus preparation (split, embed, cluster)
            corpus = exam_agent.prepare_corpus(documents, doc_ids)
            if not corpus.doc_texts:
                self.logger.warning("No documents after corpus preparation")
                return [], False

            desired_per_cluster = allocate_largest_remainder(corpus.cluster_sizes, exam_size)
            all_candidates = []
            max_rounds = 1 + examiner.max_backfill_rounds

            for wave in range(max_rounds):
                if len(all_candidates) >= exam_size:
                    break

                if wave == 0:
                    wave_size = int(exam_size * examiner.initial_candidate_multiplier)
                    cluster_deficits = None
                else:
                    survival_rate = max(len(all_candidates) / max(wave_size, 1), 0.15)
                    deficit = exam_size - len(all_candidates)
                    wave_size = min(math.ceil(deficit / survival_rate * 1.5), exam_size * 2)
                    generated_per_cluster = Counter(q.cluster_id for q in all_candidates)
                    cluster_deficits = {
                        cid: max(0, int(desired_per_cluster[cid]) - generated_per_cluster.get(cid, 0))
                        for cid in range(corpus.n_clusters)
                    }
                    self.logger.info(
                        "Backfill round %d: deficit=%d, wave_size=%d",
                        wave,
                        deficit,
                        wave_size,
                    )

                candidates = await exam_agent.generate_wave(
                    corpus,
                    wave_size,
                    exclude_questions=all_candidates,
                    cluster_deficits=cluster_deficits,
                )
                all_candidates.extend(candidates)

            # Assign readable sequential IDs to candidates
            width = len(str(len(all_candidates)))
            for i, q in enumerate(all_candidates, start=1):
                q.id = f"C{i:0{width}d}"

            # Save raw candidates (before any filtering) for cache
            try:
                candidates_json = json.dumps([q.model_dump(mode="json") for q in all_candidates], indent=2)
                candidates_path.write_text(candidates_json, encoding="utf-8")
                self.logger.info("Saved %d candidates to %s", len(all_candidates), candidates_path.name)
            except Exception:
                self.logger.warning("Failed to write candidates file", exc_info=True)

        # --- All filtering runs from here (both cached and fresh candidates) ---

        # Discriminator quality filter
        n_before_disc = len(all_candidates)
        all_candidates = exam_agent._filter_discriminator_quality(all_candidates, doc_map)
        n_removed_disc = n_before_disc - len(all_candidates)
        if n_removed_disc > 0:
            self.logger.info(
                "Discriminator quality filter: removed %d (%d remaining)", n_removed_disc, len(all_candidates)
            )

        # Validation pipeline (source_fact, retrieval difficulty, parametric leak, oracle)
        validated = await run_validation_pipeline(
            all_candidates,
            documents=doc_map,
            embedder=embedder,
            model=self.config.agent.examiner_model,
            concurrency=self.config.agent.concurrency,
            source_fact_threshold=examiner.source_fact_threshold,
            detect_parametric_leaks=examiner.detect_parametric_leaks,
            source_fact_substring_fallback=examiner.source_fact_substring_fallback,
            source_fact_min_length=examiner.source_fact_min_length,
            source_fact_window_chunk_size=examiner.source_fact_window_chunk_size,
            source_fact_window_chunk_overlap=examiner.source_fact_window_chunk_overlap,
            parametric_leak_trials=examiner.parametric_leak_trials,
            retrieval_filter_chunks=weak_index_chunks,
            retrieval_filter_embeddings=weak_index_embeddings,
            retrieval_filter_embedder=weak_index_embedder,
            retrieval_difficulty_top_k=examiner.retrieval_difficulty_top_k,
        )
        self.logger.info("Validation: %d/%d candidates passed", len(validated), len(all_candidates))

        if len(validated) < exam_size:
            self.logger.warning(
                "Exam has %d questions (target %d) after filtering",
                len(validated),
                exam_size,
            )

        exam = validated

        # Probe-based selection (optional)
        if examiner.probe_selection and len(exam) > exam_size:
            self.logger.info(
                "Running probe-based discrimination selection (%d candidates for %d slots)", len(exam), exam_size
            )

            labelled_probes = select_probe_configs(
                self.config,
                ranked_llms=ranked_llms,
                ranked_embeds=ranked_embeds,
                ranked_rerankers=ranked_rerankers,
            )
            probe_results: list[ExamResult] = []

            # In-memory index cache: reuse the weak filter index when a probe
            # has the same structural fingerprint (avoids redundant rebuild).
            exam_index_cache: dict[str, RAGIndex] = {}
            if weak_index is not None and weak_structural is not None:
                exam_index_cache[weak_structural.fingerprint()] = weak_index

            for i, (probe_label, probe_config) in enumerate(labelled_probes):
                self.logger.info(
                    "Probe %d/%d — %s | chunk=%d top_k=%d",
                    i + 1,
                    len(labelled_probes),
                    probe_label,
                    probe_config.chunk_token_size,
                    probe_config.top_k,
                )
                try:
                    probe_structural = probe_config.to_structural()
                    probe_fp = probe_structural.fingerprint()
                    if probe_fp in exam_index_cache:
                        probe_index = exam_index_cache[probe_fp]
                        self.logger.info("Reusing cached index %s", probe_fp)
                    else:
                        probe_index = await self.index_builder.build(
                            documents, probe_structural, embedding_token_limits=self.config.embedding_token_limits
                        )
                        exam_index_cache[probe_fp] = probe_index
                    probe_index.graph_store = self.graph_store if hasattr(self, "graph_store") else None
                    probe_embedder = self.index_builder.get_embedder(probe_config.embedding_model)
                    probe_cross_encoder = (
                        self.index_builder.get_cross_encoder(probe_config.reranker)
                        if probe_config.reranker and probe_config.reranker != "none"
                        else None
                    )
                    probe_pipeline = RAGPipeline(
                        vector_store=probe_index.vector_store,
                        graph_store=probe_index.graph_store,
                        config=probe_config.to_runtime(
                            reasoning_effort=self.config.search_space.reasoning_effort,
                        ),
                        embedder=probe_embedder,
                        index_type=probe_config.index_type,
                        cross_encoder=probe_cross_encoder,
                    )
                    result = await self.evaluator.evaluate(probe_pipeline, exam)
                    probe_results.append(result)
                    self.logger.info(
                        "Probe %d/%d result: %d/%d correct (%.0f%%) — %s",
                        i + 1,
                        len(labelled_probes),
                        result.n_correct,
                        result.n_total,
                        result.score * 100,
                        probe_label.split("(")[0].strip(),
                    )
                except Exception:
                    self.logger.exception("Probe %d (%s) failed; skipping", i + 1, probe_label)

            if probe_results:
                scores = score_questions_by_discrimination(probe_results, exam)
                n_zero = sum(1 for s in scores.values() if s == 0.0)
                # Identify all-wrong questions (all probes incorrect, no errors)
                question_ids = {q.id for q in exam}
                all_wrong_ids: set[str] = set()
                for qid in question_ids:
                    responses = []
                    evaluated_by_all = True
                    for result in probe_results:
                        result_map = {qr.question_id: qr.correct for qr in result.question_results}
                        if qid not in result_map:
                            evaluated_by_all = False
                            break
                        responses.append(result_map[qid])
                    if evaluated_by_all and responses and not any(responses):
                        all_wrong_ids.add(qid)
                self.logger.info(
                    "Discrimination scores: min=%.3f, max=%.3f, mean=%.3f, zero_scores=%d/%d, all_wrong=%d",
                    min(scores.values()),
                    max(scores.values()),
                    sum(scores.values()) / len(scores),
                    n_zero,
                    len(scores),
                    len(all_wrong_ids),
                )
                exam = select_exam(exam, scores, exam_size, all_wrong_ids=all_wrong_ids)
                self.logger.info("Probe selection: %d questions selected", len(exam))
            else:
                self.logger.warning("All probes failed; falling back to simple truncation")
                exam = exam[:exam_size]
        elif len(exam) > exam_size:
            exam = exam[:exam_size]
            self.logger.info("Truncated to exam_size=%d", exam_size)

        # Assign readable sequential IDs to final exam questions
        width = len(str(len(exam)))
        for i, q in enumerate(exam, start=1):
            q.id = f"Q{i:0{width}d}"

        try:
            exam_path.write_text(
                json.dumps([q.model_dump(mode="json") for q in exam], indent=2),
                encoding="utf-8",
            )
            self.logger.info("Saved exam to %s", exam_path.name)
        except Exception:
            self.logger.warning("Failed to write exam file", exc_info=True)

        return exam, False

    def _save_exam(self, exam: list[MCQQuestion]) -> None:
        """Persist the generated exam to JSON."""
        exam_path = self.output_dir / "exam.json"
        data = [q.model_dump(mode="json") for q in exam]
        exam_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_best_config(self, best: TrialRecord | None) -> None:
        """Persist the best trial configuration as YAML."""
        if best is None:
            return
        best_path = self.output_dir / "best_config.yaml"
        payload = best.config.model_dump(mode="json")
        best_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.logger.info("Saved best config to %s", best_path)

    async def _propose_next_config_with_retries(
        self,
        result: ExamResult,
        current_config: TrialConfig,
    ) -> tuple[str, TrialConfig]:
        """Call the agent up to 5 times; reuse previous config on failure."""
        for attempt in range(1, 6):
            try:
                error_trace, next_config = await self.agent.analyze_and_propose(
                    result,
                    current_config,
                )
                return error_trace, next_config
            except Exception:
                self.logger.exception("Agent proposal attempt %d/5 failed", attempt)
        self.logger.error("Agent failed after 5 retries; reusing previous config")
        return "", current_config

    @staticmethod
    def _setup_logger(output_dir: Path) -> logging.Logger:
        """Configure a run logger with console and file handlers."""
        run_logger = logging.getLogger("agentic_autorag.run")
        run_logger.setLevel(logging.DEBUG)
        run_logger.propagate = False

        for handler in list(run_logger.handlers):
            run_logger.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        run_logger.addHandler(console_handler)
        run_logger.addHandler(file_handler)
        return run_logger

    def _random_tweak(self, config: TrialConfig) -> TrialConfig:
        """Apply one random parameter change as a fallback."""
        data = config.model_dump()
        ss = self.config.search_space

        param = random.choice(["top_k", "temperature", "hybrid_alpha"])
        if param == "top_k":
            data["top_k"] = random.randint(
                int(ss.top_k.min),
                int(ss.top_k.max),
            )
        elif param == "temperature":
            data["temperature"] = round(
                random.uniform(ss.temperature.min, ss.temperature.max),
                2,
            )
        elif param == "hybrid_alpha":
            data["hybrid_alpha"] = round(
                random.uniform(ss.hybrid_alpha.min, ss.hybrid_alpha.max),
                2,
            )

        return TrialConfig.model_validate(data)

    @staticmethod
    def _print_trial_header(trial_num: int, max_trials: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"  TRIAL {trial_num}/{max_trials}")
        print(f"{'=' * 60}")

    def _log_config_summary(self, label: str, config: TrialConfig) -> None:
        reasoning_tag = " +reasoning" if config.reasoning else ""
        self.logger.info(
            "%s | chunk=%s strategy=%s embed=%s index=%s top_k=%s reranker=%s llm=%s%s temp=%s",
            label,
            config.chunk_token_size,
            config.chunking_strategy,
            config.embedding_model,
            config.index_type.value,
            config.top_k,
            config.reranker,
            config.llm_model,
            reasoning_tag,
            config.temperature,
        )

    @staticmethod
    def _print_config_summary(label: str, config: TrialConfig) -> None:
        reasoning_tag = " +reasoning" if config.reasoning else ""
        print(f"   {label}:")
        print(
            f"     chunk={config.chunk_token_size}, strategy={config.chunking_strategy}, embed={config.embedding_model}"
        )
        print(f"     index={config.index_type.value}, top_k={config.top_k}, reranker={config.reranker}")
        print(f"     llm={config.llm_model}{reasoning_tag}, temp={config.temperature}")

    @staticmethod
    def _print_config_diff(old: TrialConfig, new: TrialConfig) -> None:
        """Print which key parameters changed between configs."""
        changes: list[str] = []
        pairs = [
            ("chunk_token_size", old.chunk_token_size, new.chunk_token_size),
            ("chunk_token_overlap", old.chunk_token_overlap, new.chunk_token_overlap),
            ("chunking_strategy", old.chunking_strategy, new.chunking_strategy),
            ("embedding_model", old.embedding_model, new.embedding_model),
            ("index_type", old.index_type.value, new.index_type.value),
            ("top_k", old.top_k, new.top_k),
            ("hybrid_alpha", old.hybrid_alpha, new.hybrid_alpha),
            ("reranker", old.reranker, new.reranker),
            ("llm_model", old.llm_model, new.llm_model),
            ("temperature", old.temperature, new.temperature),
            ("reasoning", old.reasoning, new.reasoning),
            ("query_expansion", old.query_expansion, new.query_expansion),
        ]
        for name, old_val, new_val in pairs:
            if old_val != new_val:
                changes.append(f"     {name}: {old_val} → {new_val}")

        if changes:
            print("   Config changes:")
            for line in changes:
                print(line)
        else:
            print("   Config: no changes")

    def _log_config_diff(self, old: TrialConfig, new: TrialConfig) -> None:
        changes: list[str] = []
        pairs = [
            ("chunk_token_size", old.chunk_token_size, new.chunk_token_size),
            ("chunk_token_overlap", old.chunk_token_overlap, new.chunk_token_overlap),
            ("chunking_strategy", old.chunking_strategy, new.chunking_strategy),
            ("embedding_model", old.embedding_model, new.embedding_model),
            ("index_type", old.index_type.value, new.index_type.value),
            ("top_k", old.top_k, new.top_k),
            ("hybrid_alpha", old.hybrid_alpha, new.hybrid_alpha),
            ("reranker", old.reranker, new.reranker),
            ("llm_model", old.llm_model, new.llm_model),
            ("temperature", old.temperature, new.temperature),
            ("reasoning", old.reasoning, new.reasoning),
            ("query_expansion", old.query_expansion, new.query_expansion),
        ]
        for name, old_val, new_val in pairs:
            if old_val != new_val:
                changes.append(f"{name}: {old_val} -> {new_val}")

        if changes:
            self.logger.info("Config changes: %s", "; ".join(changes))
            self.logger.debug("Full config diff details: %s", changes)
        else:
            self.logger.info("Config: no changes")

    @staticmethod
    def _print_summary(
        best: TrialRecord | None,
        total_trials: int,
        elapsed: float,
    ) -> None:
        print(f"\n{'=' * 60}")
        print("  OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total trials:  {total_trials}")
        print(f"  Time elapsed:  {elapsed:.1f}s")
        if best:
            print(f"  Best score:    {best.score:.3f}")
            print(f"  Best config:   {best.summary()}")
        else:
            print("  No trials completed.")
        print(f"{'=' * 60}\n")
