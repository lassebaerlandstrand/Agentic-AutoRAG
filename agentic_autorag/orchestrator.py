"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import MCQQuestion, ProjectConfig, StructuralConfig, TrialConfig
from agentic_autorag.engine.graph_store import LightRAGStore
from agentic_autorag.engine.index_builder import IndexBuilder, IngredientCache, RAGIndex
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

    def __init__(
        self,
        config_path: str,
        debug_prompts: bool = False,
        output_dir_override: str | None = None,
    ) -> None:
        self.config: ProjectConfig = load_config(config_path)
        _check_api_keys(self.config)
        meta = self.config.meta

        # Cache dir: always meta.output_dir from the YAML — the shared root for
        # parsed-corpus cache, exam.json, ingredient cache, and graph store.
        # Multiple baseline drivers can point at the same cache_dir to reuse
        # all of these without rebuilding.
        self._cache_dir = Path(meta.output_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Output dir: per-run target for history.jsonl, run.log, best_config.yaml.
        # Baselines pass output_dir_override to keep their per-run artifacts out
        # of the agentic optimize run's directory while still sharing the cache.
        self.output_dir = Path(output_dir_override) if output_dir_override else self._cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger(self.output_dir)

        # NOTE: history.clear() is moved into run() so constructing an
        # Orchestrator (e.g. for baseline drivers that ignore self.history)
        # doesn't wipe a sibling agentic run's history.jsonl.
        self.history = HistoryLog(path=str(self.output_dir / "history.jsonl"))

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
            chunk_relevance_min_overlap_chars=self.config.examiner.chunk_relevance_min_overlap_chars,
            chunk_relevance_ngram_size=self.config.examiner.chunk_relevance_ngram_size,
            chunk_relevance_overlap_threshold=self.config.examiner.chunk_relevance_overlap_threshold,
            chunk_relevance_min_run=self.config.examiner.chunk_relevance_min_run,
        )

        parsing = self.config.parsing
        self.parser = build_parser(
            parsing.parser,
            ocr=parsing.ocr,
            table_structure=parsing.table_structure,
        )

        self.ingredient_cache = IngredientCache(
            cache_dir=self.cache_dir / ".cache" / "ingredients",
            max_bytes=int(meta.cache_max_gb * 1024**3),
        )
        self.index_builder = IndexBuilder(cache=self.ingredient_cache)

        # Graph store — only created when the config has a graph section
        self.graph_store: LightRAGStore | None = None
        if self.config.graph is not None:
            self.graph_store = LightRAGStore(
                working_dir=self.cache_dir / "lightrag",
                build_config=self.config.graph,
            )

        # vLLM server — auto-managed when any hosted_vllm/ model appears either in
        # the search space (used at trial time) or as the graph extraction model
        # (used once, during graph build).
        has_vllm_in_search = any(m.startswith("hosted_vllm/") for m in self.config.search_space.llm_models)
        has_vllm_in_graph = self.config.graph is not None and self.config.graph.extraction_model.startswith(
            "hosted_vllm/"
        )
        self.vllm_manager: VLLMServerManager | None = None
        if has_vllm_in_search or has_vllm_in_graph:
            self.vllm_manager = VLLMServerManager(self.config.vllm, self.output_dir)

        # Setup state — populated lazily by setup(), reused across evaluate_trial() calls.
        # Lets baseline drivers reuse the same parsed corpus, graph, and exam without
        # rebuilding everything from scratch.
        self._setup_done: bool = False
        self._documents: list[str] | None = None
        self._doc_ids: list[str] | None = None
        self._exam: list[MCQQuestion] | None = None

    @property
    def cache_dir(self) -> Path:
        """Shared cache root. Falls back to ``output_dir`` for tests that bypass ``__init__``."""
        return getattr(self, "_cache_dir", None) or self.output_dir

    @property
    def documents(self) -> list[str]:
        if self._documents is None:
            raise RuntimeError("Orchestrator.setup() must be called before accessing documents")
        return self._documents

    @property
    def doc_ids(self) -> list[str]:
        if self._doc_ids is None:
            raise RuntimeError("Orchestrator.setup() must be called before accessing doc_ids")
        return self._doc_ids

    @property
    def exam(self) -> list[MCQQuestion]:
        if self._exam is None:
            raise RuntimeError("Orchestrator.setup() must be called before accessing exam")
        return self._exam

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

    async def setup(self) -> None:
        """Idempotent: parse corpus, build graph (once), generate exam (or load).

        Populates ``self._documents``, ``self._doc_ids``, ``self._exam`` as instance
        state so subsequent calls to ``evaluate_trial`` and the agent loop can reuse
        them. Safe to call multiple times — second and later calls are no-ops.

        Baseline drivers call this before their proposal loop so the corpus, graph,
        and exam are shared with the agentic ``run()`` path.
        """
        if self._setup_done:
            return

        meta = self.config.meta
        self._log_config_overview()

        # 1. Parse corpus
        self.logger.info("Loading corpus from %s", meta.corpus_path)
        t0 = time.monotonic()
        parsed = self._load_and_parse_corpus()
        self.logger.info("Loaded %d document(s) in %.2fs", len(parsed), time.monotonic() - t0)
        if not parsed:
            raise RuntimeError(f"No documents found in {meta.corpus_path}")
        filenames = [name for name, _ in parsed]
        documents = [text for _, text in parsed]

        # Expose the doc-id → text map to the evaluator so its deterministic
        # chunk-relevance matcher can look up offsets for verbatim graph chunks.
        self.evaluator.documents = dict(zip(filenames, documents, strict=True))

        # 2. Build graph index (once, if graph is configured)
        if self.graph_store is not None:
            self.logger.info("Initialising LightRAG graph store")
            t0 = time.monotonic()
            corpus_hash = self._corpus_cache_key()
            # Check cache state BEFORE touching vLLM — a cached graph skips the
            # build entirely and never calls extraction_model, so there's no
            # reason to spin up the server.
            graph_already_built = self.graph_store.is_built(corpus_hash)
            extraction_model = self.config.graph.extraction_model  # type: ignore[union-attr]
            if not graph_already_built and self.vllm_manager and extraction_model.startswith("hosted_vllm/"):
                self.logger.info("Starting vLLM for graph extraction model: %s", extraction_model)
                await self.vllm_manager.ensure_model(extraction_model)
            await self.graph_store.initialize(corpus_hash)
            if graph_already_built:
                self.logger.info("Loaded existing LightRAG graph in %.2fs", time.monotonic() - t0)
            else:
                self.logger.info("Building LightRAG knowledge graph (resumable, cached across runs)")
                await self.graph_store.build(documents, corpus_hash)
                self.logger.info("Graph build complete in %.2fs", time.monotonic() - t0)

        # 3. Generate exam (or load from cache)
        self.logger.info("Generating/loading MCQ exam")
        t0 = time.monotonic()
        exam, from_cache = await self._generate_exam(
            documents,
            doc_ids=filenames,
            knowledge_base=self.knowledge_base,
            optimizer_model=self.config.agent.optimizer_model,
        )
        self._save_exam(exam)
        if from_cache:
            self.logger.info("Loaded %d questions in %.2fs", len(exam), time.monotonic() - t0)
        else:
            self.logger.info("Generated %d questions in %.2fs", len(exam), time.monotonic() - t0)
        self.logger.info("Saved exam to %s", self.cache_dir / "exam.json")

        self._documents = documents
        self._doc_ids = filenames
        self._exam = exam
        self._setup_done = True

    async def evaluate_trial(self, trial_config: TrialConfig) -> ExamResult:
        """Build/load index → ensure vLLM → run pipeline → score the MCQ exam.

        Requires ``setup()`` to have been called. Returns the ExamResult exactly as
        ``MCQEvaluator.evaluate`` produces it. Logs the same per-trial diagnostic
        lines the agentic loop has always emitted (index source, score, etc.).
        """
        if not self._setup_done:
            raise RuntimeError("Orchestrator.setup() must be called before evaluate_trial()")
        documents = self._documents
        doc_ids = self._doc_ids
        exam = self._exam
        assert documents is not None and doc_ids is not None and exam is not None

        # a. Build or load index (ingredient caching is internal to IndexBuilder)
        structural = trial_config.to_structural()
        corpus_hash = self._corpus_cache_key()
        chunks_fp = structural.chunks_fingerprint(corpus_hash)
        emb_fp = structural.embeddings_fingerprint(corpus_hash)

        t0 = time.monotonic()
        if self.ingredient_cache.has_embeddings(emb_fp):
            index_source = "cache hit (full)"
        elif self.ingredient_cache.has_chunks(chunks_fp):
            index_source = "cache hit (chunks, re-embed)"
        else:
            index_source = "build"
        self.logger.info(
            "Index %s %s (embed=%s, chunk=%d, overlap=%d, strategy=%s)",
            emb_fp,
            index_source,
            trial_config.embedding_model,
            trial_config.chunk_token_size,
            trial_config.chunk_token_overlap,
            trial_config.chunking_strategy,
        )
        if index_source == "cache hit (full)":
            self.logger.info("Rebuilding in-memory LanceDB table from cached ingredients...")
        index = await self.index_builder.build(
            documents,
            structural,
            corpus_hash=corpus_hash,
            doc_ids=doc_ids,
        )
        index.graph_store = self.graph_store
        index_elapsed = time.monotonic() - t0
        self.logger.info(
            "Index ready in %.2fs (%d chunks, %s)",
            index_elapsed,
            len(index.chunks),
            index_source,
        )

        # b. Ensure vLLM is serving the right model (no-op if unchanged)
        if self.vllm_manager and trial_config.llm_model.startswith("hosted_vllm/"):
            await self.vllm_manager.ensure_model(trial_config.llm_model)

        # c. Construct pipeline
        embedder = self.index_builder.get_embedder(trial_config.embedding_model)
        cross_encoder = (
            self.index_builder.get_cross_encoder(trial_config.reranker)
            if trial_config.reranker and trial_config.reranker != "none"
            else None
        )
        pipeline = RAGPipeline(
            vector_store=index.vector_store,
            graph_store=index.graph_store,
            config=trial_config.to_runtime(
                reasoning_effort=self.config.search_space.reasoning_effort,
            ),
            embedder=embedder,
            index_type=trial_config.index_type,
            cross_encoder=cross_encoder,
        )

        # d. Evaluate
        self.logger.info("Evaluating %d questions", len(exam))
        t0 = time.monotonic()
        result: ExamResult = await self.evaluator.evaluate(pipeline, exam)
        score_elapsed = time.monotonic() - t0
        valid_suffix = f" of {result.n_total}" if result.n_valid != result.n_total else ""
        self.logger.info(
            "Score %.3f (mcq=%.3f, rq=%.3f) (%d/%d%s) in %.2fs",
            result.score,
            result.mcq_accuracy,
            result.mean_retrieval_quality,
            result.n_correct,
            result.n_valid,
            valid_suffix,
            score_elapsed,
        )
        return result

    async def cleanup(self) -> None:
        """Release resources (graph store, vLLM). Safe to call multiple times."""
        if self.graph_store is not None:
            await self.graph_store.close()
        if self.vllm_manager is not None:
            await self.vllm_manager.shutdown()

    async def run(self) -> TrialRecord:
        """Run the full optimization loop and return the best trial."""
        t_start = time.monotonic()
        meta = self.config.meta

        # Fresh history.jsonl for each agentic run. Baseline drivers manage their
        # own HistoryLog and never touch this one.
        self.history.clear()

        await self.setup()
        exam = self.exam

        # Agent proposes initial config
        self.logger.info("Agent proposing initial configuration")
        t0 = time.monotonic()
        current_config = await self.agent.propose_initial(
            corpus_description=meta.corpus_description,
        )
        self.logger.info("Initial config received in %.2fs", time.monotonic() - t0)

        # Optimization loop
        best: TrialRecord | None = None
        for trial_num in range(1, meta.max_trials + 1):
            trial_start = time.monotonic()
            self.logger.info("%s", "=" * 60)
            self.logger.info("TRIAL %d/%d", trial_num, meta.max_trials)
            self.logger.info("%s", "=" * 60)
            self._log_config_summary("Config", current_config)

            try:
                result = await self.evaluate_trial(current_config)
            except Exception:
                self.logger.exception("Trial %d evaluation failed; skipping trial", trial_num)
                continue

            # Agent analyzes failures and proposes next config.
            # Must happen BEFORE history.add(), which clears context/response
            # fields in-place to save RAM (shared object references).
            reasoning_elapsed = 0.0
            trial_config = current_config
            stage_metrics = None
            diagnosis = None
            proposal_meta = None
            if trial_num < meta.max_trials:
                self.logger.info("Agent diagnosing and proposing next config")
                t0 = time.monotonic()
                stage_metrics, diagnosis, next_config, proposal_meta = await self._propose_next_config_with_retries(
                    result,
                    exam,
                    current_config,
                    trial_number=trial_num,
                    trials_remaining=meta.max_trials - trial_num,
                )
                reasoning_elapsed = time.monotonic() - t0
                self._log_config_diff(current_config, next_config)
                current_config = next_config

            # Record trial (mutates question_results to free RAM)
            record = TrialRecord(
                trial_number=trial_num,
                config=trial_config,
                score=result.score,
                question_results=result.question_results,
                mcq_accuracy=result.mcq_accuracy,
                mean_retrieval_quality=result.mean_retrieval_quality,
                stage_metrics=stage_metrics,
                diagnosis=diagnosis,
                meta=proposal_meta,
            )
            self.history.add(record)
            if best is None or result.score > best.score:
                best = record

            trial_elapsed = time.monotonic() - trial_start
            self.logger.info(
                "Trial %d total %.2fs | agent %.2fs",
                trial_num,
                trial_elapsed,
                reasoning_elapsed,
            )

            if trial_num == meta.max_trials:
                break

        # Summary
        elapsed = time.monotonic() - t_start
        best = self.history.get_best()
        self._save_best_config(best)
        self.logger.info("Optimization complete in %.2fs", elapsed)
        if best:
            self.logger.info("Best score %.3f", best.score)
        else:
            self.logger.info("No successful trials completed")

        await self.cleanup()

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
        """Return the path to the corpus cache file (under shared cache_dir)."""
        cache_dir = self.cache_dir / ".cache"
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
                "source_fact_min_length": examiner.source_fact_min_length,
                "source_fact_verify_fuzzy_threshold": examiner.source_fact_verify_fuzzy_threshold,
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

    def _load_and_parse_corpus(self) -> list[tuple[str, str]]:
        """Recursively discover files in corpus_path and parse to text.

        Returns a list of ``(filename, text)`` tuples. ``filename`` is the
        file's basename (e.g. ``healthcare_0038629.pdf``) and is later used
        as the source document id in generated exam questions.

        Results are cached as JSON keyed by (parser, file paths + mtimes).
        """
        corpus_path = Path(self.config.meta.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

        cache_path = self._corpus_cache_path()
        if cache_path.exists():
            self.logger.info("Loading cached parsed corpus from %s", cache_path.name)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and all(isinstance(entry, list) and len(entry) == 2 for entry in cached):
                return [(name, text) for name, text in cached]
            self.logger.info("Cached corpus has legacy format; re-parsing")

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

        documents: list[tuple[str, str]] = []
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
                    documents.append((file_path.name, text))
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
        doc_ids: list[str],
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
        exam_path = self.cache_dir / "exam.json"
        candidates_path = self.cache_dir / "candidates.json"

        if exam_path.exists():
            self.logger.info("Loading existing exam from %s", exam_path.name)
            try:
                raw = json.loads(exam_path.read_text(encoding="utf-8"))
                exam = [MCQQuestion.model_validate(q) for q in raw]
                return exam, True
            except Exception:
                self.logger.warning("Existing exam file is invalid; regenerating", exc_info=True)

        if len(doc_ids) != len(documents):
            raise ValueError(f"doc_ids length ({len(doc_ids)}) does not match documents length ({len(documents)})")
        duplicates = [name for name, count in Counter(doc_ids).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate document filenames in corpus: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
            )
        doc_map = dict(zip(doc_ids, documents, strict=True))
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
        weak_index_chunk_ranges: list[tuple[int, int]] | None = None
        weak_index_chunk_doc_ids: list[str] | None = None
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
                documents,
                weak_structural,
                corpus_hash=self._corpus_cache_key(),
                doc_ids=doc_ids,
            )
            weak_index_chunks = weak_index.chunks
            weak_index_chunk_ranges = weak_index.chunk_char_ranges
            weak_index_chunk_doc_ids = weak_index.chunk_doc_ids
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

        # Validation pipeline (source_fact verify-and-locate, retrieval difficulty, parametric leak, oracle)
        validated = await run_validation_pipeline(
            all_candidates,
            documents=doc_map,
            model=self.config.agent.examiner_model,
            concurrency=self.config.agent.concurrency,
            detect_parametric_leaks=examiner.detect_parametric_leaks,
            source_fact_min_length=examiner.source_fact_min_length,
            source_fact_verify_fuzzy_threshold=examiner.source_fact_verify_fuzzy_threshold,
            parametric_leak_trials=examiner.parametric_leak_trials,
            retrieval_filter_chunks=weak_index_chunks,
            retrieval_filter_chunk_ranges=weak_index_chunk_ranges,
            retrieval_filter_chunk_doc_ids=weak_index_chunk_doc_ids,
            retrieval_filter_embeddings=weak_index_embeddings,
            retrieval_filter_embedder=weak_index_embedder,
            retrieval_difficulty_top_k=examiner.retrieval_difficulty_top_k,
            chunk_relevance_min_overlap_chars=examiner.chunk_relevance_min_overlap_chars,
            chunk_relevance_ngram_size=examiner.chunk_relevance_ngram_size,
            chunk_relevance_overlap_threshold=examiner.chunk_relevance_overlap_threshold,
            chunk_relevance_min_run=examiner.chunk_relevance_min_run,
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
                            documents,
                            probe_structural,
                            corpus_hash=self._corpus_cache_key(),
                            doc_ids=doc_ids,
                        )
                        exam_index_cache[probe_fp] = probe_index
                    probe_index.graph_store = self.graph_store
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
                    valid_suffix = f" of {result.n_total}" if result.n_valid != result.n_total else ""
                    self.logger.info(
                        "Probe %d/%d %s: score=%.3f (mcq=%.3f, rq=%.3f) (%d/%d%s)",
                        i + 1,
                        len(labelled_probes),
                        probe_label.split("(")[0].strip(),
                        result.score,
                        result.mcq_accuracy,
                        result.mean_retrieval_quality,
                        result.n_correct,
                        result.n_valid,
                        valid_suffix,
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
        """Persist the generated exam to JSON in the shared cache_dir."""
        exam_path = self.cache_dir / "exam.json"
        data = [q.model_dump(mode="json") for q in exam]
        exam_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_best_config(self, best: TrialRecord | None) -> None:
        """Persist the best trial configuration as YAML."""
        if best is None:
            return
        best_path = self.output_dir / "best_config.yaml"
        payload = best.config.to_prompt_dump(include_graph=self.config.uses_graph())
        best_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.logger.info("Saved best config to %s", best_path)

    async def _propose_next_config_with_retries(
        self,
        result: ExamResult,
        exam: list[MCQQuestion],
        current_config: TrialConfig,
        *,
        trial_number: int,
        trials_remaining: int,
    ) -> tuple:
        """Call the agent up to 5 times; reuse previous config on failure.

        Returns ``(stage_metrics, diagnosis, next_config, proposal_meta)``.
        On persistent failure, returns ``(None, None, current_config, None)``
        so the loop can still progress with the previous config.
        """
        for attempt in range(1, 6):
            try:
                return await self.agent.analyze_and_propose(
                    result,
                    exam,
                    current_config,
                    trial_number=trial_number,
                    trials_remaining=trials_remaining,
                )
            except Exception:
                self.logger.exception("Agent proposal attempt %d/5 failed", attempt)
        self.logger.error("Agent failed after 5 retries; reusing previous config")
        return None, None, current_config, None

    @staticmethod
    def _setup_logger(output_dir: Path) -> logging.Logger:
        """Configure a run logger with console and file handlers."""
        run_logger = logging.getLogger("agentic_autorag.run")
        run_logger.setLevel(logging.DEBUG)
        run_logger.propagate = False

        for handler in list(run_logger.handlers):
            run_logger.removeHandler(handler)

        formatter = logging.Formatter("%(message)s")

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
    def _diff_pairs(old: TrialConfig, new: TrialConfig) -> list[tuple[str, object, object]]:
        """All config lever pairs the optimizer can change, for diff reporting.

        Includes secondary levers (reranker_top_n, overlap, graph_*) so the diff
        log matches what actually changed between trials.
        """
        return [
            ("chunk_token_size", old.chunk_token_size, new.chunk_token_size),
            ("chunk_token_overlap", old.chunk_token_overlap, new.chunk_token_overlap),
            ("chunking_strategy", old.chunking_strategy, new.chunking_strategy),
            ("embedding_model", old.embedding_model, new.embedding_model),
            ("index_type", old.index_type.value, new.index_type.value),
            ("top_k", old.top_k, new.top_k),
            ("hybrid_alpha", old.hybrid_alpha, new.hybrid_alpha),
            ("reranker", old.reranker, new.reranker),
            ("reranker_top_n", old.reranker_top_n, new.reranker_top_n),
            ("llm_model", old.llm_model, new.llm_model),
            ("temperature", old.temperature, new.temperature),
            ("reasoning", old.reasoning, new.reasoning),
            ("query_expansion", old.query_expansion, new.query_expansion),
            ("graph_query_mode", old.graph_query_mode, new.graph_query_mode),
            ("graph_top_k", old.graph_top_k, new.graph_top_k),
        ]

    @staticmethod
    def _print_config_diff(old: TrialConfig, new: TrialConfig) -> None:
        """Print which key parameters changed between configs."""
        changes = [
            f"     {name}: {old_val} → {new_val}"
            for name, old_val, new_val in Orchestrator._diff_pairs(old, new)
            if old_val != new_val
        ]
        if changes:
            print("   Config changes:")
            for line in changes:
                print(line)
        else:
            print("   Config: no changes")

    def _log_config_diff(self, old: TrialConfig, new: TrialConfig) -> None:
        changes = [
            f"{name}: {old_val} -> {new_val}"
            for name, old_val, new_val in self._diff_pairs(old, new)
            if old_val != new_val
        ]
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
