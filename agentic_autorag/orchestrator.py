"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    OpenEndedQuestion,
    ProjectConfig,
    TrialConfig,
)
from agentic_autorag.engine.corpus_cleaner import (
    DuplicateClusters,
    detect_near_duplicates,
)
from agentic_autorag.engine.graph_store import LightRAGStore
from agentic_autorag.engine.index_builder import IndexBuilder, IngredientCache, RAGIndex
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.engine.vllm_server import VLLMServerManager
from agentic_autorag.examiner.evaluator import ExamResult, OpenEndedEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent
from agentic_autorag.examiner.exam_validator import run_validation_pipeline
from agentic_autorag.examiner.probe_selector import (
    attach_probe_metadata,
    collect_probe_outcomes,
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
        debug_eval_samples: int = 0,
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
        # Trial-time judge defaults to the same strong model used for gate-1
        # oracle so paraphrased correct answers don't get scored as wrong.
        trial_judge_model = self.config.examiner.validator_model or self.config.agent.examiner_model
        self.evaluator = OpenEndedEvaluator(
            concurrency=self.config.agent.concurrency,
            retrieval_quality_alpha=self.config.examiner.retrieval_quality_alpha,
            judge_model=trial_judge_model,
            chunk_relevance_min_overlap_chars=self.config.examiner.chunk_relevance_min_overlap_chars,
            chunk_relevance_ngram_size=self.config.examiner.chunk_relevance_ngram_size,
            chunk_relevance_overlap_threshold=self.config.examiner.chunk_relevance_overlap_threshold,
            chunk_relevance_min_run=self.config.examiner.chunk_relevance_min_run,
            debug_eval_samples=debug_eval_samples,
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
        self._exam: list[OpenEndedQuestion] | None = None
        # Near-duplicate clusters: metadata only, never used to filter the
        # corpus that per-trial IndexBuilder.build sees.
        self._duplicate_clusters: DuplicateClusters | None = None

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
    def exam(self) -> list[OpenEndedQuestion]:
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

        # 1b. Near-duplicate detection — metadata only. The full corpus continues
        # to be passed to per-trial IndexBuilder.build, so the optimization loop
        # scores configurations against what users will actually deploy. The
        # cluster map is consumed only by exam generation, the validator BM25
        # index, and the evaluator's chunk-relevance canonicalization.
        self._duplicate_clusters = self._detect_or_load_duplicates(documents, filenames)
        self.evaluator.duplicate_alias_map = dict(self._duplicate_clusters.alias_to_canonical)

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
        self.logger.info("Generating/loading open-ended 2-hop exam")
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
        """Build/load index → ensure vLLM → run pipeline → score the open-ended exam.

        Requires ``setup()`` to have been called. Returns the ExamResult exactly as
        ``OpenEndedEvaluator.evaluate`` produces it. Logs the same per-trial diagnostic
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
        self.logger.info("Trial scored in %.2fs", score_elapsed)
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
        # (config, error_message) pairs for trials that failed before producing
        # a result. Surfaced to the agent on the next propose call so it picks
        # an alternative instead of retrying the same broken config.
        failure_history: list[tuple[TrialConfig, str]] = []
        for trial_num in range(1, meta.max_trials + 1):
            trial_start = time.monotonic()
            self.logger.info("%s", "=" * 60)
            self.logger.info("TRIAL %d/%d", trial_num, meta.max_trials)
            self.logger.info("%s", "=" * 60)
            self._log_config_summary("Config", current_config)

            try:
                result = await self.evaluate_trial(current_config)
            except Exception as exc:
                error_summary = f"{type(exc).__name__}: {exc}"
                self.logger.exception("Trial %d evaluation failed; recovering", trial_num)
                failure_history.append((current_config, error_summary))
                if trial_num == meta.max_trials:
                    self.logger.warning("Last trial failed; no further recovery possible")
                    continue
                try:
                    next_config, recovery_meta = await self.agent.propose_after_failure(
                        failed_config=current_config,
                        error_summary=error_summary,
                        failure_history=failure_history,
                    )
                except Exception:
                    self.logger.exception("Failure-recovery proposal failed; reusing current config")
                    continue
                self.logger.info(
                    "Failure-recovery: %s",
                    "; ".join(recovery_meta.changes) if recovery_meta.changes else "(no changes listed)",
                )
                self._log_config_diff(current_config, next_config)
                current_config = next_config
                continue

            # Agent analyzes failures and proposes next config.
            # Must happen BEFORE history.add(), which clears context/response
            # fields in-place to save RAM (shared object references).
            reasoning_elapsed = 0.0
            trial_config = current_config
            trial_metrics = None
            diagnosis = None
            proposal_meta = None
            if trial_num < meta.max_trials:
                self.logger.info("Agent diagnosing and proposing next config")
                t0 = time.monotonic()
                trial_metrics, diagnosis, next_config, proposal_meta = await self._propose_next_config_with_retries(
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
                answer_accuracy=result.answer_accuracy,
                mean_retrieval_quality=result.mean_retrieval_quality,
                n_em_correct=result.n_em_correct,
                n_judge_correct=result.n_judge_correct,
                n_judge_rejected=result.n_judge_rejected,
                n_judge_failed=result.n_judge_failed,
                n_no_answer=result.n_no_answer,
                n_judge_calls=result.n_judge_calls,
                mean_em=result.mean_em,
                mean_f1=result.mean_f1,
                trial_metrics=trial_metrics,
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

    def _detect_or_load_duplicates(
        self,
        documents: list[str],
        doc_ids: list[str],
    ) -> DuplicateClusters:
        """Run near-duplicate detection (or load a cached map) and persist the result.

        The map is keyed off the corpus cache key so re-runs against an
        unchanged corpus skip the all-pairs comparison. Disabling the
        feature returns an identity map.
        """
        parsing = self.config.parsing
        if not parsing.near_duplicate_detection_enabled:
            self.logger.info("Near-duplicate detection disabled; using identity alias map")
            return DuplicateClusters(
                canonical_doc_ids=list(doc_ids),
                alias_to_canonical={d: d for d in doc_ids},
            )

        cache_dir = self.cache_dir / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache key encodes the dedup threshold so changing it invalidates
        # the cached cluster map (otherwise a tweaked threshold returns the
        # previous result silently).
        dedup_key = hashlib.sha256(
            json.dumps(
                {
                    "corpus": self._corpus_cache_key(),
                    "threshold": parsing.near_duplicate_threshold,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        cache_path = cache_dir / f"duplicate_clusters_{dedup_key}.json"

        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                clusters = DuplicateClusters(
                    canonical_doc_ids=list(payload["canonical_doc_ids"]),
                    alias_to_canonical=dict(payload["alias_to_canonical"]),
                )
                self.logger.info(
                    "Loaded duplicate-cluster map from %s (%d clusters, %d duplicates)",
                    cache_path.name,
                    clusters.n_clusters,
                    clusters.n_duplicates,
                )
                return clusters
            except Exception:
                self.logger.warning("Cached duplicate clusters file is invalid; re-detecting", exc_info=True)

        t0 = time.monotonic()
        clusters = detect_near_duplicates(
            documents,
            doc_ids,
            threshold=parsing.near_duplicate_threshold,
        )
        self.logger.info(
            "Near-duplicate detection: %d documents → %d clusters (%d duplicates) in %.2fs",
            len(documents),
            clusters.n_clusters,
            clusters.n_duplicates,
            time.monotonic() - t0,
        )
        try:
            cache_path.write_text(
                json.dumps(
                    {
                        "canonical_doc_ids": clusters.canonical_doc_ids,
                        "alias_to_canonical": clusters.alias_to_canonical,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            self.logger.warning("Failed to write duplicate clusters cache", exc_info=True)
        return clusters

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
    ) -> tuple[list[OpenEndedQuestion], bool]:
        """Generate and validate the frozen open-ended 2-hop exam from the corpus.

        Pipeline:
          1. Build chunk-pair index (entity cooccurrence) and emit cross-doc seeds.
          2. Batched composition LLM calls produce candidate questions.
          3. Deterministic bridge-leak check + LLM single-hop sufficiency probe.
          4. Source-span verification + two-gate validator (oracle-pass + naive-RAG-fail).
          5. Optional probe-based discrimination selection if too many candidates survive.

        Returns:
            (exam, from_cache) — the frozen exam and whether it was loaded from cache.
        """
        exam_path = self.cache_dir / "exam.json"
        candidates_path = self.cache_dir / "candidates.json"

        if exam_path.exists():
            self.logger.info("Loading existing exam from %s", exam_path.name)
            try:
                raw = json.loads(exam_path.read_text(encoding="utf-8"))
                exam = [OpenEndedQuestion.model_validate(q) for q in raw]
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

        exam_agent = ExamAgent(
            config=examiner,
            examiner_model=self.config.agent.examiner_model,
            corpus_description=self.config.meta.corpus_description,
            temperature=examiner.composition_temperature,
            concurrency=self.config.agent.concurrency,
            # Seed the preferred-type sampler from project_name so the same
            # corpus always gets the same per-seed type assignment.
            type_sampler_seed=self.config.meta.project_name,
            reasoning_effort=self.config.agent.examiner_reasoning_effort,
        )

        # Rank models — used for probe selection AND to pick the strong oracle.
        # We always rank LLMs (not just when probe_selection is on): the oracle
        # gate represents a *ceiling* check ("if no LLM can answer with perfect
        # spans, the question is unanswerable"), so it must run on at least as
        # strong a model as the strongest probe LLM. The cheap examiner model
        # is too weak to serve as a ceiling.
        ss = self.config.search_space
        ranked_llms = await rank_models_for_probes(ss.llm_models, "llm", knowledge_base, optimizer_model)
        ranked_embeds: list[str] | None = None
        ranked_rerankers: list[str] | None = None
        if examiner.probe_selection:
            ranked_embeds = await rank_models_for_probes(
                ss.embedding_models, "embedding", knowledge_base, optimizer_model
            )
            ranked_rerankers = await rank_models_for_probes(
                ss.reranker.models, "reranker", knowledge_base, optimizer_model
            )

        if examiner.validator_model:
            validator_model = examiner.validator_model
        elif ranked_llms:
            validator_model = ranked_llms[-1]
        else:
            validator_model = self.config.agent.examiner_model
        self.logger.info("Oracle / judge validator model: %s", validator_model)
        # Trial-time judge picks up the same strong model so paraphrased
        # correct answers don't get penalised by a weak grader. Guarded for
        # tests that construct Orchestrator without going through __init__.
        evaluator = getattr(self, "evaluator", None)
        if evaluator is not None:
            evaluator.judge_model = validator_model

        # Load cached candidates or run composition fresh. The on-disk shape
        # is either a v2 bare list of questions or a v3 object with
        # ``candidates`` + ``rejections`` siblings; the loader accepts both.
        all_candidates: list[OpenEndedQuestion] | None = None
        if candidates_path.exists():
            try:
                raw_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
                raw_candidates = raw_payload.get("candidates", []) if isinstance(raw_payload, dict) else raw_payload
                all_candidates = [OpenEndedQuestion.model_validate(q) for q in raw_candidates]
                self.logger.info("Loaded %d cached candidates from %s", len(all_candidates), candidates_path.name)
            except Exception:
                self.logger.warning("Cached candidates file is invalid; regenerating", exc_info=True)

        # Subset the corpus to canonical-only docs for exam generation.
        # Per-trial IndexBuilder.build still gets the full corpus (handled by
        # evaluate_trial), so the optimizer scores against deployed conditions.
        # Tests that bypass setup() see an identity map so behaviour matches
        # "no duplicates" cleanly.
        clusters = getattr(self, "_duplicate_clusters", None) or DuplicateClusters(
            canonical_doc_ids=list(doc_ids),
            alias_to_canonical={d: d for d in doc_ids},
        )
        canonical_set = set(clusters.canonical_doc_ids)
        canonical_documents: list[str] = []
        canonical_doc_ids: list[str] = []
        for d_id, d_text in zip(doc_ids, documents, strict=True):
            if d_id in canonical_set:
                canonical_documents.append(d_text)
                canonical_doc_ids.append(d_id)
        if len(canonical_documents) < len(documents):
            self.logger.info(
                "Exam generation uses %d canonical documents (full corpus has %d, %d duplicates suppressed)",
                len(canonical_documents),
                len(documents),
                len(documents) - len(canonical_documents),
            )

        eligible_sections = frozenset(SectionLabel(name) for name in examiner.eligible_section_types)

        if all_candidates is None:
            self.logger.info("Composing typed 2-hop candidates via embedding-pair pipeline")
            all_candidates, prepared_corpus = await exam_agent.generate_exam(
                canonical_documents,
                canonical_doc_ids,
                eligible_sections=eligible_sections,
            )

            width = len(str(max(1, len(all_candidates))))
            for i, q in enumerate(all_candidates, start=1):
                q.id = f"C{i:0{width}d}"

            # Surface LLM refusals (linkable=False with a rejection_explanation)
            # next to the accepted candidates so the user can audit why each
            # seed didn't yield a question. Persist even when 0 candidates
            # survived — the rejections are then the only diagnostic we have.
            rejections: list[dict] = []
            for cr in prepared_corpus.composition_results:
                if cr.linkable or not cr.rejection_explanation:
                    continue
                rejections.append(
                    {
                        "source_chunk_ids": [cr.seed.chunk_a.chunk_id, cr.seed.chunk_b.chunk_id],
                        "explanation": cr.rejection_explanation,
                    }
                )

            try:
                payload = {
                    "candidates": [q.model_dump(mode="json") for q in all_candidates],
                    "rejections": rejections,
                }
                candidates_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self.logger.info(
                    "Saved %d candidates (+ %d rejections) to %s",
                    len(all_candidates),
                    len(rejections),
                    candidates_path.name,
                )
            except Exception:
                self.logger.warning("Failed to write candidates file", exc_info=True)

            if not all_candidates:
                self.logger.warning(
                    "No candidate questions survived composition + single-hop probe — "
                    "the corpus may be too small or topically disjoint for multi-hop synthesis. "
                    "See %s for the LLM's per-seed rejection explanations.",
                    candidates_path.name,
                )
                return [], False

        # Source-span verify → oracle answerability gate. The post-oracle
        # discrimination filter (below) replaces the old naive-RAG gate.
        validated = await run_validation_pipeline(
            all_candidates,
            documents=doc_map,
            validator_model=validator_model,
            judge_model=validator_model,
            concurrency=self.config.agent.concurrency,
            source_fact_verify_fuzzy_threshold=examiner.source_fact_verify_fuzzy_threshold,
        )
        self.logger.info("Validation: %d/%d candidates passed", len(validated), len(all_candidates))

        exam = validated

        # 4-probe discrimination filter — the new core selection mechanism.
        # Evaluates every oracle-passed candidate against 2-4 search-space
        # extremes; questions with high outcome variance (some probes solve,
        # others don't) are the most discriminating and are kept first. All-
        # pass (variance=0) and all-fail patterns score 0 and fall to the
        # bottom; ``select_exam`` truncates to exam_size after sorting.
        if examiner.probe_selection and exam:
            self.logger.info(
                "Running 4-probe discrimination filter (%d candidates, target %d)",
                len(exam),
                exam_size,
            )
            labelled_probes = select_probe_configs(
                self.config,
                ranked_llms=ranked_llms,
                ranked_embeds=ranked_embeds,
                ranked_rerankers=ranked_rerankers,
            )
            probe_results: list[ExamResult] = []
            exam_index_cache: dict[str, RAGIndex] = {}

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
                        "Probe %d/%d %s: composite=%.3f accuracy=%.3f (%d/%d%s) rq=%.3f",
                        i + 1,
                        len(labelled_probes),
                        probe_label.split("(")[0].strip(),
                        result.score,
                        result.answer_accuracy,
                        result.n_correct,
                        result.n_valid,
                        valid_suffix,
                        result.mean_retrieval_quality,
                    )
                    # DIAG per-reasoning_type accuracy for this probe — tells
                    # us whether saturation is uniform across types or
                    # concentrated in a few easy types.
                    type_to_q = {q.id: q for q in exam}
                    type_correct: dict[str, int] = {}
                    type_total: dict[str, int] = {}
                    for qr in result.question_results:
                        q_obj = type_to_q.get(qr.question_id)
                        if q_obj is None:
                            continue
                        rt = q_obj.reasoning_type
                        type_total[rt] = type_total.get(rt, 0) + 1
                        if qr.correct:
                            type_correct[rt] = type_correct.get(rt, 0) + 1
                    if type_total:
                        type_acc = ", ".join(
                            f"{rt}={type_correct.get(rt, 0)}/{type_total[rt]}"
                            f"={type_correct.get(rt, 0) / type_total[rt]:.2f}"
                            for rt in sorted(type_total.keys())
                        )
                        self.logger.info(
                            "DIAG Probe %d/%d %s by type: %s",
                            i + 1,
                            len(labelled_probes),
                            probe_label.split("(")[0].strip(),
                            type_acc,
                        )
                except Exception:
                    self.logger.exception("Probe %d (%s) failed; skipping", i + 1, probe_label)

            if probe_results:
                outcomes = collect_probe_outcomes(probe_results, exam)
                scores = score_questions_by_discrimination(probe_results, exam)
                # Persist the 4-bit correctness vector + variance score on every
                # candidate before selection so post-hoc analysis can read them
                # off the exam.json without recomputing.
                exam = attach_probe_metadata(exam, outcomes, scores)
                # Distribution of outcome patterns across all candidates —
                # tells us at a glance whether probes span the difficulty
                # range (healthy: a mix of 0001/0011/0111) or collapse
                # (everything 0000 or 1111 = saturating exam).
                pattern_counts: dict[str, int] = {}
                for vec in outcomes.values():
                    key = "".join(str(b) for b in vec)
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1
                pattern_str = ", ".join(f"{p}: {n}" for p, n in sorted(pattern_counts.items()))
                self.logger.info("Probe outcome patterns: %s", pattern_str)
                # DIAG one sample question per non-empty pattern, so the
                # next pass can eyeball what each saturation / strong-only
                # / anti-aligned bucket actually contains.
                from agentic_autorag.examiner.probe_selector import _stratum_label as _strat

                pattern_to_sample: dict[str, OpenEndedQuestion] = {}
                for q in exam:
                    if not q.probe_outcomes:
                        continue
                    key = "".join(str(b) for b in q.probe_outcomes)
                    pattern_to_sample.setdefault(key, q)
                for pat in sorted(pattern_to_sample.keys()):
                    q_sample = pattern_to_sample[pat]
                    self.logger.info(
                        "DIAG Pattern %s sample [%s/%s]: %s",
                        pat,
                        _strat(q_sample),
                        q_sample.reasoning_type,
                        q_sample.question[:140],
                    )
                # All-wrong = every probe wrong with no probe errors. These
                # are genuinely very hard items; ``select_exam`` interleaves
                # a small fraction (capped) into the final exam.
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
                n_zero = sum(1 for s in scores.values() if s == 0.0)
                self.logger.info(
                    "Discrimination scores: min=%.3f, max=%.3f, mean=%.3f, zero_scores=%d/%d, all_wrong=%d",
                    min(scores.values()),
                    max(scores.values()),
                    sum(scores.values()) / len(scores),
                    n_zero,
                    len(scores),
                    len(all_wrong_ids),
                )
                # Per-actual-type discrimination-entropy mean — tells us
                # which question types produced the most informative items
                # on this corpus.
                entropy_by_type: dict[str, list[float]] = {}
                for q in exam:
                    entropy_by_type.setdefault(q.reasoning_type, []).append(q.discrimination_entropy)
                if entropy_by_type:
                    type_lines = ", ".join(
                        f"{t}: mean={sum(v) / len(v):.3f} (n={len(v)})" for t, v in sorted(entropy_by_type.items())
                    )
                    self.logger.info("Per-type discrimination entropy: %s", type_lines)
                # DIAG per-(origin, reasoning_type) discrimination means.
                from agentic_autorag.examiner.probe_selector import _stratum_label

                entropy_by_origin_type: dict[tuple[str, str], list[float]] = {}
                for q in exam:
                    key = (_stratum_label(q), q.reasoning_type)
                    entropy_by_origin_type.setdefault(key, []).append(q.discrimination_entropy)
                if entropy_by_origin_type:
                    rows = ", ".join(
                        f"{origin}/{rt}: mean={sum(v) / len(v):.3f} (n={len(v)})"
                        for (origin, rt), v in sorted(entropy_by_origin_type.items())
                    )
                    self.logger.info("DIAG Per-(origin, type) discrimination entropy: %s", rows)
                # DIAG saturation samples: pick up to 3 all-correct and 3
                # all-wrong questions so we can read what kind of question
                # ends up in each saturation bucket.
                all_correct_samples: list[OpenEndedQuestion] = []
                all_wrong_samples: list[OpenEndedQuestion] = []
                for q in exam:
                    vec = q.probe_outcomes
                    if not vec:
                        continue
                    if all(v == 1 for v in vec) and len(all_correct_samples) < 3:
                        all_correct_samples.append(q)
                    elif all(v == 0 for v in vec) and len(all_wrong_samples) < 3:
                        all_wrong_samples.append(q)
                for j, q in enumerate(all_correct_samples, start=1):
                    self.logger.info(
                        "DIAG All-correct sample #%d [%s/%s]: %s",
                        j,
                        _stratum_label(q),
                        q.reasoning_type,
                        q.question[:160],
                    )
                for j, q in enumerate(all_wrong_samples, start=1):
                    self.logger.info(
                        "DIAG All-wrong sample #%d [%s/%s]: %s",
                        j,
                        _stratum_label(q),
                        q.reasoning_type,
                        q.question[:160],
                    )
                exam = select_exam(
                    exam,
                    scores,
                    exam_size,
                    all_wrong_ids=all_wrong_ids,
                )
                self.logger.info("Probe selection: %d questions selected", len(exam))
            else:
                self.logger.warning("All probes failed; falling back to simple truncation")
                exam = exam[:exam_size]
        elif len(exam) > exam_size:
            exam = exam[:exam_size]
            self.logger.info("Truncated to exam_size=%d", exam_size)

        if len(exam) < exam_size:
            self.logger.warning(
                "Exam has %d questions (target %d) after filtering",
                len(exam),
                exam_size,
            )

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

    def _save_exam(self, exam: list[OpenEndedQuestion]) -> None:
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
        exam: list[OpenEndedQuestion],
        current_config: TrialConfig,
        *,
        trial_number: int,
        trials_remaining: int,
    ) -> tuple:
        """Call the agent up to 5 times; reuse previous config on failure.

        Returns ``(trial_metrics, diagnosis, next_config, proposal_meta)``.
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
        """Configure a run logger with console and file handlers.

        The run logger ("agentic_autorag.run") writes the orchestrator's own
        narration. The parent logger ("agentic_autorag") captures module-level
        diagnostics (composition rejections, section-filter counts, validation
        funnel) into both ``run.log`` and the console — so users running
        without ``--verbose`` still see the high-signal setup lines (NER
        backend, entity histogram, prepared-corpus stats, etc.). LiteLLM uses
        its own logger hierarchy, so it does NOT bleed into our handlers.
        """
        formatter = logging.Formatter("%(message)s")
        log_path = output_dir / "run.log"
        # Truncate once explicitly. Both file handlers below open in "a" so
        # they cooperate via O_APPEND instead of racing on file offsets — that
        # race used to overwrite earlier lines mid-run.
        log_path.write_text("", encoding="utf-8")

        run_console = logging.StreamHandler()
        run_console.setLevel(logging.INFO)
        run_console.setFormatter(formatter)

        run_file = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        run_file.setLevel(logging.DEBUG)
        run_file.setFormatter(formatter)

        run_logger = logging.getLogger("agentic_autorag.run")
        run_logger.setLevel(logging.DEBUG)
        run_logger.propagate = False
        for handler in list(run_logger.handlers):
            run_logger.removeHandler(handler)
        run_logger.addHandler(run_console)
        run_logger.addHandler(run_file)

        parent_file = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        parent_file.setLevel(logging.INFO)
        parent_file.setFormatter(formatter)
        parent_console = logging.StreamHandler()
        parent_console.setLevel(logging.INFO)
        parent_console.setFormatter(formatter)

        parent_logger = logging.getLogger("agentic_autorag")
        parent_logger.setLevel(logging.INFO)
        parent_logger.propagate = False
        # Drop only previously-attached handlers to avoid stacking when
        # _setup_logger is called more than once in a process.
        for handler in list(parent_logger.handlers):
            if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                parent_logger.removeHandler(handler)
        parent_logger.addHandler(parent_file)
        parent_logger.addHandler(parent_console)

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
