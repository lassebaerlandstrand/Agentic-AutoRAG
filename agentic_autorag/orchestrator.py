"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import yaml
from docling_core.types.doc.document import DoclingDocument
from tqdm import tqdm

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    OpenEndedQuestion,
    ProjectConfig,
    TrialConfig,
    _describe_dim,
)
from agentic_autorag.cost_ledger import CostLedger, get_active_ledger, reset_active_ledger, set_active_ledger
from agentic_autorag.engine._io import SKIP_FILENAMES
from agentic_autorag.engine.corpus_cleaner import (
    DuplicateClusters,
    detect_near_duplicates,
)
from agentic_autorag.engine.corpus_sampler import sample_corpus
from agentic_autorag.engine.graph_store import LightRAGStore
from agentic_autorag.engine.index_builder import IndexBuilder, IngredientCache, RAGIndex
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.engine.vllm_server import VLLMServerManager
from agentic_autorag.examiner._errors import AllQuestionsErrored, ExamGenerationFailed
from agentic_autorag.examiner.evaluator import ExamResult, OpenEndedEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent, dl_doc_to_chunk_text
from agentic_autorag.examiner.exam_validator import run_validation_pipeline
from agentic_autorag.examiner.probe_selector import (
    attach_probe_metadata,
    collect_probe_outcomes,
    rank_models_for_probes,
    score_questions_by_discrimination,
    select_exam,
    select_probe_configs,
)
from agentic_autorag.litellm_runtime import install_model_aliases
from agentic_autorag.optimizer import pareto
from agentic_autorag.optimizer.diagnosis import ProposalMeta, Strategy
from agentic_autorag.optimizer.frontier_report import render_report as render_frontier_report
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent
from agentic_autorag.optimizer.state import CONFIG_LEVER_FIELDS, build_failure_cross_tab
from agentic_autorag.optimizer.verify_models import (
    assert_all_ok,
    verify_llm_endpoints,
)


def _format_per_stage_llm(config: TrialConfig) -> str:
    """Render per-stage LLM picks compactly. Collapses to a single value when
    every active stage uses the same LLM."""
    parts: dict[str, str | None] = {
        "gen": config.generator_llm,
        "comp": config.compressor_llm,
        "exp": config.expander_llm,
    }
    active = [v for v in parts.values() if v is not None]
    if active and all(v == active[0] for v in active):
        return active[0]
    return "|".join(f"{k}:{v if v is not None else 'null'}" for k, v in parts.items())


logger = logging.getLogger(__name__)

# Fraction of the requested exam_size below which exam generation is treated
# as a fatal failure. Anything smaller would force the optimizer to spend
# trials judging an exam that doesn't span enough difficulty to discriminate.
MIN_EXAM_FRACTION = 0.5

# CostBucket fields replayed when crediting a cached exam's recorded
# generation cost to the active ledger. ``n_calls`` is excluded because the
# replay is one logical call regardless of how many original calls the
# generation issued.
_REPLAYABLE_BUCKET_FIELDS = (
    "usd",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "embedding_input_tokens",
)


def _exam_gen_bucket_snapshot() -> dict:
    """Return current ``exam_generation`` bucket totals as a plain dict (zeros if absent)."""
    ledger = get_active_ledger()
    if ledger is None:
        return {k: 0 for k in _REPLAYABLE_BUCKET_FIELDS}
    bucket = ledger.buckets.get("exam_generation")
    if bucket is None:
        return {k: 0 for k in _REPLAYABLE_BUCKET_FIELDS}
    return {k: getattr(bucket, k) for k in _REPLAYABLE_BUCKET_FIELDS}


def _exam_cache_key(exam_path: Path) -> str:
    """Content-hashed cache key so two machines loading the same cached exam
    log identical keys."""
    try:
        digest = hashlib.sha256(exam_path.read_bytes()).hexdigest()[:16]
        return f"exam_{digest}"
    except OSError:
        return "exam_unreadable"

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
    """Verify required env vars are set for configured providers. Each provider
    can satisfy auth via any one of multiple alternative env-var sets (e.g.
    Bedrock accepts explicit keys, named profile, or IAM)."""
    missing: list[tuple[str, list[str]]] = []

    models_to_check: list[str] = []
    models_to_check.extend(config.search_space.all_llm_models())
    models_to_check.append(config.agent.optimizer_model)
    models_to_check.append(config.agent.examiner_model)
    if config.graph is not None:
        models_to_check.append(config.graph.extraction_model)

    checked_prefixes: set[str] = set()

    models_to_check = [config.resolve_alias(m) for m in models_to_check]

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
        objective: str = "max_score",
        seed: int | None = None,
        force_verify: bool = False,
    ) -> None:
        self.config: ProjectConfig = load_config(config_path)
        install_model_aliases(self.config.model_aliases)
        _check_api_keys(self.config)
        self._objective = pareto.SelectionPolicy.parse(objective)
        self.seed = seed
        self._force_verify = force_verify
        meta = self.config.meta
        # Score-only mode has only one meaningful selection policy. Coerce
        # silently so a knee pick on a run that never optimized cost can't slip
        # through.
        if not meta.cost_aware and self._objective.kind != "max_score":
            self._objective = pareto.SelectionPolicy(kind="max_score")

        # Cache dir always tracks meta.output_dir from YAML — the shared root
        # for parsed-corpus, exam, ingredient, and graph caches. Baseline
        # drivers pass ``output_dir_override`` to keep per-run artifacts out
        # of the agentic optimize run's directory while sharing the cache.
        self._cache_dir = Path(meta.output_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(output_dir_override) if output_dir_override else self._cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger(self.output_dir)

        # History is cleared in run() so baseline drivers can construct an
        # Orchestrator without wiping a sibling agentic run's history.jsonl.
        self.history = HistoryLog(path=str(self.output_dir / "history.jsonl"), load_existing=False)

        try:
            self.knowledge_base: KnowledgeBase | None = KnowledgeBase()
        except Exception as e:
            logger.warning("Could not load knowledge base: %s. Agent will run without model context.", e)
            self.knowledge_base = None

        # Populate embedding token limits from KB for cross-field validation
        if self.knowledge_base:
            embed_models = self.knowledge_base._embeddings.get("models", {})
            for name in self.config.search_space.embedding.models:
                entry = embed_models.get(name)
                if entry and entry.get("max_tokens"):
                    self.config.embedding_token_limits[name] = int(entry["max_tokens"])

        self.agent = ReasoningAgent(
            agent_model=self.config.agent.optimizer_model,
            config=self.config,
            history=self.history,
            debug_prompts=debug_prompts,
            knowledge_base=self.knowledge_base,
            seed=seed,
        )
        # Trial-time judge defaults to the gate-1 oracle model so paraphrased
        # correct answers don't get scored wrong. When ``agent.judge_model``
        # is None, ``_generate_exam`` auto-picks the strongest search-space LLM
        # and overwrites ``evaluator.judge_model`` later.
        trial_judge_model = self.config.agent.judge_model or self.config.agent.examiner_model
        self.evaluator = OpenEndedEvaluator(
            concurrency=self.config.agent.concurrency,
            judge_model=trial_judge_model,
            chunk_relevance_min_overlap_chars=self.config.examiner.chunk_relevance_min_overlap_chars,
            chunk_relevance_ngram_size=self.config.examiner.chunk_relevance_ngram_size,
            chunk_relevance_overlap_threshold=self.config.examiner.chunk_relevance_overlap_threshold,
            chunk_relevance_min_run=self.config.examiner.chunk_relevance_min_run,
            debug_eval_samples=debug_eval_samples,
        )

        parsing = self.config.parsing
        self.parser = build_parser(
            ocr=parsing.ocr,
            table_structure=parsing.table_structure,
        )

        self.ingredient_cache = IngredientCache(
            cache_dir=self.cache_dir / ".cache" / "ingredients",
            max_bytes=int(meta.cache_max_gb * 1024**3),
        )
        self.index_builder = IndexBuilder(cache=self.ingredient_cache)

        self.graph_store: LightRAGStore | None = None
        if self.config.graph is not None:
            self.graph_store = LightRAGStore(
                working_dir=self.cache_dir / "lightrag",
                build_config=self.config.graph,
            )

        # vLLM server is auto-managed when any hosted_vllm/ model appears in
        # the search space or as the graph extraction model.
        has_vllm_in_search = any(m.startswith("hosted_vllm/") for m in self.config.search_space.all_llm_models())
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

        # First-use-per-(method, seed) cache-credit bookkeeping. Each Orchestrator
        # instance is the lifetime of one (method, seed) run, so a per-instance
        # set is the right scope: the first time this run encounters an
        # ``emb_fp``, we credit the deterministic embedding token count to the
        # ``embedding_build`` cost bucket; subsequent encounters credit nothing.
        self._seen_emb_fps: set[str] = set()
        # Pending cache events to flush to ``cache_events.jsonl``. Each event
        # carries its own ``phase`` tag ("exam_gen" or "trial") set at queue
        # time so the audit log honestly attributes probe-phase builds — they
        # are flushed at the end of ``_load_or_generate_exam`` with no trial
        # number, not lumped into trial 1.
        self._pending_cache_events: list[dict] = []
        # Current pipeline phase, used to tag pending cache events at queue
        # time. Flipped to "exam_gen" inside ``_load_or_generate_exam`` and
        # to "trial" at the top of each trial-loop iteration.
        self._current_phase: str = "setup"
        # Near-duplicate clusters: metadata only, never used to filter the
        # corpus that per-trial IndexBuilder.build sees.
        self._duplicate_clusters: DuplicateClusters | None = None
        # Stance-observability state — set as the agent declares stances.
        # ``_last_logged_stance`` is the stance from the most recent agent
        # emission we narrated; ``_stance_run_start_trial`` is the upcoming
        # trial number at which the current stance run began. Both stay None
        # in score-only mode.
        self._last_logged_stance: str | None = None
        self._stance_run_start_trial: int | None = None

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
        all_llms = ss.all_llm_models()
        self.logger.info(
            "Search space: %d LLM(s), %d embedding(s), %d reranker(s), %d index type(s)",
            len(all_llms),
            len(ss.embedding.models),
            len(ss.reranker.models),
            len(ss.retrieval.index_types),
        )
        self.logger.info("  LLMs (generator): %s", self._truncate_list(ss.generator.models))
        self.logger.info("  LLMs (expander):  %s", self._truncate_list(ss.query_expansion.models))
        self.logger.info("  LLMs (compressor):%s", self._truncate_list(ss.passage_compressor.models))
        self.logger.info("  Embeddings: %s", self._truncate_list(ss.embedding.models))
        self.logger.info("  Rerankers: %s", self._truncate_list(ss.reranker.models))
        self.logger.info("  Index types: %s", self._truncate_list([it.value for it in ss.retrieval.index_types]))
        self.logger.info(
            "  Chunking: %s | size %s | overlap %s",
            self._truncate_list(ss.chunking.strategies),
            _describe_dim(ss.chunking.chunk_token_size),
            _describe_dim(ss.chunking.chunk_token_overlap),
        )

    async def setup(self) -> None:
        """Idempotent: parse corpus, build graph (once), generate or load exam.
        Populates instance state shared with ``evaluate_trial``. Safe to call
        multiple times; baseline drivers call it before their proposal loop
        to share the corpus, graph, and exam with the agentic ``run()``."""
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
        dl_documents = [dl_doc for _, dl_doc in parsed]
        # HybridChunker chunk-text concatenation is the canonical doc-text
        # coordinate frame: vector ``char_range``, graph chunk lookup, and the
        # source-span verifier all index into this string. Spans the composer
        # LLM extracts are guaranteed findable verbatim.
        max_chunk_words = self.config.examiner.max_chunk_words
        documents = [dl_doc_to_chunk_text(dl_doc, max_chunk_words=max_chunk_words) for dl_doc in dl_documents]

        # Expose the doc-id → text map to the evaluator so its deterministic
        # chunk-relevance matcher can look up offsets for verbatim graph chunks.
        self.evaluator.documents = dict(zip(filenames, documents, strict=True))

        # Near-duplicate detection emits metadata only — the full corpus is
        # still passed to per-trial IndexBuilder.build so trials score against
        # what users actually deploy.
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
            dl_documents,
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
        """Build/load index → ensure vLLM → run pipeline → score the exam.
        Requires ``setup()``. Returns an ``ExamResult`` from
        ``OpenEndedEvaluator.evaluate``."""
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
        self._credit_embedding_build(index)
        index.graph_store = self.graph_store
        index_elapsed = time.monotonic() - t0
        self.logger.info(
            "Index ready in %.2fs (%d chunks, %s)",
            index_elapsed,
            len(index.chunks),
            index_source,
        )

        # b. Ensure vLLM is serving every per-stage model this trial needs.
        # vLLM only hosts one model at a time, so all hosted_vllm/ models in
        # this trial must match — enforced inside ensure_model().
        if self.vllm_manager:
            for stage_model in (
                trial_config.generator_llm,
                trial_config.compressor_llm,
                trial_config.expander_llm,
            ):
                if stage_model and stage_model.startswith("hosted_vllm/"):
                    await self.vllm_manager.ensure_model(stage_model)

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
                reasoning_effort=self.config.search_space.generator.reasoning_effort,
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

        # Smoke-test every LLM endpoint in the search space before any indexing
        # work. Catches credential/region failures up-front instead of mid-trial.
        await self._verify_search_space_llms()

        # Fresh history.jsonl for each agentic run. Baseline drivers manage their
        # own HistoryLog and never touch this one.
        self.history.clear()

        ledger = CostLedger()
        ledger_token = set_active_ledger(ledger)
        try:
            return await self._run_with_ledger(t_start, meta, ledger)
        finally:
            reset_active_ledger(ledger_token)
            self._report_cost_breakdown(ledger)

    async def _verify_search_space_llms(self) -> None:
        """Ping every generator/expander/compressor model + agent/examiner/judge.

        Reuses the global verification cache so a sibling bench run in the
        same session only pays the ping cost once across methods.
        """
        ss = self.config.search_space
        models: list[str] = list(ss.all_llm_models())
        models.append(self.config.agent.optimizer_model)
        models.append(self.config.agent.examiner_model)
        if self.config.agent.judge_model:
            models.append(self.config.agent.judge_model)
        models = [self.config.resolve_alias(m) for m in models]

        results = await verify_llm_endpoints(models, force=self._force_verify, logger_=self.logger)
        assert_all_ok(results)

    async def _run_with_ledger(self, t_start: float, meta, ledger: CostLedger) -> TrialRecord:
        await self.setup()
        exam = self.exam

        # Agent proposes initial config. Snapshot the ledger BEFORE this call
        # and pre-set the phase tag so the Initial Proposer's LLM spend lands
        # in trial 1's per-trial delta (instead of escaping into the
        # unattributed gap between exam-gen and the trial loop). Matches the
        # semantic that every later trial N's bucket includes the Proposer
        # call that produced trial N's config.
        self._current_phase = "trial"
        initial_proposer_snapshot = ledger.snapshot()
        self.logger.info("Agent proposing initial configuration")
        t0 = time.monotonic()
        current_config = await self.agent.propose_initial(
            corpus_description=meta.corpus_description,
        )
        self.logger.info("Initial config received in %.2fs", time.monotonic() - t0)

        # Seed the agent's strategy. In cost-aware mode the agent owns its
        # stance (explore/refine) thereafter; in score-only mode the stance is
        # always None. The orchestrator preserves this object across trials
        # and threads it back as ``previous_strategy`` on every
        # ``analyze_and_propose`` call.
        active_strategy: Strategy | None = Strategy(
            stance="explore" if self.config.meta.cost_aware else None,
            journal="",
        )

        # Optimization loop
        best: TrialRecord | None = None
        # (config, error_message) pairs for trials that failed before producing
        # a result. Surfaced to the agent on the next propose call so it picks
        # an alternative instead of retrying the same broken config.
        failure_history: list[tuple[TrialConfig, str]] = []
        cumulative_cost_usd = 0.0
        prev_frontier_trials: set[int] = set()
        # Meta describing the Proposer call that produced the upcoming trial's
        # config. Carried across iterations so it attaches to the TrialRecord
        # of the trial it actually selected, not the one that ran before it.
        # The initial trial has no preceding Proposer call, so it stays None.
        pending_meta: ProposalMeta | None = None
        for trial_num in range(1, meta.max_trials + 1):
            self._current_phase = "trial"
            trial_start = time.monotonic()
            # Trial 1 reuses the pre-``propose_initial`` snapshot so the
            # Initial Proposer's tokens roll into trial 1's bucket; later
            # trials snapshot at the loop top (Proposer-of-N + Diagnoser-of-N
            # already ran inside trial N-1's iteration body).
            trial_ledger_before = (
                initial_proposer_snapshot if trial_num == 1 else ledger.snapshot()
            )
            self.logger.info("%s", "=" * 60)
            self.logger.info("TRIAL %d/%d", trial_num, meta.max_trials)
            self.logger.info("%s", "=" * 60)
            self._log_config_summary("Config", current_config)

            # ``delta_written`` ensures the per-trial ledger line is appended
            # exactly once per iteration. The success path calls
            # ``_finalize_trial_accounting`` to get the totals it needs for
            # ``TrialRecord``; the ``finally`` only fires on failure so the
            # spend incurred by a failed trial still appears in
            # ``trial_cost_ledger.jsonl`` (status="failed") instead of leaking
            # into the next trial's delta.
            delta_written = False
            skip_success_path = False
            try:
                try:
                    result = await self.evaluate_trial(current_config)
                    if result.all_errored:
                        raise AllQuestionsErrored(result.error_sentinel, result.n_total)
                except Exception as exc:
                    error_summary = f"{type(exc).__name__}: {exc}"
                    self.logger.exception("Trial %d evaluation failed; recovering", trial_num)
                    failure_history.append((current_config, error_summary))
                    skip_success_path = True
                    if trial_num == meta.max_trials:
                        self.logger.warning("Last trial failed; no further recovery possible")
                    else:
                        try:
                            next_config, recovery_meta = await self.agent.propose_after_failure(
                                failed_config=current_config,
                                error_summary=error_summary,
                                failure_history=failure_history,
                            )
                            recovery_changes = [
                                f"{name}: {old_val} → {new_val}"
                                for name, old_val, new_val in self._diff_pairs(current_config, next_config)
                                if old_val != new_val
                            ]
                            self.logger.info(
                                "Failure-recovery: %s",
                                "; ".join(recovery_changes) if recovery_changes else "(no levers changed)",
                            )
                            self._log_config_diff(current_config, next_config)
                            current_config = next_config
                            pending_meta = recovery_meta
                        except Exception:
                            self.logger.exception(
                                "Failure-recovery proposal failed; reusing current config"
                            )

                if not skip_success_path:
                    # Agent analyzes failures and proposes next config.
                    # Must happen BEFORE history.add(), which clears context/response
                    # fields in-place to save RAM (shared object references).
                    reasoning_elapsed = 0.0
                    trial_config = current_config
                    trial_metrics = None
                    diagnosis = None
                    proposal_meta: ProposalMeta | None = None
                    if trial_num < meta.max_trials:
                        self.logger.info("Agent diagnosing and proposing next config")
                        t0 = time.monotonic()
                        try:
                            (
                                trial_metrics,
                                diagnosis,
                                next_config,
                                proposal_meta,
                            ) = await self._propose_next_config_with_retries(
                                result,
                                exam,
                                current_config,
                                trial_number=trial_num,
                                trials_remaining=meta.max_trials - trial_num,
                                previous_strategy=active_strategy,
                            )
                            reasoning_elapsed = time.monotonic() - t0
                            self._log_config_diff(current_config, next_config)
                            current_config = next_config
                            # Persist the agent's record-side meta with the previous strategy
                            # carried over when the agent didn't manage to emit one (the
                            # agent-failure fallback returns proposal_meta=None).
                            if proposal_meta is not None and proposal_meta.strategy is not None:
                                new_strategy = proposal_meta.strategy
                                self._log_strategy_status(new_strategy, upcoming_trial=trial_num + 1)
                                active_strategy = new_strategy
                        except Exception:
                            reasoning_elapsed = time.monotonic() - t0
                            self.logger.exception(
                                "Trial %d post-evaluation agent call crashed; keeping trial result, "
                                "reusing current config for trial %d",
                                trial_num,
                                trial_num + 1,
                            )

                    # Record trial (mutates question_results to free RAM).
                    # ``meta`` is the Proposer output that *produced* ``trial_config``
                    # (emitted by the prior trial's diagnose-and-propose step, or by
                    # failure-recovery, or None for the initial trial). The meta just
                    # emitted by this trial's Proposer describes the NEXT config and
                    # is carried via ``pending_meta`` to the next iteration's record.
                    cross_tab_snapshot = build_failure_cross_tab(result.question_results, exam)
                    (
                        total_prompt_tokens,
                        total_completion_tokens,
                        total_embedding_tokens,
                    ) = self._finalize_trial_accounting(
                        trial_num, trial_ledger_before, status="ok"
                    )
                    delta_written = True
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
                        n_judge_no_answer=result.n_judge_no_answer,
                        n_judge_failed=result.n_judge_failed,
                        n_no_answer=result.n_no_answer,
                        n_judge_calls=result.n_judge_calls,
                        mean_em=result.mean_em,
                        mean_f1=result.mean_f1,
                        mean_llm_cost_per_query_usd=result.mean_llm_cost_per_query_usd,
                        total_llm_cost_usd=result.total_llm_cost_usd,
                        mean_prompt_tokens=result.mean_prompt_tokens,
                        mean_completion_tokens=result.mean_completion_tokens,
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        total_embedding_tokens=total_embedding_tokens,
                        trial_metrics=trial_metrics,
                        diagnosis=diagnosis,
                        meta=pending_meta,
                        cross_tab_snapshot=cross_tab_snapshot,
                    )
                    self.history.add(record)
                    pending_meta = proposal_meta
                    # The Pareto frontier depends on every trial's (score, cost), so
                    # ``is_pareto_optimal`` must be recomputed for ALL records on every
                    # add — a previously-frontier trial can be displaced by a new one.
                    self.history.recompute_pareto_flags()
                    self.history.rewrite_all()
                    if best is None or result.score > best.score:
                        best = record

                    cumulative_cost_usd += record.total_llm_cost_usd
                    prev_frontier_trials = self._log_pareto_state(
                        prev_frontier_trials=prev_frontier_trials,
                        current_trial_number=trial_num,
                        current_record_is_frontier=record.is_pareto_optimal,
                    )

                    trial_elapsed = time.monotonic() - trial_start
                    pareto_tag = " ★Pareto" if record.is_pareto_optimal else ""
                    self.logger.info(
                        "Trial %d total %.2fs | agent %.2fs | cost=$%.4f/q (trial $%.3f, run $%.3f)%s",
                        trial_num,
                        trial_elapsed,
                        reasoning_elapsed,
                        record.mean_llm_cost_per_query_usd,
                        record.total_llm_cost_usd,
                        cumulative_cost_usd,
                        pareto_tag,
                    )
            finally:
                if not delta_written:
                    # Failed trial: still write a delta line so the wasted
                    # spend is visible in ``trial_cost_ledger.jsonl`` and
                    # doesn't bleed into the next trial's snapshot.
                    self._finalize_trial_accounting(
                        trial_num, trial_ledger_before, status="failed"
                    )

        # Summary
        elapsed = time.monotonic() - t_start
        recommended = self._save_frontier_artifacts()
        max_score = self.history.get_best()
        self.logger.info(
            "Optimization complete in %.2fs (rag_eval cost: $%.3f across %d trial(s) — used for Pareto)",
            elapsed,
            cumulative_cost_usd,
            len(self.history.records),
        )
        if max_score:
            self.logger.info(
                "Max-score trial %d: score=%.3f cost=$%.4f/q",
                max_score.trial_number,
                max_score.score,
                max_score.mean_llm_cost_per_query_usd,
            )
            if recommended is not None and recommended.trial_number != max_score.trial_number:
                self.logger.info(
                    "Recommended trial %d (policy=%s): score=%.3f cost=$%.4f/q",
                    recommended.trial_number,
                    self._objective.kind,
                    recommended.score,
                    recommended.mean_llm_cost_per_query_usd,
                )
            elif recommended is None:
                self.logger.info(
                    "No frontier member satisfies policy=%s — see frontier_report.md for alternatives",
                    self._objective.kind,
                )
            self._log_pareto_frontier_summary()
        else:
            self.logger.info("No successful trials completed")

        await self.cleanup()

        return recommended if recommended is not None else max_score

    def _log_strategy_status(self, new: Strategy, *, upcoming_trial: int) -> None:
        """Log the agent's stance — hold or flip — every trial in cost-aware
        mode. Trial 1 is implicitly ``explore`` (``propose_initial`` doesn't
        declare a stance, but the first config is score-chasing by
        construction); we seed it so trial 2 reads as a continuation."""
        if new.stance is None:
            return
        if self._last_logged_stance is None:
            self._last_logged_stance = "explore"
            self._stance_run_start_trial = 1
        if self._last_logged_stance == new.stance:
            start = self._stance_run_start_trial or upcoming_trial
            run_len = upcoming_trial - start + 1
            self.logger.info(
                "Strategy: stance=%s (held, %d trial(s) since trial %d)",
                new.stance,
                run_len,
                start,
            )
            return
        self.logger.info(
            "Strategy: stance=%s → %s (flipped at trial %d)",
            self._last_logged_stance,
            new.stance,
            upcoming_trial,
        )
        self._last_logged_stance = new.stance
        self._stance_run_start_trial = upcoming_trial

    def _log_pareto_state(
        self,
        *,
        prev_frontier_trials: set[int],
        current_trial_number: int,
        current_record_is_frontier: bool,
    ) -> set[int]:
        """Log frontier diff, hypervolume, and knee after every trial. Returns
        the new set of frontier trial numbers so the next call can diff. HV
        uses the same cost reference (max observed cost) the state card uses."""
        from agentic_autorag.optimizer import pareto

        records = list(self.history.records)
        if not records:
            return set()
        frontier = pareto.compute_frontier(records)
        new_frontier_trials = {int(r.trial_number) for r in frontier}

        cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
        cost_ref = max(cost_values) if cost_values else 0.0
        if cost_ref <= 0.0:
            cost_ref = 1.0
        hv = pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))
        knee = pareto.find_knee(frontier)
        knee_str = (
            f"trial {knee.trial_number} (score={knee.score:.3f}, cost=${knee.mean_llm_cost_per_query_usd:.4f}/q)"
            if knee is not None
            else "n/a"
        )
        self.logger.info(
            "Pareto: frontier=%d (HV=%.4f, ref_cost=$%.4f/q) | knee=%s",
            len(frontier),
            hv,
            cost_ref,
            knee_str,
        )

        if current_record_is_frontier and current_trial_number not in prev_frontier_trials:
            self.logger.info("Pareto: trial %d added to frontier", current_trial_number)
        displaced = prev_frontier_trials - new_frontier_trials
        if displaced:
            displaced_str = ", ".join(f"trial {t}" for t in sorted(displaced))
            self.logger.info("Pareto: displaced from frontier: %s", displaced_str)

        return new_frontier_trials

    def _log_pareto_frontier_summary(self) -> None:
        """Log the final Pareto frontier (one line per non-dominated trial) + knee."""
        from agentic_autorag.optimizer import pareto

        records = self.history.records
        if not records:
            return
        frontier = pareto.compute_frontier(list(records))
        if not frontier:
            return
        knee = pareto.find_knee(frontier)
        knee_trial = knee.trial_number if knee is not None else None
        self.logger.info("Pareto frontier (%d non-dominated trials):", len(frontier))
        for r in sorted(frontier, key=lambda x: x.score):
            tag = "  ★knee" if r.trial_number == knee_trial else ""
            self.logger.info(
                "  trial %d: score=%.3f cost=$%.4f/q%s",
                r.trial_number,
                r.score,
                r.mean_llm_cost_per_query_usd,
                tag,
            )

    def _detect_or_load_duplicates(
        self,
        documents: list[str],
        doc_ids: list[str],
    ) -> DuplicateClusters:
        """Run near-duplicate detection (or load a cached map). Keyed off the
        corpus cache key + threshold so re-runs skip the all-pairs scan."""
        parsing = self.config.parsing
        if not parsing.near_duplicate_detection_enabled:
            self.logger.info("Near-duplicate detection disabled; using identity alias map")
            return DuplicateClusters(
                canonical_doc_ids=list(doc_ids),
                alias_to_canonical={d: d for d in doc_ids},
            )

        cache_dir = self.cache_dir / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Threshold goes into the cache key so a tweaked threshold doesn't
        # silently return the previous result.
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

    def _finalize_trial_accounting(
        self,
        trial_number: int,
        before_snapshot: dict,
        *,
        status: str = "ok",
    ) -> tuple[int, int, int]:
        """Write per-trial bucket delta and pending cache events; return
        ``(prompt_tokens, completion_tokens, embedding_tokens)``. Failed trials
        write ``status="failed"`` so downstream analyzers can keep wasted
        spend out of headline aggregates."""
        ledger = get_active_ledger()
        if ledger is None:
            self._flush_pending_cache_events(trial_number)
            return 0, 0, 0

        full_delta = ledger.delta_since(before_snapshot)
        delta = {k: v for k, v in full_delta.items() if any(vv != 0 for vv in v.values())}
        trial_ledger_path = self.output_dir / "trial_cost_ledger.jsonl"
        try:
            with trial_ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"trial_number": trial_number, "status": status, "buckets": delta})
                    + "\n"
                )
        except OSError:
            self.logger.warning("Failed to append %s", trial_ledger_path, exc_info=True)

        self._flush_pending_cache_events(trial_number)

        total_prompt = sum(int(b["prompt_tokens"]) for b in delta.values())
        total_completion = sum(int(b["completion_tokens"]) for b in delta.values())
        total_embedding = sum(int(b["embedding_input_tokens"]) for b in delta.values())
        return total_prompt, total_completion, total_embedding

    def _flush_pending_cache_events(self, trial_number: int) -> None:
        """Drain queued trial-phase cache events to ``cache_events.jsonl``.
        Only ``phase="trial"`` events get the trial number; any leftover
        exam-phase events get ``trial_number=null``."""
        if not self._pending_cache_events:
            return
        events_path = self.output_dir / "cache_events.jsonl"
        try:
            with events_path.open("a", encoding="utf-8") as fh:
                for event in self._pending_cache_events:
                    tn = trial_number if event.get("phase") == "trial" else None
                    fh.write(json.dumps({"trial_number": tn, **event}) + "\n")
        except OSError:
            self.logger.warning("Failed to append %s", events_path, exc_info=True)
        finally:
            self._pending_cache_events.clear()

    def _flush_exam_gen_cache_events(self) -> None:
        """Drain queued exam-phase cache events with ``trial_number=null`` —
        keeps probe-phase embeddings and exam-replay events from being
        lumped into trial 1's flush."""
        if not self._pending_cache_events:
            return
        events_path = self.output_dir / "cache_events.jsonl"
        try:
            with events_path.open("a", encoding="utf-8") as fh:
                for event in self._pending_cache_events:
                    fh.write(json.dumps({"trial_number": None, **event}) + "\n")
        except OSError:
            self.logger.warning("Failed to append %s", events_path, exc_info=True)
        finally:
            self._pending_cache_events.clear()

    def _persist_exam_cost(self, exam_cost_path: Path, before_bucket: dict) -> None:
        """Snapshot the exam_generation bucket delta after a fresh generation.

        Written unconditionally so a later orchestrator can replay the cost
        via ``_replay_exam_cost`` even when no ledger was active at exam-gen
        time (the bench installs per-method ledgers after setup). All-zero
        deltas are correct for the bench's exam-gen exclusion rule.
        """
        after_bucket = _exam_gen_bucket_snapshot()
        delta = {k: after_bucket[k] - before_bucket[k] for k in _REPLAYABLE_BUCKET_FIELDS}
        try:
            exam_cost_path.write_text(json.dumps(delta), encoding="utf-8")
        except OSError:
            self.logger.warning("Failed to write %s", exam_cost_path, exc_info=True)

    def _replay_exam_cost(self, exam_cost_path: Path) -> None:
        """Credit a cached exam's recorded generation cost to the active
        ledger. Raises on missing ``exam_cost.json`` — silent under-attribution
        would corrupt paper numbers. Remediation: wipe cache and rerun."""
        ledger = get_active_ledger()
        if ledger is None:
            return
        if not exam_cost_path.exists():
            raise RuntimeError(
                f"Exam cache at {exam_cost_path.parent / 'exam.json'} is missing its "
                f"exam_cost.json sidecar. This cache was built before exam-token "
                f"accounting; delete the cache dir and rerun."
            )
        cached = json.loads(exam_cost_path.read_text(encoding="utf-8"))
        ledger.record("exam_generation", **{k: cached.get(k, 0) for k in _REPLAYABLE_BUCKET_FIELDS})
        self._pending_cache_events.append(
            {
                "cache_kind": "exam",
                # Stable, environment-independent key derived from the exam
                # content rather than the absolute path on this machine.
                "cache_key": _exam_cache_key(exam_cost_path.parent / "exam.json"),
                "tokens_credited": int(cached.get("prompt_tokens", 0) + cached.get("completion_tokens", 0)),
                "phase": self._current_phase,
            }
        )

    def _credit_embedding_build(self, index: RAGIndex) -> None:
        """First-use-per-(method, seed) credit for an embeddings cache key.

        Credits the deterministic token count to the ``embedding_build``
        bucket and queues a cache event. Subsequent encounters of the same
        ``emb_fp`` within this Orchestrator instance no-op.
        """
        emb_fp = index.emb_fp
        if emb_fp is None or emb_fp in self._seen_emb_fps:
            return
        self._seen_emb_fps.add(emb_fp)
        ledger = get_active_ledger()
        if ledger is not None and index.embedding_input_tokens > 0:
            ledger.record(
                "embedding_build",
                usd=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                embedding_input_tokens=index.embedding_input_tokens,
            )
        self._pending_cache_events.append(
            {
                "cache_kind": "embeddings",
                "cache_key": emb_fp,
                "tokens_credited": int(index.embedding_input_tokens),
                "embedding_model": index.embedding_model,
                "phase": self._current_phase,
            }
        )

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
            if file_path.name in SKIP_FILENAMES:
                continue
            stat = file_path.stat()
            rel = str(file_path.relative_to(corpus_path))
            file_signatures.append((rel, stat.st_mtime_ns, stat.st_size))

        key_data = json.dumps(
            {
                # Cache schema version: bump when DoclingDocument JSON shape
                # changes or the parsing pipeline gains/loses fields.
                "schema": 2,
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

    def _load_and_parse_corpus(self) -> list[tuple[str, DoclingDocument]]:
        """Recursively parse files in ``corpus_path`` via Docling, returning
        ``(filename, DoclingDocument)`` tuples. Filename is the basename used
        as ``source doc id`` in generated questions. Cached by file mtimes +
        parsing options."""
        corpus_path = Path(self.config.meta.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

        cache_path = self._corpus_cache_path()
        if cache_path.exists():
            self.logger.info("Loading cached parsed corpus from %s", cache_path.name)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return [(name, DoclingDocument.model_validate(doc_dict)) for name, doc_dict in cached]

        parser_extensions = frozenset(self.parser.supported_extensions())
        eligible = sample_corpus(
            corpus_path=corpus_path,
            parser_extensions=parser_extensions,
            word_budget=self.config.meta.corpus_word_budget,
            sample_seed=self.config.meta.corpus_sample_seed,
            cache_dir=self.cache_dir,
        )

        documents: list[tuple[str, DoclingDocument]] = []
        skipped = 0
        failed = 0
        for file_path in tqdm(eligible, desc="   Parsing files", unit="file", smoothing=0):
            suffix = file_path.suffix.lower()
            if suffix not in self.parser.supported_extensions():
                skipped += 1
                continue
            try:
                dl_doc = self.parser.parse(file_path)
            except Exception:
                failed += 1
                logger.warning("Failed to parse %s, skipping", file_path, exc_info=True)
                continue
            if dl_doc_to_chunk_text(dl_doc, max_chunk_words=self.config.examiner.max_chunk_words).strip():
                documents.append((file_path.name, dl_doc))

        if skipped:
            self.logger.info("Skipped %d unsupported file(s)", skipped)
        if failed:
            self.logger.warning("Failed to parse %d file(s)", failed)

        try:
            payload = [(name, dl_doc.export_to_dict()) for name, dl_doc in documents]
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Cached parsed corpus to %s", cache_path.name)
        except Exception:
            self.logger.warning("Failed to write corpus cache", exc_info=True)

        return documents

    async def _generate_exam(
        self,
        documents: list[DoclingDocument],
        doc_ids: list[str],
        knowledge_base: KnowledgeBase | None = None,
        optimizer_model: str | None = None,
    ) -> tuple[list[OpenEndedQuestion], bool]:
        """Generate and validate the frozen open-ended exam from the corpus.

        Returns ``(exam, from_cache)``.
        """
        exam_path = self.cache_dir / "exam.json"
        exam_cost_path = self.cache_dir / "exam_cost.json"
        candidates_path = self.cache_dir / "candidates.json"
        exam_size = self.config.examiner.exam_size

        # First-use rule for exam generation: snapshot the ``exam_generation``
        # bucket before any work so we can persist the per-run delta after a
        # fresh generation, and replay it from disk on subsequent cache hits.
        before_bucket = _exam_gen_bucket_snapshot()

        # Tag every cache event queued in this scope (probe-phase embedding
        # builds + exam-replay) with phase="exam_gen" so the bench can filter
        # them out of per-trial accounting. Cleanup is in the matching finally
        # below so the phase always restores even on a surprising exception.
        prev_phase = self._current_phase
        self._current_phase = "exam_gen"

        if exam_path.exists():
            self.logger.info("Loading existing exam from %s", exam_path.name)
            try:
                raw = json.loads(exam_path.read_text(encoding="utf-8"))
                exam = [OpenEndedQuestion.model_validate(q) for q in raw]
                # Cached exams that fall below the minimum fraction are as
                # poisonous to the optimizer as a freshly-generated degenerate
                # exam — fail fast rather than billing trials against them.
                if len(exam) < MIN_EXAM_FRACTION * exam_size:
                    raise ExamGenerationFailed(
                        n_actual=len(exam),
                        n_target=exam_size,
                        candidates_path=str(candidates_path),
                        top_rejection_reasons=[],
                        stage_counts={"loaded_from_cache": len(exam)},
                    )
                self._replay_exam_cost(exam_cost_path)
                self._flush_exam_gen_cache_events()
                self._current_phase = prev_phase
                return exam, True
            except ExamGenerationFailed:
                self._flush_exam_gen_cache_events()
                self._current_phase = prev_phase
                raise
            except Exception:
                self.logger.warning("Existing exam file is invalid; regenerating", exc_info=True)

        if len(doc_ids) != len(documents):
            raise ValueError(f"doc_ids length ({len(doc_ids)}) does not match documents length ({len(documents)})")
        duplicates = [name for name, count in Counter(doc_ids).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate document filenames in corpus: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
            )
        # doc_map carries the canonical HybridChunker chunk-text-concat for
        # each document. The span verifier searches this text (matching what
        # the composer was shown); naive-RAG and probe paths also consume it
        # so production retrieval and verification share one coordinate frame.
        max_chunk_words = self.config.examiner.max_chunk_words
        doc_map = {
            doc_id: dl_doc_to_chunk_text(dl_doc, max_chunk_words=max_chunk_words)
            for doc_id, dl_doc in zip(doc_ids, documents, strict=True)
        }
        examiner = self.config.examiner

        exam_agent = ExamAgent(
            config=examiner,
            examiner_model=self.config.agent.examiner_model,
            corpus_description=self.config.meta.corpus_description,
            temperature=examiner.composition_temperature,
            concurrency=self.config.agent.concurrency,
            # Seed the preferred-type sampler from project_name so the same
            # corpus always gets the same per-seed type assignment.
            anchor_sampler_seed=self.config.meta.project_name,
            reasoning_effort=self.config.agent.examiner_reasoning_effort,
            # TEMPORARY: dump every composition LLM call (chunks + parsed
            # response) to a pretty JSON file so we can inspect the
            # parsed-but-unstored ``reasoning`` field during composition-
            # prompt iteration. Remove this kwarg when the corresponding
            # ExamAgent parameter and helpers are removed.
            composition_log_path=self.output_dir / "composition_log.json",
            # TEMPORARY DEBUG: per-question / per-span outcomes of the
            # source-span verifier (verbatim / tolerant / snap / not_found),
            # so we can diagnose the rejection rate instead of staring at a
            # summary counter. Remove this kwarg when the corresponding
            # ExamAgent parameter and ``verify_source_facts(report_path=...)``
            # are removed.
            span_verification_report_path=self.output_dir / "span_verification.json",
        )

        # Rank models — used for probe selection AND to pick the strong oracle.
        # We always rank LLMs (not just when probe_selection is on): the oracle
        # gate represents a *ceiling* check ("if no LLM can answer with perfect
        # spans, the question is unanswerable"), so it must run on at least as
        # strong a model as the strongest probe LLM. The cheap examiner model
        # is too weak to serve as a ceiling.
        ss = self.config.search_space
        all_llms = ss.all_llm_models()
        reasoning_allowed_for_rank = {m: ss.is_reasoning_allowed(m) for m in all_llms}
        ranked_llms = await rank_models_for_probes(
            all_llms,
            "llm",
            knowledge_base,
            optimizer_model,
            reasoning_allowed=reasoning_allowed_for_rank,
            reasoning_effort=ss.generator.reasoning_effort,
        )
        ranked_embeds: list[str] | None = None
        ranked_rerankers: list[str] | None = None
        if examiner.probe_selection:
            ranked_embeds = await rank_models_for_probes(
                ss.embedding.models, "embedding", knowledge_base, optimizer_model
            )
            ranked_rerankers = await rank_models_for_probes(
                ss.reranker.models, "reranker", knowledge_base, optimizer_model
            )

        if self.config.agent.judge_model:
            validator_model = self.config.agent.judge_model
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
        canonical_documents: list[DoclingDocument] = []
        canonical_doc_ids: list[str] = []
        for d_id, dl_doc in zip(doc_ids, documents, strict=True):
            if d_id in canonical_set:
                canonical_documents.append(dl_doc)
                canonical_doc_ids.append(d_id)
        if len(canonical_documents) < len(documents):
            self.logger.info(
                "Exam generation uses %d canonical documents (full corpus has %d, %d duplicates suppressed)",
                len(canonical_documents),
                len(documents),
                len(documents) - len(canonical_documents),
            )

        excluded_sections = frozenset(SectionLabel(name) for name in examiner.excluded_section_types)
        eligible_sections = frozenset(SectionLabel) - excluded_sections

        # Track stage-by-stage survival so ExamGenerationFailed can surface
        # exactly where the funnel collapsed.
        composition_rejection_counter: Counter[str] = Counter()
        stage_funnel: dict[str, int] = {}

        if all_candidates is None:
            self.logger.info("Composing typed 2-hop candidates via embedding-pair pipeline")
            all_candidates, prepared_corpus = await exam_agent.generate_exam(
                canonical_documents,
                canonical_doc_ids,
                eligible_sections=eligible_sections,
                doc_text_map=doc_map,
                source_fact_verify_fuzzy_threshold=examiner.source_fact_verify_fuzzy_threshold,
            )
            composition_rejection_counter = Counter(exam_agent.last_composition_rejections)

            width = len(str(max(1, len(all_candidates))))
            for i, q in enumerate(all_candidates, start=1):
                q.id = f"C{i:0{width}d}"

            # Surface LLM refusals (linkable=False with a rejection_explanation)
            # next to the accepted candidates so the user can audit why each
            # neighborhood didn't yield a question. Persist even when 0
            # candidates survived — the rejections are then the only
            # diagnostic we have.
            rejections: list[dict] = []
            for cr in prepared_corpus.composition_results:
                if cr.linkable or not cr.rejection_explanation:
                    continue
                rejections.append(
                    {
                        "anchor_chunk_id": cr.neighborhood.anchor.chunk_id,
                        "neighborhood_chunk_ids": [c.chunk_id for c in cr.neighborhood.chunks],
                        "reason": "llm_refused",
                        "explanation": cr.rejection_explanation,
                    }
                )
            # Post-LLM filter rejections (self_contained, empty_span_*,
            # formula_*, pydantic_validation) — recorded inside
            # ``_compositions_to_questions``.
            rejections.extend(exam_agent.last_downstream_rejections)

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

        stage_funnel["after_composition"] = len(all_candidates)
        if not all_candidates:
            self.logger.warning(
                "No candidate questions survived composition — "
                "the corpus may be too small or topically disjoint for multi-hop synthesis. "
                "See %s for the LLM's per-seed rejection explanations.",
                candidates_path.name,
            )
            raise ExamGenerationFailed(
                n_actual=0,
                n_target=exam_size,
                candidates_path=str(candidates_path),
                top_rejection_reasons=composition_rejection_counter.most_common(3),
                stage_counts=stage_funnel,
            )

        # Oracle answerability gate. For multi-hop candidates, this same
        # call also judges decomposability (the DeBERTa probe is gone); a
        # per-candidate audit log lands in multi_hop_rejections.json.
        validated = await run_validation_pipeline(
            all_candidates,
            documents=doc_map,
            validator_model=validator_model,
            judge_model=validator_model,
            concurrency=self.config.agent.concurrency,
            cache_dir=self.cache_dir,
        )
        self.logger.info("Validation: %d/%d candidates passed", len(validated), len(all_candidates))
        stage_funnel["after_validation"] = len(validated)

        exam = validated

        # Probe discrimination filter — the core selection mechanism.
        # Evaluates every oracle-passed candidate against 2-4 search-space
        # extremes; questions with high outcome variance (some probes solve,
        # others don't) are the most discriminating and are kept first. All-
        # pass (variance=0) and all-fail patterns score 0 and fall to the
        # bottom; ``select_exam`` truncates to exam_size after sorting.
        if examiner.probe_selection and exam:
            labelled_probes = select_probe_configs(
                self.config,
                ranked_llms=ranked_llms,
                ranked_embeds=ranked_embeds,
                ranked_rerankers=ranked_rerankers,
            )
            self.logger.info(
                "Running %d-probe discrimination filter (%d candidates, target %d)",
                len(labelled_probes),
                len(exam),
                exam_size,
            )
            probe_results: list[ExamResult] = []
            successful_probe_labels: list[str] = []
            exam_index_cache: dict[str, RAGIndex] = {}
            # Probe trial-time pipeline takes markdown strings; ``documents`` in
            # this scope is a list[DoclingDocument]. Derive aligned markdown via
            # the same export already cached in doc_map.
            probe_documents = [doc_map[doc_id] for doc_id in doc_ids]

            # Probe runs are diagnostic — per-question MISS/SLOW lines clutter
            # the log without adding signal (we already log per-probe summary
            # stats and outcome patterns). Trial-time optimisation keeps the
            # per-question lines because they help the user understand WHY a
            # specific trial regressed. Tests construct Orchestrator without
            # going through ``__init__`` so ``self.evaluator`` may be absent.
            probe_evaluator = getattr(self, "evaluator", None)
            prev_quiet = getattr(probe_evaluator, "quiet_per_question", False) if probe_evaluator else False
            if probe_evaluator is not None:
                probe_evaluator.quiet_per_question = True

            try:
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
                                probe_documents,
                                probe_structural,
                                corpus_hash=self._corpus_cache_key(),
                                doc_ids=doc_ids,
                            )
                            self._credit_embedding_build(probe_index)
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
                                reasoning_effort=self.config.search_space.generator.reasoning_effort,
                            ),
                            embedder=probe_embedder,
                            index_type=probe_config.index_type,
                            cross_encoder=probe_cross_encoder,
                        )
                        result = await self.evaluator.evaluate(probe_pipeline, exam)
                        probe_results.append(result)
                        successful_probe_labels.append(probe_label)
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
                        # per-reasoning_type accuracy for this probe — tells
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
            finally:
                if probe_evaluator is not None:
                    probe_evaluator.quiet_per_question = prev_quiet

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
                # Per-probe diagnostic dump for top-tier-split patterns
                # (0010, 0001) and all-wrong (0000). The split items are
                # where T3 and T4 disagree — the key discrimination signal
                # at the top of the ladder. The 0000 items are included so
                # we can check whether the gold chunks were retrieved by
                # the strong probes: if yes, the failure was generation
                # (LLM-bottlenecked, low value); if no, the failure was
                # retrieval (genuinely hard, worth keeping). Dump each
                # probe's selected_answer + retrieval_status + retrieved
                # doc ids so we can analyse offline.
                top_split_patterns = {"0010", "0001", "0000"}
                probe_question_maps = [{qr.question_id: qr for qr in pr.question_results} for pr in probe_results]
                audit_path = self.output_dir / "probe_audit_top_split.json"
                audit_records: list[dict] = []
                for q in exam:
                    if not q.probe_outcomes:
                        continue
                    pat = "".join(str(b) for b in q.probe_outcomes)
                    if pat not in top_split_patterns:
                        continue
                    probe_entries = []
                    for probe_idx, qmap in enumerate(probe_question_maps):
                        label = successful_probe_labels[probe_idx]
                        qr = qmap.get(q.id)
                        if qr is None:
                            probe_entries.append({"tier": label, "evaluated": False})
                            continue
                        probe_entries.append(
                            {
                                "tier": label,
                                "correct": bool(qr.correct),
                                "selected_answer": qr.selected_answer,
                                "judge": qr.judge,
                                "refused": bool(qr.refused),
                                "retrieval_status": qr.retrieval_status,
                                "retrieved_doc_ids": list(qr.retrieved_doc_ids),
                            }
                        )
                    gold_cited_chunks = [
                        {"chunk_id": cid, "doc_id": did, "span": span}
                        for cid, did, span in zip(q.source_chunk_ids, q.source_doc_ids, q.source_spans, strict=False)
                    ]
                    audit_records.append(
                        {
                            "question_id": q.id,
                            "pattern": pat,
                            "reasoning_type": q.reasoning_type,
                            "num_hops": q.num_hops,
                            "question": q.question,
                            "canonical_answer": q.canonical_answer,
                            "answer_variants": list(q.answer_variants),
                            "gold_cited_chunks": gold_cited_chunks,
                            "probes": probe_entries,
                        }
                    )
                audit_path.write_text(
                    json.dumps(audit_records, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                self.logger.info(
                    "Probe audit dump (0000/0010/0001 patterns): %d records → %s",
                    len(audit_records),
                    audit_path.name,
                )
                # one sample question per non-empty pattern, so the
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
                # per-(origin, reasoning_type) discrimination means.
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
                # saturation samples: pick up to 3 all-correct and 3
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

        stage_funnel["after_selection"] = len(exam)

        if len(exam) < MIN_EXAM_FRACTION * exam_size:
            self.logger.error(
                "Exam has %d questions (target %d, minimum %d) — failing fast before optimization",
                len(exam),
                exam_size,
                int(MIN_EXAM_FRACTION * exam_size),
            )
            raise ExamGenerationFailed(
                n_actual=len(exam),
                n_target=exam_size,
                candidates_path=str(candidates_path),
                top_rejection_reasons=composition_rejection_counter.most_common(3),
                stage_counts=stage_funnel,
            )
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

        self._persist_exam_cost(exam_cost_path, before_bucket)
        self._flush_exam_gen_cache_events()
        self._current_phase = prev_phase

        return exam, False

    def _save_exam(self, exam: list[OpenEndedQuestion]) -> None:
        """Persist the generated exam to JSON in the shared cache_dir."""
        exam_path = self.cache_dir / "exam.json"
        data = [q.model_dump(mode="json") for q in exam]
        exam_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_frontier_artifacts(self) -> TrialRecord | None:
        """Persist the Pareto frontier (YAMLs, JSON, markdown, ``recommended.yaml``).

        Returns the recommended trial per the configured selection policy,
        or ``None`` when no frontier member satisfies the policy (e.g.
        ``cheapest_above`` with an unmet score threshold).
        """
        records = list(self.history.records)
        if not records:
            return None
        frontier = pareto.compute_frontier(records)
        if not frontier:
            return None

        recommended = pareto.select(records, policy=self._objective)
        recommended_trial = recommended.trial_number if recommended is not None else None
        knee_record = pareto.find_knee(frontier)
        knee_trial = knee_record.trial_number if knee_record is not None else None
        max_score_record = max(frontier, key=lambda r: r.score)
        max_score_trial = max_score_record.trial_number

        cost_values = [float(r.mean_llm_cost_per_query_usd) for r in records]
        cost_ref = max(cost_values) if cost_values else 1.0
        if cost_ref <= 0.0:
            cost_ref = 1.0
        hv = pareto.compute_hypervolume(frontier, ref_point=(0.0, cost_ref))

        self._write_frontier_dir(
            frontier=frontier,
            recommended_trial=recommended_trial,
            knee_trial=knee_trial,
            max_score_trial=max_score_trial,
        )
        self._write_frontier_json(
            frontier=frontier,
            recommended_trial=recommended_trial,
            knee_trial=knee_trial,
            max_score_trial=max_score_trial,
            hypervolume=hv,
        )
        self._write_frontier_report(
            records=records,
            recommended_trial=recommended_trial,
        )
        if recommended is not None:
            self._write_recommended(recommended)
        return recommended

    def _write_frontier_dir(
        self,
        *,
        frontier: list[TrialRecord],
        recommended_trial: int | None,
        knee_trial: int | None,
        max_score_trial: int,
    ) -> None:
        frontier_dir = self.output_dir / "frontier"
        frontier_dir.mkdir(parents=True, exist_ok=True)
        for record in frontier:
            tags: list[str] = []
            if record.trial_number == recommended_trial:
                tags.append("recommended")
            if record.trial_number == knee_trial:
                tags.append("knee")
            if record.trial_number == max_score_trial:
                tags.append("max-score")
            header = [
                f"# Frontier member: trial {record.trial_number}",
                f"# score:    {record.score:.4f}",
                f"# cost/q:   ${record.mean_llm_cost_per_query_usd:.4f}",
                f"# total:    ${record.total_llm_cost_usd:.4f}",
            ]
            if tags:
                header.append(f"# tags:     {', '.join(tags)}")
            payload = record.config.to_prompt_dump(include_graph=self.config.uses_graph())
            body = yaml.safe_dump(payload, sort_keys=False)
            target = frontier_dir / f"trial_{record.trial_number:02d}.yaml"
            target.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
        self.logger.info("Saved frontier configs to %s/ (%d members)", frontier_dir, len(frontier))

    def _write_frontier_json(
        self,
        *,
        frontier: list[TrialRecord],
        recommended_trial: int | None,
        knee_trial: int | None,
        max_score_trial: int,
        hypervolume: float,
    ) -> None:
        members = []
        for record in sorted(frontier, key=lambda r: r.score):
            members.append(
                {
                    "trial_number": record.trial_number,
                    "score": record.score,
                    "cost_per_query_usd": record.mean_llm_cost_per_query_usd,
                    "total_cost_usd": record.total_llm_cost_usd,
                    "is_knee": record.trial_number == knee_trial,
                    "is_max_score": record.trial_number == max_score_trial,
                    "is_recommended": record.trial_number == recommended_trial,
                    "config": record.config.to_prompt_dump(include_graph=self.config.uses_graph()),
                }
            )
        payload = {
            "policy": self._objective.model_dump(mode="json"),
            "recommended_trial": recommended_trial,
            "knee_trial": knee_trial,
            "max_score_trial": max_score_trial,
            "hypervolume": hypervolume,
            "frontier": members,
        }
        target = self.output_dir / "frontier.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved frontier index to %s", target)

    def _write_frontier_report(
        self,
        *,
        records: list[TrialRecord],
        recommended_trial: int | None,
    ) -> None:
        report = render_frontier_report(
            records=records,
            policy=self._objective,
            recommended_trial=recommended_trial,
            include_graph=self.config.uses_graph(),
        )
        target = self.output_dir / "frontier_report.md"
        target.write_text(report, encoding="utf-8")
        self.logger.info("Saved frontier report to %s", target)

    def _write_recommended(self, record: TrialRecord) -> None:
        payload = record.config.to_prompt_dump(include_graph=self.config.uses_graph())
        body = yaml.safe_dump(payload, sort_keys=False)
        header = (
            f"# Recommended config: trial {record.trial_number}\n"
            f"# Selection policy:  {self._objective.describe()}\n"
            f"# score: {record.score:.4f}  cost/q: ${record.mean_llm_cost_per_query_usd:.4f}\n"
        )
        target = self.output_dir / "recommended.yaml"
        target.write_text(header + body, encoding="utf-8")
        self.logger.info("Saved recommended config to %s", target)

    def _report_cost_breakdown(self, ledger: CostLedger) -> None:
        """Log + persist a per-category LLM cost breakdown for the run.

        ``rag_eval`` covers the trial-time generation calls (Pareto-relevant);
        the other buckets cover everything else the optimizer spends on LLMs.
        """
        if not ledger.buckets:
            return
        total = ledger.total_usd()
        # Stable display order; unknown categories appear last alphabetically.
        order = ("rag_eval", "exam_generation", "judge", "agent_proposal", "graph_build")
        ordered = [c for c in order if c in ledger.buckets]
        ordered += sorted(c for c in ledger.buckets if c not in order)
        self.logger.info("LLM cost breakdown:")
        for category in ordered:
            bucket = ledger.buckets[category]
            cache_parts: list[str] = []
            if bucket.cache_read_input_tokens > 0:
                cache_parts.append(f"{bucket.cache_read_input_tokens} cache-read")
            if bucket.cache_creation_input_tokens > 0:
                cache_parts.append(f"{bucket.cache_creation_input_tokens} cache-write")
            cache_suffix = f" [of which {' + '.join(cache_parts)}]" if cache_parts else ""
            self.logger.info(
                "  %-18s $%.4f  (%d call(s), %d prompt + %d completion tokens%s)",
                category,
                bucket.usd,
                bucket.n_calls,
                bucket.prompt_tokens,
                bucket.completion_tokens,
                cache_suffix,
            )
        self.logger.info("  %-18s $%.4f", "TOTAL", total)
        try:
            (self.output_dir / "cost_breakdown.json").write_text(
                json.dumps(ledger.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception:
            self.logger.warning("Failed to write cost_breakdown.json", exc_info=True)

    async def _propose_next_config_with_retries(
        self,
        result: ExamResult,
        exam: list[OpenEndedQuestion],
        current_config: TrialConfig,
        *,
        trial_number: int,
        trials_remaining: int,
        previous_strategy: Strategy | None,
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
                    previous_strategy=previous_strategy,
                )
            except Exception:
                self.logger.exception("Agent proposal attempt %d/5 failed", attempt)
        self.logger.error("Agent failed after 5 retries; reusing previous config")
        return None, None, current_config, None

    @staticmethod
    def _setup_logger(output_dir: Path) -> logging.Logger:
        """Configure run logger + parent ``agentic_autorag`` logger so high-
        signal setup lines (NER backend, entity histogram, prepared-corpus
        stats) reach both ``run.log`` and the console even without
        ``--verbose``."""
        formatter = logging.Formatter("%(message)s")
        log_path = output_dir / "run.log"
        # Truncate once explicitly; both file handlers below open in "a" so
        # they cooperate via O_APPEND instead of racing on file offsets.
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

    def _log_config_summary(self, label: str, config: TrialConfig) -> None:
        reasoning_tag = " +reasoning" if config.reasoning else ""
        llm_summary = _format_per_stage_llm(config)
        self.logger.info(
            "%s | chunk=%s strategy=%s embed=%s index=%s top_k=%s alpha=%s fusion=%s reorder=%s "
            "compressor=%s expansion=%s reranker=%s rerank_top_n=%s llm=%s%s temp=%s",
            label,
            config.chunk_token_size,
            config.chunking_strategy,
            config.embedding_model,
            config.index_type.value,
            config.top_k,
            config.hybrid_alpha,
            config.bm25_vector_fusion,
            config.long_context_reorder,
            config.passage_compressor,
            config.query_expansion,
            config.reranker,
            config.reranker_top_n,
            llm_summary,
            reasoning_tag,
            config.temperature,
        )

    @staticmethod
    def _diff_pairs(old: TrialConfig, new: TrialConfig) -> list[tuple[str, object, object]]:
        """All config lever pairs the optimizer can change. Iterates
        ``CONFIG_LEVER_FIELDS`` so this stays in sync with the agent-facing
        diff (``state._config_diff_summary``)."""
        pairs: list[tuple[str, object, object]] = []
        for name in CONFIG_LEVER_FIELDS:
            ov = getattr(old, name, None)
            nv = getattr(new, name, None)
            ov = getattr(ov, "value", ov)
            nv = getattr(nv, "value", nv)
            pairs.append((name, ov, nv))
        return pairs

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
