"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import MCQQuestion, ProjectConfig, TrialConfig
from agentic_autorag.engine.graph_store import LightRAGStore
from agentic_autorag.engine.index_builder import IndexBuilder, RAGIndex
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner.evaluator import ExamResult, MCQEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent
from agentic_autorag.examiner.exam_refiner import ExamRefiner
from agentic_autorag.examiner.irt import IRTAnalyzer
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

        if provider_prefix in ("ollama", "sentence-transformers"):
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
            knowledge_base = KnowledgeBase()
        except Exception as e:
            logger.warning("Could not load knowledge base: %s. Agent will run without model context.", e)
            knowledge_base = None

        self.agent = ReasoningAgent(
            agent_model=self.config.agent.optimizer_model,
            config=self.config,
            history=self.history,
            debug_prompts=debug_prompts,
            knowledge_base=knowledge_base,
        )
        self.evaluator = MCQEvaluator(concurrency=self.config.agent.concurrency)
        self.irt_analyzer = IRTAnalyzer(
            discrimination_threshold=self.config.examiner.irt_discrimination_threshold,
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

        self._exam_chunks: list[str] = []
        self._exam_chunk_ids: list[str] = []
        self._exam_embeddings: np.ndarray | None = None
        self._exam_embedding_model: SentenceTransformer | None = None
        self._latest_irt_summary: str = ""

    async def run(self) -> TrialRecord:
        """Run the full optimization loop and return the best trial."""
        t_start = time.monotonic()
        meta = self.config.meta

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
        exam, chunks, chunk_ids, embeddings, exam_embedding_model, from_cache = await self._generate_exam(documents)
        self._exam_chunks = chunks
        self._exam_chunk_ids = chunk_ids
        self._exam_embeddings = embeddings
        self._exam_embedding_model = exam_embedding_model
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
                        current_config.chunk_size,
                        current_config.chunking_strategy,
                    )
                    index = await self.index_builder.build(
                        documents,
                        current_config.to_structural(),
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

            # b. Construct pipeline
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
                "Score %.3f (%d/%d) in %.2fs",
                result.score,
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
            )
            self.history.add(record)
            if best is None or result.score > best.score:
                best = record

            refresh_interval = self.config.examiner.refresh_interval_trials
            if trial_num >= 2 and trial_num % refresh_interval == 0 and len(self.history.records) >= 2:
                self.logger.info("Running IRT exam refinement")
                response_matrix = self.history.get_response_matrix_for_exam({question.id for question in exam})
                if (
                    response_matrix is not None
                    and self._exam_embeddings is not None
                    and self._exam_embedding_model is not None
                ):
                    try:
                        exam_refiner = ExamRefiner(
                            irt_analyzer=self.irt_analyzer,
                            exam_agent=ExamAgent(
                                config=self.config.examiner,
                                examiner_model=self.config.agent.examiner_model,
                                embedding_model=self._exam_embedding_model,
                                corpus_description=self.config.meta.corpus_description,
                                concurrency=self.config.agent.concurrency,
                            ),
                            drop_ratio=0.1,
                        )
                        exam = await exam_refiner.refine(
                            exam=exam,
                            response_matrix=response_matrix,
                            chunks=self._exam_chunks,
                            chunk_ids=self._exam_chunk_ids,
                            embeddings=self._exam_embeddings,
                        )
                        self._save_exam(exam)

                        irt_result = self.irt_analyzer.fit(response_matrix)
                        weak_questions = self.irt_analyzer.identify_weak_questions(irt_result.discriminations)
                        self._latest_irt_summary = (
                            "## Exam Quality (IRT Analysis)\n"
                            "- Questions below discrimination threshold: "
                            f"{len(weak_questions)}/{len(irt_result.discriminations)}\n"
                            f"- Mean discrimination: {float(np.mean(irt_result.discriminations)):.2f}\n"
                            f"- Mean difficulty: {float(np.mean(irt_result.difficulties)):.2f}\n"
                            f"- Ability range across trials: "
                            f"[{float(np.min(irt_result.abilities)):.2f}, {float(np.max(irt_result.abilities)):.2f}]"
                        )
                        self.logger.info("IRT refinement complete")
                    except Exception:
                        self.logger.exception("IRT refinement failed")

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
                "diversity_clusters": examiner.diversity_clusters,
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
    ) -> tuple[list[MCQQuestion], list[str], list[str], np.ndarray, SentenceTransformer]:
        """Chunk, embed, and generate MCQ exam from the corpus.

        Chunks and embeddings are always computed (needed for IRT refinement).
        The initial MCQ list is cached at .cache/exam_{key}.json so that
        re-runs skip the expensive LLM generation calls.
        """
        self.logger.info("Chunking documents for exam generation")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks: list[str] = []
        for doc in documents:
            chunks.extend(splitter.split_text(doc))
        self.logger.info("Created %d chunks from %d documents", len(chunks), len(documents))

        chunk_ids = [f"exam_chunk_{i}" for i in range(len(chunks))]

        self.logger.info("Embedding exam chunks")
        embedder = self.index_builder.get_embedder(self.config.examiner.embedding_model)
        embeddings = np.asarray(embedder.encode(chunks, show_progress_bar=True), dtype=np.float32)
        self.logger.info("Exam embeddings shape: %s", embeddings.shape)

        # Check cache — only the initial (pre-IRT) exam is cached
        cache_dir = self.output_dir / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        exam_cache_path = cache_dir / f"exam_{self._exam_cache_key()}.json"

        if exam_cache_path.exists():
            self.logger.info("Loading cached initial exam from %s", exam_cache_path.name)
            try:
                raw = json.loads(exam_cache_path.read_text(encoding="utf-8"))
                exam = [MCQQuestion.model_validate(q) for q in raw]
                return exam, chunks, chunk_ids, embeddings, embedder, True
            except Exception:
                self.logger.warning("Exam cache corrupted; regenerating", exc_info=True)

        exam_agent = ExamAgent(
            config=self.config.examiner,
            examiner_model=self.config.agent.examiner_model,
            embedding_model=embedder,
            corpus_description=self.config.meta.corpus_description,
            concurrency=self.config.agent.concurrency,
        )
        exam = await exam_agent.generate_exam(chunks, chunk_ids, embeddings)

        try:
            exam_cache_path.write_text(
                json.dumps([q.model_dump(mode="json") for q in exam], indent=2),
                encoding="utf-8",
            )
            self.logger.info("Cached initial exam to %s", exam_cache_path.name)
        except Exception:
            self.logger.warning("Failed to write exam cache", exc_info=True)

        return exam, chunks, chunk_ids, embeddings, embedder, False

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
                    irt_summary=self._latest_irt_summary,
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
            config.chunk_size,
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
        print(f"     chunk={config.chunk_size}, strategy={config.chunking_strategy}, embed={config.embedding_model}")
        print(f"     index={config.index_type.value}, top_k={config.top_k}, reranker={config.reranker}")
        print(f"     llm={config.llm_model}{reasoning_tag}, temp={config.temperature}")

    @staticmethod
    def _print_config_diff(old: TrialConfig, new: TrialConfig) -> None:
        """Print which key parameters changed between configs."""
        changes: list[str] = []
        pairs = [
            ("chunk_size", old.chunk_size, new.chunk_size),
            ("chunk_overlap", old.chunk_overlap, new.chunk_overlap),
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
            ("chunk_size", old.chunk_size, new.chunk_size),
            ("chunk_overlap", old.chunk_overlap, new.chunk_overlap),
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
