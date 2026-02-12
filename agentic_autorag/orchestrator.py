"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import MCQQuestion, SearchSpace, TrialConfig
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


class Orchestrator:
    """Main optimization loop that ties all components together."""

    def __init__(self, config_path: str) -> None:
        self.search_space: SearchSpace = load_config(config_path)
        meta = self.search_space.meta

        self.output_dir = Path(meta.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger(self.output_dir)

        self.history = HistoryLog(path=str(self.output_dir / "history.jsonl"))
        self.agent = ReasoningAgent(
            agent_model=self.search_space.agent.optimizer_model,
            search_space=self.search_space,
            history=self.history,
        )
        self.evaluator = MCQEvaluator()
        self.irt_analyzer = IRTAnalyzer(
            discrimination_threshold=self.search_space.examiner.irt_discrimination_threshold,
        )

        parser_name = self.search_space.structural.parsers[0]
        self.parser = build_parser(parser_name)

        self.index_builder = IndexBuilder(
            db_path=str(self.output_dir / "lancedb"),
        )
        self.registry = IndexRegistry(str(self.output_dir / "indices")) if meta.index_registry else None
        self._exam_chunks: list[str] = []
        self._exam_chunk_ids: list[str] = []
        self._exam_embeddings: np.ndarray | None = None
        self._exam_embedding_model: SentenceTransformer | None = None
        self._latest_irt_summary: str = ""

    async def run(self) -> TrialRecord:
        """Run the full optimization loop and return the best trial."""
        t_start = time.monotonic()
        meta = self.search_space.meta

        # 1. Parse corpus
        self.logger.info("Loading corpus from %s", meta.corpus_path)
        t0 = time.monotonic()
        documents = self._load_and_parse_corpus()
        self.logger.info("Loaded %d document(s) in %.2fs", len(documents), time.monotonic() - t0)
        if not documents:
            raise RuntimeError(f"No documents found in {meta.corpus_path}")

        # 2. Generate exam
        self.logger.info("Generating MCQ exam")
        t0 = time.monotonic()
        exam, chunks, chunk_ids, embeddings, exam_embedding_model = await self._generate_exam(documents)
        self._exam_chunks = chunks
        self._exam_chunk_ids = chunk_ids
        self._exam_embeddings = embeddings
        self._exam_embedding_model = exam_embedding_model
        self._save_exam(exam)
        self.logger.info("Generated %d questions in %.2fs", len(exam), time.monotonic() - t0)
        self.logger.info("Saved exam to %s", self.output_dir / "exam.json")

        # 3. Agent proposes initial config
        self.logger.info("Agent proposing initial configuration")
        t0 = time.monotonic()
        current_config = await self.agent.propose_initial(
            corpus_description=meta.corpus_description,
        )
        self.logger.info("Initial config received in %.2fs", time.monotonic() - t0)
        self._log_config_summary("Initial config", current_config)

        # 4. Optimization loop
        best: TrialRecord | None = None
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
                if self.registry and self.registry.has(fingerprint):
                    index = RAGIndex.load(self.registry.get(fingerprint))
                    index_source = "cache"
                    self.logger.info("Loaded cached index %s", fingerprint)
                else:
                    index = await self.index_builder.build(
                        documents,
                        current_config.structural,
                        current_config.graph,
                    )
                    if self.registry:
                        staging = self.output_dir / ".index_staging" / fingerprint
                        if staging.exists():
                            shutil.rmtree(staging)
                        index.save(staging)
                        self.registry.register(fingerprint, staging, current_config.structural)
                        shutil.rmtree(staging)
                        self.logger.info("Registered index %s in cache", fingerprint)
                index_elapsed = time.monotonic() - t0
            except Exception:
                self.logger.exception("Index build/load failed for trial %d; skipping trial", trial_num)
                continue

            # b. Construct pipeline
            embedder = SentenceTransformer(current_config.structural.embedding_model)
            pipeline = RAGPipeline(
                vector_store=index.vector_store,
                graph_store=index.graph_store,
                config=current_config.runtime,
                embedder=embedder,
                index_type=current_config.structural.index_type,
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

            refresh_interval = self.search_space.examiner.refresh_interval_trials
            if trial_num >= 2 and trial_num % refresh_interval == 0 and len(self.history.records) >= 2:
                self.logger.info("Running IRT exam refinement")
                response_matrix = self.history.get_response_matrix_for_exam(
                    {question.id for question in exam}
                )
                if (
                    response_matrix is not None
                    and self._exam_embeddings is not None
                    and self._exam_embedding_model is not None
                ):
                    try:
                        exam_refiner = ExamRefiner(
                            irt_analyzer=self.irt_analyzer,
                            exam_agent=ExamAgent(
                                config=self.search_space.examiner,
                                examiner_model=self.search_space.agent.examiner_model,
                                embedding_model=self._exam_embedding_model,
                                corpus_description=self.search_space.meta.corpus_description,
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

        # 5. Summary
        elapsed = time.monotonic() - t_start
        best = self.history.get_best()
        self._save_best_config(best)
        self.logger.info("Optimization complete in %.2fs", elapsed)
        if best:
            self.logger.info("Best score %.3f", best.score)
        else:
            self.logger.info("No successful trials completed")
        return best

    def _load_and_parse_corpus(self) -> list[str]:
        """Recursively discover files in corpus_path and parse to text."""
        corpus_path = Path(self.search_space.meta.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

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

        return documents

    async def _generate_exam(
        self,
        documents: list[str],
    ) -> tuple[list[MCQQuestion], list[str], list[str], np.ndarray, SentenceTransformer]:
        """Chunk, embed, and generate MCQ exam from the corpus."""
        self.logger.info("Chunking documents for exam generation")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks: list[str] = []
        for doc in documents:
            chunks.extend(splitter.split_text(doc))
        self.logger.info("Created %d chunks from %d documents", len(chunks), len(documents))

        chunk_ids = [f"exam_chunk_{i}" for i in range(len(chunks))]

        self.logger.info("Embedding exam chunks")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.asarray(embedder.encode(chunks, show_progress_bar=True), dtype=np.float32)
        self.logger.info("Exam embeddings shape: %s", embeddings.shape)

        exam_agent = ExamAgent(
            config=self.search_space.examiner,
            examiner_model=self.search_space.agent.examiner_model,
            embedding_model=embedder,
            corpus_description=self.search_space.meta.corpus_description,
        )
        exam = await exam_agent.generate_exam(chunks, chunk_ids, embeddings)
        return exam, chunks, chunk_ids, embeddings, embedder

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

        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        run_logger.addHandler(console_handler)
        run_logger.addHandler(file_handler)
        return run_logger

    def _random_tweak(self, config: TrialConfig) -> TrialConfig:
        """Apply one random runtime parameter change as a fallback."""
        data = config.model_dump()
        runtime = data["runtime"]
        ss = self.search_space.runtime

        param = random.choice(["top_k", "temperature", "hybrid_alpha"])
        if param == "top_k":
            runtime["top_k"] = random.randint(
                int(ss.retrieval.top_k.min),
                int(ss.retrieval.top_k.max),
            )
        elif param == "temperature":
            runtime["temperature"] = round(
                random.uniform(ss.generation.temperature.min, ss.generation.temperature.max),
                2,
            )
        elif param == "hybrid_alpha":
            runtime["hybrid_alpha"] = round(
                random.uniform(ss.retrieval.hybrid_alpha.min, ss.retrieval.hybrid_alpha.max),
                2,
            )

        return TrialConfig.model_validate(data)

    @staticmethod
    def _print_trial_header(trial_num: int, max_trials: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"  TRIAL {trial_num}/{max_trials}")
        print(f"{'=' * 60}")

    def _log_config_summary(self, label: str, config: TrialConfig) -> None:
        s = config.structural
        r = config.runtime
        self.logger.info(
            "%s | chunk=%s strategy=%s embed=%s index=%s top_k=%s reranker=%s llm=%s temp=%s",
            label,
            s.chunk_size,
            s.chunking_strategy,
            s.embedding_model,
            s.index_type.value,
            r.top_k,
            r.reranker,
            r.llm_model,
            r.temperature,
        )

    @staticmethod
    def _print_config_summary(label: str, config: TrialConfig) -> None:
        s = config.structural
        r = config.runtime
        print(f"   {label}:")
        print(f"     chunk={s.chunk_size}, strategy={s.chunking_strategy}, embed={s.embedding_model}")
        print(f"     index={s.index_type.value}, top_k={r.top_k}, reranker={r.reranker}")
        print(f"     llm={r.llm_model}, temp={r.temperature}")

    @staticmethod
    def _print_config_diff(old: TrialConfig, new: TrialConfig) -> None:
        """Print which key parameters changed between configs."""
        changes: list[str] = []
        pairs = [
            ("chunk_size", old.structural.chunk_size, new.structural.chunk_size),
            ("chunk_overlap", old.structural.chunk_overlap, new.structural.chunk_overlap),
            ("chunking_strategy", old.structural.chunking_strategy, new.structural.chunking_strategy),
            ("embedding_model", old.structural.embedding_model, new.structural.embedding_model),
            ("index_type", old.structural.index_type.value, new.structural.index_type.value),
            ("top_k", old.runtime.top_k, new.runtime.top_k),
            ("hybrid_alpha", old.runtime.hybrid_alpha, new.runtime.hybrid_alpha),
            ("reranker", old.runtime.reranker, new.runtime.reranker),
            ("llm_model", old.runtime.llm_model, new.runtime.llm_model),
            ("temperature", old.runtime.temperature, new.runtime.temperature),
            ("query_expansion", old.runtime.query_expansion, new.runtime.query_expansion),
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
            ("chunk_size", old.structural.chunk_size, new.structural.chunk_size),
            ("chunk_overlap", old.structural.chunk_overlap, new.structural.chunk_overlap),
            ("chunking_strategy", old.structural.chunking_strategy, new.structural.chunking_strategy),
            ("embedding_model", old.structural.embedding_model, new.structural.embedding_model),
            ("index_type", old.structural.index_type.value, new.structural.index_type.value),
            ("top_k", old.runtime.top_k, new.runtime.top_k),
            ("hybrid_alpha", old.runtime.hybrid_alpha, new.runtime.hybrid_alpha),
            ("reranker", old.runtime.reranker, new.runtime.reranker),
            ("llm_model", old.runtime.llm_model, new.runtime.llm_model),
            ("temperature", old.runtime.temperature, new.runtime.temperature),
            ("query_expansion", old.runtime.query_expansion, new.runtime.query_expansion),
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
