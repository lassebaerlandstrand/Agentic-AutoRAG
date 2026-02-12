"""Main orchestration loop: build → eval → diagnose → propose."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import MCQQuestion, SearchSpace, TrialConfig
from agentic_autorag.engine.index_builder import IndexBuilder
from agentic_autorag.engine.parsers import build_parser
from agentic_autorag.engine.pipeline import RAGPipeline
from agentic_autorag.examiner.evaluator import ExamResult, MCQEvaluator
from agentic_autorag.examiner.exam_agent import ExamAgent
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent

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

        self.history = HistoryLog(path=str(self.output_dir / "history.jsonl"))
        self.agent = ReasoningAgent(
            agent_model=self.search_space.agent.optimizer_model,
            search_space=self.search_space,
            history=self.history,
        )
        self.evaluator = MCQEvaluator()

        parser_name = self.search_space.structural.parsers[0]
        self.parser = build_parser(parser_name)

        self.index_builder = IndexBuilder(
            db_path=str(self.output_dir / "lancedb"),
        )

    async def run(self) -> TrialRecord:
        """Run the full optimization loop and return the best trial."""
        t_start = time.monotonic()
        meta = self.search_space.meta

        # 1. Parse corpus
        print(f"\n📂 Loading corpus from {meta.corpus_path}")
        t0 = time.monotonic()
        documents = self._load_and_parse_corpus()
        print(f"   ✓ {len(documents)} document(s) loaded in {time.monotonic() - t0:.1f}s")
        if not documents:
            raise RuntimeError(f"No documents found in {meta.corpus_path}")

        # 2. Generate exam
        print("\n📝 Generating MCQ exam...")
        t0 = time.monotonic()
        exam = await self._generate_exam(documents)
        self._save_exam(exam)
        print(f"   ✓ {len(exam)} questions generated in {time.monotonic() - t0:.1f}s")
        print(f"   Saved to {self.output_dir / 'exam.json'}")

        # 3. Agent proposes initial config
        print("\n🤖 Agent proposing initial configuration...")
        t0 = time.monotonic()
        current_config = await self.agent.propose_initial(
            corpus_description=meta.corpus_description,
        )
        print(f"   ✓ Initial config received in {time.monotonic() - t0:.1f}s")
        self._print_config_summary("Initial config", current_config)

        # 4. Optimization loop
        best: TrialRecord | None = None
        for trial_num in range(1, meta.max_trials + 1):
            self._print_trial_header(trial_num, meta.max_trials)

            # a. Build index (rebuild every time in v1)
            s = current_config.structural
            print(f"   🏗️  Building index (chunk={s.chunk_size}, embed={s.embedding_model})...")
            t0 = time.monotonic()
            index = await self.index_builder.build(
                documents,
                current_config.structural,
                current_config.graph,
            )
            print(f"   ✓ Index built in {time.monotonic() - t0:.1f}s")

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
            print(f"   📊 Evaluating {len(exam)} questions...")
            t0 = time.monotonic()
            result: ExamResult = await self.evaluator.evaluate(pipeline, exam)
            score_elapsed = time.monotonic() - t0
            print(f"   ✓ Score: {result.score:.3f} ({result.n_correct}/{result.n_total}) in {score_elapsed:.1f}s")

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

            # e. Last trial — no need to propose next
            if trial_num == meta.max_trials:
                break

            # f. Agent analyzes failures and proposes next config
            try:
                print("   🧠 Agent diagnosing and proposing next config...")
                t0 = time.monotonic()
                error_trace, next_config = await self.agent.analyze_and_propose(
                    result,
                    current_config,
                )
                print(f"   ✓ Next config received in {time.monotonic() - t0:.1f}s")
                # Update the record's error_trace retroactively
                record.error_trace = error_trace

                self._print_config_diff(current_config, next_config)
                current_config = next_config

            except Exception:
                logger.exception("Agent failed to propose next config — applying random tweak")
                current_config = self._random_tweak(current_config)
                print("   ⚠ Agent failed; applied random parameter tweak as fallback.")

        # 5. Summary
        elapsed = time.monotonic() - t_start
        best = self.history.get_best()
        self._print_summary(best, meta.max_trials, elapsed)
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
            print(f"   Skipped {skipped} unsupported file(s)")
        if failed:
            print(f"   ⚠ Failed to parse {failed} file(s)")

        return documents

    async def _generate_exam(self, documents: list[str]) -> list[MCQQuestion]:
        """Chunk, embed, and generate MCQ exam from the corpus."""
        print("   Chunking documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks: list[str] = []
        for doc in documents:
            chunks.extend(splitter.split_text(doc))
        print(f"   ✓ {len(chunks)} chunks from {len(documents)} documents")

        chunk_ids = [f"exam_chunk_{i}" for i in range(len(chunks))]

        print("   Embedding chunks...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.asarray(embedder.encode(chunks, show_progress_bar=True), dtype=np.float32)
        print(f"   ✓ Embeddings: {embeddings.shape}")

        exam_agent = ExamAgent(
            config=self.search_space.examiner,
            examiner_model=self.search_space.agent.examiner_model,
        )
        return await exam_agent.generate_exam(chunks, chunk_ids, embeddings)

    def _save_exam(self, exam: list[MCQQuestion]) -> None:
        """Persist the generated exam to JSON."""
        exam_path = self.output_dir / "exam.json"
        data = [q.model_dump(mode="json") for q in exam]
        exam_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
