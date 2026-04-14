"""Two-stage reasoning agent for RAG optimization.

Stage 1 (diagnose): Analyze why a configuration failed.
Stage 2 (propose): Propose the next configuration based on diagnosis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import litellm
import yaml

from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.models import ProjectConfig, TrialConfig
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.history import HistoryLog

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

DIAGNOSTIC_PROMPT = (_PROMPTS_DIR / "diagnostic.txt").read_text(encoding="utf-8")
PROPOSAL_PROMPT = (_PROMPTS_DIR / "proposal.txt").read_text(encoding="utf-8")
INITIAL_PROPOSAL_PROMPT = (_PROMPTS_DIR / "initial_proposal.txt").read_text(encoding="utf-8")

MAX_RETRIES = 3


class ReasoningAgent:
    """Two-stage reasoning agent for RAG optimization.

    Uses a shared HistoryLog (JSONL) as the single source of truth for trial
    history, rather than maintaining its own internal list.
    """

    def __init__(
        self,
        agent_model: str,
        config: ProjectConfig,
        history: HistoryLog,
        debug_prompts: bool = False,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self.model = agent_model
        self.config = config
        self.history = history
        self.debug_prompts = debug_prompts
        self.knowledge_base = knowledge_base

    def _log_exchange(self, stage: str, prompt: str, response: str) -> None:
        """Write a formatted prompt/response block to run.log at DEBUG level."""
        if not self.debug_prompts:
            return
        sep = "═" * 64
        logging.getLogger("agentic_autorag.run").debug(
            "\n%s\n  PROMPT → %s\n%s\n%s\n\n%s\n  RESPONSE ← %s\n%s\n%s\n%s",
            sep,
            stage,
            sep,
            prompt,
            sep,
            stage,
            sep,
            response,
            sep,
        )

    async def propose_initial(self, corpus_description: str) -> TrialConfig:
        """Propose the first configuration based on corpus description."""
        prompt = INITIAL_PROPOSAL_PROMPT.format(
            corpus_description=corpus_description,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
        )
        return await self._call_and_validate(prompt, stage="Initial Proposer")

    async def analyze_and_propose(
        self,
        exam_result: ExamResult,
        current_config: TrialConfig,
    ) -> tuple[str, TrialConfig]:
        """Run the two-stage loop: diagnose failures, then propose next config.

        Returns (error_trace, next_config). The caller is responsible for
        adding the completed trial to the history log.
        """
        error_trace = await self._diagnose(exam_result, current_config)
        next_config = await self._propose(error_trace, current_config)
        return error_trace, next_config

    async def _diagnose(self, result: ExamResult, config: TrialConfig) -> str:
        """Produce a structured error trace from failed exam questions."""
        failed = [q for q in result.question_results if not q.correct]
        sample = failed[:15]

        # Surface "lucky" questions: MCQ correct but poor retrieval (rank > 5 or not found)
        lucky = [q for q in result.question_results if q.correct and q.retrieval_mrr < 0.2]
        lucky_section = ""
        if lucky:
            lucky_section = (
                f"\n\n### Lucky questions (correct answer despite poor retrieval): {len(lucky)}\n"
                + "\n".join(
                    f"- {q.question_id}: source_fact_rank={q.source_fact_rank} (MRR={q.retrieval_mrr:.2f})"
                    for q in lucky[:5]
                )
            )

        failed_questions = self._format_failures(sample) + lucky_section

        prompt = DIAGNOSTIC_PROMPT.format(
            failed_questions=failed_questions,
            current_config=config.model_dump_json(indent=2),
        )
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        self._log_exchange("Diagnoser", prompt, raw)
        return raw

    async def _propose(self, error_trace: str, current_config: TrialConfig) -> TrialConfig:
        """Propose the next configuration based on error trace and history."""
        history_text = self.history.format_for_agent(
            last_n=self.config.agent.max_history_trials,
        )

        prompt = PROPOSAL_PROMPT.format(
            error_trace=error_trace,
            current_config=current_config.model_dump_json(indent=2),
            history=history_text,
            search_space=self.config.to_agent_prompt(),
            knowledge_base=self._kb_text(),
        )
        return await self._call_and_validate(prompt, stage="Proposer")

    async def _call_and_validate(self, prompt: str, stage: str = "Proposer") -> TrialConfig:
        """Call LLM, extract YAML, validate, and retry on failure."""
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                )
                raw = response.choices[0].message.content
                self._log_exchange(stage, messages[-1]["content"], raw)
                yaml_dict = self._extract_yaml(raw)
                config = TrialConfig.model_validate(yaml_dict)

                # Check search space violations
                violations = self.config.validate_trial(config)
                if violations:
                    violation_msg = "Search space violations:\n" + "\n".join(f"- {v}" for v in violations)
                    raise ValueError(violation_msg)

                return config

            except Exception as e:
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    # Feed error back to LLM for self-healing
                    messages.append({"role": "assistant", "content": raw if "raw" in dir() else ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response had an error: {e}\n\n"
                                "Please fix the issue and output a corrected ```yaml block."
                            ),
                        }
                    )

        raise RuntimeError(f"Failed to get valid config after {MAX_RETRIES} attempts")

    def _kb_text(self) -> str:
        """Return formatted knowledge base text, or empty string if not available."""
        if self.knowledge_base is None:
            return ""
        ss = self.config.search_space
        reasoning_allowed = {m: ss.is_reasoning_allowed(m) for m in ss.llm_models}
        return self.knowledge_base.format_for_prompt(
            llm_models=ss.llm_models,
            embedding_models=ss.embedding_models,
            reranker_models=ss.reranker.models,
            reasoning_allowed=reasoning_allowed,
        )

    @staticmethod
    def _extract_yaml(text: str) -> dict:
        """Extract a YAML block from agent response text."""
        # Try ```yaml ... ``` first, then bare ``` ... ```
        match = re.search(r"```ya?ml\n(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in agent response")
        return yaml.safe_load(match.group(1))

    @staticmethod
    def _format_failures(failures: list[QuestionResult]) -> str:
        """Format failed questions as readable blocks for the diagnostic prompt."""
        blocks = []
        for i, qr in enumerate(failures, 1):
            block = (
                f"### Failure {i}\n"
                f"Question ID: {qr.question_id}\n"
                f"Correct answer: {qr.correct_answer}\n"
                f"Selected answer: {qr.selected_answer}\n"
                f"Source fact rank: {qr.source_fact_rank} (MRR: {qr.retrieval_mrr:.2f})\n"
                f"Generated response: {qr.generated_response}\n"
                f"Retrieved context:\n{qr.retrieved_context}\n"
            )
            blocks.append(block)
        return "\n".join(blocks)
