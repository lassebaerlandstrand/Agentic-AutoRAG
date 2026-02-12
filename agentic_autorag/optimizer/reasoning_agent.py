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

from agentic_autorag.config.models import SearchSpace, TrialConfig
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
        search_space: SearchSpace,
        history: HistoryLog,
    ) -> None:
        self.model = agent_model
        self.search_space = search_space
        self.history = history

    async def propose_initial(self, corpus_description: str) -> TrialConfig:
        """Propose the first configuration based on corpus description."""
        prompt = INITIAL_PROPOSAL_PROMPT.format(
            corpus_description=corpus_description,
            search_space=self.search_space.to_agent_prompt(),
        )
        return await self._call_and_validate(prompt)

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

        prompt = DIAGNOSTIC_PROMPT.format(
            failed_questions=self._format_failures(sample),
            current_config=config.model_dump_json(indent=2),
        )
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def _propose(self, error_trace: str, current_config: TrialConfig) -> TrialConfig:
        """Propose the next configuration based on error trace and history."""
        history_text = self.history.format_for_agent(
            last_n=self.search_space.agent.max_history_trials,
        )

        prompt = PROPOSAL_PROMPT.format(
            error_trace=error_trace,
            current_config=current_config.model_dump_json(indent=2),
            history=history_text,
            search_space=self.search_space.to_agent_prompt(),
        )
        return await self._call_and_validate(prompt)

    async def _call_and_validate(self, prompt: str) -> TrialConfig:
        """Call LLM, extract YAML, validate, and retry on failure."""
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                )
                raw = response.choices[0].message.content
                yaml_dict = self._extract_yaml(raw)
                config = TrialConfig.model_validate(yaml_dict)

                # Check search space violations
                violations = self.search_space.validate_trial(config)
                if violations:
                    violation_msg = "Search space violations:\n" + "\n".join(f"- {v}" for v in violations)
                    raise ValueError(violation_msg)

                return config

            except Exception as e:
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    # Feed error back to LLM for self-healing
                    messages.append({"role": "assistant", "content": raw if "raw" in dir() else ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your response had an error: {e}\n\n"
                            "Please fix the issue and output a corrected ```yaml block."
                        ),
                    })

        raise RuntimeError(f"Failed to get valid config after {MAX_RETRIES} attempts")

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
                f"Generated response: {qr.generated_response}\n"
                f"Retrieved context:\n{qr.retrieved_context}\n"
            )
            blocks.append(block)
        return "\n".join(blocks)
