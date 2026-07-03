"""Tests for the OPRO compact-history proposer mode.

OPRO = the naive LLM-proposer baseline: ``compact_history=True`` +
``knowledge_base=None`` + ``use_diagnosis=False``. The proposer sees only a
one-line ``config -> accuracy`` trajectory, with no knowledge base, no
diagnosis, and no rich per-trial blocks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agentic_autorag.config.models import (
    AgentConfig,
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    NumericRange,
    OpenEndedQuestion,
    ProjectConfig,
    RetrievalSearchSpace,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord, _compact_history, _config_signature
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent

_ALL_TUNABLE: set[str] = {
    "chunking_strategy",
    "chunk_token_size",
    "chunk_token_overlap",
    "embedding_model",
    "index_type",
    "top_k",
    "hybrid_alpha",
    "bm25_vector_fusion",
    "long_context_reorder",
    "passage_compressor",
    "reranker",
    "reranker_top_n",
    "query_expansion",
    "generator_llm",
    "temperature",
    "reasoning",
}


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        reranker_top_n=5,
        generator_llm="ollama/llama3.2",
        temperature=0.0,
        reasoning=False,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_record(trial_number: int, score: float, config: TrialConfig | None = None) -> TrialRecord:
    return TrialRecord(
        trial_number=trial_number,
        config=config or _make_config(),
        answer_accuracy=score,
        question_results=[],
    )


def _make_project_config() -> ProjectConfig:
    return ProjectConfig(
        search_space=SearchSpace(
            embedding=EmbeddingSearchSpace(models=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]),
            retrieval=RetrievalSearchSpace(index_types=[IndexType.VECTOR_ONLY]),
            generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/llama3.1"]),
            temperature=NumericRange(min=0.0, max=1.0),
        ),
        agent=AgentConfig(
            optimizer_model="test-model",
            examiner_model="test-model",
            judge_model="test-model",
        ),
    )


def _make_exam_question(qid: str = "q1") -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"What does {qid} ask?",
        canonical_answer="alpha",
        answer_variants=["alpha-2"],
        reasoning_type="bridge",
        source_doc_ids=["docA", "docB"],
        source_spans=[f"chunk A span for {qid}", f"chunk B span for {qid}"],
    )


VALID_PROPOSER_YAML = """\
Trying a stronger embedding model.

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: BAAI/bge-m3
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
generator_llm: ollama/llama3.2
temperature: 0.0
reasoning: false
meta:
  rationale: "Score history is flat; trying bge-m3."
  strategy:
    phase: ceiling
    plan: "score history flat; retrieval limits now; trying a bigger embedder next."
    notes: "flat scores so far across small embedders."
```
"""


VALID_PROPOSER_YAML_NO_STRATEGY = """\
Trying a stronger embedding model. No campaign plan (OPRO).

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: BAAI/bge-m3
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
generator_llm: ollama/llama3.2
temperature: 0.0
reasoning: false
meta:
  rationale: "Score history is flat; trying bge-m3."
```
"""


VALID_INITIAL_CONFIG_YAML = """\
Starting with a capable LLM and a strong embedder.

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: BAAI/bge-m3
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
generator_llm: ollama/llama3.2
temperature: 0.0
reasoning: false
```
"""


def _mock_completion(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestCompactHistoryRenderer:
    def test_empty_history(self) -> None:
        assert _compact_history([], _ALL_TUNABLE) == "No previous trials."

    def test_one_line_per_trial_with_score(self) -> None:
        records = [_make_record(1, 0.42), _make_record(2, 0.55)]
        text = _compact_history(records, _ALL_TUNABLE)
        lines = text.splitlines()
        assert len(lines) == 2
        assert lines[0] == f"trial 1: {_config_signature(records[0].config, _ALL_TUNABLE)} -> answer_accuracy=0.420"
        assert lines[1].startswith("trial 2:")
        assert lines[1].endswith("-> answer_accuracy=0.550")

    def test_appends_current_trial(self) -> None:
        records = [_make_record(1, 0.42)]
        current = _make_record(2, 0.61)
        text = _compact_history(records, _ALL_TUNABLE, current_trial=current)
        assert len(text.splitlines()) == 2
        assert text.splitlines()[1].endswith("-> answer_accuracy=0.610")

    def test_no_rich_markers(self) -> None:
        """Compact render carries none of the rich-history scaffolding."""
        text = _compact_history([_make_record(1, 0.42), _make_record(2, 0.55)], _ALL_TUNABLE)
        for marker in ("Configs already tried", "phase:", "changes vs prior", "verdicts:"):
            assert marker not in text

    def test_robust_across_index_types_no_keyerror(self) -> None:
        """Records spanning index types / inapplicable levers must render
        without indexing a lever absent from a stage-gated config."""
        records = [
            _make_record(1, 0.3, _make_config(index_type=IndexType.VECTOR_ONLY)),
            _make_record(
                2,
                0.4,
                _make_config(
                    index_type=IndexType.HYBRID_BM25_VECTOR,
                    bm25_vector_fusion="rrf",
                    query_expansion="hyde",
                    expander_llm="ollama/llama3.2",
                ),
            ),
        ]
        text = _compact_history(records, _ALL_TUNABLE)
        assert len(text.splitlines()) == 2


class TestCompactHistoryFlag:
    def test_defaults_to_false(self, tmp_path) -> None:
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        assert agent.compact_history is False

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_opro_proposer_sees_compact_history_no_kb_no_diagnosis(self, mock_litellm, tmp_path) -> None:
        """OPRO mode: a single proposer LLM call, compact score-history in the
        prompt, no rich-history scaffolding, and an empty knowledge base."""
        mock_litellm.acompletion = AsyncMock(side_effect=[_mock_completion(VALID_PROPOSER_YAML)])
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(
            agent_model="test-model",
            config=cfg,
            history=history,
            knowledge_base=None,
            use_diagnosis=False,
            compact_history=True,
        )

        assert agent.compact_history is True
        assert agent._kb_text() == ""

        exam = [_make_exam_question("q1"), _make_exam_question("q2")]
        exam_result = ExamResult(
            answer_accuracy=0.5,
            n_correct=1,
            n_total=2,
            n_valid=2,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=True,
                    selected_answer="A",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="A",
                    retrieved_spans=2,
                    n_spans=2,
                ),
                QuestionResult(
                    question_id="q2",
                    correct=False,
                    selected_answer="B",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="B",
                    retrieved_spans=0,
                    n_spans=2,
                ),
            ],
        )

        _trial_metrics, _diagnosis, next_config, _meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=1,
            trials_remaining=9,
        )

        # Exactly one LLM call (the Proposer); the Diagnoser never ran.
        assert mock_litellm.acompletion.call_count == 1
        proposer_call = mock_litellm.acompletion.call_args_list[0]
        proposer_prompt = (proposer_call.kwargs.get("messages") or proposer_call.args[0])[0]["content"]

        # OPRO uses a DEDICATED template (not the rich one with sections blanked):
        # the compact config->accuracy trajectory is present...
        assert "-> answer_accuracy=0.500" in proposer_prompt
        assert "## Configurations tried so far" in proposer_prompt
        assert "best accuracy so far" in proposer_prompt
        # ...and the agentic sections are ABSENT entirely (not blanked), as is the
        # dangling "Knowledge Base" instruction and the rich-render data lines.
        assert "## Diagnosis" not in proposer_prompt
        assert "## State card" not in proposer_prompt
        assert "## Key evidence" not in proposer_prompt
        assert "Knowledge Base" not in proposer_prompt
        assert "Journal" not in proposer_prompt
        assert "(acc=" not in proposer_prompt

        assert isinstance(next_config, TrialConfig)
        assert next_config.embedding_model == "BAAI/bge-m3"

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_opro_does_not_require_strategy_block(self, mock_litellm, tmp_path) -> None:
        """OPRO must not be forced to maintain a journal/stance: a proposal with
        no meta.strategy is accepted on the first call (no retry/fallback)."""
        mock_litellm.acompletion = AsyncMock(side_effect=[_mock_completion(VALID_PROPOSER_YAML_NO_STRATEGY)])
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(
            agent_model="test-model",
            config=cfg,
            history=history,
            knowledge_base=None,
            use_diagnosis=False,
            compact_history=True,
        )

        exam = [_make_exam_question("q1")]
        exam_result = ExamResult(
            answer_accuracy=1.0,
            n_correct=1,
            n_total=1,
            n_valid=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=True,
                    selected_answer="A",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="A",
                    retrieved_spans=2,
                    n_spans=2,
                ),
            ],
        )

        _tm, _diag, next_config, meta = await agent.analyze_and_propose(
            exam_result, exam, _make_config(), trial_number=1, trials_remaining=9
        )

        # Accepted on the first call — the missing strategy did not trigger a retry.
        assert mock_litellm.acompletion.call_count == 1
        assert isinstance(next_config, TrialConfig)
        assert next_config.embedding_model == "BAAI/bge-m3"
        assert meta.strategy is None

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_opro_initial_proposal_uses_clean_template(self, mock_litellm, tmp_path) -> None:
        """The OPRO initial proposal uses a dedicated template with no KB block
        and no 'Use the Knowledge Base' instruction — only the corpus + options."""
        mock_litellm.acompletion = AsyncMock(side_effect=[_mock_completion(VALID_INITIAL_CONFIG_YAML)])
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(
            agent_model="test-model",
            config=cfg,
            history=history,
            knowledge_base=None,
            use_diagnosis=False,
            compact_history=True,
        )

        config = await agent.propose_initial("A tiny corpus of news articles.")

        call = mock_litellm.acompletion.call_args
        prompt = (call.kwargs.get("messages") or call.args[0])[0]["content"]
        assert "Knowledge Base" not in prompt
        assert "## Diagnosis" not in prompt
        assert "Search Space" in prompt  # it still sees the available options
        assert "A tiny corpus of news articles." in prompt
        assert isinstance(config, TrialConfig)
