"""Tests for the reasoning agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_autorag.config.models import (
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    OpenEndedQuestion,
    ProjectConfig,
    RetrievalSearchSpace,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import ExamResult, QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    FrontierContext,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog
from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent
from agentic_autorag.optimizer.state import FailureAttribution


def _make_project_config(llm_models: list[str] | None = None) -> ProjectConfig:
    """Build a minimal ProjectConfig. ``llm_models`` populates the generator
    stage pool (and is also implicitly available for compressor/expander
    pools when those stages are enabled per-test)."""
    generators = list(llm_models) if llm_models else ["ollama/llama3.2"]
    return ProjectConfig(
        search_space=SearchSpace(
            embedding=EmbeddingSearchSpace(models=["sentence-transformers/all-MiniLM-L6-v2"]),
            retrieval=RetrievalSearchSpace(index_types=[IndexType.VECTOR_ONLY]),
            generator=GeneratorSearchSpace(models=generators),
        ),
    )


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


def _make_exam_question(qid: str = "q1") -> OpenEndedQuestion:
    return OpenEndedQuestion(
        id=qid,
        question=f"What does {qid} ask?",
        canonical_answer="alpha",
        answer_variants=["alpha-2"],
        reasoning_type="bridge",
        source_chunk_ids=["docA::chunk_0", "docB::chunk_0"],
        source_doc_ids=["docA", "docB"],
        source_spans=[f"chunk A span for {qid}", f"chunk B span for {qid}"],
    )


VALID_INITIAL_YAML = """\
Reasoning text here...

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: sentence-transformers/all-MiniLM-L6-v2
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

VALID_DIAGNOSIS_YAML = """\
Narrative: retrieval is underperforming.

```yaml
narrative: "retrieval_miss=0.50; the retriever is missing both spans on most failures."
failure_attribution:
  retrieval: 0.8
  ranking: 0.0
  generation: 0.2
  composition: 0.0
confirmed_findings:
  - "12 of 20 failures are retrieval_miss"
  - "3 of 20 failures are generation_wrong on arithmetic"
regression_detected: false
regression_axes: []
notable_deltas: []
illustrative_qids:
  - q1
```
"""

VALID_PROPOSER_YAML = """\
Changing embedding model per diagnosis.

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
  changes:
    - "embedding_model: sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-m3"
  rationale: "Diagnoser flagged retrieval primary; bge-m3 has higher MTEB."
  strategy:
    stance: explore
    journal: "MiniLM misses span_B on this corpus; trying bge-m3 first."
```
"""


def _mock_completion(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestExtractYaml:
    def test_yaml_block(self) -> None:
        text = "some text\n```yaml\nfoo: bar\n```\nmore text"
        result = ReasoningAgent._extract_yaml(text)
        assert result == {"foo": "bar"}

    def test_no_block_raises(self) -> None:
        with pytest.raises(ValueError, match="No YAML block found"):
            ReasoningAgent._extract_yaml("no yaml here")

    def test_repairs_missing_space_after_colon(self) -> None:
        text = (
            "```yaml\n"
            "failure_attribution:\n"
            "  retrieval: 0.82\n"
            "  ranking:   0.00\n"
            "  generation:0.18\n"
            "  composition:0.00\n"
            "```"
        )
        result = ReasoningAgent._extract_yaml(text)
        assert result["failure_attribution"] == {
            "retrieval": 0.82,
            "ranking": 0.00,
            "generation": 0.18,
            "composition": 0.00,
        }


class TestRenderFailureBlock:
    def _make_question_with_chunks(self, span_a: str, span_b: str) -> OpenEndedQuestion:
        return OpenEndedQuestion(
            id="q1",
            question="What does q1 ask?",
            canonical_answer="alpha",
            reasoning_type="bridge",
            source_chunk_ids=["docA::chunk_0", "docB::chunk_0"],
            source_doc_ids=["docA", "docB"],
            source_spans=[span_a, span_b],
        )

    def test_generation_wrong_renders_only_span_windows(self) -> None:
        # Gold spans appear in chunks rank 1 & 2; rank 3 is a distractor.
        chunk1 = "Unrelated lead-in. " + ("y" * 100) + " The alpha was nine. " + ("z" * 100)
        chunk2 = ("z" * 80) + " The beta was eleven. " + ("z" * 80)
        chunk3 = "Totally unrelated distractor content goes here."
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="wrong",
            correct_answer="alpha",
            retrieved_context="\n".join([chunk1, chunk2, chunk3]),
            generated_response="wrong",
            retrieved_doc_ids=["docA", "docB", "docZ"],
            retrieved_chunks=[chunk1, chunk2, chunk3],
            retrieved_spans=2,
            n_spans=2,
        )
        q = self._make_question_with_chunks("The alpha was nine.", "The beta was eleven.")

        text = ReasoningAgent._render_failure_block(qr, q, mode="generation_wrong")

        assert "### generation_wrong" in text
        assert "q_id=q1" in text
        assert "The alpha was nine." in text
        assert "The beta was eleven." in text
        # The distractor chunk's prefix should NOT appear (generation_wrong = windows only).
        assert "Totally unrelated distractor" not in text
        # Rank labels render.
        assert "[rank=1 | doc=docA]" in text
        assert "[rank=2 | doc=docB]" in text

    def test_retrieval_miss_shows_only_chunk_prefixes(self) -> None:
        chunk1 = "Wrong topic chunk 1 about something else entirely."
        chunk2 = "Wrong topic chunk 2 again unrelated."
        chunk3 = "Wrong topic chunk 3."
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="?",
            correct_answer="alpha",
            retrieved_context="\n".join([chunk1, chunk2, chunk3]),
            generated_response="?",
            retrieved_doc_ids=["docX", "docY", "docZ"],
            retrieved_chunks=[chunk1, chunk2, chunk3],
            retrieved_spans=0,
            n_spans=2,
        )
        q = self._make_question_with_chunks("The alpha was nine.", "The beta was eleven.")

        text = ReasoningAgent._render_failure_block(qr, q, mode="retrieval_miss")

        assert "Wrong topic chunk 1" in text
        assert "Wrong topic chunk 2" in text
        # Only top-2 chunks render.
        assert "Wrong topic chunk 3" not in text
        # No gold-span window markers since no span was retrieved.
        assert "[span:" not in text

    def test_falls_back_to_chunk_prefix_when_span_not_found(self) -> None:
        # Gold spans are listed but DO NOT appear in retrieved_chunks (the
        # matcher may have used a fuzzy/n-gram path, so retrieved_spans>0 but
        # exact strings aren't recoverable from the chunk text).
        chunk1 = "approximate match for the alpha story exists here."
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="wrong",
            correct_answer="alpha",
            retrieved_context=chunk1,
            generated_response="wrong",
            retrieved_doc_ids=["docA"],
            retrieved_chunks=[chunk1],
            retrieved_spans=1,
            n_spans=2,
        )
        q = self._make_question_with_chunks("never appears verbatim", "neither does this")

        text = ReasoningAgent._render_failure_block(qr, q, mode="generation_wrong")
        # The fallback message should appear and the chunk prefix should render.
        assert "no gold spans located" in text
        assert "approximate match for the alpha story" in text

    def test_retrieval_complete_with_unicode_dash_locates_span(self) -> None:
        """Bug regression: en-dash in gold span vs hyphen in chunk text used to
        prevent the renderer from locating the span. The unicode-fold fallback
        now handles this case so the window renders correctly."""
        # Gold span uses an en-dash ("1999–2000"); chunk text uses a regular hyphen.
        gold_span = "The 1999–2000 Season of BAI Basket (31st edition) ran with 8 teams."
        chunk = "# Article header\n\nThe 1999-2000 Season of BAI Basket (31st edition) ran with 8 teams. " + (
            "Many additional sentences " * 20
        )
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="0",
            correct_answer="4",
            retrieved_context=chunk,
            generated_response="0",
            retrieved_doc_ids=["1999_2000_bai_basket.md"],
            retrieved_chunks=[chunk],
            chunk_satisfies_spans=[[0]],
            retrieved_spans=1,
            n_spans=2,
        )
        q = OpenEndedQuestion(
            id="q1",
            question="What is the difference?",
            canonical_answer="4 teams",
            reasoning_type="comparison",
            source_chunk_ids=["1999_2000_bai_basket.md::0", "2007_08_bai_basket.md::0"],
            source_doc_ids=["1999_2000_bai_basket.md", "2007_08_bai_basket.md"],
            source_spans=[gold_span, "The 2007-2008 Season had 12 teams."],
        )

        text = ReasoningAgent._render_failure_block(qr, q, mode="generation_wrong")

        # Verbatim window for span_1 is rendered (unicode-fold matched the en-dash).
        assert "[span_1:" in text
        assert "window:" in text
        assert "1999–2000 Season" in text  # the gold span text appears in the window label
        # Approximate-match fallback NOT used (we found the span verbatim).
        assert "approximate match" not in text

    def test_renders_evaluator_credited_chunk_when_text_search_fails(self) -> None:
        """Bug regression: evaluator can credit a chunk via n-gram coverage on a
        non-source-doc chunk. The renderer must surface that chunk (as an
        approximate-match prefix), not skip it silently."""
        # chunk0 satisfies span_1 verbatim. chunk1 satisfies span_2 per the
        # evaluator (via n-gram coverage, NOT by containing the span text).
        chunk0 = "The 30th edition ran with 12 teams in three stages."
        chunk1 = "Earlier seasons of this league featured around 8 to 10 participating sides per cycle."
        qr = QuestionResult(
            question_id="q1",
            correct=False,
            selected_answer="0",
            correct_answer="4",
            retrieved_context=f"{chunk0}\n{chunk1}",
            generated_response="0",
            retrieved_doc_ids=["2007_08_bai_basket.md", "league_overview.md"],
            retrieved_chunks=[chunk0, chunk1],
            chunk_satisfies_spans=[[0], [1]],
            retrieved_spans=2,
            n_spans=2,
        )
        q = OpenEndedQuestion(
            id="q1",
            question="diff?",
            canonical_answer="4 teams",
            reasoning_type="comparison",
            source_chunk_ids=["docA::0", "docB::0"],
            source_doc_ids=["2007_08_bai_basket.md", "1999_2000_bai_basket.md"],
            source_spans=[
                "The 30th edition ran with 12 teams in three stages.",
                "The 31st edition (1999-2000) ran with 8 teams.",
            ],
        )

        text = ReasoningAgent._render_failure_block(qr, q, mode="generation_wrong")

        # Both chunks render:
        # - chunk0 with a verbatim window for span_1
        # - chunk1 with an approximate-match prefix for span_2 (since its text
        #   doesn't contain the span literally, only n-gram-related content).
        assert "[span_1:" in text
        assert "[span_2 approximate match]" in text
        assert "league_overview.md" in text

    def test_handles_missing_question_metadata(self) -> None:
        qr = QuestionResult(
            question_id="qX",
            correct=False,
            selected_answer="some prediction",
            correct_answer="some gold",
            retrieved_context="ctx",
            generated_response="some prediction",
            retrieved_chunks=["ctx"],
        )
        text = ReasoningAgent._render_failure_block(qr, None, mode="retrieval_miss")
        assert "<question text unavailable>" in text


class TestRenderFailureList:
    def test_renders_one_line_per_failure(self) -> None:
        failures = [
            QuestionResult(
                question_id="q1",
                correct=False,
                selected_answer="wrong text",
                correct_answer="alpha",
                retrieved_context="",
                generated_response="wrong",
                retrieved_chunks=[],
                retrieved_spans=2,
                n_spans=2,
            ),
            QuestionResult(
                question_id="q2",
                correct=False,
                selected_answer="cannot answer",
                correct_answer="alpha",
                retrieved_context="",
                generated_response="cannot answer",
                retrieved_chunks=[],
                refused=True,
                retrieved_spans=0,
                n_spans=2,
            ),
        ]
        questions_by_id = {qid: _make_exam_question(qid) for qid in ("q1", "q2")}
        text = ReasoningAgent._render_failure_list(failures, questions_by_id)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 2
        assert "q1" in lines[0]
        assert "generation_wrong" in lines[0]
        assert "q2" in lines[1]
        assert "refused" in lines[1]


class TestSelectStratifiedFailures:
    def _qr(self, qid: str, *, retrieved_spans: int, n_spans: int = 2) -> QuestionResult:
        return QuestionResult(
            question_id=qid,
            correct=False,
            selected_answer="B",
            correct_answer="A",
            retrieved_context="",
            generated_response="B",
            retrieved_spans=retrieved_spans,
            n_spans=n_spans,
        )

    def test_prioritises_flipped_questions(self) -> None:
        # All currently failing; q3/q4 were correct last trial.
        failures = [self._qr(f"q{i}", retrieved_spans=0) for i in range(1, 7)]
        questions_by_id = {qr.question_id: _make_exam_question(qr.question_id) for qr in failures}
        prev = {"q3": True, "q4": True, "q1": False, "q2": False, "q5": False, "q6": False}

        picked = ReasoningAgent._select_stratified_failures(failures, questions_by_id, prev, n=3, seed=7)

        picked_ids = [sf.result.question_id for sf in picked]
        # Flipped questions come first.
        assert picked_ids[0] in {"q3", "q4"}
        assert picked_ids[1] in {"q3", "q4"}
        assert len(picked) == 3

    def test_deterministic_for_same_seed(self) -> None:
        failures = [self._qr(f"q{i}", retrieved_spans=0) for i in range(1, 9)]
        questions_by_id = {qr.question_id: _make_exam_question(qr.question_id) for qr in failures}
        a = ReasoningAgent._select_stratified_failures(failures, questions_by_id, {}, n=4, seed=11)
        b = ReasoningAgent._select_stratified_failures(failures, questions_by_id, {}, n=4, seed=11)
        assert [sf.result.question_id for sf in a] == [sf.result.question_id for sf in b]

    def test_regression_vs_best_band_pulls_qids_correct_in_best_so_far(self) -> None:
        """Q5 was correct in the best-so-far trial but is wrong now → must surface
        as ``source=regression_vs_best`` even though it didn't flip vs the
        immediate prior trial."""
        failures = [self._qr(f"q{i}", retrieved_spans=0) for i in range(1, 7)]
        questions_by_id = {qr.question_id: _make_exam_question(qr.question_id) for qr in failures}
        # q5 was wrong last trial but correct in best-so-far.
        prev = {f"q{i}": False for i in range(1, 7)}
        best = {"q5": (True, False), "q1": (False, False)}

        picked = ReasoningAgent._select_stratified_failures(failures, questions_by_id, prev, best, n=6, seed=7)

        by_qid = {sf.result.question_id: sf for sf in picked}
        assert "q5" in by_qid
        assert by_qid["q5"].source == "regression_vs_best"
        assert by_qid["q5"].judge_only is False

    def test_regression_vs_best_judge_only_flag(self) -> None:
        """When best-so-far's correctness on q5 came from the judge (not EM),
        the regression-vs-best selection must carry ``judge_only=True``."""
        failures = [self._qr(f"q{i}", retrieved_spans=0) for i in range(1, 4)]
        questions_by_id = {qr.question_id: _make_exam_question(qr.question_id) for qr in failures}
        prev = {f"q{i}": False for i in range(1, 4)}
        best = {"q2": (True, True)}  # correct via judge only

        picked = ReasoningAgent._select_stratified_failures(failures, questions_by_id, prev, best, n=3, seed=7)
        by_qid = {sf.result.question_id: sf for sf in picked}
        assert by_qid["q2"].source == "regression_vs_best"
        assert by_qid["q2"].judge_only is True


class TestDiagnoseClassification:
    """Failure classification surfaces per-question failure_mode tags."""

    def _make_result(
        self,
        *,
        qid: str,
        correct: bool,
        retrieved_spans: int,
        n_spans: int = 2,
        refused: bool = False,
        generated_response: str = "B",
    ) -> QuestionResult:
        return QuestionResult(
            question_id=qid,
            correct=correct,
            selected_answer="A" if correct else "B",
            correct_answer="A",
            retrieved_context=f"retrieved {qid}",
            generated_response=generated_response,
            chunk_precision=0.2 if retrieved_spans > 0 else 0.0,
            source_fact_rank=1 if retrieved_spans > 0 else 0,
            retrieved_doc_ids=[f"doc_{qid}"],
            retrieved_spans=retrieved_spans,
            n_spans=n_spans,
            refused=refused,
        )

    def _build_agent(self, tmp_path):
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        return ReasoningAgent(agent_model="test-model", config=cfg, history=history)

    async def test_failure_mode_tags_render_in_prompt(self, tmp_path) -> None:
        results = [
            self._make_result(qid="q1", correct=False, retrieved_spans=0),
            self._make_result(qid="q2", correct=False, retrieved_spans=1),
            self._make_result(qid="q3", correct=False, retrieved_spans=1),
            self._make_result(
                qid="q4",
                correct=False,
                retrieved_spans=0,
                refused=True,
                generated_response="cannot answer based on provided context",
            ),
            self._make_result(qid="q5", correct=False, retrieved_spans=2),
        ]
        exam_result = ExamResult(score=0.0, n_correct=0, n_total=5, question_results=results)
        exam_questions = [_make_exam_question(qr.question_id) for qr in results]

        agent = self._build_agent(tmp_path)
        captured: dict[str, str] = {}

        async def _capture_prompt(messages):
            captured["prompt"] = messages[-1]["content"]
            return VALID_DIAGNOSIS_YAML

        with patch.object(agent, "_llm_complete_messages", side_effect=_capture_prompt):
            await agent._diagnose(
                exam_result=exam_result,
                exam_questions=exam_questions,
                current_config=_make_config(),
                trial_metrics=TrialMetrics(answer_accuracy=0.0, retrieval_complete=0.2),
                trial_number=1,
                trials_remaining=9,
                frontier_context=FrontierContext(),
                previous_strategy=None,
            )

        prompt = captured["prompt"]
        # All four failure-mode headers should appear in the deep blocks
        # rendered by _render_failure_block (formatted as "### <mode>  q_id=…").
        assert "### retrieval_miss" in prompt
        assert "### retrieval_partial" in prompt
        assert "### refused" in prompt
        assert "### generation_wrong" in prompt
        # Each failure block carries question text. Quotes appear because
        # the new block renders the question as a repr() value.
        for qid in ("q1", "q2", "q3", "q4", "q5"):
            assert f"What does {qid} ask?" in prompt
        # The new Tier-1 cross-tab is rendered.
        assert "failure_mode × reasoning_type × n_spans" in prompt
        # The new Tier-2 one-line list is rendered.
        assert "gold=" in prompt and "pred=" in prompt


class TestProposeInitial:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_valid_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert isinstance(config, TrialConfig)
        assert config.chunk_token_size == 512
        mock_litellm.acompletion.assert_called_once()

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_retry_on_invalid_yaml(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion("no yaml here"),
                _mock_completion(VALID_INITIAL_YAML),
            ]
        )
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert isinstance(config, TrialConfig)
        assert mock_litellm.acompletion.call_count == 2

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_raises_after_max_retries(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion("no yaml at all"))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        with pytest.raises(RuntimeError, match="Failed to get valid config"):
            await agent.propose_initial("A test corpus.")

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_initial_prompt_pipeline_rules_appear_when_tunable(self, mock_litellm, tmp_path) -> None:
        """When pipeline levers (query_decompose, passage_compressor,
        long_context_reorder, bm25_vector_fusion) are tunable in the search
        space, their guidance blocks must appear in the prompt."""
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        # Use 2 llm_models so single-LLM pinning of compressor_llm/expander_llm
        # doesn't conflict with the dependent-field defaults in the mock YAML.
        cfg = _make_project_config(llm_models=["ollama/llama3.2", "ollama/llama3.1"])
        cfg.search_space.query_expansion.strategies = ["none", "query_decompose"]
        cfg.search_space.query_expansion.models = ["ollama/llama3.2"]
        cfg.search_space.passage_compressor.strategies = ["none", "tree_summarize"]
        cfg.search_space.passage_compressor.models = ["ollama/llama3.2"]
        cfg.search_space.retrieval.long_context_reorder = [False, True]
        cfg.search_space.retrieval.bm25_vector_fusion = ["alpha", "rrf"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        await agent.propose_initial("A test corpus.")

        kwargs = mock_litellm.acompletion.call_args.kwargs
        messages = kwargs.get("messages") or mock_litellm.acompletion.call_args.args[0]
        sent_prompt = messages[0]["content"]
        assert "query_decompose" in sent_prompt
        assert "passage_compressor" in sent_prompt
        assert "long_context_reorder" in sent_prompt
        assert "bm25_vector_fusion" in sent_prompt

    def test_pinned_levers_are_skipped_from_parameter_guide(self, tmp_path) -> None:
        """Every pinned lever (single configured value) is dropped from the
        Parameter Guide. The agent reads the pinned value from the search
        space's "Fixed values" block instead — duplicating a description for a
        knob the agent cannot turn is waste."""
        from agentic_autorag.config.knowledge_base import KnowledgeBase

        cfg = _make_project_config()
        cfg.search_space.passage_compressor.strategies = ["tree_summarize"]
        cfg.search_space.passage_compressor.models = ["ollama/llama3.2"]
        cfg.search_space.retrieval.index_types = [IndexType.HYBRID_BM25_VECTOR]
        cfg.search_space.retrieval.bm25_vector_fusion = ["rrf"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        try:
            kb = KnowledgeBase()
        except Exception:
            pytest.skip("KnowledgeBase data not available in this environment")
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history, knowledge_base=kb)
        kb_text = agent._kb_text()
        # Both are pinned in this search space → omitted from Parameter Guide.
        assert "- **passage_compressor**:" not in kb_text
        assert "- **bm25_vector_fusion**:" not in kb_text

    def test_derived_stage_llms_are_skipped_from_parameter_guide(self, tmp_path) -> None:
        """Mixed strategies + single-LLM pool → compressor_llm / expander_llm
        are derived at injection time (not emitted by the agent). Their
        parameter-guide entries must be skipped — the "Derived values" block
        in the prompt already explains the resolution rule, and a guide
        telling the agent how to choose would contradict that."""
        from agentic_autorag.config.knowledge_base import KnowledgeBase

        cfg = _make_project_config()
        cfg.search_space.passage_compressor.strategies = ["none", "tree_summarize"]
        cfg.search_space.passage_compressor.models = ["ollama/llama3.2"]
        cfg.search_space.query_expansion.strategies = ["none", "hyde"]
        cfg.search_space.query_expansion.models = ["ollama/llama3.2"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        try:
            kb = KnowledgeBase()
        except Exception:
            pytest.skip("KnowledgeBase data not available in this environment")
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history, knowledge_base=kb)
        kb_text = agent._kb_text()
        assert cfg.search_space.compressor_llm_is_derived()
        assert cfg.search_space.expander_llm_is_derived()
        assert "- **compressor_llm**:" not in kb_text
        assert "- **expander_llm**:" not in kb_text

    def test_options_filtered_to_configured_set(self, tmp_path) -> None:
        """Per-option descriptions render only for option-values the agent can
        actually pick. If `query_expansion=["none","hyde"]`, the guide must
        not describe `multi_query` or `query_decompose`."""
        from agentic_autorag.config.knowledge_base import KnowledgeBase

        cfg = _make_project_config()
        cfg.search_space.query_expansion.strategies = ["none", "hyde"]
        cfg.search_space.query_expansion.models = ["ollama/llama3.2"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        try:
            kb = KnowledgeBase()
        except Exception:
            pytest.skip("KnowledgeBase data not available in this environment")
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history, knowledge_base=kb)
        kb_text = agent._kb_text()
        # The query_expansion bullet must include "none" and "hyde" option
        # descriptions but NOT the multi_query / query_decompose ones.
        assert "- none:" in kb_text
        assert "- hyde:" in kb_text
        assert "- multi_query:" not in kb_text
        assert "- query_decompose:" not in kb_text

    def test_parameter_guide_preserves_multiline_constraints(self, tmp_path) -> None:
        """Multi-line guidance entries must not be flattened into a single
        run-on line by the YAML loader / renderer. The reasoning parameter
        carries a multi-sentence guidance block that should render with line
        breaks intact so the agent can read each constraint independently."""
        from agentic_autorag.config.knowledge_base import KnowledgeBase

        cfg = _make_project_config(llm_models=["ollama/llama3.2", "ollama/llama3.1"])
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        try:
            kb = KnowledgeBase()
        except Exception:
            pytest.skip("KnowledgeBase data not available in this environment")
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history, knowledge_base=kb)
        kb_text = agent._kb_text()
        # The reasoning parameter's guidance carries two distinct constraints
        # separated by a sentence boundary. Both should survive rendering.
        assert "Only the generator's final-answer call uses reasoning_effort" in kb_text
        assert "(reasoning)" in kb_text


class TestSeedPlumbing:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_seed_forwarded_when_set(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history, seed=42)

        await agent.propose_initial("A test corpus.")

        assert mock_litellm.acompletion.call_args.kwargs["seed"] == 42

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_seed_omitted_when_none(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_INITIAL_YAML))
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        await agent.propose_initial("A test corpus.")

        assert "seed" not in mock_litellm.acompletion.call_args.kwargs


class TestAnalyzeAndPropose:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_full_tuple(self, mock_litellm, tmp_path) -> None:
        # First call is Diagnoser, second is Proposer.
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
            ]
        )
        cfg = _make_project_config(llm_models=["ollama/llama3.2", "ollama/llama3.1"])
        cfg.search_space.embedding.models = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
        # Enable pipeline levers so the rules block remains in the prompt;
        # pinned levers no longer produce guidance text under the new contract.
        cfg.search_space.query_expansion.strategies = ["none", "query_decompose"]
        cfg.search_space.query_expansion.models = ["ollama/llama3.2"]
        cfg.search_space.passage_compressor.strategies = ["none", "tree_summarize"]
        cfg.search_space.passage_compressor.models = ["ollama/llama3.2"]
        cfg.search_space.retrieval.long_context_reorder = [False, True]
        cfg.search_space.retrieval.bm25_vector_fusion = ["alpha", "rrf"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        exam = [_make_exam_question("q1"), _make_exam_question("q2")]
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
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

        trial_metrics, diagnosis, next_config, meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=1,
            trials_remaining=9,
        )

        assert isinstance(trial_metrics, TrialMetrics)
        assert isinstance(diagnosis, Diagnosis)
        assert isinstance(next_config, TrialConfig)
        assert isinstance(meta, ProposalMeta)
        assert next_config.embedding_model == "BAAI/bge-m3"
        # failure_attribution was dropped from Diagnosis — orchestrator now
        # surfaces mechanical attribution to the Proposer's state card directly.
        assert "retrieval_miss" in diagnosis.confirmed_findings[0]

        # The proposer's prompt (second litellm call) must include the
        # pipeline-rules guidance so the agent can reason about the
        # compressor / reorder / fusion / decompose dimensions.
        proposer_call = mock_litellm.acompletion.call_args_list[1]
        proposer_prompt = (proposer_call.kwargs.get("messages") or proposer_call.args[0])[0]["content"]
        assert "query_decompose" in proposer_prompt
        assert "passage_compressor" in proposer_prompt
        assert "long_context_reorder" in proposer_prompt
        assert "bm25_vector_fusion" in proposer_prompt
        assert diagnosis.illustrative_qids == ["q1"]
        # ProposalMeta.changes was removed — the renderer derives the diff
        # mechanically from configs. The lever-change assertion on
        # next_config.embedding_model (above) already covers what mattered.


class TestProposerParseFailureFallback:
    """Proposer must not raise when YAML can't be parsed after retries — fall
    back to a random single-lever perturbation so the run keeps going."""

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_falls_back_to_random_perturbation_after_retries(self, mock_litellm, tmp_path) -> None:
        garbage = _mock_completion("not yaml")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),  # Diagnoser succeeds.
                garbage,
                garbage,
                garbage,
            ]
        )

        cfg = _make_project_config(llm_models=["ollama/llama3.2", "ollama/llama3.1"])
        cfg.search_space.embedding.models = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        exam = [_make_exam_question("q1")]
        exam_result = ExamResult(
            score=0.0,
            n_correct=0,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
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

        current = _make_config()
        _, _, next_config, meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            current,
            trial_number=1,
            trials_remaining=9,
        )

        assert isinstance(next_config, TrialConfig)
        assert isinstance(meta, ProposalMeta)
        assert meta.rationale.startswith("Proposer parse failed")
        # The picked lever shows up inline in rationale (no separate `changes` field).
        assert "->" in meta.rationale
        # validate_trial returns a list of violations; empty = valid.
        assert cfg.validate_trial(next_config) == []
        # Strategy carries over even on fallback so the agent's journal isn't lost.
        assert meta.strategy is not None

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_fallback_returns_current_config_if_no_perturbation_possible(self, mock_litellm, tmp_path) -> None:
        """When the search space pins every safe lever, the fallback re-uses
        the current config rather than producing an invalid one."""
        garbage = _mock_completion("not yaml")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                garbage,
                garbage,
                garbage,
            ]
        )

        cfg = _make_project_config(llm_models=["ollama/llama3.2"])
        # Single embedding model, single generator, single chunking strategy,
        # all numeric dims pinned to a single value → no alternative exists.
        cfg.search_space.embedding.models = ["sentence-transformers/all-MiniLM-L6-v2"]
        cfg.search_space.chunking.strategies = ["recursive"]
        cfg.search_space.reranker.models = ["none"]
        from agentic_autorag.config.models import NumericRange

        cfg.search_space.chunking.chunk_token_size = NumericRange(min=512, max=512)
        cfg.search_space.chunking.chunk_token_overlap = NumericRange(min=64, max=64)
        cfg.search_space.retrieval.top_k = NumericRange(min=5, max=5)
        cfg.search_space.temperature = NumericRange(min=0.0, max=0.0)
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        exam = [_make_exam_question("q1")]
        exam_result = ExamResult(
            score=0.0,
            n_correct=0,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
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

        current = _make_config()
        _, _, next_config, meta = await agent.analyze_and_propose(
            exam_result,
            exam,
            current,
            trial_number=1,
            trials_remaining=9,
        )

        assert next_config == current
        assert "no perturbation found" in meta.rationale


class TestDuplicateConfigDetection:
    """Proposer must reject configs identical to a prior trial; orchestrator
    accepts the duplicate after MAX_DUPLICATE_RETRIES re-prompts."""

    @staticmethod
    def _seed_history_with_trial(history: HistoryLog, config: TrialConfig, trial_number: int = 1) -> None:
        from agentic_autorag.optimizer.history import TrialRecord

        history.records.append(
            TrialRecord(
                trial_number=trial_number,
                config=config,
                score=0.5,
                question_results=[],
            )
        )

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_dup_triggers_retry_then_accepts_different_config(self, mock_litellm, tmp_path) -> None:
        """First proposal duplicates trial 1; retry produces a different config which is accepted."""
        cfg = _make_project_config(llm_models=["ollama/llama3.2"])
        cfg.search_space.embedding.models = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # Trial 1's config must match what the proposer-parser produces after
        # `_inject_pinned` rewrites pinned levers — namely hybrid_alpha=0 (no
        # hybrid index types) and reranker_top_n=3 (search space minimum).
        prior_config = _make_config(
            embedding_model="BAAI/bge-m3",
            hybrid_alpha=0.0,
            reranker_top_n=3,
        )
        self._seed_history_with_trial(history, prior_config, trial_number=1)

        # Second proposal swaps to a non-dup value.
        retry_yaml = VALID_PROPOSER_YAML.replace("BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(VALID_PROPOSER_YAML),  # dup of trial 1
                _mock_completion(retry_yaml),  # different
            ]
        )

        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        exam = [_make_exam_question("q1")]
        exam_result = ExamResult(
            score=0.5,
            n_correct=0,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
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

        _, _, next_config, _ = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=2,
            trials_remaining=8,
        )
        assert next_config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        # Diagnoser + 2 proposer calls expected (dup + retry).
        assert mock_litellm.acompletion.await_count == 3
        retry_call = mock_litellm.acompletion.await_args_list[2]
        retry_messages = retry_call.kwargs.get("messages") or retry_call.args[0]
        assert "Trial 1 already had this exact config" in retry_messages[-1]["content"]

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_dup_accepted_after_max_duplicate_retries(self, mock_litellm, tmp_path) -> None:
        """After MAX_DUPLICATE_RETRIES persistent duplicates, accept with a warning."""
        cfg = _make_project_config(llm_models=["ollama/llama3.2"])
        cfg.search_space.embedding.models = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        prior_config = _make_config(
            embedding_model="BAAI/bge-m3",
            hybrid_alpha=0.0,
            reranker_top_n=3,
        )
        self._seed_history_with_trial(history, prior_config, trial_number=1)

        # Every proposer call emits the same duplicate.
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
                _mock_completion(VALID_PROPOSER_YAML),
            ]
        )

        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        exam = [_make_exam_question("q1")]
        exam_result = ExamResult(
            score=0.5,
            n_correct=0,
            n_total=1,
            question_results=[
                QuestionResult(
                    question_id="q1",
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

        _, _, next_config, _ = await agent.analyze_and_propose(
            exam_result,
            exam,
            _make_config(),
            trial_number=2,
            trials_remaining=8,
        )
        assert next_config == prior_config
        # Diagnoser + 1 initial + 2 retries = 4 total.
        assert mock_litellm.acompletion.await_count == 4


class TestBuildDiagnosis:
    def test_parses_narrative_and_findings(self, tmp_path) -> None:
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        diagnosis = agent._build_diagnosis(
            raw=VALID_DIAGNOSIS_YAML,
            trial_metrics=TrialMetrics(answer_accuracy=0.4, retrieval_complete=0.5),
        )

        assert isinstance(diagnosis, Diagnosis)
        assert "retriever is missing both spans" in diagnosis.narrative
        # Trial metrics merged in mechanically, not from YAML.
        assert diagnosis.trial_metrics.retrieval_complete == 0.5
        assert diagnosis.illustrative_qids == ["q1"]
        assert any("retrieval_miss" in f for f in diagnosis.confirmed_findings)

    def test_extra_legacy_fields_silently_ignored(self, tmp_path) -> None:
        """Old YAMLs with failure_attribution / regression_detected fields
        parse without error — the slim Diagnosis just ignores them."""
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        raw = """
```yaml
narrative: "score dropped 5 points after the embedding swap."
failure_attribution:
  retrieval: 0.5
  generation: 0.5
regression_detected: true
regression_axes:
  - score
illustrative_qids: [q3]
```
"""
        diagnosis = agent._build_diagnosis(raw=raw, trial_metrics=TrialMetrics())

        assert "score dropped" in diagnosis.narrative
        assert diagnosis.illustrative_qids == ["q3"]

    def test_falls_back_when_yaml_missing_fields(self, tmp_path) -> None:
        cfg = _make_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        empty_yaml = "narrative-only.\n\n```yaml\nnarrative: empty\n```\n"
        diagnosis = agent._build_diagnosis(
            raw=empty_yaml,
            trial_metrics=TrialMetrics(),
        )

        assert diagnosis.narrative == "empty"
        assert diagnosis.confirmed_findings == []
        assert diagnosis.illustrative_qids == []


VALID_RECOVERY_YAML = """\
The reranker failed to load — swapping it.

```yaml
chunking_strategy: recursive
chunk_token_size: 512
chunk_token_overlap: 64
embedding_model: sentence-transformers/all-MiniLM-L6-v2
index_type: vector_only
top_k: 5
hybrid_alpha: 0.5
reranker: BAAI/bge-reranker-v2-m3
reranker_top_n: 5
query_expansion: none
generator_llm: ollama/llama3.2
temperature: 0.0
reasoning: false
meta:
  changes:
    - "reranker: jinaai/jina-reranker-v2-base-multilingual → BAAI/bge-reranker-v2-m3"
  rationale: "Jina reranker requires trust_remote_code which is not enabled."
```
"""


class TestProposeAfterFailure:
    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_returns_alternative_config(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(VALID_RECOVERY_YAML))
        cfg = _make_project_config()
        cfg.search_space.reranker.models = [
            "jinaai/jina-reranker-v2-base-multilingual",
            "BAAI/bge-reranker-v2-m3",
            "none",
        ]
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        failed = _make_config(reranker="jinaai/jina-reranker-v2-base-multilingual")
        config, meta = await agent.propose_after_failure(
            failed_config=failed,
            error_summary="ValueError: requires trust_remote_code=True",
            failure_history=[(failed, "ValueError: requires trust_remote_code=True")],
        )

        assert isinstance(config, TrialConfig)
        assert config.reranker == "BAAI/bge-reranker-v2-m3"
        assert isinstance(meta, ProposalMeta)
        # Recovery rationale should explain the swap; the actual lever change
        # is observable from the returned TrialConfig (reranker assertion above).
        assert meta.rationale


class TestModelDataIntegrity:
    """Pydantic types must reject invalid inputs and accept valid ones."""

    def test_failure_attribution_rejects_out_of_range(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FailureAttribution(retrieval=1.5)  # type: ignore[call-arg]

    def test_diagnosis_clamps_narrative_length(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Diagnosis(trial_metrics=TrialMetrics(), narrative="x" * 3000)


def _make_pinned_project_config() -> ProjectConfig:
    """A search space modeled on hotpot_dev_project: chunk/overlap pinned at 0,
    reranker pinned to ``none``, temperature pinned at 1.0, only embedding /
    top_k / generator_llm are tunable."""
    return ProjectConfig.model_validate(
        {
            "search_space": {
                "chunking": {
                    "strategies": ["recursive"],
                    "chunk_token_size": {"min": 256, "max": 256},
                    "chunk_token_overlap": {"min": 0, "max": 0},
                },
                "embedding": {
                    "models": [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "BAAI/bge-m3",
                    ],
                },
                "retrieval": {
                    "index_types": ["vector_only"],
                    "top_k": {"min": 3, "max": 10},
                    "hybrid_alpha": {"min": 0.0, "max": 1.0},
                },
                "reranker": {"models": ["none"], "top_n": {"min": 3, "max": 5}},
                "query_expansion": {"strategies": ["none"], "models": []},
                "generator": {
                    "models": ["ollama/llama3.2", "ollama/mistral"],
                    "reasoning": False,
                },
                "temperature": {"min": 1.0, "max": 1.0},
            }
        }
    )


# Agent response that obeys the new prompt instructions: emits ONLY the
# tunable fields. Pinned values (chunking, reranker, temperature, etc.) are
# missing and must be filled in by ``_inject_pinned``.
TUNABLE_ONLY_PROPOSER_YAML = """\
Picking a stronger embedding.

```yaml
embedding_model: BAAI/bge-m3
top_k: 8
generator_llm: ollama/mistral
meta:
  changes:
    - "embedding_model: sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-m3"
  rationale: "Diagnoser flagged retrieval primary; bge-m3 has higher MTEB."
  strategy:
    stance: explore
    journal: "MiniLM misses span_B on this corpus; trying bge-m3 first."
```
"""

# Agent ignored the prompt and emitted ``chunk_token_overlap: 64`` even though
# the search space pins it at 0. This is the exact failure pattern from the
# reported run — injection must override the agent's bad value so the trial
# can still proceed.
PINNED_VIOLATING_PROPOSER_YAML = """\
Going wide.

```yaml
chunking_strategy: recursive
chunk_token_size: 256
chunk_token_overlap: 64
embedding_model: BAAI/bge-m3
index_type: vector_only
top_k: 8
hybrid_alpha: 0.5
reranker: none
reranker_top_n: 5
query_expansion: none
generator_llm: ollama/mistral
temperature: 1.0
meta:
  changes:
    - "embedding_model: sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-m3"
  rationale: "Increasing overlap to reduce span loss."
  strategy:
    stance: explore
    journal: "overlap was rejected by injection; relying on embedding swap."
```
"""

TUNABLE_ONLY_INITIAL_YAML = """\
Initial picks.

```yaml
embedding_model: BAAI/bge-m3
top_k: 6
generator_llm: ollama/llama3.2
```
"""


class TestPinnedInjectionInProposer:
    """End-to-end: agent omits pinned fields → injection fills them; agent emits
    a wrong value for a pinned field → injection overrides; trial validates."""

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_proposer_yaml_without_pinned_fields_succeeds(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(TUNABLE_ONLY_PROPOSER_YAML),
            ]
        )
        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        current = _make_config(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
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
        _, _, next_config, _ = await agent.analyze_and_propose(
            exam_result,
            [_make_exam_question("q1"), _make_exam_question("q2")],
            current,
            trial_number=1,
            trials_remaining=9,
        )

        # Pinned values came from the search space, not the agent's YAML.
        assert next_config.chunk_token_size == 256
        assert next_config.chunk_token_overlap == 0
        assert next_config.chunking_strategy == "recursive"
        assert next_config.reranker == "none"
        assert next_config.temperature == 1.0
        # Tunable values came from the agent.
        assert next_config.embedding_model == "BAAI/bge-m3"
        assert next_config.top_k == 8
        assert next_config.generator_llm == "ollama/mistral"

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_agent_emitting_pinned_field_with_wrong_value_is_overridden(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_completion(VALID_DIAGNOSIS_YAML),
                _mock_completion(PINNED_VIOLATING_PROPOSER_YAML),
            ]
        )
        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        current = _make_config(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        exam_result = ExamResult(
            score=0.5,
            n_correct=1,
            n_total=2,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    correct=False,
                    selected_answer="B",
                    correct_answer="A",
                    retrieved_context="ctx",
                    generated_response="B",
                    retrieved_spans=0,
                    n_spans=2,
                )
            ],
        )
        _, _, next_config, _ = await agent.analyze_and_propose(
            exam_result,
            [_make_exam_question("q1")],
            current,
            trial_number=1,
            trials_remaining=9,
        )

        # Agent emitted chunk_token_overlap=64, but the pinned value (0) wins.
        # Without this, the validator would have raised a search-space
        # violation and the proposer would have burned retries.
        assert next_config.chunk_token_overlap == 0
        # The first attempt succeeded — no retries happened.
        assert mock_litellm.acompletion.call_count == 2  # diagnose + propose

    @patch("agentic_autorag.litellm_runtime.litellm")
    async def test_initial_proposer_injects_pinned(self, mock_litellm, tmp_path) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_completion(TUNABLE_ONLY_INITIAL_YAML))
        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)

        config = await agent.propose_initial("A test corpus.")

        assert config.chunk_token_size == 256
        assert config.chunk_token_overlap == 0
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.top_k == 6


class TestInjectPinnedHelper:
    """The injection helper itself, isolated."""

    def test_inject_adds_missing_pinned_fields(self, tmp_path) -> None:
        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        yaml_dict: dict = {"embedding_model": "BAAI/bge-m3", "top_k": 7, "generator_llm": "ollama/mistral"}
        agent._inject_pinned(yaml_dict)
        assert yaml_dict["chunk_token_size"] == 256
        assert yaml_dict["chunk_token_overlap"] == 0
        assert yaml_dict["chunking_strategy"] == "recursive"
        assert yaml_dict["reranker"] == "none"
        assert yaml_dict["index_type"] == "vector_only"

    def test_inject_overrides_agent_emitted_pinned_field(self, tmp_path) -> None:
        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        yaml_dict: dict = {"chunk_token_overlap": 64, "embedding_model": "BAAI/bge-m3"}
        agent._inject_pinned(yaml_dict)
        assert yaml_dict["chunk_token_overlap"] == 0  # pinned wins

    def test_inject_warns_on_mismatch(self, tmp_path, caplog) -> None:
        """When the agent emits a pinned field with a wrong value, log warns.

        Lets us count search-space violations from the run logs without
        re-introducing the validator-rejection retry loop the injection
        replaced.
        """
        import logging

        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        with caplog.at_level(logging.WARNING, logger="agentic_autorag.optimizer.reasoning_agent"):
            agent._inject_pinned({"chunk_token_overlap": 64, "embedding_model": "BAAI/bge-m3"})
        assert any("chunk_token_overlap" in r.getMessage() for r in caplog.records)
        assert any("64" in r.getMessage() for r in caplog.records)

    def test_inject_does_not_warn_when_agent_omits_pinned(self, tmp_path, caplog) -> None:
        import logging

        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        with caplog.at_level(logging.WARNING, logger="agentic_autorag.optimizer.reasoning_agent"):
            agent._inject_pinned({"embedding_model": "BAAI/bge-m3", "top_k": 8})
        assert caplog.records == []

    def test_inject_does_not_warn_when_agent_emits_correct_pinned_value(self, tmp_path, caplog) -> None:
        """Agent ignored 'don't emit' instruction but used the right value
        — not a violation, no warning."""
        import logging

        cfg = _make_pinned_project_config()
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        with caplog.at_level(logging.WARNING, logger="agentic_autorag.optimizer.reasoning_agent"):
            agent._inject_pinned({"chunk_token_overlap": 0, "embedding_model": "BAAI/bge-m3"})
        assert caplog.records == []

    def test_inject_is_noop_when_search_space_fully_tunable(self, tmp_path) -> None:
        cfg = ProjectConfig.model_validate(
            {
                "search_space": {
                    "chunking": {
                        "strategies": ["recursive", "fixed"],
                        "chunk_token_size": {"min": 256, "max": 1024},
                        "chunk_token_overlap": {"min": 0, "max": 128},
                    },
                    "embedding": {"models": ["e1", "e2"]},
                    "retrieval": {
                        "index_types": ["vector_only", "hybrid_bm25_vector"],
                        "top_k": {"min": 3, "max": 15},
                        "bm25_vector_fusion": ["alpha", "rrf"],
                        "long_context_reorder": [False, True],
                    },
                    "passage_compressor": {
                        "strategies": ["none", "tree_summarize"],
                        "models": ["m1", "m2"],
                    },
                    "reranker": {"models": ["none", "real"], "top_n": {"min": 3, "max": 8}},
                    "query_expansion": {
                        "strategies": ["none", "hyde"],
                        "models": ["m1", "m2"],
                    },
                    "generator": {"models": ["m1", "m2"]},
                    "temperature": {"min": 0.0, "max": 1.0},
                }
            }
        )
        history = HistoryLog(path=str(tmp_path / "history.jsonl"))
        agent = ReasoningAgent(agent_model="test-model", config=cfg, history=history)
        yaml_dict: dict = {"top_k": 7, "embedding_model": "e1"}
        before = dict(yaml_dict)
        agent._inject_pinned(yaml_dict)
        assert yaml_dict == before


class TestStateCardRenderAndCheatsheet:
    """State-card render + cost-cheatsheet conditional rendering."""

    def test_render_includes_total_budget_trials_since_best_and_coverage(self) -> None:
        from agentic_autorag.optimizer.diagnosis import StateCard
        from agentic_autorag.optimizer.reasoning_agent import _format_state_card

        card = StateCard(
            cost_aware=True,
            trial_number=12,
            trials_remaining=3,
            best_score_so_far=0.837,
            best_trial_number=4,
            last_trial_delta=-0.041,
            trials_since_best_score=8,
            coverage=[
                {"label": "generators", "tried": 3, "total": 13},
                {"label": "embeddings", "tried": 2, "total": 8},
                {"label": "rerankers", "tried": 2, "total": 5},
            ],
        )
        rendered = _format_state_card(card)

        assert "trials_remaining=3 (of 15 total)" in rendered
        assert "trials_since_best_score=8" in rendered
        assert "search space coverage: generators 3/13; embeddings 2/8; rerankers 2/5" in rendered

    def test_render_omits_coverage_line_when_empty(self) -> None:
        from agentic_autorag.optimizer.diagnosis import StateCard
        from agentic_autorag.optimizer.reasoning_agent import _format_state_card

        card = StateCard(
            cost_aware=False,
            trial_number=1,
            trials_remaining=9,
            best_score_so_far=0.5,
            best_trial_number=1,
            last_trial_delta=0.0,
            trials_since_best_score=0,
            coverage=[],
        )
        rendered = _format_state_card(card)
        assert "search space coverage" not in rendered

    def test_render_trial_summary_includes_retrieval_complete(self) -> None:
        from agentic_autorag.optimizer.diagnosis import StateCard
        from agentic_autorag.optimizer.reasoning_agent import _format_state_card

        card = StateCard(
            cost_aware=True,
            trial_number=3,
            trials_remaining=7,
            best_score_so_far=0.833,
            best_trial_number=2,
            last_trial_delta=-0.03,
            trials_since_best_score=1,
            trial_summaries=[
                {
                    "trial_number": 2,
                    "score": 0.833,
                    "cost_usd": 0.0039,
                    "retrieval_complete": 0.93,
                    "what_changed_from_prev": ["embedding_model: A → B"],
                    "top_failure_modes": ["retrieval", "generation"],
                },
            ],
        )
        rendered = _format_state_card(card)
        assert "retrieval_complete=0.93" in rendered
        # Order: score then retrieval_complete then cost on the same line.
        line = next(line for line in rendered.splitlines() if "trial 2:" in line)
        assert line.index("score=") < line.index("retrieval_complete=") < line.index("cost=")

    def test_cost_cheatsheet_present_in_cost_aware_mode(self) -> None:
        from agentic_autorag.optimizer.reasoning_agent import (
            PROPOSAL_PROMPT,
            _proposal_template_sections,
        )

        sections = _proposal_template_sections(cost_aware=True)
        assert "How to read cost" in sections["cost_cheatsheet"]
        assert "reranker_top_n" in sections["cost_cheatsheet"]
        assert "expander_llm" in sections["cost_cheatsheet"]

        rendered = PROPOSAL_PROMPT.format(
            diagnosis="<d>",
            state_card="<sc>",
            current_config="<cfg>",
            history="<h>",
            key_evidence="<ke>",
            search_space="<ss>",
            knowledge_base="<kb>",
            graph_rules="",
            **sections,
        )
        assert "How to read cost" in rendered
        assert "Disregard cost in this stance" in rendered
        assert "Budget intuition" in rendered

    def test_cost_cheatsheet_absent_in_score_only_mode(self) -> None:
        from agentic_autorag.optimizer.reasoning_agent import (
            PROPOSAL_PROMPT,
            _proposal_template_sections,
        )

        sections = _proposal_template_sections(cost_aware=False)
        assert sections["cost_cheatsheet"] == ""
        assert sections["stance_section"] == ""

        rendered = PROPOSAL_PROMPT.format(
            diagnosis="<d>",
            state_card="<sc>",
            current_config="<cfg>",
            history="<h>",
            key_evidence="<ke>",
            search_space="<ss>",
            knowledge_base="<kb>",
            graph_rules="",
            **sections,
        )
        assert "How to read cost" not in rendered
        assert "Stances" not in rendered
        assert "Disregard cost" not in rendered
        assert "Budget intuition" not in rendered
