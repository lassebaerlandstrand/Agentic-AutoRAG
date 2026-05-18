"""Tests for agentic_autorag.engine.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_autorag.config.models import IndexType, RuntimeConfig
from agentic_autorag.engine.pipeline import RAGPipeline, RetrievalResult, RetrievalTiming, RetrievedDocument


def _make_doc(doc_id: str, text: str = "doc text", score: float = 0.9) -> dict:
    """Return a minimal raw document dict (as returned by LanceDB / graph_store)."""
    return {"id": doc_id, "text": text, "score": score}


def _mock_embedder():
    """Return a mock embedder whose `encode` returns a fixed numpy vector."""
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.zeros(384))
    return embedder


def _default_config(**overrides) -> RuntimeConfig:
    defaults = {"generator_llm": "test/model"}
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


def _pipeline(
    *,
    index_type: IndexType = IndexType.VECTOR_ONLY,
    config: RuntimeConfig | None = None,
    vector_store: MagicMock | None = None,
    graph_store: MagicMock | None = None,
    embedder: MagicMock | None = None,
    cross_encoder: MagicMock | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        vector_store=vector_store or MagicMock(),
        graph_store=graph_store,
        config=config or _default_config(),
        embedder=embedder or _mock_embedder(),
        index_type=index_type,
        cross_encoder=cross_encoder,
    )


class TestRetrieveVectorOnly:
    async def test_returns_documents(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc("a"), _make_doc("b")])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=2))

        result = await pipe.retrieve("hello")

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 2
        assert all(isinstance(d, RetrievedDocument) for d in result.documents)
        vs.search_vector.assert_called_once()

    async def test_timing_populated(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc("a")])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=2))

        result = await pipe.retrieve("hello")

        assert isinstance(result.timing, RetrievalTiming)
        assert result.timing.total_s >= 0
        assert result.timing.embed_search_s >= 0

    async def test_respects_top_k(self):
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc(f"d{i}") for i in range(10)])
        pipe = _pipeline(vector_store=vs, config=_default_config(top_k=3))

        result = await pipe.retrieve("q")

        assert len(result.documents) == 3


class TestRetrieveHybridBM25:
    async def test_dispatches_to_hybrid(self):
        vs = MagicMock()
        vs.search_hybrid = MagicMock(return_value=[_make_doc("h1")])
        pipe = _pipeline(
            index_type=IndexType.HYBRID_BM25_VECTOR,
            vector_store=vs,
            config=_default_config(top_k=5, hybrid_alpha=0.7),
        )

        result = await pipe.retrieve("query")

        assert len(result.documents) == 1
        vs.search_hybrid.assert_called_once()
        assert vs.search_hybrid.call_args.kwargs["hybrid_alpha"] == 0.7

    async def test_rrf_fusion_calls_both_paths_and_merges(self):
        """``bm25_vector_fusion='rrf'`` runs vector + BM25 separately and fuses by RRF."""
        vs = MagicMock()
        vs.search_vector = MagicMock(return_value=[_make_doc("v1"), _make_doc("shared")])
        vs.search_bm25 = MagicMock(return_value=[_make_doc("shared"), _make_doc("b1")])
        pipe = _pipeline(
            index_type=IndexType.HYBRID_BM25_VECTOR,
            vector_store=vs,
            config=_default_config(top_k=5, bm25_vector_fusion="rrf"),
        )

        result = await pipe.retrieve("query")

        assert vs.search_vector.called
        assert vs.search_bm25.called
        assert not vs.search_hybrid.called
        # ``shared`` ranks first because it appears in both lists; uniqueness
        # preserved by _rrf_merge's per-id dedup.
        ids = [d.id for d in result.documents]
        assert ids[0] == "shared"
        assert set(ids) == {"shared", "v1", "b1"}

    async def test_alpha_fusion_ignores_search_bm25(self):
        """``bm25_vector_fusion='alpha'`` (default) keeps the old search_hybrid path."""
        vs = MagicMock()
        vs.search_hybrid = MagicMock(return_value=[_make_doc("h1")])
        vs.search_bm25 = MagicMock(return_value=[_make_doc("b1")])
        pipe = _pipeline(
            index_type=IndexType.HYBRID_BM25_VECTOR,
            vector_store=vs,
            config=_default_config(top_k=5, hybrid_alpha=0.4, bm25_vector_fusion="alpha"),
        )

        await pipe.retrieve("q")

        vs.search_hybrid.assert_called_once()
        assert not vs.search_bm25.called


class TestRetrieveGraphOnly:
    async def test_dispatches_to_graph_store(self):
        gs = AsyncMock()
        gs.query = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])
        pipe = _pipeline(
            index_type=IndexType.GRAPH_ONLY,
            graph_store=gs,
            config=_default_config(top_k=5, graph_query_mode="hybrid", graph_top_k=60),
        )

        result = await pipe.retrieve("graph query")

        assert len(result.documents) == 2
        gs.query.assert_called_once_with("graph query", mode="hybrid", top_k=60)

    async def test_returns_empty_when_no_graph_store(self):
        pipe = _pipeline(
            index_type=IndexType.GRAPH_ONLY,
            graph_store=None,
            config=_default_config(top_k=5),
        )

        result = await pipe.retrieve("no graph")

        assert len(result.documents) == 0


class TestRetrieveHybridGraphVector:
    async def test_merges_vector_and_graph_results(self):
        vs = MagicMock()
        vs.search_hybrid = MagicMock(return_value=[_make_doc("v1"), _make_doc("v2")])
        gs = AsyncMock()
        gs.query = AsyncMock(return_value=[_make_doc("g1"), _make_doc("g2")])

        pipe = _pipeline(
            index_type=IndexType.HYBRID_GRAPH_VECTOR,
            vector_store=vs,
            graph_store=gs,
            config=_default_config(top_k=5, hybrid_alpha=0.3, graph_query_mode="hybrid", graph_top_k=60),
        )

        result = await pipe.retrieve("hybrid")

        # All 4 unique docs should be returned (< top_k=5).
        assert len(result.documents) == 4
        vs.search_hybrid.assert_called_once()
        gs.query.assert_called_once_with("hybrid", mode="hybrid", top_k=60)
        assert vs.search_hybrid.call_args.kwargs["hybrid_alpha"] == 0.3


class TestPrepareContext:
    async def test_disabled_preserves_original_order_and_single_newline_join(self):
        """Default config: prepare_context returns the same string as the
        legacy inline ``"\\n".join(doc.text ...)`` (byte-for-byte)."""
        pipe = _pipeline()
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id="a", text="A", score=0.1),
                RetrievedDocument(id="b", text="B", score=0.9),
                RetrievedDocument(id="c", text="C", score=0.5),
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        context, cost = await pipe.prepare_context("q", result)

        assert context == "A\nB\nC"
        assert cost == {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def test_long_context_reorder_appends_top_by_score(self):
        """``long_context_reorder=True``: input order is preserved, with the
        top-scored passage duplicated at the end."""
        pipe = _pipeline(config=_default_config(long_context_reorder=True))
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id="a", text="A", score=0.1),
                RetrievedDocument(id="b", text="B", score=0.9),
                RetrievedDocument(id="c", text="C", score=0.3),
                RetrievedDocument(id="d", text="D", score=0.7),
                RetrievedDocument(id="e", text="E", score=0.5),
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        context, _ = await pipe.prepare_context("q", result)

        # Input order preserved; top-scored 'B' (0.9) appended at the end.
        assert context == "A\nB\nC\nD\nE\nB"

    async def test_long_context_reorder_noop_for_single_passage(self):
        pipe = _pipeline(config=_default_config(long_context_reorder=True))
        result = RetrievalResult(
            documents=[RetrievedDocument(id="a", text="only", score=0.5)],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        context, _ = await pipe.prepare_context("q", result)

        assert context == "only"

    async def test_long_context_reorder_noop_for_empty(self):
        pipe = _pipeline(config=_default_config(long_context_reorder=True))
        result = RetrievalResult(
            documents=[],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        context, _ = await pipe.prepare_context("q", result)

        assert context == ""

    async def test_tree_summarize_calls_llm_per_batch_and_collapses(self):
        """tree_summarize batches by 16 and recurses until ≤1 passage remains.
        For N=3 passages, one batch → one LLM call → single summary."""
        pipe = _pipeline(config=_default_config(passage_compressor="tree_summarize"))
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id="a", text="alpha says X", score=0.9),
                RetrievedDocument(id="b", text="beta says Y", score=0.5),
                RetrievedDocument(id="c", text="gamma says Z", score=0.1),
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        calls = []

        async def fake_generate(prompt: str, **kwargs):
            calls.append(prompt)
            return "summarised", {"usd": 0.01, "prompt_tokens": 10, "completion_tokens": 5}

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("Q?", result)

        assert context == "summarised"
        assert len(calls) == 1
        # All passages should be in the single batch's context_str.
        assert "alpha says X" in calls[0]
        assert "beta says Y" in calls[0]
        assert "gamma says Z" in calls[0]
        assert "Q?" in calls[0]
        # tree_summarize uses the llama_index TreeSummarize default (the
        # "multiple sources" variant), not the single-source text_qa default.
        assert "from multiple sources" in calls[0]
        assert cost == {"usd": 0.01, "prompt_tokens": 10, "completion_tokens": 5}

    async def test_tree_summarize_recurses_for_more_than_one_batch(self):
        """20 passages → batch_size=16 → 2 batches at level 1 → 1 batch at level 2 = 3 LLM calls."""
        pipe = _pipeline(config=_default_config(passage_compressor="tree_summarize"))
        result = RetrievalResult(
            documents=[RetrievedDocument(id=f"d{i}", text=f"P{i}", score=1.0 - i * 0.01) for i in range(20)],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        call_counts = [0]

        async def fake_generate(prompt: str, **kwargs):
            call_counts[0] += 1
            return f"L{call_counts[0]}", {"usd": 0.005, "prompt_tokens": 5, "completion_tokens": 5}

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("q", result)

        assert call_counts[0] == 3
        assert cost["usd"] == 0.015

    async def test_refine_iterates_serial_through_passages(self):
        """refine seeds with the first passage and threads through remaining N-1."""
        pipe = _pipeline(config=_default_config(passage_compressor="refine"))
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id="a", text="A first", score=0.9),
                RetrievedDocument(id="b", text="B second", score=0.5),
                RetrievedDocument(id="c", text="C third", score=0.1),
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        seen_prompts = []
        seen_answers = ["seeded", "refined-once", "refined-twice"]

        async def fake_generate(prompt: str, **kwargs):
            seen_prompts.append(prompt)
            return seen_answers[len(seen_prompts) - 1], {
                "usd": 0.02,
                "prompt_tokens": 7,
                "completion_tokens": 3,
            }

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("Q?", result)

        assert context == "refined-twice"
        assert len(seen_prompts) == 3
        # Seed uses QA prompt; rest use Refine prompt.
        assert "A first" in seen_prompts[0]
        assert "Refined Answer" in seen_prompts[1]
        assert "seeded" in seen_prompts[1]  # existing_answer threaded in
        assert "B second" in seen_prompts[1]
        assert "refined-once" in seen_prompts[2]
        assert "C third" in seen_prompts[2]
        assert cost == {"usd": 0.06, "prompt_tokens": 21, "completion_tokens": 9}

    async def test_compressor_collapses_makes_reorder_noop(self):
        """When compression is on, the result is a single passage, so reorder
        cannot duplicate (len ≤ 1) — output stays single string."""
        pipe = _pipeline(
            config=_default_config(passage_compressor="refine", long_context_reorder=True),
        )
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id="a", text="A", score=0.9),
                RetrievedDocument(id="b", text="B", score=0.1),
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        async def fake_generate(_prompt: str, **kwargs):
            return "final", {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

        with patch.object(pipe, "generate", new=fake_generate):
            context, _ = await pipe.prepare_context("q", result)

        assert context == "final"  # no duplication, no \n

    async def test_compressor_empty_retrieval_is_noop_and_no_llm_calls(self):
        """Empty retrieval → no compression LLM call; empty context out."""
        pipe = _pipeline(config=_default_config(passage_compressor="tree_summarize"))
        result = RetrievalResult(
            documents=[],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        called = [False]

        async def fake_generate(_prompt: str, **kwargs):
            called[0] = True
            return "x", {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("q", result)

        assert context == ""
        assert called[0] is False
        assert cost == {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def test_refine_with_single_passage_calls_qa_only(self):
        """Single passage → seed call only; no refine pass."""
        pipe = _pipeline(config=_default_config(passage_compressor="refine"))
        result = RetrievalResult(
            documents=[RetrievedDocument(id="a", text="lone", score=0.8)],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        calls = []

        async def fake_generate(prompt: str, **kwargs):
            calls.append(prompt)
            return "seeded-answer", {"usd": 0.01, "prompt_tokens": 5, "completion_tokens": 3}

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("q", result)

        assert context == "seeded-answer"
        assert len(calls) == 1
        # Single passage takes the QA path, not the refine path.
        assert "Context information is below" in calls[0]
        assert "Refined Answer" not in calls[0]
        assert cost == {"usd": 0.01, "prompt_tokens": 5, "completion_tokens": 3}

    async def test_tree_summarize_with_single_passage_emits_zero_llm_calls(self):
        """tree_summarize with 1 passage: ``while len > 1`` never enters →
        the single passage is returned verbatim, no LLM cost."""
        pipe = _pipeline(config=_default_config(passage_compressor="tree_summarize"))
        result = RetrievalResult(
            documents=[RetrievedDocument(id="a", text="solo", score=0.8)],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )

        async def fake_generate(_prompt: str, **kwargs):
            raise AssertionError("generate should not be called for single passage")

        with patch.object(pipe, "generate", new=fake_generate):
            context, cost = await pipe.prepare_context("q", result)

        assert context == "solo"
        assert cost["usd"] == 0.0


class TestDeduplication:
    async def test_removes_duplicate_ids(self):
        """When query expansion returns multiple queries, duplicates are removed."""
        vs = MagicMock()
        # Both calls return the same doc.
        vs.search_vector = MagicMock(return_value=[_make_doc("dup")])

        config = _default_config(top_k=5, query_expansion="hyde")
        pipe = _pipeline(vector_store=vs, config=config)

        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=("hypothetical", {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}),
        ):
            result = await pipe.retrieve("q")

        # Two queries (original + HyDE), but the duplicate should be collapsed.
        assert len(result.documents) == 1
        assert result.documents[0].id == "dup"


class TestReranking:
    def test_requires_cross_encoder_when_reranker_enabled(self):
        config = _default_config(
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        with pytest.raises(ValueError, match="cross_encoder"):
            _pipeline(config=config)

    async def test_fetches_more_candidates_and_truncates(self):
        """When reranking is active, fetch top_k*3, rerank, return reranker_top_n."""
        vs = MagicMock()
        docs = [_make_doc(f"d{i}") for i in range(15)]
        vs.search_vector = MagicMock(return_value=docs)

        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(return_value=list(range(15)))

        config = _default_config(
            top_k=5,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_n=3,
        )
        pipe = _pipeline(vector_store=vs, config=config, cross_encoder=mock_ce)

        result = await pipe.retrieve("rerank me")

        assert len(result.documents) == 3
        # The vector store should have been asked for top_k*3 = 15 candidates.
        call_args = vs.search_vector.call_args
        assert call_args.kwargs.get("top_k") == 15


def _mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    """Build a litellm-shaped response mock with usable .choices and .usage."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestGenerate:
    async def test_calls_litellm_and_returns_content_and_cost(self):
        pipe = _pipeline(config=_default_config(generator_llm="ollama/llama3.2", temperature=0.1))

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer", prompt_tokens=10, completion_tokens=4),
            ) as mock_llm,
            patch(
                "agentic_autorag.litellm_runtime.litellm.completion_cost",
                return_value=0.0123,
            ),
        ):
            content, cost = await pipe.generate("prompt text")

        assert content == "answer"
        assert cost == {"usd": 0.0123, "prompt_tokens": 10, "completion_tokens": 4}
        mock_llm.assert_called_once_with(
            model="ollama/llama3.2",
            messages=[{"role": "user", "content": "prompt text"}],
            temperature=0.1,
            num_retries=0,
            timeout=100.0,
        )

    async def test_passes_reasoning_effort_when_reasoning_enabled(self):
        config = _default_config(
            generator_llm="vertex_ai/gemini-2.5-flash",
            temperature=0.0,
            reasoning=True,
            reasoning_effort="high",
        )
        pipe = _pipeline(config=config)

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("reasoned answer"),
            ) as mock_llm,
            patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.0),
        ):
            content, _ = await pipe.generate("complex question")

        assert content == "reasoned answer"
        mock_llm.assert_called_once_with(
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "complex question"}],
            temperature=0.0,
            num_retries=0,
            timeout=100.0,
            reasoning_effort="high",
        )

    async def test_no_reasoning_effort_when_reasoning_disabled(self):
        config = _default_config(
            generator_llm="vertex_ai/gemini-2.5-flash",
            reasoning=False,
        )
        pipe = _pipeline(config=config)

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer"),
            ) as mock_llm,
            patch("agentic_autorag.litellm_runtime.litellm.completion_cost", return_value=0.0),
        ):
            await pipe.generate("simple question")

        call_kwargs = mock_llm.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

    async def test_cost_falls_back_to_zero_on_completion_cost_error(self):
        """When LiteLLM has no pricing for a model, cost gracefully degrades to 0."""
        pipe = _pipeline(config=_default_config(generator_llm="ollama/local-model"))

        with (
            patch(
                "agentic_autorag.litellm_runtime.litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_mock_response("answer"),
            ),
            patch(
                "agentic_autorag.litellm_runtime.litellm.completion_cost",
                side_effect=Exception("no pricing for model"),
            ),
        ):
            _, cost = await pipe.generate("prompt")

        assert cost["usd"] == 0.0


class TestParseDecompose:
    """The magic ``"the question needs no decomposition"`` string and any
    malformed input fall back to ``[query]``; otherwise sub-queries are
    extracted from each ``"N: question"`` line."""

    def test_question_needs_no_decomposition_returns_query(self):
        from agentic_autorag.engine.pipeline import _parse_decompose

        assert _parse_decompose("The question needs no decomposition", "X?") == ["X?"]
        # Case-insensitive.
        assert _parse_decompose("THE QUESTION NEEDS NO DECOMPOSITION", "X?") == ["X?"]

    def test_numbered_lines_yield_questions(self):
        from agentic_autorag.engine.pipeline import _parse_decompose

        raw = "1: Where is Paris?\n2: When was the Eiffel Tower built?"
        assert _parse_decompose(raw, "Q") == [
            "Where is Paris?",
            "When was the Eiffel Tower built?",
        ]

    def test_strips_decompositions_header(self):
        from agentic_autorag.engine.pipeline import _parse_decompose

        raw = "Decompositions:\n1: A\n2: B"
        assert _parse_decompose(raw, "Q") == ["A", "B"]

    def test_empty_or_unparseable_returns_query(self):
        from agentic_autorag.engine.pipeline import _parse_decompose

        assert _parse_decompose("", "fallback") == ["fallback"]
        # No colons → no questions extracted.
        assert _parse_decompose("just some text\nwith no structure", "fallback") == ["fallback"]


class TestExpandQuery:
    async def test_none_returns_original(self):
        pipe = _pipeline(config=_default_config(query_expansion="none"))

        queries, cost = await pipe._expand_query("hello")

        assert queries == ["hello"]
        assert cost == {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def test_query_decompose_replaces_with_subqueries(self):
        """query_decompose: REPLACE the original — does NOT prepend."""
        pipe = _pipeline(config=_default_config(query_expansion="query_decompose"))

        with patch.object(
            pipe,
            "generate",
            new=AsyncMock(
                return_value=(
                    "Decompositions:\n1: Where is Paris?\n2: When built?",
                    {"usd": 0.001, "prompt_tokens": 30, "completion_tokens": 20},
                )
            ),
        ) as mock_gen:
            queries, cost = await pipe._expand_query("original")

        assert queries == ["Where is Paris?", "When built?"]
        assert "original" not in queries  # critical: NOT prepended
        assert cost["usd"] == 0.001
        # The 6-shot prompt was passed to generate.
        passed_prompt = mock_gen.call_args.args[0]
        assert "Decompose a question" in passed_prompt
        assert "original" in passed_prompt

    async def test_query_decompose_no_decomposition_falls_back(self):
        pipe = _pipeline(config=_default_config(query_expansion="query_decompose"))

        with patch.object(
            pipe,
            "generate",
            new=AsyncMock(
                return_value=(
                    "The question needs no decomposition",
                    {"usd": 0.0001, "prompt_tokens": 5, "completion_tokens": 6},
                )
            ),
        ):
            queries, cost = await pipe._expand_query("atomic Q")

        assert queries == ["atomic Q"]
        assert cost["usd"] == 0.0001  # The LLM was called regardless.

    async def test_hyde_returns_two_queries(self):
        pipe = _pipeline(config=_default_config(query_expansion="hyde"))

        gen_cost = {"usd": 0.001, "prompt_tokens": 5, "completion_tokens": 7}
        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=("hypothetical answer", gen_cost),
        ):
            queries, cost = await pipe._expand_query("question")

        assert len(queries) == 2
        assert queries[0] == "question"
        assert queries[1] == "hypothetical answer"
        assert cost == gen_cost

    async def test_multi_query_returns_up_to_four(self):
        pipe = _pipeline(config=_default_config(query_expansion="multi_query"))

        rephrasings = "rephrasing 1\nrephrasing 2\nrephrasing 3\nrephrasing 4"
        with patch.object(
            pipe,
            "generate",
            new_callable=AsyncMock,
            return_value=(rephrasings, {"usd": 0.002, "prompt_tokens": 8, "completion_tokens": 16}),
        ):
            queries, cost = await pipe._expand_query("original")

        # original + 3 rephrasings (cap at 3)
        assert len(queries) == 4
        assert queries[0] == "original"
        assert cost["usd"] == 0.002


class TestRRFMerge:
    def test_merges_two_lists(self):
        list_a = [_make_doc("a"), _make_doc("b")]
        list_b = [_make_doc("b"), _make_doc("c")]

        merged = RAGPipeline._rrf_merge(list_a, list_b, k=60)

        ids = [d["id"] for d in merged]
        # "b" appears in both lists, so it should have the highest fused score.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_empty_lists(self):
        assert RAGPipeline._rrf_merge([], []) == []

    def test_one_empty_list(self):
        list_a = [_make_doc("x")]
        merged = RAGPipeline._rrf_merge(list_a, [])
        assert len(merged) == 1
        assert merged[0]["id"] == "x"

    def test_preserves_doc_data(self):
        doc = _make_doc("z", text="important", score=0.99)
        merged = RAGPipeline._rrf_merge([doc], [])
        assert merged[0]["text"] == "important"
