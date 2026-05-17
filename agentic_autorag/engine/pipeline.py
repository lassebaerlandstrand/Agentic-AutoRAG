"""RAG pipeline: retrieval (with query expansion, dedup, reranking) and generation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import CrossEncoder

from agentic_autorag.config.models import IndexType, RuntimeConfig
from agentic_autorag.litellm_runtime import acompletion_with_cost

logger = logging.getLogger(__name__)

# Single-thread executor for CPU-bound model inference (embedding, reranking).
# Serializes to avoid thread contention while keeping the event loop free.
_model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# llama_index's response_synthesizers defaults. ``_DEFAULT_TREE_SUMMARIZE_PROMPT``
# is the completion-mode template used by ``TreeSummarize`` (note the "multiple
# sources" framing — distinct from ``_DEFAULT_TEXT_QA_PROMPT``). ``_DEFAULT_TEXT_QA_PROMPT``
# is the seed-call template for ``Refine``. ``_DEFAULT_REFINE_PROMPT`` is the
# refinement template for ``Refine``. Keeping all three in sync with the AutoRAG
# / llama_index defaults lets us pin both sides of the comparison to identical text.
_DEFAULT_TREE_SUMMARIZE_PROMPT_TMPL = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

_DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

_DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

_PASSAGE_COMPRESSOR_BATCH_SIZE = 16


# 6-shot multi-hop decomposition prompt (Visconde / StrategyQA style). The
# ``{question}`` placeholder is substituted with the live query at format time.
_QUERY_DECOMPOSE_PROMPT = """Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.

    Example 1:

    Question: Is Hamlet more common on IMDB than Comedy of Errors?
    Decompositions:
    1: How many listings of Hamlet are there on IMDB?
    2: How many listing of Comedy of Errors is there on IMDB?

    Example 2:

    Question: Are birds important to badminton?

    Decompositions:
    The question needs no decomposition

    Example 3:

    Question: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?

    Decompositions:
    1: What is the minimum driving age in the US?
    2: What is the minimum age for someone to be employed in the US?

    Example 4:

    Question: Are all cucumbers the same texture?

    Decompositions:
    The question needs no decomposition

    Example 5:

    Question: Hydrogen's atomic number squared exceeds number of Spice Girls?

    Decompositions:
    1: What is the atomic number of hydrogen?
    2: How many Spice Girls are there?

    Example 6:

    Question: {question}

    Decompositions:
    """


def _parse_decompose(answer: str, query: str) -> list[str]:
    """Parse a decomposition response into sub-queries.

    The magic string ``"The question needs no decomposition"`` (case
    insensitive) and any malformed output fall back to ``[query]``. The
    original query is NOT prepended — sub-queries fully replace it.
    """
    if answer.lower().strip() == "the question needs no decomposition":
        return [query]
    try:
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if lines and lines[0].startswith("Decompositions:"):
            lines.pop(0)
        questions = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
        if not questions:
            return [query]
        return questions
    except (IndexError, ValueError):
        return [query]


@dataclass(slots=True)
class RetrievedDocument:
    """A single document returned by the retrieval stage."""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    # (char_start, char_end) of this chunk in its source document, when known.
    # Populated by vector/hybrid retrieval (LanceDB metadata). Left ``None`` for
    # graph retrieval where offsets aren't stored; the evaluator looks them up at
    # query time via ``str.find`` for verbatim graph chunks, or falls back to
    # n-gram matching for synthesized entity/relationship descriptions.
    char_range: tuple[int, int] | None = None


@dataclass(slots=True)
class RetrievalTiming:
    """Wall-clock breakdown of retrieval sub-stages (seconds)."""

    expand_s: float = 0.0
    embed_search_s: float = 0.0
    rerank_s: float = 0.0
    model_s: float = 0.0  # actual encode+rerank compute (excludes queue wait)
    total_s: float = 0.0


def _zero_cost() -> dict[str, float | int]:
    return {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def _accumulate_cost(target: dict[str, float | int], delta: dict[str, float | int]) -> None:
    target["usd"] = float(target.get("usd", 0.0)) + float(delta.get("usd", 0.0))
    target["prompt_tokens"] = int(target.get("prompt_tokens", 0)) + int(delta.get("prompt_tokens", 0))
    target["completion_tokens"] = int(target.get("completion_tokens", 0)) + int(delta.get("completion_tokens", 0))


@dataclass(slots=True)
class RetrievalResult:
    """Wrapper around a list of retrieved documents."""

    documents: list[RetrievedDocument]
    timing: RetrievalTiming = field(default_factory=RetrievalTiming)
    expansion_cost: dict[str, float | int] = field(default_factory=_zero_cost)


class RAGPipeline:
    """Configurable RAG pipeline constructed from runtime parameters.

    The pipeline is instantiated *per trial* by the evaluator.  Structural
    parameters (chunking, embedding model, index type) are already baked into
    the pre-built ``vector_store`` and ``graph_store`` — this class only needs
    ``RuntimeConfig`` to control retrieval and generation behaviour.
    """

    def __init__(
        self,
        vector_store: Any,
        graph_store: Any | None,
        config: RuntimeConfig,
        embedder: Any,
        index_type: IndexType,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        if config.reranker != "none" and cross_encoder is None:
            raise ValueError(
                "Reranking is enabled but no cross_encoder was provided. "
                "Pass a CrossEncoder when runtime.reranker is not 'none'."
            )
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config
        self.embedder = embedder
        self.index_type = index_type
        self._cross_encoder: CrossEncoder | None = cross_encoder

    async def _run_model(self, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Run *fn* in the model executor, returning ``(result, compute_seconds)``."""

        def _timed() -> tuple[Any, float]:
            t = time.monotonic()
            result = fn(*args, **kwargs)
            return result, time.monotonic() - t

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_model_executor, _timed)

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents using the configured strategy."""
        t_start = time.monotonic()

        t0 = time.monotonic()
        queries, expansion_cost = await self._expand_query(query)
        expand_s = time.monotonic() - t0

        reranking = self.config.reranker != "none"
        fetch_k = self.config.top_k * 3 if reranking else self.config.top_k

        t0 = time.monotonic()
        model_compute_s = 0.0
        all_docs: list[dict] = []
        for q in queries:
            q_embedding, encode_s = await self._run_model(self.embedder.encode, q, show_progress_bar=False)
            model_compute_s += encode_s
            docs = await self._dispatch_search(q, q_embedding, fetch_k)
            all_docs.extend(docs)
        embed_search_s = time.monotonic() - t0

        unique_docs = self._deduplicate(all_docs)

        rerank_s = 0.0
        if reranking:
            t0 = time.monotonic()
            unique_docs, rerank_compute_s = await self._rerank(query, unique_docs)
            model_compute_s += rerank_compute_s
            rerank_s = time.monotonic() - t0
            final = unique_docs[: self.config.reranker_top_n]
        else:
            final = unique_docs[: self.config.top_k]

        timing = RetrievalTiming(
            expand_s=expand_s,
            embed_search_s=embed_search_s,
            rerank_s=rerank_s,
            model_s=model_compute_s,
            total_s=time.monotonic() - t_start,
        )
        logger.debug(
            "Retrieval timing: expand=%.3fs embed_search=%.3fs rerank=%.3fs model=%.3fs total=%.3fs",
            timing.expand_s,
            timing.embed_search_s,
            timing.rerank_s,
            timing.model_s,
            timing.total_s,
        )
        return RetrievalResult(
            documents=[self._to_retrieved_doc(d) for d in final],
            timing=timing,
            expansion_cost=expansion_cost,
        )

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        apply_reasoning_effort: bool | None = None,
    ) -> tuple[str, dict[str, float | int]]:
        """Generate a response using the configured LLM via LiteLLM.

        ``model`` selects which per-stage LLM to use. Defaults to
        ``config.generator_llm`` so external callers (evaluators) get the
        final-answer LLM without having to pass it explicitly.
        ``apply_reasoning_effort`` is True for the generator call and False
        for compressor/expander stages — reasoning only applies to the final
        answer step.

        Returns the answer text plus a cost dict
        ``{"usd", "prompt_tokens", "completion_tokens"}``. ``usd`` is 0.0
        when LiteLLM has no pricing for the model.
        """
        if model is None:
            model = self.config.generator_llm
        if apply_reasoning_effort is None:
            apply_reasoning_effort = model == self.config.generator_llm
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "num_retries": 0,
            "timeout": self.config.llm_timeout_s,
        }
        if self.config.reasoning and apply_reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response, cost = await acompletion_with_cost(cost_category="rag_eval", **kwargs)
        return response.choices[0].message.content, cost

    async def prepare_context(
        self, query: str, retrieval_result: RetrievalResult
    ) -> tuple[str, dict[str, float | int]]:
        """Build the joined-passage context string consumed by the grader's prompt.

        Applies ``passage_compressor`` (if any), then ``long_context_reorder``
        (no-op after compression collapses to one passage), then joins on a
        single newline. Returns the joined string and the accumulated LLM
        cost from any compressor calls.
        """
        passages = [doc.text for doc in retrieval_result.documents]
        cost = _zero_cost()
        if self.config.passage_compressor != "none" and passages:
            compressed, c_cost = await self._compress_passages(query, passages)
            _accumulate_cost(cost, c_cost)
            passages = [compressed]
        if self.config.long_context_reorder and len(passages) > 1:
            scores = [doc.score for doc in retrieval_result.documents]
            top_idx = max(range(len(passages)), key=lambda i: scores[i])
            passages = passages + [passages[top_idx]]
        return "\n".join(passages), cost

    async def _compress_passages(
        self, query: str, passages: list[str]
    ) -> tuple[str, dict[str, float | int]]:
        if not passages:
            return "", _zero_cost()
        method = self.config.passage_compressor
        if method == "tree_summarize":
            return await self._tree_summarize(query, passages)
        if method == "refine":
            return await self._refine(query, passages)
        raise ValueError(f"Unknown passage_compressor {method!r}")

    async def _tree_summarize(
        self, query: str, passages: list[str]
    ) -> tuple[str, dict[str, float | int]]:
        """Batch passages by ``_PASSAGE_COMPRESSOR_BATCH_SIZE``, summarise each
        batch concurrently, recurse until one passage remains."""
        cost = _zero_cost()
        current = list(passages)
        batch_size = _PASSAGE_COMPRESSOR_BATCH_SIZE
        compressor_model = self.config.compressor_llm
        while len(current) > 1:
            tasks = []
            for i in range(0, len(current), batch_size):
                batch = current[i : i + batch_size]
                context_str = "\n\n".join(batch)
                prompt = _DEFAULT_TREE_SUMMARIZE_PROMPT_TMPL.format(context_str=context_str, query_str=query)
                tasks.append(self.generate(prompt, model=compressor_model, apply_reasoning_effort=False))
            results = await asyncio.gather(*tasks)
            new_passages = []
            for answer, gen_cost in results:
                new_passages.append(answer)
                _accumulate_cost(cost, gen_cost)
            current = new_passages
        return current[0], cost

    async def _refine(
        self, query: str, passages: list[str]
    ) -> tuple[str, dict[str, float | int]]:
        """Seed an answer from the first passage, then refine serially through
        the remaining passages."""
        cost = _zero_cost()
        compressor_model = self.config.compressor_llm
        seed_prompt = _DEFAULT_TEXT_QA_PROMPT_TMPL.format(context_str=passages[0], query_str=query)
        answer, gen_cost = await self.generate(seed_prompt, model=compressor_model, apply_reasoning_effort=False)
        _accumulate_cost(cost, gen_cost)
        for passage in passages[1:]:
            refine_prompt = _DEFAULT_REFINE_PROMPT_TMPL.format(
                query_str=query,
                existing_answer=answer,
                context_msg=passage,
            )
            answer, gen_cost = await self.generate(refine_prompt, model=compressor_model, apply_reasoning_effort=False)
            _accumulate_cost(cost, gen_cost)
        return answer, cost

    async def _expand_query(self, query: str) -> tuple[list[str], dict[str, float | int]]:
        """Return one or more queries depending on the expansion strategy.

        Returns the queries plus the summed cost across any LLM expansion calls.
        """
        strategy = self.config.query_expansion
        accumulated = _zero_cost()
        expander_model = self.config.expander_llm

        if strategy == "hyde":
            hypothetical, cost = await self.generate(
                f"Write a short paragraph that would answer: {query}",
                model=expander_model,
                apply_reasoning_effort=False,
            )
            _accumulate_cost(accumulated, cost)
            return [query, hypothetical], accumulated

        if strategy == "multi_query":
            raw, cost = await self.generate(
                f"Generate 3 different phrasings of this question:\n{query}\nReturn each on a new line.",
                model=expander_model,
                apply_reasoning_effort=False,
            )
            _accumulate_cost(accumulated, cost)
            variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            return [query] + variants[:3], accumulated

        if strategy == "query_decompose":
            raw, cost = await self.generate(
                _QUERY_DECOMPOSE_PROMPT.format(question=query),
                model=expander_model,
                apply_reasoning_effort=False,
            )
            _accumulate_cost(accumulated, cost)
            return _parse_decompose(raw, query), accumulated

        return [query], accumulated

    async def _dispatch_search(
        self,
        query: str,
        query_embedding: Any,
        top_k: int,
    ) -> list[dict]:
        """Route to the correct search backend based on ``self.index_type``."""
        if self.index_type == IndexType.VECTOR_ONLY:
            return self.vector_store.search_vector(
                query_embedding,
                top_k=top_k,
            )

        if self.index_type == IndexType.HYBRID_BM25_VECTOR:
            if self.config.bm25_vector_fusion == "rrf":
                vector_docs = self.vector_store.search_vector(
                    query_embedding,
                    top_k=top_k,
                )
                bm25_docs = self.vector_store.search_bm25(
                    query,
                    top_k=top_k,
                )
                return self._rrf_merge(vector_docs, bm25_docs)
            return self.vector_store.search_hybrid(
                query,
                query_embedding,
                top_k=top_k,
                hybrid_alpha=self.config.hybrid_alpha,
            )

        if self.index_type == IndexType.GRAPH_ONLY:
            if self.graph_store is None:
                logger.warning("Graph search requested but no graph_store provided.")
                return []
            return await self.graph_store.query(
                query,
                mode=self.config.graph_query_mode,
                top_k=self.config.graph_top_k,
            )

        if self.index_type == IndexType.HYBRID_GRAPH_VECTOR:
            vector_docs = self.vector_store.search_hybrid(
                query,
                query_embedding,
                top_k=top_k,
                hybrid_alpha=self.config.hybrid_alpha,
            )
            graph_docs: list[dict] = []
            if self.graph_store is not None:
                graph_docs = await self.graph_store.query(
                    query,
                    mode=self.config.graph_query_mode,
                    top_k=self.config.graph_top_k,
                )
            else:
                logger.warning(
                    "hybrid_graph_vector index but no graph_store; falling back to vector-only.",
                )
            return self._rrf_merge(vector_docs, graph_docs)

        logger.warning(
            "Unrecognised index_type '%s'; defaulting to vector search.",
            self.index_type,
        )
        return self.vector_store.search_vector(
            query_embedding,
            top_k=top_k,
        )

    @staticmethod
    def _deduplicate(docs: list[dict]) -> list[dict]:
        """Remove duplicate documents by ``id``, preserving first occurrence."""
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in docs:
            doc_id = doc.get("id", "")
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique

    async def _rerank(self, query: str, docs: list[dict]) -> tuple[list[dict], float]:
        """Rerank *docs* using a cross-encoder model. Returns (ranked_docs, compute_seconds)."""
        if not docs:
            return docs, 0.0

        if self._cross_encoder is None:
            raise RuntimeError("cross_encoder is required for reranking but was not set")

        pairs = [(query, doc.get("text", "")) for doc in docs]
        scores, compute_s = await self._run_model(self._cross_encoder.predict, pairs, show_progress_bar=False)

        scored = sorted(
            zip(scores, docs, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in scored], compute_s

    @staticmethod
    def _rrf_merge(
        list_a: list[dict],
        list_b: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion to merge two ranked result lists.

        ``score(doc) = Σ  1 / (k + rank + 1)``  over both lists.
        """
        scores: dict[str, float] = {}
        for rank, doc in enumerate(list_a):
            doc_id = doc.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc in enumerate(list_b):
            doc_id = doc.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        all_docs: dict[str, dict] = {}
        for doc in list_a + list_b:
            doc_id = doc.get("id", "")
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs[doc_id] for doc_id, _ in ranked if doc_id in all_docs]

    @staticmethod
    def _to_retrieved_doc(raw: dict) -> RetrievedDocument:
        """Convert a raw dict to a ``RetrievedDocument``."""
        cs, ce = raw.get("char_start"), raw.get("char_end")
        char_range: tuple[int, int] | None = (int(cs), int(ce)) if cs is not None and ce is not None else None
        excluded = {"id", "text", "score", "_distance", "vector", "char_start", "char_end"}
        return RetrievedDocument(
            id=raw.get("id", ""),
            text=raw.get("text", ""),
            score=float(raw.get("score", raw.get("_distance", 0.0))),
            metadata={k: v for k, v in raw.items() if k not in excluded},
            char_range=char_range,
        )
