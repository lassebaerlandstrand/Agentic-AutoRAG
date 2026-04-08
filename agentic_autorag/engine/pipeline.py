"""RAG pipeline: retrieval (with query expansion, dedup, reranking) and generation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from sentence_transformers import CrossEncoder

from agentic_autorag.config.models import IndexType, RuntimeConfig

logger = logging.getLogger(__name__)

# Single-thread executor for CPU-bound model inference (embedding, reranking).
# Serializes to avoid thread contention while keeping the event loop free.
_model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


@dataclass(slots=True)
class RetrievedDocument:
    """A single document returned by the retrieval stage."""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalTiming:
    """Wall-clock breakdown of retrieval sub-stages (seconds)."""

    expand_s: float = 0.0
    embed_search_s: float = 0.0
    rerank_s: float = 0.0
    total_s: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    """Wrapper around a list of retrieved documents."""

    documents: list[RetrievedDocument]
    timing: RetrievalTiming = field(default_factory=RetrievalTiming)


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

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents using the configured strategy."""
        t_start = time.monotonic()

        t0 = time.monotonic()
        queries = await self._expand_query(query)
        expand_s = time.monotonic() - t0

        reranking = self.config.reranker != "none"
        fetch_k = self.config.top_k * 3 if reranking else self.config.top_k

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        all_docs: list[dict] = []
        for q in queries:
            q_embedding = await loop.run_in_executor(_model_executor, self.embedder.encode, q)
            docs = await self._dispatch_search(q, q_embedding, fetch_k)
            all_docs.extend(docs)
        embed_search_s = time.monotonic() - t0

        unique_docs = self._deduplicate(all_docs)

        rerank_s = 0.0
        if reranking:
            t0 = time.monotonic()
            unique_docs = await self._rerank(query, unique_docs)
            rerank_s = time.monotonic() - t0
            final = unique_docs[: self.config.reranker_top_n]
        else:
            final = unique_docs[: self.config.top_k]

        timing = RetrievalTiming(
            expand_s=expand_s,
            embed_search_s=embed_search_s,
            rerank_s=rerank_s,
            total_s=time.monotonic() - t_start,
        )
        logger.debug(
            "Retrieval timing: expand=%.3fs embed_search=%.3fs rerank=%.3fs total=%.3fs",
            timing.expand_s,
            timing.embed_search_s,
            timing.rerank_s,
            timing.total_s,
        )
        return RetrievalResult(
            documents=[self._to_retrieved_doc(d) for d in final],
            timing=timing,
        )

    async def generate(self, prompt: str) -> str:
        """Generate a response using the configured LLM via LiteLLM."""
        kwargs: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "num_retries": 0,
            "timeout": self.config.llm_timeout_s,
        }
        if self.config.reasoning:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content

    async def _expand_query(self, query: str) -> list[str]:
        """Return one or more queries depending on the expansion strategy."""
        strategy = self.config.query_expansion

        if strategy == "hyde":
            hypothetical = await self.generate(f"Write a short paragraph that would answer: {query}")
            return [query, hypothetical]

        if strategy == "multi_query":
            raw = await self.generate(
                f"Generate 3 different phrasings of this question:\n{query}\nReturn each on a new line."
            )
            variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            return [query] + variants[:3]

        return [query]

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

    async def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Rerank *docs* using a cross-encoder model."""
        if not docs:
            return docs

        if self._cross_encoder is None:
            raise RuntimeError("cross_encoder is required for reranking but was not set")

        pairs = [(query, doc.get("text", "")) for doc in docs]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(_model_executor, self._cross_encoder.predict, pairs)

        scored = sorted(
            zip(scores, docs, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in scored]

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
        return RetrievedDocument(
            id=raw.get("id", ""),
            text=raw.get("text", ""),
            score=float(raw.get("score", raw.get("_distance", 0.0))),
            metadata={k: v for k, v in raw.items() if k not in {"id", "text", "score", "_distance", "vector"}},
        )
