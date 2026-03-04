"""LightRAG knowledge graph store wrapper.

Handles building and querying the LightRAG graph. The graph is built once
and persisted to disk in LightRAG's working_dir. Subsequent runs reload
the existing graph automatically by pointing to the same working_dir.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import litellm
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

from agentic_autorag.config.models import GraphBuildConfig

# Silence LiteLLM's "Give Feedback / Get Help" banners — we log retries ourselves.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Files LightRAG writes after a successful graph build.
_GRAPH_MARKER_FILES = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_full_docs.json",
    "kv_store_entity_chunks.json",
]


class LightRAGStore:
    """Manages a LightRAG knowledge graph instance.

    Wraps LightRAG's async API and normalises its output into the standard
    retrieval dict format used by the rest of the engine.

    Persistence:  LightRAG writes all state (graph, vectors, KV cache) to
    ``working_dir``.  Re-initialising with the same directory automatically
    reloads the existing graph — no rebuild required.
    """

    def __init__(
        self,
        working_dir: str | Path,
        build_config: GraphBuildConfig,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._build_config = build_config
        self._rag: LightRAG | None = None
        self._embedder: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the LightRAG instance and initialise its storage backends.

        Must be called before any insert or query operations.
        """
        if self._rag is not None:
            return

        cfg = self._build_config

        # Load the SentenceTransformer once; reuse it for all embed calls.
        loop = asyncio.get_event_loop()
        self._embedder = await loop.run_in_executor(None, SentenceTransformer, cfg.embedding_model)
        embedding_dim = self._embedder.get_sentence_embedding_dimension()

        rag_kwargs: dict[str, Any] = {
            "working_dir": str(self.working_dir),
            "llm_model_func": self._make_llm_func(cfg.extraction_model, cfg.llm_model_max_retries),
            "llm_model_name": cfg.extraction_model,
            "llm_model_max_async": cfg.llm_model_max_async,
            "embedding_func": self._make_embedding_func(self._embedder, embedding_dim),
            "max_parallel_insert": cfg.max_parallel_insert,
        }

        if cfg.chunk_token_size is not None:
            rag_kwargs["chunk_token_size"] = cfg.chunk_token_size
        if cfg.chunk_overlap_token_size is not None:
            rag_kwargs["chunk_overlap_token_size"] = cfg.chunk_overlap_token_size
        if cfg.entity_types is not None:
            rag_kwargs["addon_params"] = {"entity_types": cfg.entity_types}

        self._rag = LightRAG(**rag_kwargs)
        await self._rag.initialize_storages()
        logger.info("LightRAGStore initialised (working_dir=%s)", self.working_dir)

    async def close(self) -> None:
        """Finalise storages and release resources."""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def is_built(self) -> bool:
        """Return True if a completed graph index already exists on disk."""
        return all((self.working_dir / f).exists() for f in _GRAPH_MARKER_FILES)

    async def build(self, documents: list[str]) -> None:
        """Insert all documents into LightRAG to build the knowledge graph.

        Idempotent: skips if ``is_built()`` is already True.
        """
        if self.is_built():
            logger.info("Graph already built at %s — skipping build", self.working_dir)
            return

        self._assert_initialized()
        logger.info("Building LightRAG graph from %d documents …", len(documents))
        await self._rag.ainsert(documents)  # type: ignore[union-attr]
        logger.info("LightRAG graph build complete")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, query: str, mode: str = "hybrid", top_k: int = 60) -> list[dict]:
        """Query the graph and return results as standard retrieval dicts.

        Uses ``aquery_data`` which returns structured entities, relationships,
        and chunks without invoking an LLM — our pipeline handles generation.

        Returns a list of dicts with keys ``id``, ``text``, and ``score``.
        """
        self._assert_initialized()

        param = QueryParam(mode=mode, top_k=top_k, enable_rerank=False)
        result = await self._rag.aquery_data(query, param)  # type: ignore[union-attr]

        if result.get("status") != "success":
            logger.warning(
                "LightRAG query returned non-success status '%s': %s",
                result.get("status"),
                result.get("message"),
            )
            return []

        return self._normalise_result(result.get("data", {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        if self._rag is None:
            raise RuntimeError("LightRAGStore.initialize() must be called before insert or query operations.")

    @staticmethod
    def _make_llm_func(model: str, num_retries: int = 5):
        """Return a LightRAG-compatible async LLM function backed by LiteLLM.

        Implements explicit exponential back-off so transient 429/503 errors
        are logged visibly (at WARNING level) rather than swallowed silently.
        Falls back to raising the last exception after ``num_retries`` attempts.
        """
        _log = logging.getLogger(__name__)

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            last_exc: Exception | None = None
            for attempt in range(num_retries + 1):
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        # Strip LightRAG kwargs that LiteLLM doesn't understand
                        **{k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens"}},
                    )
                    return response.choices[0].message.content
                except Exception as exc:
                    last_exc = exc
                    if attempt < num_retries:
                        wait = 5 * 2**attempt  # 5s, 10s, 20s, 40s, 80s, 160s
                        _log.warning(
                            "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                            attempt + 1,
                            num_retries + 1,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return llm_func

    @staticmethod
    def _make_embedding_func(embedder: SentenceTransformer, embedding_dim: int) -> EmbeddingFunc:
        """Return a LightRAG EmbeddingFunc backed by a local SentenceTransformer.

        Runs the encode call in a thread-pool executor so it doesn't block the
        async event loop during the (potentially slow) graph build phase.
        """
        loop = asyncio.get_event_loop()

        async def embed_func(texts: list[str]) -> np.ndarray:
            return await loop.run_in_executor(None, lambda: embedder.encode(texts, show_progress_bar=False))

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embed_func,
        )

    @staticmethod
    def _normalise_result(data: dict) -> list[dict]:
        """Convert LightRAG's structured query result to our standard format.

        Combines text chunks (primary context), entity descriptions, and
        relationship descriptions into a ranked list of retrieval dicts.
        The ordering preserves LightRAG's internal ranking: chunks first
        (highest relevance), then entities, then relationships.
        """
        docs: list[dict] = []

        # Text chunks are the most directly relevant context
        for i, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("content", "").strip()
            if not content:
                continue
            docs.append(
                {
                    "id": chunk.get("chunk_id", f"lgchunk_{i}"),
                    "text": content,
                    "score": 1.0 / (i + 1),  # rank-based score: first chunk = highest
                }
            )

        # Entity descriptions provide named-entity context
        for i, entity in enumerate(data.get("entities", [])):
            description = entity.get("description", "").strip()
            if not description:
                continue
            entity_name = entity.get("entity_name", f"entity_{i}")
            docs.append(
                {
                    "id": f"lgentity_{entity_name}",
                    "text": f"[Entity: {entity_name}] {description}",
                    "score": 0.5 / (i + 1),
                }
            )

        # Relationship descriptions provide relational context
        for i, rel in enumerate(data.get("relationships", [])):
            description = rel.get("description", "").strip()
            if not description:
                continue
            src = rel.get("src_id", "?")
            tgt = rel.get("tgt_id", "?")
            docs.append(
                {
                    "id": f"lgrel_{src}_{tgt}_{i}",
                    "text": f"[Relation: {src} → {tgt}] {description}",
                    "score": 0.25 / (i + 1),
                }
            )

        return docs
