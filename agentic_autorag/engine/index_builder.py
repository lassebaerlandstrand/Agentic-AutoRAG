"""Build search indices from parsed documents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.models import GraphConfig, IndexType, StructuralConfig
from agentic_autorag.engine.vector_store import LanceDBStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGIndex:
    """In-memory handle to a built retrieval index."""

    vector_store: LanceDBStore
    chunks: list[str]
    embeddings: np.ndarray
    index_type: IndexType
    graph_store: Any | None = None

    def search_vector(self, query_embedding: np.ndarray | Sequence[float], top_k: int = 5) -> list[dict]:
        return self.vector_store.search_vector(query_embedding, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 5,
    ) -> list[dict]:
        return self.vector_store.search_hybrid(query, query_embedding, top_k=top_k)

    async def search_graph(self, query: str, top_k: int = 5) -> list[dict]:
        if self.graph_store is None:
            logger.warning("Graph search requested but graph store is not available. Returning no results.")
            return []
        return await self.graph_store.search(query, top_k=top_k)


class IndexBuilder:
    """Builds searchable indices from parsed text documents."""

    SPLITTER_MAP = {
        "recursive": RecursiveCharacterTextSplitter,
        "fixed": CharacterTextSplitter,
    }

    def __init__(self, db_path: str | Path = "./data/lancedb", table_name: str = "documents") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._embedder_cache: dict[str, SentenceTransformer] = {}

    async def build(
        self,
        documents: list[str],
        config: StructuralConfig,
        graph_config: GraphConfig | None = None,
    ) -> RAGIndex:
        """Build a retrieval index from already-parsed documents."""
        del graph_config  # Reserved for future graph index implementation.

        splitter_cls = self.SPLITTER_MAP.get(config.chunking_strategy)
        if splitter_cls is None:
            supported = ", ".join(sorted(self.SPLITTER_MAP))
            raise ValueError(f"Unsupported chunking_strategy '{config.chunking_strategy}'. Supported: {supported}")

        splitter = splitter_cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = self._chunk_documents(documents, splitter)
        if not chunks:
            raise ValueError("No chunks were produced from the provided documents.")

        embedder = self._get_embedder(config.embedding_model)
        embeddings = np.asarray(embedder.encode(chunks, show_progress_bar=False), dtype=np.float32)

        records = [
            {"id": f"chunk_{i}", "text": chunk, "vector": embedding.tolist()}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]

        vector_store = LanceDBStore(db_path=self.db_path)
        vector_store.create_index(records, table_name=self.table_name, mode="overwrite")

        if config.index_type in (IndexType.GRAPH, IndexType.HYBRID_GRAPH_VECTOR):
            logger.warning(
                "Index type '%s' requested, but graph indexing is not implemented yet. "
                "Proceeding with vector index only.",
                config.index_type.value,
            )

        return RAGIndex(
            vector_store=vector_store,
            chunks=chunks,
            embeddings=embeddings,
            index_type=config.index_type,
            graph_store=None,
        )

    @staticmethod
    def _chunk_documents(documents: list[str], splitter: Any) -> list[str]:
        chunks: list[str] = []
        for document in tqdm(documents, desc="Chunking documents", unit="doc"):
            if not document.strip():
                continue
            for chunk in splitter.split_text(document):
                chunk_text = chunk.strip()
                if chunk_text:
                    chunks.append(chunk_text)
        return chunks

    def _get_embedder(self, model_name: str) -> SentenceTransformer:
        embedder = self._embedder_cache.get(model_name)
        if embedder is None:
            embedder = SentenceTransformer(model_name)
            self._embedder_cache[model_name] = embedder
        return embedder
