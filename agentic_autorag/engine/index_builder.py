"""Build search indices from parsed documents."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.models import IndexType, StructuralConfig
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
        return await self.graph_store.query(query, top_k=top_k)

    def save(self, path: str | Path) -> None:
        """Persist the vector index to a directory for later reuse.

        The graph index (if any) lives in its own working_dir managed by
        LightRAGStore and is not serialised here.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        vector_snapshot_dir = target / "vector_store.lance"
        self.vector_store.snapshot(vector_snapshot_dir)

        metadata = {
            "index_type": self.index_type.value,
            "n_chunks": len(self.chunks),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (target / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        np.save(target / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> RAGIndex:
        """Restore a previously persisted vector index from a directory.

        ``graph_store`` is always None after loading — the orchestrator
        injects the already-initialised LightRAGStore instance.
        """
        source = Path(path)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Index path does not exist or is not a directory: {source}")

        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        chunks = json.loads((source / "chunks.json").read_text(encoding="utf-8"))
        embeddings = np.load(source / "embeddings.npy")
        vector_store = LanceDBStore.from_snapshot(source / "vector_store.lance")

        return cls(
            vector_store=vector_store,
            chunks=chunks,
            embeddings=embeddings,
            index_type=IndexType(metadata["index_type"]),
            graph_store=None,
        )


class IndexBuilder:
    """Builds searchable vector indices from parsed text documents."""

    SPLITTER_SEPARATORS = {
        "recursive": ["\n\n", "\n", " ", ""],
        "fixed": ["\n", " ", ""],
    }

    def __init__(self, db_path: str | Path = "./data/lancedb", table_name: str = "documents") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._embedder_cache: dict[str, SentenceTransformer] = {}
        self._cross_encoder_cache: dict[str, CrossEncoder] = {}

    async def build(
        self,
        documents: list[str],
        config: StructuralConfig,
    ) -> RAGIndex:
        """Build a vector retrieval index from already-parsed documents.

        Graph indices are built separately by LightRAGStore and not handled here.
        The returned RAGIndex always has ``graph_store=None``; the orchestrator
        attaches the LightRAGStore after the fact.
        """
        separators = self.SPLITTER_SEPARATORS.get(config.chunking_strategy)
        if separators is None:
            supported = ", ".join(sorted(self.SPLITTER_SEPARATORS))
            raise ValueError(f"Unsupported chunking_strategy '{config.chunking_strategy}'. Supported: {supported}")

        # Load embedder first so we can use its tokenizer for token-based splitting
        embedder = self.get_embedder(config.embedding_model)

        tokenizer = getattr(embedder, "tokenizer", None)
        if tokenizer is not None:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=config.chunk_token_size,
                chunk_overlap=config.chunk_token_overlap,
                separators=separators,
            )
        else:
            # Fallback for models without a tokenizer (e.g., API-based embedders)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_token_size,
                chunk_overlap=config.chunk_token_overlap,
                separators=separators,
            )
        chunks = self._chunk_documents(documents, splitter)
        if not chunks:
            raise ValueError("No chunks were produced from the provided documents.")

        logger.info("Embedding %d chunks with %s", len(chunks), config.embedding_model)
        embeddings = np.asarray(
            embedder.encode(chunks, show_progress_bar=True),
            dtype=np.float32,
        )

        records = [
            {"id": f"chunk_{i}", "text": chunk, "vector": embedding.tolist()}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]

        logger.info("Creating LanceDB vector index (%d records)", len(records))
        vector_store = LanceDBStore(db_path=self.db_path)
        vector_store.create_index(records, table_name=self.table_name, mode="overwrite")

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

    def get_embedder(self, model_name: str) -> SentenceTransformer:
        """Return a cached SentenceTransformer, evicting any other cached embedder first."""
        if model_name not in self._embedder_cache:
            self._evict_models(self._embedder_cache, {model_name})
            model_kwargs = {"dtype": torch.float16} if torch.cuda.is_available() else {}
            self._embedder_cache[model_name] = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        return self._embedder_cache[model_name]

    def get_cross_encoder(self, model_name: str) -> CrossEncoder:
        """Return a cached CrossEncoder, evicting any other cached cross-encoder first."""
        if model_name not in self._cross_encoder_cache:
            self._evict_models(self._cross_encoder_cache, {model_name})
            self._cross_encoder_cache[model_name] = CrossEncoder(model_name)
        return self._cross_encoder_cache[model_name]

    @staticmethod
    def _evict_models(cache: dict, keep: set[str]) -> None:
        """Delete all cache entries not in *keep*, then free CUDA allocator pages."""
        to_remove = [k for k in cache if k not in keep]
        if not to_remove:
            return
        for k in to_remove:
            del cache[k]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
