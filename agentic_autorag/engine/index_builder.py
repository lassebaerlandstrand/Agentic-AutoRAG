"""Build search indices from parsed documents."""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import json
import logging
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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
    """In-memory handle to a built retrieval index.

    Not serialised. The LanceDB table is rebuilt per trial from cached chunks
    and embeddings (see ``IngredientCache``); the graph store is a separate
    singleton attached by the orchestrator.
    """

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


class IngredientCache:
    """Persistent LRU cache for chunks + embeddings, keyed by content hash.

    Each cache entry is one directory containing chunks.json and embeddings.npy,
    together representing all the expensive-to-compute state needed to
    reconstruct a RAGIndex. The LanceDB table itself is rebuilt on demand from
    these ingredients — BM25 FTS indexing on ~10k chunks takes seconds, whereas
    re-embedding takes minutes. Writes are atomic via tempfile + os.replace.
    LRU eviction runs after every store() and keeps total size <= max_bytes.
    """

    def __init__(self, cache_dir: str | Path, max_bytes: int) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.max_bytes = max_bytes
        self.manifest: dict[str, dict] = self._load_manifest()

    def has(self, fingerprint: str) -> bool:
        """Return True iff a live entry exists for *fingerprint*."""
        entry_dir = self.root / fingerprint
        return (
            fingerprint in self.manifest
            and (entry_dir / "chunks.json").exists()
            and (entry_dir / "embeddings.npy").exists()
        )

    def load(self, fingerprint: str) -> tuple[list[str], np.ndarray] | None:
        """Return ``(chunks, embeddings)`` if cached, else None."""
        entry_dir = self.root / fingerprint
        chunks_path = entry_dir / "chunks.json"
        embeddings_path = entry_dir / "embeddings.npy"

        if fingerprint not in self.manifest or not chunks_path.exists() or not embeddings_path.exists():
            if fingerprint in self.manifest:
                # Stale manifest entry — the dir was removed externally.
                del self.manifest[fingerprint]
                self._save_manifest()
            return None

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        self.manifest[fingerprint]["last_accessed"] = datetime.now(UTC).isoformat()
        self._save_manifest()
        return chunks, embeddings

    def store(self, fingerprint: str, chunks: list[str], embeddings: np.ndarray) -> None:
        entry_dir = self.root / fingerprint
        entry_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = entry_dir / "chunks.json"
        self._atomic_write_text(chunks_path, json.dumps(chunks))

        embeddings_path = entry_dir / "embeddings.npy"
        self._atomic_write_npy(embeddings_path, embeddings)

        size_bytes = chunks_path.stat().st_size + embeddings_path.stat().st_size
        self.manifest[fingerprint] = {
            "size_bytes": size_bytes,
            "last_accessed": datetime.now(UTC).isoformat(),
        }
        self._evict_if_over_budget(protect={fingerprint})
        self._save_manifest()

    def _evict_if_over_budget(self, protect: set[str]) -> None:
        total = sum(entry["size_bytes"] for entry in self.manifest.values())
        if total <= self.max_bytes:
            return

        candidates = sorted(
            ((fp, e) for fp, e in self.manifest.items() if fp not in protect),
            key=lambda item: item[1]["last_accessed"],
        )
        for fp, entry in candidates:
            if total <= self.max_bytes:
                break
            shutil.rmtree(self.root / fp, ignore_errors=True)
            del self.manifest[fp]
            total -= entry["size_bytes"]
            logger.info("Evicted cache entry %s (%.1f MB)", fp, entry["size_bytes"] / 1e6)

    def _load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt manifest at %s; starting fresh", self.manifest_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_manifest(self) -> None:
        self._atomic_write_text(self.manifest_path, json.dumps(self.manifest, indent=2))

    @staticmethod
    def _atomic_write_text(path: Path, data: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
        # np.save appends ".npy" to string/Path arguments unless passed a file object.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, array)
        os.replace(tmp, path)


class IndexBuilder:
    """Builds searchable vector indices from parsed text documents."""

    SPLITTER_SEPARATORS = {
        "recursive": ["\n\n", "\n", " ", ""],
        "fixed": ["\n", " ", ""],
    }

    def __init__(
        self,
        cache: IngredientCache | None = None,
        table_name: str = "documents",
    ) -> None:
        self.table_name = table_name
        self.cache = cache
        self._embedder_cache: dict[str, SentenceTransformer] = {}
        self._cross_encoder_cache: dict[str, CrossEncoder] = {}

    async def build(
        self,
        documents: list[str],
        config: StructuralConfig,
        corpus_hash: str,
        embedding_token_limits: dict[str, int] | None = None,
    ) -> RAGIndex:
        """Build a vector retrieval index from parsed documents.

        Chunks and embeddings are loaded from ``self.cache`` when available
        (keyed by ``config.embeddings_fingerprint(corpus_hash)``); otherwise
        they are computed and cached for reuse. Each call produces a fresh
        in-memory LanceDB table so concurrently-cached ``RAGIndex`` objects
        (e.g. the probe-selector's ``exam_index_cache``) stay isolated. Graph
        indices are attached by the orchestrator after this method returns.
        """
        fingerprint = config.embeddings_fingerprint(corpus_hash)

        cached = self.cache.load(fingerprint) if self.cache else None
        if cached is not None:
            chunks, embeddings = cached
            logger.info("Cache hit %s: %d chunks, embed_dim=%d", fingerprint, len(chunks), embeddings.shape[-1])
        else:
            chunks, embeddings = await self._compute_chunks_and_embeddings(documents, config, embedding_token_limits)
            if self.cache:
                self.cache.store(fingerprint, chunks, embeddings)
                logger.info("Cached %s (%d chunks)", fingerprint, len(chunks))

        vector_store = self._build_vector_store(chunks, embeddings)

        return RAGIndex(
            vector_store=vector_store,
            chunks=chunks,
            embeddings=embeddings,
            index_type=config.index_type,
            graph_store=None,
        )

    async def _compute_chunks_and_embeddings(
        self,
        documents: list[str],
        config: StructuralConfig,
        embedding_token_limits: dict[str, int] | None,
    ) -> tuple[list[str], np.ndarray]:
        separators = self.SPLITTER_SEPARATORS.get(config.chunking_strategy)
        if separators is None:
            supported = ", ".join(sorted(self.SPLITTER_SEPARATORS))
            raise ValueError(f"Unsupported chunking_strategy '{config.chunking_strategy}'. Supported: {supported}")

        embedder = self.get_embedder(config.embedding_model)

        tokenizer = getattr(embedder, "tokenizer", None)
        if tokenizer is not None:
            # Temporarily raise model_max_length to suppress spurious tokenizer
            # warnings during splitting. Real token-limit enforcement is in
            # _enforce_token_limit() below.
            saved_max_length = tokenizer.model_max_length
            tokenizer.model_max_length = 10**7
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=config.chunk_token_size,
                chunk_overlap=config.chunk_token_overlap,
                separators=separators,
            )
        else:
            saved_max_length = None
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_token_size,
                chunk_overlap=config.chunk_token_overlap,
                separators=separators,
            )
        chunks = self._chunk_documents(documents, splitter)
        if tokenizer is not None:
            tokenizer.model_max_length = saved_max_length
        if not chunks:
            raise ValueError("No chunks were produced from the provided documents.")

        max_tokens = (embedding_token_limits or {}).get(config.embedding_model)
        if max_tokens and tokenizer is not None:
            chunks = self._enforce_token_limit(chunks, tokenizer, max_tokens, separators)

        logger.info("Embedding %d chunks with %s", len(chunks), config.embedding_model)
        embeddings = np.asarray(
            embedder.encode(chunks, show_progress_bar=True),
            dtype=np.float32,
        )
        return chunks, embeddings

    def _build_vector_store(self, chunks: list[str], embeddings: np.ndarray) -> LanceDBStore:
        """Build a fresh in-memory LanceDB table from the given chunks + vectors.

        Each call uses a unique ``memory://<uuid>`` URI so the resulting
        ``LanceDBStore`` owns an isolated backend — required because callers
        (orchestrator trial loop, probe selector, bench script) keep multiple
        ``RAGIndex`` handles alive and would otherwise clobber each other.
        """
        records = [
            {"id": f"chunk_{i}", "text": chunk, "vector": embedding.tolist()}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        logger.info("Creating LanceDB vector index (%d records)", len(records))
        vector_store = LanceDBStore(db_path=f"memory://{uuid.uuid4()}")
        vector_store.create_index(records, table_name=self.table_name, mode="overwrite")
        return vector_store

    @staticmethod
    def _chunk_documents(documents: list[str], splitter: Any) -> list[str]:
        def _chunk_one(document: str) -> list[str]:
            if not document.strip():
                return []
            return [c.strip() for c in splitter.split_text(document) if c.strip()]

        results: list[list[str]] = [[] for _ in documents]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(_chunk_one, doc): i for i, doc in enumerate(documents)}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(documents),
                desc="Chunking documents",
                unit="doc",
            ):
                results[future_to_idx[future]] = future.result()

        return [chunk for doc_chunks in results for chunk in doc_chunks]

    @staticmethod
    def _enforce_token_limit(
        chunks: list[str],
        tokenizer: Any,
        max_tokens: int,
        separators: list[str],
    ) -> list[str]:
        """Re-split chunks that exceed *max_tokens* for the embedding model.

        The recursive text splitter targets a token count but can overshoot
        when no suitable break point exists.  This post-pass catches those
        oversized chunks and splits them with a tighter target, preventing
        silent truncation during encoding.
        """
        resplit_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=max_tokens,
            chunk_overlap=min(max_tokens // 10, 32),
            separators=separators,
        )
        result: list[str] = []
        n_resplit = 0
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
            if token_count <= max_tokens:
                result.append(chunk)
            else:
                sub_chunks = resplit_splitter.split_text(chunk)
                result.extend(c.strip() for c in sub_chunks if c.strip())
                n_resplit += 1

        if n_resplit:
            logger.info(
                "Token limit enforcement: re-split %d oversized chunks (limit=%d), %d → %d total",
                n_resplit,
                max_tokens,
                len(chunks),
                len(result),
            )
        return result

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
        """Delete all cache entries not in *keep*, moving to CPU first to free GPU memory."""
        to_remove = [k for k in cache if k not in keep]
        if not to_remove:
            return
        for k in to_remove:
            model = cache.pop(k)
            if hasattr(model, "to"):
                with contextlib.suppress(Exception):
                    model.to("cpu")
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
