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
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from agentic_autorag.config.models import IndexType, StructuralConfig
from agentic_autorag.engine._io import atomic_write_text
from agentic_autorag.engine.vector_store import LanceDBStore

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64

# HuggingFace tokenizers warn when a sequence exceeds ``model_max_length``,
# but we only use the tokenizer to compute offset boundaries — the tokens
# never reach the model. Large enough to disable the cap for any realistic
# document, small enough to avoid int32 overflow.
_TOKENIZER_NO_TRUNCATE_SENTINEL = 10**9


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
    chunk_doc_ids: list[str] | None = None  # parallel to chunks; source document id per chunk
    chunk_char_ranges: list[tuple[int, int]] | None = None  # parallel to chunks; (start, end) in source doc

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
    """Persistent two-layer LRU cache for chunks and embeddings.

    Chunks and embeddings are stored as separate entries so different embedding
    models can reuse the same chunks when chunking params match. Layout::

        root/
          manifest.json
          chunks/<chunks_fp>/chunks.json
          embeddings/<emb_fp>/embeddings.npy

    The manifest records ``chunks_fp`` on every embeddings entry, so eviction
    can enforce the dependency: an embeddings entry is useless without its
    chunks. Eviction walks entries oldest-first and preferentially drops
    embeddings; a chunks entry is only evicted when no surviving embeddings
    entry still names it as its ``chunks_fp``. All writes are atomic via
    tempfile + ``os.replace``. LRU eviction runs after every store.
    """

    _CHUNKS = "chunks"
    _EMBEDDINGS = "embeddings"

    def __init__(self, cache_dir: str | Path, max_bytes: int) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self._CHUNKS).mkdir(exist_ok=True)
        (self.root / self._EMBEDDINGS).mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.max_bytes = max_bytes
        self.manifest: dict[str, dict] = self._load_manifest()

    def has_chunks(self, chunks_fp: str) -> bool:
        key = self._chunks_key(chunks_fp)
        return key in self.manifest and self._chunks_path(chunks_fp).exists()

    def has_embeddings(self, emb_fp: str) -> bool:
        key = self._embeddings_key(emb_fp)
        return key in self.manifest and self._embeddings_path(emb_fp).exists()

    def load_chunks(self, chunks_fp: str) -> tuple[list[str], list[int], list[tuple[int, int]]] | None:
        """Load cached chunks, their source document indices, and character ranges.

        Returns None on miss. On hit returns ``(chunks, doc_indices, char_ranges)``.
        """
        key = self._chunks_key(chunks_fp)
        path = self._chunks_path(chunks_fp)
        if key not in self.manifest or not path.exists():
            if key in self.manifest:
                del self.manifest[key]
                self._save_manifest()
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = raw["chunks"]
        doc_indices = raw.get("doc_indices", [-1] * len(chunks))
        char_ranges = [tuple(r) for r in raw["char_ranges"]]
        self.manifest[key]["last_accessed"] = _now_iso()
        self._save_manifest()
        return chunks, doc_indices, char_ranges

    def load_embeddings(self, emb_fp: str) -> np.ndarray | None:
        key = self._embeddings_key(emb_fp)
        path = self._embeddings_path(emb_fp)
        if key not in self.manifest or not path.exists():
            if key in self.manifest:
                del self.manifest[key]
                self._save_manifest()
            return None
        embeddings = np.load(path)
        self.manifest[key]["last_accessed"] = _now_iso()
        self._save_manifest()
        return embeddings

    def store_chunks(
        self,
        chunks_fp: str,
        chunks: list[str],
        doc_indices: list[int],
        char_ranges: list[tuple[int, int]],
    ) -> None:
        entry_dir = self._chunks_path(chunks_fp).parent
        entry_dir.mkdir(parents=True, exist_ok=True)
        path = self._chunks_path(chunks_fp)
        payload = {"chunks": chunks, "doc_indices": doc_indices, "char_ranges": char_ranges}
        atomic_write_text(path, json.dumps(payload))
        self.manifest[self._chunks_key(chunks_fp)] = {
            "size_bytes": path.stat().st_size,
            "last_accessed": _now_iso(),
        }
        self._evict_if_over_budget(protect_chunks={chunks_fp}, protect_embeddings=set())
        self._save_manifest()

    def store_embeddings(self, emb_fp: str, chunks_fp: str, embeddings: np.ndarray) -> None:
        entry_dir = self._embeddings_path(emb_fp).parent
        entry_dir.mkdir(parents=True, exist_ok=True)
        path = self._embeddings_path(emb_fp)
        _atomic_write_npy(path, embeddings)
        self.manifest[self._embeddings_key(emb_fp)] = {
            "size_bytes": path.stat().st_size,
            "last_accessed": _now_iso(),
            "chunks_fp": chunks_fp,
        }
        self._evict_if_over_budget(protect_chunks={chunks_fp}, protect_embeddings={emb_fp})
        self._save_manifest()

    def _evict_if_over_budget(self, protect_chunks: set[str], protect_embeddings: set[str]) -> None:
        total = sum(entry["size_bytes"] for entry in self.manifest.values())
        if total <= self.max_bytes:
            return

        # Walk entries oldest-first, evict embeddings freely and chunks only when
        # no surviving embeddings entry still references them.
        candidates = sorted(self.manifest.items(), key=lambda item: item[1]["last_accessed"])
        for key, entry in candidates:
            if total <= self.max_bytes:
                break
            kind, fp = key.split("/", 1)
            if kind == self._EMBEDDINGS:
                if fp in protect_embeddings:
                    continue
                self._delete_embeddings(fp)
                total -= entry["size_bytes"]
            elif kind == self._CHUNKS:
                if fp in protect_chunks or self._has_live_referrer(fp):
                    continue
                self._delete_chunks(fp)
                total -= entry["size_bytes"]

    def _has_live_referrer(self, chunks_fp: str) -> bool:
        for key, entry in self.manifest.items():
            if key.startswith(f"{self._EMBEDDINGS}/") and entry.get("chunks_fp") == chunks_fp:
                return True
        return False

    def _delete_chunks(self, chunks_fp: str) -> None:
        shutil.rmtree(self._chunks_path(chunks_fp).parent, ignore_errors=True)
        self.manifest.pop(self._chunks_key(chunks_fp), None)
        logger.info("Evicted chunks entry %s", chunks_fp)

    def _delete_embeddings(self, emb_fp: str) -> None:
        shutil.rmtree(self._embeddings_path(emb_fp).parent, ignore_errors=True)
        self.manifest.pop(self._embeddings_key(emb_fp), None)
        logger.info("Evicted embeddings entry %s", emb_fp)

    def _chunks_key(self, chunks_fp: str) -> str:
        return f"{self._CHUNKS}/{chunks_fp}"

    def _embeddings_key(self, emb_fp: str) -> str:
        return f"{self._EMBEDDINGS}/{emb_fp}"

    def _chunks_path(self, chunks_fp: str) -> Path:
        return self.root / self._CHUNKS / chunks_fp / "chunks.json"

    def _embeddings_path(self, emb_fp: str) -> Path:
        return self.root / self._EMBEDDINGS / emb_fp / "embeddings.npy"

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
        atomic_write_text(self.manifest_path, json.dumps(self.manifest, indent=2))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _resolve_chunk_doc_ids(
    doc_indices: list[int] | None,
    doc_ids: list[str] | None,
    n_chunks: int,
) -> list[str]:
    """Map per-chunk source document indices back to human-readable doc IDs.

    Legacy caches may carry ``doc_indices`` of ``[-1] * n_chunks`` (no provenance
    recorded at chunk time). When doc_ids is missing or an index is out of range,
    fall back to an empty string so the diagnostic signal degrades gracefully.
    """
    if doc_indices is None or doc_ids is None:
        return [""] * n_chunks
    resolved: list[str] = []
    for idx in doc_indices:
        if 0 <= idx < len(doc_ids):
            resolved.append(doc_ids[idx])
        else:
            resolved.append("")
    return resolved


def _chunk_docs_by_tokens(
    documents: list[str],
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Parallel chunking: tokenize each document once, then split on token boundaries.

    Returns ``(chunks, chunk_doc_indices, char_ranges)`` where ``chunks[i]`` is
    the stripped chunk text, ``chunk_doc_indices[i]`` is the 0-based source
    document index, and ``char_ranges[i] == (start, end)`` satisfies
    ``documents[chunk_doc_indices[i]][start:end] == chunks[i]``.
    """

    def _one(doc: str) -> list[tuple[str, int, int]]:
        return _split_one_doc(doc, tokenizer, chunk_size, chunk_overlap, separators)

    # We only use the tokenizer to locate token-level offset boundaries for
    # chunking — the tokens themselves are never fed to the model, so the
    # "sequence longer than max_seq_length" warning HF emits for long docs is
    # just noise. Temporarily disable the cap so the warning stays silent.
    prev_max_len = getattr(tokenizer, "model_max_length", None)
    if prev_max_len is not None:
        tokenizer.model_max_length = _TOKENIZER_NO_TRUNCATE_SENTINEL
    try:
        results: list[list[tuple[str, int, int]]] = [[] for _ in documents]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(_one, doc): i for i, doc in enumerate(documents)}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(documents),
                desc="Chunking documents",
                unit="doc",
            ):
                results[future_to_idx[future]] = future.result()
    finally:
        if prev_max_len is not None:
            tokenizer.model_max_length = prev_max_len

    chunks: list[str] = []
    doc_indices: list[int] = []
    char_ranges: list[tuple[int, int]] = []
    for doc_idx, doc_chunks in enumerate(results):
        for text, cs, ce in doc_chunks:
            chunks.append(text)
            doc_indices.append(doc_idx)
            char_ranges.append((cs, ce))
    return chunks, doc_indices, char_ranges


def _split_one_doc(
    doc: str,
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[tuple[str, int, int]]:
    """Split a document into chunks of ≤chunk_size tokens, carrying char ranges.

    Each returned tuple is ``(stripped_text, start, end)`` with
    ``doc[start:end] == stripped_text``. Callers rely on this invariant for
    offset-based chunk-relevance scoring.
    """
    if not doc.strip():
        return []
    enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = list(enc["offset_mapping"])
    if not offsets:
        return []
    if len(offsets) <= chunk_size:
        emitted = _strip_with_range(doc, 0, len(doc))
        return [emitted] if emitted is not None else []

    token_starts = [cs for cs, _ in offsets]

    def tok_at_or_after(char_pos: int) -> int:
        return bisect_left(token_starts, char_pos)

    def recurse(char_start: int, char_end: int, seps: list[str]) -> list[tuple[str, int, int]]:
        tok_lo = tok_at_or_after(char_start)
        tok_hi = tok_at_or_after(char_end)
        if tok_hi - tok_lo <= chunk_size:
            emitted = _strip_with_range(doc, char_start, char_end)
            return [emitted] if emitted is not None else []

        sep_idx, sep = _pick_separator(doc, char_start, char_end, seps)
        next_seps = seps[sep_idx + 1 :] if sep_idx + 1 < len(seps) else [""]

        if sep == "":
            return _hard_split_by_tokens(doc, offsets, tok_lo, tok_hi, chunk_size, chunk_overlap)

        pieces = _split_char_range_on_separator(doc, char_start, char_end, sep)
        pieces = [(ps, pe) for (ps, pe) in pieces if tok_at_or_after(pe) - tok_at_or_after(ps) > 0]
        return _merge_pieces_with_overlap(
            pieces=pieces,
            tok_at_or_after=tok_at_or_after,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recurse=lambda cs, ce: recurse(cs, ce, next_seps),
            emit=lambda cs, ce: _strip_with_range(doc, cs, ce),
        )

    return recurse(0, len(doc), separators)


def _pick_separator(doc: str, char_start: int, char_end: int, seps: list[str]) -> tuple[int, str]:
    for i, s in enumerate(seps):
        if s == "":
            return i, s
        if doc.find(s, char_start, char_end) != -1:
            return i, s
    return len(seps) - 1, seps[-1]


def _split_char_range_on_separator(doc: str, char_start: int, char_end: int, sep: str) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    i = char_start
    while i < char_end:
        j = doc.find(sep, i, char_end)
        if j == -1:
            if i < char_end:
                pieces.append((i, char_end))
            break
        if j > i:
            pieces.append((i, j))
        i = j + len(sep)
    return pieces


def _hard_split_by_tokens(
    doc: str,
    offsets: list[tuple[int, int]],
    tok_lo: int,
    tok_hi: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    pos = tok_lo
    stride = max(1, chunk_size - chunk_overlap)
    while pos < tok_hi:
        end = min(pos + chunk_size, tok_hi)
        emitted = _strip_with_range(doc, offsets[pos][0], offsets[end - 1][1])
        if emitted is not None:
            out.append(emitted)
        if end >= tok_hi:
            break
        pos += stride
    return out


def _merge_pieces_with_overlap(
    pieces: list[tuple[int, int]],
    tok_at_or_after,
    chunk_size: int,
    chunk_overlap: int,
    recurse,
    emit,
) -> list[tuple[str, int, int]]:
    """Greedy-merge separator pieces into ≤chunk_size groups, preserving overlap.

    Follows LangChain's _merge_splits semantics: when a group would overflow, pop
    pieces from the front until the remainder is ≤chunk_overlap tokens, then use
    those remaining pieces as the seed for the next group.
    """
    out: list[tuple[str, int, int]] = []
    group: list[tuple[int, int, int]] = []  # (char_start, char_end, token_count)
    group_tok = 0

    def flush() -> None:
        if not group:
            return
        emitted = emit(group[0][0], group[-1][1])
        if emitted is not None:
            out.append(emitted)

    for ps, pe in pieces:
        p_tok = tok_at_or_after(pe) - tok_at_or_after(ps)
        if p_tok > chunk_size:
            flush()
            group.clear()
            group_tok = 0
            out.extend(recurse(ps, pe))
            continue
        if group_tok + p_tok > chunk_size and group:
            flush()
            # Retain trailing pieces up to chunk_overlap tokens to seed the next group.
            while group and group_tok > chunk_overlap:
                group_tok -= group[0][2]
                group.pop(0)
        group.append((ps, pe, p_tok))
        group_tok += p_tok
    flush()
    return out


def _strip_with_range(doc: str, char_start: int, char_end: int) -> tuple[str, int, int] | None:
    """Strip whitespace from doc[char_start:char_end] and return the tight range.

    The invariant ``doc[new_start:new_end] == stripped_text`` is preserved; this
    matters for offset-based chunk-relevance scoring. Returns None when the slice
    is empty after stripping.
    """
    raw = doc[char_start:char_end]
    stripped = raw.strip()
    if not stripped:
        return None
    lstrip = len(raw) - len(raw.lstrip())
    rstrip = len(raw) - len(raw.rstrip())
    return stripped, char_start + lstrip, char_end - rstrip


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
        doc_ids: list[str] | None = None,
    ) -> RAGIndex:
        """Build a vector retrieval index from parsed documents.

        Three-way cache path keyed by ``config.chunks_fingerprint(corpus_hash)``
        and ``config.embeddings_fingerprint(corpus_hash)``: both hit → load
        both; chunks hit, embeddings miss → re-embed only; both miss → chunk +
        embed. Each call produces a fresh in-memory LanceDB table. Graph
        indices are attached by the orchestrator after this method returns.

        ``doc_ids`` is a parallel list of source document identifiers matching
        ``documents``. When provided, each retrieved chunk carries its source
        doc_id in ``RetrievedDocument.metadata["doc_id"]``, which the diagnoser
        uses to detect cross-document retrieval.
        """
        chunks_fp = config.chunks_fingerprint(corpus_hash)
        emb_fp = config.embeddings_fingerprint(corpus_hash)

        loaded = self.cache.load_chunks(chunks_fp) if self.cache else None
        if loaded is None:
            chunks, doc_indices, char_ranges = None, None, None
        else:
            chunks, doc_indices, char_ranges = loaded
        embeddings = self.cache.load_embeddings(emb_fp) if self.cache else None

        if chunks is None:
            chunks, doc_indices, char_ranges, embeddings = await self._compute_chunks_and_embeddings(documents, config)
            if self.cache:
                self.cache.store_chunks(chunks_fp, chunks, doc_indices, char_ranges)
                self.cache.store_embeddings(emb_fp, chunks_fp, embeddings)
            logger.info("Built %s (%d chunks)", emb_fp, len(chunks))
        elif embeddings is None:
            logger.info(
                "Chunks cache hit %s (%d chunks); re-embedding with %s", chunks_fp, len(chunks), config.embedding_model
            )
            embeddings = _encode_chunks(self.get_embedder(config.embedding_model), chunks)
            if self.cache:
                self.cache.store_embeddings(emb_fp, chunks_fp, embeddings)
        else:
            logger.info("Cache hit %s: %d chunks, embed_dim=%d", emb_fp, len(chunks), embeddings.shape[-1])

        chunk_doc_ids = _resolve_chunk_doc_ids(doc_indices, doc_ids, n_chunks=len(chunks))
        vector_store = self._build_vector_store(chunks, embeddings, chunk_doc_ids, char_ranges)

        return RAGIndex(
            vector_store=vector_store,
            chunks=chunks,
            embeddings=embeddings,
            index_type=config.index_type,
            graph_store=None,
            chunk_doc_ids=chunk_doc_ids,
            chunk_char_ranges=char_ranges,
        )

    async def _compute_chunks_and_embeddings(
        self,
        documents: list[str],
        config: StructuralConfig,
    ) -> tuple[list[str], list[int], list[tuple[int, int]], np.ndarray]:
        separators = self.SPLITTER_SEPARATORS.get(config.chunking_strategy)
        if separators is None:
            supported = ", ".join(sorted(self.SPLITTER_SEPARATORS))
            raise ValueError(f"Unsupported chunking_strategy '{config.chunking_strategy}'. Supported: {supported}")

        embedder = self.get_embedder(config.embedding_model)
        tokenizer = getattr(embedder, "tokenizer", None)
        if tokenizer is None or not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                f"Embedding model '{config.embedding_model}' lacks a fast HuggingFace tokenizer; "
                "offset-mapping chunking requires one."
            )

        chunks, doc_indices, char_ranges = _chunk_docs_by_tokens(
            documents,
            tokenizer,
            chunk_size=config.chunk_token_size,
            chunk_overlap=config.chunk_token_overlap,
            separators=separators,
        )
        if not chunks:
            raise ValueError("No chunks were produced from the provided documents.")

        logger.info("Embedding %d chunks with %s", len(chunks), config.embedding_model)
        embeddings = _encode_chunks(embedder, chunks)
        return chunks, doc_indices, char_ranges, embeddings

    def _build_vector_store(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        chunk_doc_ids: list[str],
        char_ranges: list[tuple[int, int]],
    ) -> LanceDBStore:
        """Build a fresh in-memory LanceDB table from the given chunks + vectors.

        Each call uses a unique ``memory://<uuid>`` URI so the resulting
        ``LanceDBStore`` owns an isolated backend — required because callers
        (orchestrator trial loop, probe selector, bench script) keep multiple
        ``RAGIndex`` handles alive and would otherwise clobber each other.

        ``char_ranges[i]`` carries the (start, end) offset of ``chunks[i]`` in
        its source document text, persisted as two int columns so the evaluator
        can compute deterministic interval-overlap chunk relevance.
        """
        records = [
            {
                "id": f"chunk_{i}",
                "text": chunk,
                "vector": embedding.tolist(),
                "doc_id": doc_id,
                "chunk_index": i,
                "char_start": int(char_range[0]),
                "char_end": int(char_range[1]),
            }
            for i, (chunk, embedding, doc_id, char_range) in enumerate(
                zip(chunks, embeddings, chunk_doc_ids, char_ranges, strict=True)
            )
        ]
        logger.info("Creating LanceDB vector index (%d records)", len(records))
        vector_store = LanceDBStore(db_path=f"memory://{uuid.uuid4()}")
        vector_store.create_index(records, table_name=self.table_name, mode="overwrite")
        return vector_store

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
            ce = CrossEncoder(model_name)
            _ensure_pad_token(ce)
            self._cross_encoder_cache[model_name] = ce
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _encode_chunks(embedder: Any, chunks: list[str]) -> np.ndarray:
    return np.asarray(
        embedder.encode(chunks, show_progress_bar=True, batch_size=EMBED_BATCH_SIZE),
        dtype=np.float32,
    )


def _ensure_pad_token(ce: CrossEncoder) -> None:
    tok = getattr(ce, "tokenizer", None)
    if tok is None:
        return
    if getattr(tok, "pad_token", None) is None:
        eos = getattr(tok, "eos_token", None)
        if eos is not None:
            tok.pad_token = eos
    model = getattr(ce, "model", None)
    if model is not None and hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            model.config.pad_token_id = pad_id
