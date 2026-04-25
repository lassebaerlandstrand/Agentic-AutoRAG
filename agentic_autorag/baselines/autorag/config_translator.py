"""Translate AutoRAG's winning pipeline back into our flat ``TrialConfig``.

After ``autorag evaluate`` finishes, the winning pipeline is materialised as
``extracted_sample.yaml`` (one resolved module per node, no remaining choices).
Because ``native_config.py`` constrained AutoRAG's search space to exactly our
dimensions, every winning module corresponds 1:1 to a ``TrialConfig`` field —
this translator is a structured field extraction, not a lossy mapping.

Any genuinely missing fields fall back to the search-space lower bound
(deterministic + always valid). The set of expected exclusions
(``passage_compressor`` etc.) is recorded in the input ``translation_notes``
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_autorag.config.models import IndexType, NumericRange, SearchSpace, TrialConfig


def _midpoint(r: NumericRange) -> float:
    return (r.min + r.max) / 2.0


def _walk_nodes(extracted: dict) -> dict[str, dict]:
    """Flatten ``node_lines[].nodes[]`` into ``{node_type: node_dict}``."""
    nodes: dict[str, dict] = {}
    for line in extracted.get("node_lines", []) or []:
        for node in line.get("nodes", []) or []:
            ntype = node.get("node_type")
            if ntype:
                nodes[ntype] = node
    return nodes


def _winning_module(node: dict) -> dict:
    """Return the (single) module from a resolved ``extracted_sample`` node."""
    modules = node.get("modules", []) or []
    if not modules:
        return {}
    # extracted_sample has one module per node, but tolerate older formats.
    return modules[0]


def translate_extracted_to_trial_config(
    extracted_yaml_path: Path | str,
    search_space: SearchSpace,
) -> TrialConfig:
    """Parse ``extracted_sample.yaml`` and produce a valid ``TrialConfig``.

    Falls back to the search-space minimum for any unrepresented field.
    """
    raw = yaml.safe_load(Path(extracted_yaml_path).read_text(encoding="utf-8"))
    nodes = _walk_nodes(raw)

    fields: dict = {
        "chunking_strategy": search_space.chunking.strategies[0],
        "chunk_token_size": int(search_space.chunking.chunk_token_size.min),
        "chunk_token_overlap": int(search_space.chunking.chunk_token_overlap.min),
        "embedding_model": search_space.embedding_models[0],
        "index_type": search_space.index_types[0],
        "top_k": int(search_space.top_k.min),
        "hybrid_alpha": round(_midpoint(search_space.hybrid_alpha), 4),
        "reranker": "none" if "none" in search_space.reranker.models else search_space.reranker.models[0],
        "reranker_top_n": int(search_space.reranker.top_n.min),
        "query_expansion": "none" if "none" in search_space.query_expansion else search_space.query_expansion[0],
        "llm_model": search_space.llm_models[0],
        "temperature": float(search_space.temperature.min),
        "reasoning": False,
    }

    # Chunker
    chunker_node = nodes.get("chunker") or nodes.get("chunking")
    if chunker_node:
        m = _winning_module(chunker_node)
        cm = m.get("chunk_method")
        if cm:
            # AutoRAG yaml stores `chunk_method` as a list (search) or scalar (resolved).
            cm_val = cm[0] if isinstance(cm, list) else cm
            if cm_val == "token":
                fields["chunking_strategy"] = "recursive"
            elif cm_val in search_space.chunking.strategies:
                fields["chunking_strategy"] = cm_val
        if "chunk_size" in m:
            cs = m["chunk_size"]
            fields["chunk_token_size"] = int(cs[0] if isinstance(cs, list) else cs)
        if "chunk_overlap" in m:
            co = m["chunk_overlap"]
            fields["chunk_token_overlap"] = int(co[0] if isinstance(co, list) else co)
        # Enforce overlap < size invariant from TrialConfig validator.
        if fields["chunk_token_overlap"] >= fields["chunk_token_size"]:
            fields["chunk_token_overlap"] = max(0, fields["chunk_token_size"] - 1)

    # Retrieval
    retrieve_node = nodes.get("retrieval") or nodes.get("retrieve")
    if retrieve_node:
        m = _winning_module(retrieve_node)
        mtype = m.get("module_type", "")
        if mtype == "vectordb":
            fields["index_type"] = IndexType.VECTOR_ONLY
            embed = m.get("embedding_model")
            if embed:
                embed_val = embed[0] if isinstance(embed, list) else embed
                if embed_val in search_space.embedding_models:
                    fields["embedding_model"] = embed_val
        elif mtype in {"hybrid_cc", "hybrid_rrf", "bm25"}:
            fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
            weight = m.get("weight")
            if weight is not None:
                # AutoRAG weight is BM25's; we store vector's complement.
                w_val = weight[0] if isinstance(weight, list) else weight
                fields["hybrid_alpha"] = round(max(0.0, min(1.0, 1.0 - float(w_val))), 4)
        if "top_k" in m:
            tk = m["top_k"]
            fields["top_k"] = int(tk[0] if isinstance(tk, list) else tk)
        else:
            # Fallback: top_k is on the strategy block at the node level.
            strat = retrieve_node.get("strategy", {}) or {}
            tk = strat.get("top_k")
            if tk:
                fields["top_k"] = int(tk[0] if isinstance(tk, list) else tk)

    # Reranker
    rer_node = nodes.get("passage_reranker")
    if rer_node:
        m = _winning_module(rer_node)
        mtype = m.get("module_type", "")
        if mtype == "pass_passage_reranker":
            fields["reranker"] = "none"
        else:
            model_name = m.get("model_name") or m.get("model")
            if model_name and model_name in search_space.reranker.models:
                fields["reranker"] = model_name
            elif "none" in search_space.reranker.models:
                fields["reranker"] = "none"
        if "top_k" in m and fields["reranker"] != "none":
            tk = m["top_k"]
            fields["reranker_top_n"] = int(tk[0] if isinstance(tk, list) else tk)

    # Query expansion
    qe_node = nodes.get("query_expansion")
    if qe_node:
        m = _winning_module(qe_node)
        mtype = m.get("module_type", "")
        if mtype == "pass_query_expansion":
            fields["query_expansion"] = "none"
        elif mtype == "hyde" and "hyde" in search_space.query_expansion:
            fields["query_expansion"] = "hyde"
        elif mtype == "multi_query_expansion" and "multi_query" in search_space.query_expansion:
            fields["query_expansion"] = "multi_query"
        elif mtype in search_space.query_expansion:
            fields["query_expansion"] = mtype

    # Generator
    gen_node = nodes.get("generator")
    if gen_node:
        m = _winning_module(gen_node)
        llm = m.get("llm") or m.get("model")
        if llm and llm in search_space.llm_models:
            fields["llm_model"] = llm
        if "temperature" in m:
            t = m["temperature"]
            fields["temperature"] = float(t[0] if isinstance(t, list) else t)

    # Final invariants
    if fields["reranker"] != "none" and fields["reranker_top_n"] > fields["top_k"]:
        fields["reranker_top_n"] = fields["top_k"]

    return TrialConfig.model_validate(fields)
