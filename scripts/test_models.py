"""Validate that all embedding models and rerankers from configs/full.yaml can be loaded."""

import sys
import time
from pathlib import Path

import torch
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "full.yaml"
TEST_SENTENCES = ["This is a test sentence.", "Another test sentence for validation."]
TEST_PAIRS = [("What is machine learning?", "Machine learning is a subset of AI.")]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_embedding_model(model_name: str) -> bool:
    start = time.time()
    try:
        model_kwargs = {"dtype": torch.float16} if torch.cuda.is_available() else {}
        model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        embeddings = model.encode(TEST_SENTENCES)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        elapsed = time.time() - start
        print(f"  OK  {model_name} (dim={embeddings.shape[1]}, {elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL {model_name} ({elapsed:.1f}s): {e}")
        return False


def test_reranker_model(model_name: str) -> bool:
    start = time.time()
    try:
        model = CrossEncoder(model_name)
        scores = model.predict(TEST_PAIRS)
        elapsed = time.time() - start
        print(f"  OK  {model_name} (score={float(scores[0]):.4f}, {elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL {model_name} ({elapsed:.1f}s): {e}")
        return False


def main() -> None:
    config = load_config()
    search_space = config["search_space"]

    embedding_models = search_space.get("embedding_models", [])
    reranker_models = [m for m in search_space.get("reranker", {}).get("models", []) if m != "none"]

    print(f"\n{'=' * 60}")
    print(f"Embedding models ({len(embedding_models)})")
    print(f"{'=' * 60}")
    embed_results = []
    for model_name in embedding_models:
        embed_results.append(test_embedding_model(model_name))

    print(f"\n{'=' * 60}")
    print(f"Reranker models ({len(reranker_models)})")
    print(f"{'=' * 60}")
    rerank_results = []
    for model_name in reranker_models:
        rerank_results.append(test_reranker_model(model_name))

    # Summary
    embed_ok = sum(embed_results)
    rerank_ok = sum(rerank_results)
    total = len(embed_results) + len(rerank_results)
    total_ok = embed_ok + rerank_ok

    print(f"\n{'=' * 60}")
    print(f"Summary: {total_ok}/{total} passed")
    print(f"  Embeddings: {embed_ok}/{len(embed_results)}")
    print(f"  Rerankers:  {rerank_ok}/{len(rerank_results)}")
    print(f"{'=' * 60}")

    if total_ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
