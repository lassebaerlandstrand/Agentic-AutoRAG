"""Tests for the Random + Optuna conditional samplers."""

from __future__ import annotations

import random

import optuna
import pytest

from agentic_autorag.baselines.sampler import (
    sample_trial_config_optuna,
    sample_trial_config_random,
)
from agentic_autorag.config.models import IndexType, ProjectConfig

# Suppress Optuna's default INFO log spam during tests.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_search_space_dict(*, with_graph: bool = False) -> dict:
    raw: dict = {
        "meta": {
            "project_name": "sampler-test",
            "corpus_path": "./fake",
            "output_dir": "./fake_out",
        },
        "search_space": {
            "chunking": {
                "strategies": ["recursive", "fixed"],
                "chunk_token_size": {"min": 128, "max": 1024},
                "chunk_token_overlap": {"min": 0, "max": 128},
            },
            "embedding_models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            "index_types": ["vector_only", "hybrid_bm25_vector"],
            "top_k": {"min": 3, "max": 20},
            "hybrid_alpha": {"min": 0.0, "max": 1.0},
            "reranker": {
                "models": ["none", "BAAI/bge-reranker-v2-m3"],
                "top_n": {"min": 3, "max": 10},
            },
            "query_expansion": ["none", "hyde"],
            "llm_models": ["ollama/llama3.2", "gemini/gemini-2.5-flash-lite"],
            "temperature": {"min": 0.0, "max": 1.0},
            "reasoning": False,
        },
        "examiner": {"exam_size": 5},
        "agent": {
            "optimizer_model": "test/model",
            "examiner_model": "test/model",
        },
    }
    if with_graph:
        raw["search_space"]["index_types"] = ["vector_only", "graph_only"]
        raw["search_space"]["graph_retrieval"] = {
            "graph_query_modes": ["local", "global", "hybrid"],
            "graph_top_k": {"min": 20, "max": 100},
        }
        raw["graph"] = {
            "extraction_model": "azure/gpt-4.1-nano",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    return raw


def _make_project_config(**kwargs) -> ProjectConfig:
    raw = _make_search_space_dict(**kwargs)
    return ProjectConfig.model_validate(raw)


class TestRandomSampler:
    """Random sampler honours all SearchSpace constraints."""

    def test_reproducible_under_fixed_seed(self) -> None:
        """Same seed → same sequence of TrialConfigs."""
        cfg = _make_project_config()
        ss = cfg.search_space
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        seq_a = [sample_trial_config_random(rng_a, ss).model_dump() for _ in range(20)]
        seq_b = [sample_trial_config_random(rng_b, ss).model_dump() for _ in range(20)]
        assert seq_a == seq_b

    def test_different_seeds_diverge(self) -> None:
        cfg = _make_project_config()
        ss = cfg.search_space
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        seq_a = [sample_trial_config_random(rng_a, ss).model_dump() for _ in range(20)]
        seq_b = [sample_trial_config_random(rng_b, ss).model_dump() for _ in range(20)]
        assert seq_a != seq_b

    def test_all_samples_pass_validate_trial(self) -> None:
        """100 random samples all pass ProjectConfig.validate_trial."""
        cfg = _make_project_config()
        rng = random.Random(0)
        for _ in range(100):
            trial = sample_trial_config_random(rng, cfg.search_space)
            violations = cfg.validate_trial(trial)
            assert violations == [], f"violations: {violations} for trial {trial}"

    def test_overlap_strictly_less_than_chunk_size(self) -> None:
        """Hard invariant from TrialConfig validator."""
        cfg = _make_project_config()
        rng = random.Random(0)
        for _ in range(100):
            trial = sample_trial_config_random(rng, cfg.search_space)
            assert trial.chunk_token_overlap < trial.chunk_token_size

    def test_reranker_top_n_capped_at_top_k_when_reranker_active(self) -> None:
        cfg = _make_project_config()
        rng = random.Random(0)
        for _ in range(100):
            trial = sample_trial_config_random(rng, cfg.search_space)
            if trial.reranker != "none":
                assert trial.reranker_top_n <= trial.top_k

    def test_hybrid_alpha_random_only_when_hybrid(self) -> None:
        """For non-hybrid configs, hybrid_alpha must be a deterministic default."""
        cfg = _make_project_config()
        rng = random.Random(0)
        non_hybrid_alphas = set()
        for _ in range(100):
            trial = sample_trial_config_random(rng, cfg.search_space)
            if trial.index_type != IndexType.HYBRID_BM25_VECTOR:
                non_hybrid_alphas.add(trial.hybrid_alpha)
        # When non-hybrid, every sample uses the midpoint of the range — so a
        # single deterministic value across all such trials.
        assert len(non_hybrid_alphas) == 1, f"non-hybrid hybrid_alpha varied across trials: {non_hybrid_alphas}"

    def test_with_graph_search_space(self) -> None:
        """Graph dimensions are sampled iff index_type is graph-based."""
        cfg = _make_project_config(with_graph=True)
        rng = random.Random(0)
        for _ in range(50):
            trial = sample_trial_config_random(rng, cfg.search_space)
            violations = cfg.validate_trial(trial)
            assert violations == [], f"violations: {violations}"
            if trial.index_type.value in {"graph_only", "hybrid_graph_vector"}:
                assert trial.graph_query_mode in cfg.search_space.graph_retrieval.graph_query_modes
                assert (
                    cfg.search_space.graph_retrieval.graph_top_k.min
                    <= trial.graph_top_k
                    <= cfg.search_space.graph_retrieval.graph_top_k.max
                )

    def test_respects_embedding_token_limits(self) -> None:
        """When chunk_token_size exceeds the embedding cap, the embedding is filtered."""
        cfg = _make_project_config()
        # MiniLM caps at 256 tokens; force a chunk size that excludes it.
        cfg.embedding_token_limits = {"sentence-transformers/all-MiniLM-L6-v2": 256}
        rng = random.Random(0)
        for _ in range(50):
            trial = sample_trial_config_random(rng, cfg.search_space, cfg.embedding_token_limits)
            violations = cfg.validate_trial(trial)
            assert violations == [], f"violations: {violations}"
            if trial.embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
                assert trial.chunk_token_size <= 256


class TestOptunaSampler:
    """Optuna define-by-run sampler produces valid configs."""

    def _objective_factory(self, cfg: ProjectConfig):
        results = []

        def objective(trial: optuna.Trial) -> float:
            config = sample_trial_config_optuna(trial, cfg.search_space, cfg.embedding_token_limits)
            violations = cfg.validate_trial(config)
            assert violations == [], f"violations: {violations}"
            results.append((trial.params, config))
            # Return a synthetic deterministic score so TPE has something to learn from.
            return float(config.top_k) / 10.0

        return objective, results

    def test_all_optuna_trials_pass_validate(self) -> None:
        cfg = _make_project_config()
        objective, results = self._objective_factory(cfg)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=30)
        assert len(results) == 30

    def test_optuna_omits_hybrid_alpha_when_not_hybrid(self) -> None:
        """define-by-run: hybrid_alpha must NOT be in trial.params when index_type is non-hybrid."""
        cfg = _make_project_config()
        objective, results = self._objective_factory(cfg)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        for params, config in results:
            if config.index_type != IndexType.HYBRID_BM25_VECTOR:
                assert "hybrid_alpha" not in params

    def test_optuna_omits_reranker_top_n_when_reranker_none(self) -> None:
        cfg = _make_project_config()
        objective, results = self._objective_factory(cfg)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        for params, config in results:
            if config.reranker == "none":
                assert "reranker_top_n" not in params

    def test_optuna_with_graph_search_space(self) -> None:
        cfg = _make_project_config(with_graph=True)
        objective, results = self._objective_factory(cfg)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        graph_seen = sum(1 for _, c in results if c.index_type.value in {"graph_only", "hybrid_graph_vector"})
        # With ~50% prior probability and 20 trials we expect at least one graph sample.
        assert graph_seen > 0

    def test_optuna_reproducible_with_seed(self) -> None:
        cfg = _make_project_config()

        def run_one() -> list[dict]:
            objective, results = self._objective_factory(cfg)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
            study.optimize(objective, n_trials=10)
            return [c.model_dump() for _, c in results]

        run_a = run_one()
        run_b = run_one()
        assert run_a == run_b


class TestEdgeCases:
    def test_ollama_model_forbids_reasoning(self) -> None:
        """Ollama prefix is blocklisted in is_reasoning_allowed → reasoning must be False."""
        raw = _make_search_space_dict()
        raw["search_space"]["llm_models"] = ["ollama/llama3.2"]
        raw["search_space"]["reasoning"] = True
        cfg = ProjectConfig.model_validate(raw)
        rng = random.Random(0)
        for _ in range(20):
            trial = sample_trial_config_random(rng, cfg.search_space)
            assert trial.reasoning is False

    def test_pruned_when_no_embedding_supports_chunk_size(self) -> None:
        """Optuna sampler raises TrialPruned when limits exclude all embeddings."""
        raw = _make_search_space_dict()
        raw["search_space"]["chunking"]["chunk_token_size"]["min"] = 9000
        raw["search_space"]["chunking"]["chunk_token_size"]["max"] = 9999
        cfg = ProjectConfig.model_validate(raw)
        # Both embeddings cap below 9000.
        cfg.embedding_token_limits = {
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "sentence-transformers/all-mpnet-base-v2": 384,
        }

        prune_count = 0
        ok_count = 0

        def objective(trial: optuna.Trial) -> float:
            nonlocal prune_count, ok_count
            try:
                cfg_out = sample_trial_config_optuna(trial, cfg.search_space, cfg.embedding_token_limits)
            except optuna.TrialPruned:
                prune_count += 1
                raise
            ok_count += 1
            return float(cfg_out.top_k)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=10, catch=(optuna.TrialPruned,))
        # All trials should prune since no embedding fits.
        assert prune_count == 10 and ok_count == 0


@pytest.mark.parametrize("seed", [0, 1, 42, 999])
def test_random_sampler_smoke_across_seeds(seed: int) -> None:
    """Sanity check: sampler runs for several seeds without crashing."""
    cfg = _make_project_config()
    rng = random.Random(seed)
    for _ in range(50):
        trial = sample_trial_config_random(rng, cfg.search_space)
        assert cfg.validate_trial(trial) == []
