"""Tests for examiner clustering and allocation logic."""

import numpy as np

from agentic_autorag.examiner.clustering import (
    allocate_difficulty_weighted,
    allocate_largest_remainder,
    compute_clusters,
    compute_difficulty_scores,
    resolve_n_clusters,
)


class TestResolveNClusters:
    def test_sqrt(self) -> None:
        result = resolve_n_clusters(n_items=100, target_size=50)

        assert result == 10

    def test_capped_by_target_size(self) -> None:
        result = resolve_n_clusters(n_items=10000, target_size=20)

        assert result == 20

    def test_single_item(self) -> None:
        result = resolve_n_clusters(n_items=1, target_size=50)

        assert result == 1

    def test_floor_of_one(self) -> None:
        result = resolve_n_clusters(n_items=0, target_size=50)

        assert result >= 1

    def test_large_count(self) -> None:
        result = resolve_n_clusters(n_items=2500, target_size=100)

        assert result == 50


class TestComputeClusters:
    def test_returns_correct_shape(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((50, 8))
        n_clusters = 5

        labels = compute_clusters(embeddings, n_clusters=n_clusters)

        assert labels.shape == (50,)

    def test_label_range(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((30, 4))
        n_clusters = 3

        labels = compute_clusters(embeddings, n_clusters=n_clusters)

        assert set(labels).issubset({0, 1, 2})

    def test_deterministic(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((40, 4))
        n_clusters = 4

        labels_a = compute_clusters(embeddings, n_clusters=n_clusters)
        labels_b = compute_clusters(embeddings, n_clusters=n_clusters)

        np.testing.assert_array_equal(labels_a, labels_b)

    def test_well_separated_clusters(self) -> None:
        rng = np.random.default_rng(0)
        cluster_a = rng.standard_normal((20, 2)) + np.array([0, 0])
        cluster_b = rng.standard_normal((20, 2)) + np.array([50, 50])
        cluster_c = rng.standard_normal((20, 2)) + np.array([-50, 50])
        embeddings = np.vstack([cluster_a, cluster_b, cluster_c])
        n_clusters = 3

        labels = compute_clusters(embeddings, n_clusters=n_clusters)

        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:40])) == 1
        assert len(set(labels[40:])) == 1
        assert len({labels[0], labels[20], labels[40]}) == 3


class TestAllocateLargestRemainder:
    def test_sums_to_exam_size(self) -> None:
        sizes = np.array([100, 50, 30, 20])
        exam_size = 50

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 50

    def test_equal_clusters(self) -> None:
        sizes = np.array([100, 100, 100, 100])
        exam_size = 20

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 20
        assert alloc.min() >= 4
        assert alloc.max() <= 6

    def test_unequal_clusters(self) -> None:
        sizes = np.array([400, 25, 25, 25])
        exam_size = 40

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 40
        assert alloc[0] > alloc[1]

    def test_single_cluster(self) -> None:
        sizes = np.array([200])
        exam_size = 30

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 30
        assert alloc[0] == 30

    def test_cluster_smaller_than_allocation(self) -> None:
        sizes = np.array([3, 100, 100])
        exam_size = 50

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc[0] <= 3
        assert alloc.sum() == 50

    def test_all_zero_clusters(self) -> None:
        sizes = np.array([0, 0, 0])
        exam_size = 10

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 0

    def test_exam_size_exceeds_total_chunks(self) -> None:
        sizes = np.array([5, 5, 5])
        exam_size = 100

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() <= 15

    def test_no_negative_allocations(self) -> None:
        sizes = np.array([10, 1, 50, 2, 100])
        exam_size = 30

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert (alloc >= 0).all()
        assert alloc.sum() == 30

    def test_many_small_clusters(self) -> None:
        sizes = np.ones(20, dtype=int) * 5
        exam_size = 50

        alloc = allocate_largest_remainder(sizes, exam_size=exam_size)

        assert alloc.sum() == 50
        assert (alloc <= 5).all()


class TestComputeDifficultyScores:
    def test_returns_correct_shape(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((50, 8)).astype(np.float32)

        scores = compute_difficulty_scores(embeddings)

        assert scores.shape == (50,)

    def test_scores_in_zero_one(self) -> None:
        embeddings = np.random.default_rng(42).standard_normal((30, 8)).astype(np.float32)

        scores = compute_difficulty_scores(embeddings)

        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_single_doc(self) -> None:
        embeddings = np.array([[1.0, 2.0, 3.0]])

        scores = compute_difficulty_scores(embeddings)

        assert scores.shape == (1,)
        assert scores[0] == 0.0

    def test_identical_embeddings_high_difficulty(self) -> None:
        embeddings = np.ones((10, 4), dtype=np.float32)

        scores = compute_difficulty_scores(embeddings)

        # All identical -> cosine sim = 1.0 everywhere -> high difficulty
        assert (scores > 0.9).all()

    def test_orthogonal_embeddings_low_difficulty(self) -> None:
        embeddings = np.eye(10, dtype=np.float32)

        scores = compute_difficulty_scores(embeddings)

        # Orthogonal -> cosine sim = 0.0 to all neighbors -> low difficulty
        assert (scores < 0.1).all()

    def test_cluster_has_higher_difficulty_than_isolated(self) -> None:
        rng = np.random.default_rng(42)
        # 5 similar docs (tight cluster)
        cluster = rng.standard_normal((5, 8)).astype(np.float32) * 0.1 + 10.0
        # 5 isolated docs (spread out)
        isolated = np.eye(5, 8, dtype=np.float32) * 50.0

        embeddings = np.vstack([cluster, isolated])
        scores = compute_difficulty_scores(embeddings)

        cluster_mean = scores[:5].mean()
        isolated_mean = scores[5:].mean()
        assert cluster_mean > isolated_mean


class TestAllocateDifficultyWeighted:
    def test_sums_to_exam_size(self) -> None:
        sizes = np.array([20, 20, 20])
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        diff = np.concatenate([np.full(20, 0.8), np.full(20, 0.3), np.full(20, 0.1)])

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=30)

        assert alloc.sum() == 30

    def test_floor_respected(self) -> None:
        sizes = np.array([20, 20, 20])
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        diff = np.concatenate([np.full(20, 0.9), np.full(20, 0.01), np.full(20, 0.01)])

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=30, min_per_cluster=2)

        # Even low-difficulty clusters get the floor
        assert alloc[1] >= 2
        assert alloc[2] >= 2

    def test_high_difficulty_gets_more(self) -> None:
        sizes = np.array([50, 50, 50])
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        diff = np.concatenate([np.full(50, 0.9), np.full(50, 0.1), np.full(50, 0.1)])

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=30, min_per_cluster=1)

        # Cluster 0 (high difficulty) should get the most
        assert alloc[0] > alloc[1]
        assert alloc[0] > alloc[2]

    def test_single_cluster(self) -> None:
        sizes = np.array([50])
        labels = np.zeros(50, dtype=int)
        diff = np.full(50, 0.5, dtype=np.float32)

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=20)

        assert alloc[0] == 20

    def test_no_negative_allocations(self) -> None:
        sizes = np.array([10, 3, 50, 2, 100])
        labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
        diff = np.random.default_rng(42).random(len(labels)).astype(np.float32)

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=30)

        assert (alloc >= 0).all()
        assert alloc.sum() == 30

    def test_floor_zero_allowed(self) -> None:
        sizes = np.array([50, 50])
        labels = np.array([0] * 50 + [1] * 50)
        # Cluster 1 has zero difficulty
        diff = np.concatenate([np.full(50, 0.8), np.zeros(50)])

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=20, min_per_cluster=0)

        # Cluster 1 may get 0 questions
        assert alloc.sum() == 20
        assert alloc[0] > alloc[1]

    def test_capped_at_cluster_size(self) -> None:
        sizes = np.array([3, 100])
        labels = np.array([0, 0, 0] + [1] * 100)
        diff = np.concatenate([np.full(3, 0.9), np.full(100, 0.1)])

        alloc = allocate_difficulty_weighted(sizes, diff, labels, exam_size=50, min_per_cluster=1)

        assert alloc[0] <= 3
