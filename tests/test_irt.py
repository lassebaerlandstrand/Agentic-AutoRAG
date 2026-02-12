"""Tests for the 3PL IRT model implementation."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_autorag.examiner.irt import IRTAnalyzer, IRTResult


class TestIRTFitting:
    """Test the core 3PL IRT fitting procedure."""

    def test_fit_basic(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.integers(0, 2, size=(8, 10))

        analyzer = IRTAnalyzer()
        result = analyzer.fit(matrix)

        assert isinstance(result, IRTResult)
        assert result.abilities.shape == (8,)
        assert result.discriminations.shape == (10,)
        assert result.difficulties.shape == (10,)
        assert result.guessings.shape == (10,)

    def test_fit_bounds_respected(self) -> None:
        rng = np.random.default_rng(9)
        matrix = rng.integers(0, 2, size=(6, 12))

        result = IRTAnalyzer().fit(matrix)

        assert np.all(result.abilities >= -3.0)
        assert np.all(result.abilities <= 3.0)
        assert np.all(result.discriminations >= 0.1)
        assert np.all(result.discriminations <= 1.5)
        assert np.all(result.difficulties >= 0.01)
        assert np.all(result.difficulties <= 1.0)
        assert np.all(result.guessings >= 0.2)
        assert np.all(result.guessings <= 0.4)

    def test_easy_question_low_difficulty(self) -> None:
        matrix = np.array(
            [
                [1, 1],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
            ]
        )
        result = IRTAnalyzer().fit(matrix)
        assert result.difficulties[0] <= result.difficulties[1]

    def test_high_ability_system(self) -> None:
        matrix = np.array(
            [
                [1, 1, 1, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )
        result = IRTAnalyzer().fit(matrix)
        assert result.abilities[0] > result.abilities[-1]

    def test_requires_minimum_2_systems(self) -> None:
        matrix = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="at least 2 systems"):
            IRTAnalyzer().fit(matrix)


class TestItemInformation:
    """Test the Item Information Function (Equation 3)."""

    def test_peaks_near_difficulty(self) -> None:
        theta = np.linspace(-3, 3, 301)
        b = 0.8
        info = IRTAnalyzer.item_information(theta=theta, a=1.2, b=b, g=0.25)
        peak_theta = theta[int(np.argmax(info))]
        assert abs(peak_theta - b) < 0.5

    def test_higher_discrimination_more_info(self) -> None:
        theta = np.array([0.2])
        info_low = IRTAnalyzer.item_information(theta=theta, a=0.3, b=0.2, g=0.25)[0]
        info_high = IRTAnalyzer.item_information(theta=theta, a=1.4, b=0.2, g=0.25)[0]
        assert info_high > info_low

    def test_non_negative(self) -> None:
        theta = np.linspace(-3, 3, 100)
        info = IRTAnalyzer.item_information(theta=theta, a=1.0, b=0.5, g=0.25)
        assert np.all(info >= 0)


class TestQuestionCulling:
    """Test question identification for removal."""

    def test_identify_weak_questions(self) -> None:
        analyzer = IRTAnalyzer(discrimination_threshold=0.5)
        weak = analyzer.identify_weak_questions(np.array([0.2, 0.4, 0.5, 0.8]))
        assert weak == [0, 1]

    def test_cull_candidates_ratio(self) -> None:
        disc = np.array([0.2, 0.6, 0.9, 1.2, 0.4])
        cull = IRTAnalyzer.identify_cull_candidates(disc, drop_ratio=0.2)
        assert len(cull) == 1
        assert cull[0] == 0


class TestIRTIntegration:
    """Test IRT with realistic-sized data."""

    def test_fit_realistic_scale(self) -> None:
        rng = np.random.default_rng(123)
        n_trials = 10
        n_questions = 50

        true_ability = np.linspace(-1.5, 1.5, n_trials)
        a = np.clip(rng.normal(1.0, 0.2, n_questions), 0.1, 1.5)
        b = np.clip(rng.normal(0.5, 0.2, n_questions), 0.01, 1.0)
        g = np.clip(rng.normal(0.25, 0.04, n_questions), 0.2, 0.4)

        probs = np.zeros((n_trials, n_questions), dtype=float)
        for i in range(n_trials):
            probs[i] = IRTAnalyzer.icc(true_ability[i], a, b, g)

        matrix = rng.binomial(1, probs).astype(int)
        result = IRTAnalyzer().fit(matrix)

        rank_corr = np.corrcoef(np.argsort(true_ability), np.argsort(result.abilities))[0, 1]
        assert rank_corr > 0.5
