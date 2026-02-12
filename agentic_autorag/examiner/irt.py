"""3-Parameter Logistic IRT model for exam quality analysis and refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from agentic_autorag.config.models import MCQ_OPTIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IRTResult:
    """Results of a fitted 3PL IRT model."""

    abilities: np.ndarray
    discriminations: np.ndarray
    difficulties: np.ndarray
    guessings: np.ndarray
    converged: bool
    neg_log_likelihood: float


class IRTAnalyzer:
    """3PL IRT analysis with paper-aligned bounds and initialization."""

    ABILITY_BOUNDS = (-3.0, 3.0)
    DISCRIMINATION_BOUNDS = (0.1, 1.5)
    DIFFICULTY_BOUNDS = (0.01, 1.0)
    GUESSING_BOUNDS = (0.2, 0.4)

    ABILITY_INIT = 0.0
    DISCRIMINATION_INIT = 1.0
    DIFFICULTY_INIT = 0.0
    GUESSING_INIT = 1.0 / MCQ_OPTIONS

    def __init__(self, discrimination_threshold: float = 0.3) -> None:
        self.discrimination_threshold = discrimination_threshold

    def fit(self, response_matrix: np.ndarray) -> IRTResult:
        """Fit 3PL IRT to a binary matrix with shape (n_systems, n_questions)."""
        n_systems, n_questions = response_matrix.shape
        if n_systems < 2:
            raise ValueError(f"IRT requires at least 2 systems, got {n_systems}")
        if n_questions < 1:
            raise ValueError("IRT requires at least 1 question")

        x0 = np.concatenate(
            [
                np.full(n_systems, self.ABILITY_INIT),
                np.full(n_questions, self.DISCRIMINATION_INIT),
                np.full(n_questions, self.DIFFICULTY_INIT),
                np.full(n_questions, self.GUESSING_INIT),
            ]
        )

        bounds = (
            [self.ABILITY_BOUNDS] * n_systems
            + [self.DISCRIMINATION_BOUNDS] * n_questions
            + [self.DIFFICULTY_BOUNDS] * n_questions
            + [self.GUESSING_BOUNDS] * n_questions
        )

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(response_matrix.astype(float), n_systems, n_questions),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        params = result.x
        abilities = params[:n_systems]
        discriminations = params[n_systems : n_systems + n_questions]
        difficulties = params[n_systems + n_questions : n_systems + 2 * n_questions]
        guessings = params[n_systems + 2 * n_questions :]

        if not result.success:
            logger.warning("IRT optimization did not converge: %s", result.message)

        return IRTResult(
            abilities=abilities,
            discriminations=discriminations,
            difficulties=difficulties,
            guessings=guessings,
            converged=bool(result.success),
            neg_log_likelihood=float(result.fun),
        )

    @staticmethod
    def icc(theta: np.ndarray | float, a: float, b: float, g: float) -> np.ndarray:
        """Item characteristic curve"""
        theta_arr = np.asarray(theta, dtype=float)
        logistic = 1.0 / (1.0 + np.exp(-a * (theta_arr - b)))
        p = g + (1.0 - g) * logistic
        return np.clip(p, 1e-10, 1.0 - 1e-10)

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        response_matrix: np.ndarray,
        n_systems: int,
        n_questions: int,
    ) -> float:
        """Joint negative log-likelihood"""
        abilities = params[:n_systems]
        discriminations = params[n_systems : n_systems + n_questions]
        difficulties = params[n_systems + n_questions : n_systems + 2 * n_questions]
        guessings = params[n_systems + 2 * n_questions :]

        theta = abilities[:, np.newaxis]
        a = discriminations[np.newaxis, :]
        b = difficulties[np.newaxis, :]
        g = guessings[np.newaxis, :]

        logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        p = g + (1.0 - g) * logistic
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        ll = response_matrix * np.log(p) + (1.0 - response_matrix) * np.log(1.0 - p)
        return float(-ll.sum())

    @staticmethod
    def item_information(theta: np.ndarray, a: float, b: float, g: float) -> np.ndarray:
        """Item information function"""
        p = IRTAnalyzer.icc(theta, a, b, g)
        info = (a**2) * (((p - g) ** 2) / ((1.0 - g) ** 2)) * ((1.0 - p) / p)
        return info

    @staticmethod
    def aggregated_information(
        theta: np.ndarray,
        discriminations: np.ndarray,
        difficulties: np.ndarray,
        guessings: np.ndarray,
        question_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Aggregated information function over selected questions."""
        if question_indices is not None:
            a_arr = discriminations[question_indices]
            b_arr = difficulties[question_indices]
            g_arr = guessings[question_indices]
        else:
            a_arr = discriminations
            b_arr = difficulties
            g_arr = guessings

        total_info = np.zeros_like(theta, dtype=float)
        for idx in range(len(a_arr)):
            total_info += IRTAnalyzer.item_information(theta, a_arr[idx], b_arr[idx], g_arr[idx])
        return total_info / max(1, len(a_arr))

    def identify_weak_questions(self, discriminations: np.ndarray) -> list[int]:
        """Return indices of questions below discrimination threshold."""
        return [idx for idx, disc in enumerate(discriminations) if disc < self.discrimination_threshold]

    @staticmethod
    def identify_cull_candidates(discriminations: np.ndarray, drop_ratio: float = 0.1) -> list[int]:
        """Return bottom drop_ratio of question indices by discrimination."""
        if len(discriminations) == 0:
            return []
        n_drop = max(1, int(len(discriminations) * drop_ratio))
        sorted_indices = np.argsort(discriminations)
        return sorted(sorted_indices[:n_drop].tolist())
