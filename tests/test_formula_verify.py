"""Tests for the safe formula verifier."""

from __future__ import annotations

import pytest

from agentic_autorag.examiner.formula_verify import (
    FormulaError,
    evaluate_formula,
    matches_canonical,
    verify_formula,
)


class TestArithmetic:
    def test_simple_difference(self) -> None:
        assert evaluate_formula("2012 - 1948", "arithmetic") == 64.0

    def test_compound_expression(self) -> None:
        assert evaluate_formula("(300 + 250) / 2", "arithmetic") == 275.0

    def test_unary_minus(self) -> None:
        assert evaluate_formula("-5 + 12", "arithmetic") == 7.0

    def test_power(self) -> None:
        assert evaluate_formula("2 ** 10", "arithmetic") == 1024.0

    def test_modulo(self) -> None:
        assert evaluate_formula("13 % 5", "arithmetic") == 3.0

    def test_float_literals(self) -> None:
        assert abs(evaluate_formula("1.5 * 2.0", "arithmetic") - 3.0) < 1e-9

    def test_division_by_zero_raises(self) -> None:
        with pytest.raises(FormulaError, match="division by zero"):
            evaluate_formula("10 / 0", "arithmetic")

    def test_empty_formula_raises(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("", "arithmetic")

    def test_syntax_error_raises(self) -> None:
        with pytest.raises(FormulaError, match="syntax error"):
            evaluate_formula("2 + + ", "arithmetic")


class TestSandbox:
    """Anything outside the whitelist must be rejected."""

    def test_rejects_name_lookup(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("os", "arithmetic")

    def test_rejects_function_call(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("__import__('os')", "arithmetic")

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("(2).bit_length", "arithmetic")

    def test_rejects_subscript(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("[1, 2][0]", "arithmetic")

    def test_rejects_string_constant(self) -> None:
        with pytest.raises(FormulaError):
            evaluate_formula("'hello'", "arithmetic")


class TestUnknownKind:
    def test_unknown_formula_kind_raises(self) -> None:
        with pytest.raises(FormulaError, match="unknown formula_kind"):
            evaluate_formula("anything", "ratio")

    def test_date_diff_days_no_longer_supported(self) -> None:
        with pytest.raises(FormulaError, match="unknown formula_kind"):
            evaluate_formula("days('2020-01-01', '2019-01-01')", "date_diff_days")


class TestMatchesCanonical:
    def test_int_match(self) -> None:
        assert matches_canonical(64.0, "64 years")
        assert matches_canonical(64.0, "64")

    def test_int_mismatch(self) -> None:
        assert not matches_canonical(64.0, "65 years")

    def test_negative_match(self) -> None:
        assert matches_canonical(-3.0, "-3 points")

    def test_float_within_tolerance(self) -> None:
        assert matches_canonical(0.3333333333, "0.3333333333")

    def test_with_unit_after_number(self) -> None:
        assert matches_canonical(11.0, "11 points")
        assert matches_canonical(50.0, "$50 million")

    def test_comma_separated_number(self) -> None:
        assert matches_canonical(30495.0, "30,495")

    def test_no_number_in_canonical(self) -> None:
        assert not matches_canonical(64.0, "many years")


class TestVerifyFormula:
    def test_passes_when_formula_matches_answer(self) -> None:
        assert verify_formula("2012 - 1948", "arithmetic", "64 years") is True

    def test_fails_when_formula_disagrees_with_answer(self) -> None:
        assert verify_formula("2012 - 1948", "arithmetic", "65 years") is False

    def test_fails_on_malformed_formula(self) -> None:
        assert verify_formula("not a formula", "arithmetic", "64") is False

    def test_fails_on_unknown_kind(self) -> None:
        assert verify_formula("64", "ratio", "64") is False

    def test_fails_on_legacy_date_diff_days(self) -> None:
        assert verify_formula("days('1871-05-10', '1871-03-01')", "date_diff_days", "70 days") is False
