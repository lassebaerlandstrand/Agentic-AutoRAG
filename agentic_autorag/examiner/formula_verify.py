"""External math verification for ``numeric`` exam questions.

The composition LLM emits ``formula`` and ``formula_kind`` alongside
``canonical_answer`` for every ``numeric`` question. This module
evaluates the formula in a restricted Python AST and compares the
result against the canonical answer; questions where the math
disagrees are rejected before any LLM-based gate runs.

Only one formula kind is supported:

  ``arithmetic`` — a Python expression over numeric literals.
                   Whitelist: int / float constants, ``+ - * / **
                   %`` binary ops, unary minus, parenthesisation.

Anything outside the whitelist (names, function calls, attributes,
imports) raises ``FormulaError``.
"""

from __future__ import annotations

import ast
import re

_LEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class FormulaError(ValueError):
    """The formula could not be parsed or evaluated safely."""


def evaluate_formula(formula: str, kind: str) -> float:
    if kind == "arithmetic":
        return _safe_eval_arithmetic(formula)
    raise FormulaError(f"unknown formula_kind: {kind!r}")


def matches_canonical(result: float, canonical_answer: str) -> bool:
    """True when the formula result matches the leading number in the answer.

    Decimal-aware: the canonical answer's *displayed* precision sets the
    comparison precision. ``"33.3%"`` declares 1 decimal, so the formula's
    evaluation is rounded to 1 decimal and compared with absolute tolerance
    of half-a-unit at that precision (±0.05). This handles human-rounded
    answers (33.333… written as 33.3) while still rejecting answers off by
    more than a rounding step (30% vs 33.3%).
    """
    token = _leading_number_token(canonical_answer)
    if token is None:
        return False
    if "." in token:
        decimals = len(token.split(".", 1)[1])
        expected = float(token)
        rounded_result = round(result, decimals)
        tol = 0.5 * 10 ** (-decimals)
        return abs(rounded_result - expected) <= tol
    try:
        expected_int = int(token)
    except ValueError:
        return False
    return int(round(result)) == expected_int


def verify_formula(formula: str, kind: str, canonical_answer: str) -> bool:
    """Evaluate the formula and compare to ``canonical_answer``.

    Returns True iff the formula is well-formed AND its result matches
    the leading numeric token of ``canonical_answer`` within tolerance.
    """
    try:
        result = evaluate_formula(formula, kind)
    except FormulaError:
        return False
    except Exception:
        return False
    return matches_canonical(result, canonical_answer)


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _safe_eval_arithmetic(expr: str) -> float:
    if not expr or not expr.strip():
        raise FormulaError("empty formula")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"syntax error: {exc}") from exc
    return _walk_arithmetic(tree.body)


def _walk_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError(f"non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise FormulaError(f"binary op not allowed: {type(node.op).__name__}")
        left = _walk_arithmetic(node.left)
        right = _walk_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise FormulaError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise FormulaError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise FormulaError("division by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise FormulaError(f"unary op not allowed: {type(node.op).__name__}")
        operand = _walk_arithmetic(node.operand)
        return -operand if isinstance(node.op, ast.USub) else operand
    raise FormulaError(f"node type not allowed: {type(node).__name__}")


def _leading_number_token(text: str) -> str | None:
    """Extract the raw leading-number substring, preserving displayed decimals."""
    if not text:
        return None
    cleaned = text.replace(",", "").strip()
    m = _LEADING_NUMBER_RE.search(cleaned)
    return m.group(0) if m else None
