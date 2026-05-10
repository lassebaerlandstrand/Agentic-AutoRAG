"""Tests for the AA reasoning-mode classifier.

Each AA naming pattern in the wild is pinned with a representative case.
"""

from __future__ import annotations

from agentic_autorag.config.reasoning_mode import classify, select_pair


def _make(slug: str, ii: float, base_slug: str = "", variant_type: str = "") -> dict:
    entry: dict = {"slug": slug, "benchmarks": {"artificial_analysis_intelligence_index": ii}}
    if base_slug:
        entry["base_slug"] = base_slug
    if variant_type:
        entry["variant_type"] = variant_type
    return entry


class TestClassifyVariant:
    def test_reasoning(self) -> None:
        assert classify(_make("x-reasoning", 30, "x", "reasoning")) == "on"

    def test_non_reasoning(self) -> None:
        assert classify(_make("x-non-reasoning", 20, "x", "non-reasoning")) == "off"

    def test_non_reasoning_low_effort(self) -> None:
        # Anthropic's Claude Sonnet 4.6 uses this token
        assert classify(_make("x-non-reasoning-low-effort", 22, "x", "non-reasoning-low-effort")) == "off"

    def test_thinking(self) -> None:
        # Claude Sonnet 4.5 + extended thinking
        assert classify(_make("x-thinking", 43, "x", "thinking")) == "on"

    def test_adaptive(self) -> None:
        # Claude Sonnet 4.6 adaptive mode (auto-thinking)
        assert classify(_make("x-adaptive", 51, "x", "adaptive")) == "on"

    def test_low_medium_high(self) -> None:
        for vt in ("low", "medium", "high"):
            assert classify(_make(f"x-{vt}", 30, "x", vt)) == "on", vt

    def test_reasoning_effort_levels(self) -> None:
        # AWS Nova: "-reasoning-low", "-reasoning-medium"
        for vt in ("reasoning-low", "reasoning-medium", "reasoning-high"):
            assert classify(_make(f"x-{vt}", 25, "x", vt)) == "on", vt


class TestClassifyBase:
    def test_pattern_a_reasoning_sibling(self) -> None:
        # base + -reasoning → base is OFF
        base = _make("haiku", 31)
        siblings = [_make("haiku-reasoning", 35, "haiku", "reasoning")]
        assert classify(base, siblings) == "off"

    def test_pattern_b_non_reasoning_sibling(self) -> None:
        # base + -non-reasoning → base is ON
        base = _make("glm-4-7", 34)
        siblings = [_make("glm-4-7-non-reasoning", 22, "glm-4-7", "non-reasoning")]
        assert classify(base, siblings) == "on"

    def test_pattern_c_thinking_sibling(self) -> None:
        # base + -thinking → base is OFF (default mode is non-thinking)
        base = _make("sonnet-4-5", 37)
        siblings = [_make("sonnet-4-5-thinking", 43, "sonnet-4-5", "thinking")]
        assert classify(base, siblings) == "off"

    def test_pattern_d_reasoning_effort_only(self) -> None:
        # base + -reasoning-medium/-low → base is OFF (the variants are "reasoning ON")
        base = _make("nova-lite", 18)
        siblings = [
            _make("nova-lite-reasoning", 34, "nova-lite", "reasoning"),
            _make("nova-lite-reasoning-low", 25, "nova-lite", "reasoning-low"),
        ]
        assert classify(base, siblings) == "off"

    def test_pattern_e_effort_only(self) -> None:
        # base + -low (only) → base is ON (highest effort), variant is lower effort
        base = _make("gpt-oss-120b", 33)
        siblings = [_make("gpt-oss-120b-low", 25, "gpt-oss-120b", "low")]
        assert classify(base, siblings) == "on"

    def test_pattern_f_no_siblings(self) -> None:
        # No siblings → cannot classify
        base = _make("standalone", 25)
        assert classify(base, []) is None

    def test_mixed_siblings_off_wins(self) -> None:
        # gpt-5-4-mini: base + -medium + -non-reasoning → base is ON (because of non-reasoning)
        base = _make("gpt-5-4-mini", 49)
        siblings = [
            _make("gpt-5-4-mini-medium", 37, "gpt-5-4-mini", "medium"),
            _make("gpt-5-4-mini-non-reasoning", 23, "gpt-5-4-mini", "non-reasoning"),
        ]
        assert classify(base, siblings) == "on"


class TestSelectPair:
    def test_pattern_a_returns_base_off_variant_on(self) -> None:
        base = _make("haiku", 31)
        variants = [_make("haiku-reasoning", 38, "haiku", "reasoning")]
        off, on = select_pair(base, variants)
        assert off is base
        assert on is variants[0]

    def test_pattern_b_returns_variant_off_base_on(self) -> None:
        base = _make("glm-4-7", 34)
        variants = [_make("glm-4-7-non-reasoning", 22, "glm-4-7", "non-reasoning")]
        off, on = select_pair(base, variants)
        assert off is variants[0]
        assert on is base

    def test_pattern_c_thinking(self) -> None:
        base = _make("sonnet-4-5", 37)
        variants = [_make("sonnet-4-5-thinking", 43, "sonnet-4-5", "thinking")]
        off, on = select_pair(base, variants)
        assert off is base
        assert on is variants[0]

    def test_pattern_d_prefers_medium_for_on(self) -> None:
        base = _make("nova-lite", 18)
        var_default = _make("nova-lite-reasoning", 34, "nova-lite", "reasoning")
        var_medium = _make("nova-lite-reasoning-medium", 30, "nova-lite", "reasoning-medium")
        var_low = _make("nova-lite-reasoning-low", 25, "nova-lite", "reasoning-low")
        off, on = select_pair(base, [var_default, var_medium, var_low])
        assert off is base
        # `-reasoning-medium` matches LiteLLM `reasoning_effort=medium` best
        assert on is var_medium

    def test_pattern_e_no_off_entry(self) -> None:
        # gpt-oss-120b: only ON variants — there's no AA off benchmark
        base = _make("gpt-oss-120b", 33)
        variants = [_make("gpt-oss-120b-low", 25, "gpt-oss-120b", "low")]
        off, on = select_pair(base, variants)
        assert off is None  # AA has no off-mode for gpt-oss
        assert on is base  # base is reasoning-on (high effort)

    def test_gpt_5_4_mini_prefers_medium(self) -> None:
        base = _make("gpt-5-4-mini", 49)
        var_medium = _make("gpt-5-4-mini-medium", 37, "gpt-5-4-mini", "medium")
        var_off = _make("gpt-5-4-mini-non-reasoning", 23, "gpt-5-4-mini", "non-reasoning")
        off, on = select_pair(base, [var_medium, var_off])
        assert off is var_off
        # ON should be `-medium` (matches LiteLLM medium effort), not the base
        assert on is var_medium

    def test_no_variants(self) -> None:
        base = _make("standalone", 25)
        off, on = select_pair(base, [])
        assert off is None
        assert on is None
