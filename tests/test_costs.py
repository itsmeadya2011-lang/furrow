"""Tests for the cost estimation module."""

from __future__ import annotations

from typing import Any

import pytest

from furrow.costs import (
    DEFAULT_PRICING,
    ModelPricing,
    estimate_cost,
    price_for_model,
)


def test_model_pricing_defaults():
    p = ModelPricing()
    assert p.input_per_million == 0.0
    assert p.output_per_million == 0.0


def test_model_pricing_custom_values():
    p = ModelPricing(input_per_million=1.5, output_per_million=3.0)
    assert p.input_per_million == 1.5
    assert p.output_per_million == 3.0


def test_model_pricing_is_frozen():
    p = ModelPricing(input_per_million=1.0, output_per_million=2.0)
    with pytest.raises(Exception):
        p.input_per_million = 9.0


def test_default_pricing_contains_known_models():
    assert "claude-sonnet-4" in DEFAULT_PRICING
    assert "gpt-4o-mini" in DEFAULT_PRICING
    assert "claude-3-5-haiku" in DEFAULT_PRICING


def test_price_for_model_exact_match():
    entry = price_for_model("claude-sonnet-4-20250514")
    assert entry is not None
    assert entry.input_per_million == 3.0


def test_price_for_model_strips_provider_prefix():
    entry = price_for_model("anthropic/claude-sonnet-4-20250514")
    assert entry is not None
    assert entry.input_per_million == 3.0


def test_price_for_model_longest_prefix_match():
    # claude-sonnet-4-20250514 should resolve via the "claude-sonnet-4" prefix
    entry = price_for_model("claude-sonnet-4-20250514")
    assert entry is not None
    assert entry.input_per_million == 3.0


def test_price_for_model_unknown_returns_none():
    assert price_for_model("totally-unknown-model") is None


def test_price_for_model_empty_string_returns_none():
    assert price_for_model("") is None


def test_price_for_model_none_returns_none():
    assert price_for_model(None) is None  # type: ignore[arg-type]


def test_estimate_cost_claude_sonnet_4():
    from furrow.config import Usage

    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    cost = estimate_cost("claude-sonnet-4-20250514", usage)
    # claude-sonnet-4: $3.00 input / $15.00 output per 1M tokens
    expected = 3.0 + 15.0
    assert cost == pytest.approx(expected)


def test_estimate_cost_gpt_4o_mini():
    from furrow.config import Usage

    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    cost = estimate_cost("gpt-4o-mini", usage)
    # gpt-4o-mini: $0.15 input / $0.60 output per 1M tokens
    expected = 0.15 + 0.60
    assert cost == pytest.approx(expected)


def test_estimate_cost_unknown_model_returns_zero():
    from furrow.config import Usage

    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    cost = estimate_cost("totally-unknown-model", usage)
    assert cost == 0.0


def test_estimate_cost_zero_tokens_returns_zero():
    from furrow.config import Usage

    usage = Usage()
    cost = estimate_cost("claude-sonnet-4-20250514", usage)
    assert cost == 0.0


def test_estimate_cost_rounds_to_six_decimal_places():
    from furrow.config import Usage

    usage = Usage(input_tokens=1, output_tokens=1, calls=1)
    cost = estimate_cost("claude-sonnet-4-20250514", usage)
    # Should be rounded to 6 decimal places
    assert cost == round(cost, 6)


def test_estimate_cost_never_raises():
    from furrow.config import Usage

    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    # Even with garbage model, never raises
    cost = estimate_cost("!!not-a-model!!", usage)
    assert cost == 0.0


def test_price_for_model_normalizes_dots_and_underscores():
    # claude.sonnet-4 should match claude-sonnet-4
    entry = price_for_model("claude.sonnet-4")
    assert entry is not None
    assert entry.input_per_million == 3.0


def test_price_for_model_case_insensitive():
    entry = price_for_model("CLAUDE-SONNET-4-20250514")
    assert entry is not None
    assert entry.input_per_million == 3.0


def test_merged_pricing_overrides_win(monkeypatch: Any):
    from furrow.costs import _merged_pricing

    overrides = {
        "claude-sonnet-4": ModelPricing(input_per_million=99.0, output_per_million=99.0)
    }
    table = _merged_pricing(overrides)
    assert table["claude-sonnet-4"].input_per_million == 99.0


def test_env_pricing_overrides(monkeypatch: Any):
    monkeypatch.setenv(
        "FURROW_MODEL_PRICING",
        '{"my-model":{"input_per_million":1.0,"output_per_million":2.0}}',
    )
    # Need to reload settings to pick up env var
    from furrow.config import Settings

    s = Settings(_env_file=None)
    assert "my-model" in s.model_pricing
    assert s.model_pricing["my-model"].input_per_million == 1.0
    assert s.model_pricing["my-model"].output_per_million == 2.0


def test_estimate_cost_uses_env_overrides(monkeypatch: Any):
    monkeypatch.setenv(
        "FURROW_MODEL_PRICING",
        '{"my-model":{"input_per_million":10.0,"output_per_million":20.0}}',
    )
    from furrow.config import Settings, Usage

    s = Settings(_env_file=None)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    cost = estimate_cost("my-model", usage, pricing=s.pricing_overrides())
    assert cost == pytest.approx(30.0)


def test_estimate_cost_negative_tokens_clamped_to_zero():
    from furrow.config import Usage

    usage = Usage(input_tokens=-100, output_tokens=-100, calls=1)
    cost = estimate_cost("claude-sonnet-4-20250514", usage)
    assert cost == 0.0


def test_default_pricing_all_entries_have_non_negative_prices():
    for key, entry in DEFAULT_PRICING.items():
        assert entry.input_per_million >= 0.0, f"{key} has negative input price"
        assert entry.output_per_million >= 0.0, f"{key} has negative output price"
