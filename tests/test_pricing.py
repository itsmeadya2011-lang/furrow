"""Tests for the pricing tables and cost computation."""

from __future__ import annotations

from furrow.config import Settings
from furrow.pricing import (
    DEFAULT_COST_PER_1K,
    DEFAULT_PRICING,
    compute_cost_usd,
    model_key,
    price_for,
)


def test_default_pricing_has_anthropic_and_openai():
    assert "anthropic" in DEFAULT_PRICING
    assert "openai" in DEFAULT_PRICING
    assert "ollama" in DEFAULT_PRICING


def test_anthropic_sonnet_pricing():
    inp, out = DEFAULT_PRICING["anthropic"]["claude-3-5-sonnet-20241022"]
    assert inp == 3.0
    assert out == 15.0


def test_openai_gpt4o_mini_pricing():
    inp, out = DEFAULT_PRICING["openai"]["gpt-4o-mini"]
    assert inp == 0.15
    assert out == 0.60


def test_ollama_default_is_free():
    inp, out = DEFAULT_PRICING["ollama"]["default"]
    assert inp == 0.0
    assert out == 0.0


def test_price_for_known_model():
    inp, out = price_for("anthropic", "claude-3-5-sonnet-20241022")
    assert (inp, out) == (3.0, 15.0)


def test_price_for_unknown_model_uses_default():
    inp, out = price_for("anthropic", "some-unknown-model")
    assert (inp, out) == DEFAULT_COST_PER_1K


def test_price_for_unknown_provider_uses_default():
    inp, out = price_for("unknown", "whatever")
    assert (inp, out) == DEFAULT_COST_PER_1K


def test_price_for_ollama_model_uses_free_default():
    # Any ollama model falls through to the "default" entry in the table.
    inp, out = price_for("ollama", "llama3")
    assert (inp, out) == (0.0, 0.0)


def test_price_for_custom_override_wins():
    custom = {"anthropic:custom-model": (1.0, 2.0)}
    inp, out = price_for("anthropic", "custom-model", custom=custom)
    assert (inp, out) == (1.0, 2.0)


def test_price_for_custom_default_per_1k():
    inp, out = price_for(
        "anthropic", "unknown", default_per_1k=(0.5, 1.0)
    )
    assert (inp, out) == (0.5, 1.0)


def test_compute_cost_usd():
    # 1000 input tokens @ $3/1k + 500 output tokens @ $15/1k = 3 + 7.5 = 10.5
    cost = compute_cost_usd(1000, 500, 3.0, 15.0)
    assert cost == 10.5


def test_compute_cost_usd_zero_tokens():
    assert compute_cost_usd(0, 0, 3.0, 15.0) == 0.0


def test_model_key():
    assert (
        model_key("anthropic", "claude-3-5-sonnet-20241022")
        == "anthropic:claude-3-5-sonnet-20241022"
    )


def test_settings_track_cost_defaults_true():
    s = Settings(_env_file=None)
    assert s.track_cost is True


def test_settings_default_cost_per_1k_is_free():
    s = Settings(_env_file=None)
    assert s.default_cost_per_1k == (0.0, 0.0)


def test_settings_model_pricing_default_empty():
    s = Settings(_env_file=None)
    assert s.model_pricing == {}


def test_settings_model_pricing_can_be_set():
    s = Settings(_env_file=None, model_pricing={"anthropic:x": (1.0, 2.0)})
    assert s.model_pricing == {"anthropic:x": (1.0, 2.0)}