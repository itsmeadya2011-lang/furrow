from __future__ import annotations

from furrow.config import Usage
from furrow.costs import (
    DEFAULT_PRICING,
    ModelPricing,
    _candidates,
    _merged_pricing,
    _normalize,
    _strip_provider_prefix,
    _token_count,
    estimate_cost,
    price_for_model,
)


def test_normalize_dotted_and_dashed_spellings() -> None:
    assert _normalize("GPT-4o") == "gpt-4o"
    assert _normalize("claude.sonnet.4") == "claude-sonnet-4"
    assert _normalize("  Mixed_Case-Model  ") == "mixed-case-model"


def test_strip_provider_prefix_handles_known_prefixes() -> None:
    assert _strip_provider_prefix("anthropic/claude-3-5-haiku") == "claude-3-5-haiku"
    assert _strip_provider_prefix("openai/gpt-4o") == "gpt-4o"
    assert _strip_provider_prefix("ollama/llama3") == "llama3"
    assert _strip_provider_prefix("gpt-4o") == "gpt-4o"


def test_strip_provider_prefix_handles_unknown_prefix() -> None:
    assert _strip_provider_prefix("unknown/gpt-4o") == "unknown/gpt-4o"


def test_candidates_returns_raw_and_stripped() -> None:
    candidates = _candidates("anthropic/claude-3-5-haiku")
    assert "anthropic/claude-3-5-haiku" in candidates
    assert "claude-3-5-haiku" in candidates
    assert len(candidates) == 2


def test_candidates_deduplicates() -> None:
    assert _candidates("gpt-4o") == ["gpt-4o"]


def test_merged_pricing_user_overrides_win() -> None:
    overrides = {"gpt-4o": ModelPricing(input_per_million=99.0, output_per_million=1.0)}
    merged = _merged_pricing(overrides)
    assert merged["gpt-4o"].input_per_million == 99.0
    # Built-in entries are still present.
    assert "claude-3-5-haiku" in merged


def test_merged_pricing_none_returns_defaults() -> None:
    merged = _merged_pricing(None)
    assert merged is not DEFAULT_PRICING  # a copy, not the same object
    assert "gpt-4o" in merged


def test_merged_pricing_invalid_entries_are_skipped() -> None:
    merged = _merged_pricing({"bad": "not-a-price", "also-bad": "still-not-a-price"})
    assert "bad" not in merged
    assert "also-bad" not in merged
    # Built-in entries remain untouched.
    assert "gpt-4o" in merged


def test_token_count_handles_weird_inputs() -> None:
    class _Weird:
        input_tokens = None
        output_tokens = "not-a-number"

    assert _token_count(_Weird(), "input_tokens") == 0.0
    assert _token_count(_Weird(), "output_tokens") == 0.0
    assert _token_count(None, "input_tokens") == 0.0
    assert _token_count(-5, "input_tokens") == 0.0


def test_estimate_cost_uses_override_table() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    overrides = {"my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
    assert estimate_cost("my-model", usage, overrides) == 3.0


def test_estimate_cost_unknown_model_is_zero() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    assert estimate_cost("totally-unknown-model", usage) == 0.0


def test_estimate_cost_zero_tokens_is_zero() -> None:
    usage = Usage(input_tokens=0, output_tokens=0, calls=1)
    assert estimate_cost("gpt-4o", usage) == 0.0


def test_estimate_cost_negative_tokens_never_costs_negative() -> None:
    usage = Usage(input_tokens=-1000, output_tokens=-500, calls=1)
    cost = estimate_cost("gpt-4o", usage)
    assert cost >= 0
    assert cost == 0.0


def test_estimate_cost_never_raises() -> None:
    class _Weird:
        input_tokens = None
        output_tokens = "not-a-number"

    assert estimate_cost("gpt-4o", _Weird()) == 0.0
    assert estimate_cost("gpt-4o", None) == 0.0  # type: ignore[arg-type]


def test_estimate_cost_rounds_to_six_decimal_places() -> None:
    usage = Usage(input_tokens=1, output_tokens=1, calls=1)
    cost = estimate_cost("gpt-4o-mini", usage)
    # 1/1e6 * 0.15 + 1/1e6 * 0.60 = 0.00000075 -> rounded to 0.000001
    assert cost == round(0.00000075, 6)


def test_default_pricing_covers_the_mainstream_models() -> None:
    for model in (
        "claude-sonnet-4",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-opus-4",
        "gpt-4o",
        "gpt-4o-mini",
    ):
        assert model in DEFAULT_PRICING, model


def test_default_pricing_entries_are_model_pricing_instances() -> None:
    for entry in DEFAULT_PRICING.values():
        assert isinstance(entry, ModelPricing)
        assert entry.input_per_million >= 0
        assert entry.output_per_million >= 0


def test_price_for_model_exact_id() -> None:
    price = price_for_model("gpt-4o-mini")
    assert price is not None
    assert price.input_per_million == 0.15
    assert price.output_per_million == 0.60


def test_price_for_model_exact_id_for_anthropic() -> None:
    price = price_for_model("claude-3-5-haiku")
    assert price is not None
    assert price.input_per_million == 0.80
    assert price.output_per_million == 4.0


def test_price_for_model_dated_id_resolves_through_the_prefix() -> None:
    price = price_for_model("claude-sonnet-4-20250514")
    assert price is not None
    assert price == price_for_model("claude-sonnet-4")
    assert price.input_per_million == 3.0
    assert price.output_per_million == 15.0


def test_price_for_model_uses_the_longest_matching_prefix() -> None:
    # "gpt-4o-mini" is more specific than "gpt-4o" and must win.
    price = price_for_model("gpt-4o-mini-2024-07-18")
    assert price is not None
    assert price == price_for_model("gpt-4o-mini")
    assert price.input_per_million == 0.15


def test_price_for_model_strips_a_provider_prefix() -> None:
    anthropic_price = price_for_model("anthropic/claude-3-5-haiku-20241022")
    assert anthropic_price is not None
    assert anthropic_price == price_for_model("claude-3-5-haiku")

    openai_price = price_for_model("openai/gpt-4o")
    assert openai_price is not None
    assert openai_price == price_for_model("gpt-4o")


def test_price_for_model_unknown_model_returns_none() -> None:
    assert price_for_model("totally-unknown-model") is None
    assert price_for_model("") is None


def test_price_for_model_never_raises() -> None:
    assert price_for_model("gpt-4o", {"broken": "not-a-price"}) is not None
    assert price_for_model(None) is None  # type: ignore[arg-type]


def test_price_for_model_overrides_beat_the_built_in_table() -> None:
    overrides = {"gpt-4o-mini": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
    price = price_for_model("gpt-4o-mini", overrides)
    assert price is not None
    assert price.input_per_million == 1.0
    assert price.output_per_million == 2.0


def test_price_for_model_override_adds_a_new_model() -> None:
    overrides = {"my-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
    price = price_for_model("my-model", overrides)
    assert price is not None
    assert price.input_per_million == 1.0
    assert price.output_per_million == 2.0


def test_estimate_cost_for_a_known_model() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    cost = estimate_cost("gpt-4o-mini", usage)
    assert cost == 0.75


def test_estimate_cost_matches_the_formula() -> None:
    usage = Usage(input_tokens=1000, output_tokens=500, calls=1)
    cost = estimate_cost("claude-3-5-haiku", usage)
    expected = 1000 / 1e6 * 0.80 + 500 / 1e6 * 4.0
    assert cost == round(expected, 6)
    assert cost > 0



