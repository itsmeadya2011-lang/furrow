"""Model pricing tables for currency cost estimation.

Prices are USD per 1,000 tokens (input, output). The tables cover the most
commonly used models for each provider. Unknown models fall back to a
configurable default so a run never crashes on a missing price.

A custom price can be supplied via ``Settings.model_pricing`` (a mapping of
``provider:model`` -> ``(input_per_1k, output_per_1k)``). When neither the
table nor a custom override has an entry, ``default_per_1k`` is used.
"""

from __future__ import annotations

# provider -> { model: (input_per_1k_usd, output_per_1k_usd) }
DEFAULT_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "anthropic": {
        # Claude 4.x / Sonnet family
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.80, 4.0),
        "claude-3-7-sonnet-20250219": (3.0, 15.0),
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
    },
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o-2024-08-06": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4-turbo-2024-04-09": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1": (15.00, 60.00),
        "o1-mini": (1.10, 4.40),
        "o3-mini": (1.10, 4.40),
    },
    "ollama": {
        # Local models are free to run; record zero cost.
        "default": (0.0, 0.0),
    },
}

# Fallback when a model is unknown: treat as free (conservative = do not
# invent a price). Override via ``Settings.default_cost_per_1k``.
DEFAULT_COST_PER_1K: tuple[float, float] = (0.0, 0.0)


def price_for(
    provider: str,
    model: str,
    custom: dict[str, tuple[float, float]] | None = None,
    default_per_1k: tuple[float, float] = DEFAULT_COST_PER_1K,
) -> tuple[float, float]:
    """Return ``(input_per_1k, output_per_1k)`` for *provider*:*model*.

    Lookup order: custom overrides, then the built-in table, then the
    configured default. Unknown providers also fall through to the default.
    """
    if custom:
        key = f"{provider}:{model}"
        if key in custom:
            return custom[key]
        if provider in custom and custom[provider] == ():
            return default_per_1k
    table = DEFAULT_PRICING.get(provider)
    if table is not None:
        if model in table:
            return table[model]
        if "default" in table:
            return table["default"]
    return default_per_1k


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_per_1k: float,
    output_per_1k: float,
) -> float:
    """Return the USD cost for the given token counts and per-1k prices."""
    return (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k


def model_key(provider: str, model: str) -> str:
    """Return the canonical ``provider:model`` key used by the pricing map."""
    return f"{provider}:{model}"


__all__ = [
    "DEFAULT_COST_PER_1K",
    "DEFAULT_PRICING",
    "compute_cost_usd",
    "model_key",
    "price_for",
]