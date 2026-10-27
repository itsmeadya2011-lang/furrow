"""Currency cost accounting for LLM token usage.

Prices are looked up per model and multiplied by the token counts the provider
reported for a call.  The built-in table is only a *snapshot* of published
prices taken at a point in time: providers change their prices regularly, so
these numbers go stale and are meant as an approximation.  Override or extend
them with the ``FURROW_MODEL_PRICING`` environment variable, which takes a JSON
object mapping model ids to prices::

    FURROW_MODEL_PRICING='{"my-model":{"input_per_million":1.0,"output_per_million":2.0}}'

This module must not import :mod:`furrow.config` at module scope -- the config
module imports :class:`ModelPricing` from here, so a runtime import would be
circular.  Token counts are therefore read by duck typing instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelPricing(BaseModel):
    """USD price per 1,000,000 tokens for a single model.

    The model is immutable (frozen) so a pricing entry cannot be changed by
    accident, and it is a pydantic model so it can be parsed straight out of
    the JSON ``FURROW_MODEL_PRICING`` environment variable.
    """

    model_config = ConfigDict(frozen=True)

    input_per_million: float = Field(
        default=0.0, description="USD per 1,000,000 input tokens"
    )
    output_per_million: float = Field(
        default=0.0, description="USD per 1,000,000 output tokens"
    )


# DEFAULT SNAPSHOT of published per-1,000,000-token USD prices.
#
# These figures go stale: providers raise and cut prices regularly, and new
# models appear all the time.  Treat them as a rough estimate and override
# them with the FURROW_MODEL_PRICING environment variable, for example
# FURROW_MODEL_PRICING='{"my-model":{"input_per_million":1.0,"output_per_million":2.0}}'
#
# Keys are family names so that dated model ids ("claude-sonnet-4-20250514",
# "gpt-4o-mini-2024-07-18", ...) resolve through a longest-prefix match.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-opus-4-1": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-sonnet-4": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-haiku-4-5": ModelPricing(input_per_million=1.0, output_per_million=5.0),
    "claude-3-7-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-3-5-opus": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-3-5-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-3-5-haiku": ModelPricing(input_per_million=0.8, output_per_million=4.0),
    "claude-3-opus": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-3-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-3-haiku": ModelPricing(input_per_million=0.25, output_per_million=1.25),
    "gpt-4.1": ModelPricing(input_per_million=2.0, output_per_million=8.0),
    "gpt-4.1-mini": ModelPricing(input_per_million=0.4, output_per_million=1.6),
    "gpt-4.1-nano": ModelPricing(input_per_million=0.1, output_per_million=0.4),
    "gpt-4o": ModelPricing(input_per_million=2.5, output_per_million=10.0),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.6),
    "gpt-4": ModelPricing(input_per_million=30.0, output_per_million=60.0),
    "gpt-4-turbo": ModelPricing(input_per_million=10.0, output_per_million=30.0),
    "gpt-3.5-turbo": ModelPricing(input_per_million=0.5, output_per_million=1.5),
    "o1": ModelPricing(input_per_million=15.0, output_per_million=60.0),
    "o1-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
    "o3": ModelPricing(input_per_million=2.0, output_per_million=8.0),
    "o3-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
    "o4-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
}

#: Provider prefixes that are stripped from a model id before it is looked up.
_PROVIDER_PREFIXES = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "deepseek",
        "gemini",
        "google",
        "groq",
        "mistral",
        "ollama",
        "openai",
        "vertex",
        "vertex_ai",
        "xai",
    }
)


def _normalize(name: str) -> str:
    """Normalize a model id so that dotted and dashed spellings both match."""
    return name.strip().lower().replace(".", "-").replace("_", "-")


def _strip_provider_prefix(model: str) -> str:
    """Drop a leading provider prefix such as ``anthropic/`` from a model id."""
    head, sep, tail = model.partition("/")
    if sep and tail and head.lower() in _PROVIDER_PREFIXES:
        return tail
    return model


def _candidates(model: str) -> list[str]:
    """The spellings of *model* to try, most specific first."""
    raw = model.strip()
    stripped = _strip_provider_prefix(raw)
    candidates: list[str] = []
    for candidate in (raw, stripped):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _merged_pricing(pricing: Mapping[str, ModelPricing] | None) -> dict[str, ModelPricing]:
    """The default table with the user-supplied overrides applied on top."""
    table: dict[str, ModelPricing] = dict(DEFAULT_PRICING)
    if not pricing:
        return table
    for key, value in pricing.items():
        if not isinstance(value, ModelPricing):
            try:
                value = ModelPricing.model_validate(value)
            except Exception:
                continue
        table[str(key)] = value
    return table


def price_for_model(
    model: Any, pricing: Mapping[str, ModelPricing] | None = None
) -> ModelPricing | None:
    """Resolve a model id to its price, or ``None`` when the model is unknown.

    A provider prefix (``anthropic/``, ``openai/``, ...) is stripped first, then
    an exact match is attempted against the merged table (user overrides win
    over :data:`DEFAULT_PRICING`), and finally the longest matching key prefix
    is used (so ``claude-sonnet-4-20250514`` resolves through
    ``claude-sonnet-4``).  Never raises.
    """
    try:
        if not isinstance(model, str) or not model.strip():
            return None
        table = _merged_pricing(pricing)
        exact_candidates = _candidates(model)
        for candidate in exact_candidates:
            if candidate in table:
                return table[candidate]
        normalized = {_normalize(key): value for key, value in table.items()}
        normalized_candidates = [_normalize(candidate) for candidate in exact_candidates]
        for candidate in normalized_candidates:
            if candidate in normalized:
                return normalized[candidate]
        for candidate in reversed(normalized_candidates):
            best_key: str | None = None
            for key in normalized:
                if candidate.startswith(key) and (best_key is None or len(key) > len(best_key)):
                    best_key = key
            if best_key is not None:
                return normalized[best_key]
        return None
    except Exception:
        return None


def _token_count(usage: Any, field: str) -> float:
    """Read a token count from *usage* by duck typing (never negative)."""
    try:
        value = getattr(usage, field, 0) or 0
        count = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, count)


def estimate_cost(
    model: str, usage: Any, pricing: Mapping[str, ModelPricing] | None = None
) -> float:
    """Estimate the USD cost of *usage* for *model*.

    Returns ``0.0`` for an unknown model, rounds to six decimal places and
    never returns a negative amount.  Never raises.
    """
    try:
        entry = price_for_model(model, pricing)
        if entry is None:
            return 0.0
        cost = (
            _token_count(usage, "input_tokens") / 1_000_000.0
            * float(entry.input_per_million)
            + _token_count(usage, "output_tokens") / 1_000_000.0
            * float(entry.output_per_million)
        )
        if cost <= 0:
            return 0.0
        return round(cost, 6)
    except Exception:
        return 0.0
