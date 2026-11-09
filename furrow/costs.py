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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from furrow.config import Usage


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
    cache_write_per_million: float = Field(
        default=0.0, description="USD per 1,000,000 cache write tokens"
    )
    cache_read_per_million: float = Field(
        default=0.0, description="USD per 1,000,000 cache read tokens"
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
    # Anthropic Claude 4 family
    "claude-opus-4": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-opus-4-1": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-sonnet-4": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-haiku-4-5": ModelPricing(input_per_million=1.0, output_per_million=5.0),
    
    # Anthropic Claude 3.7 family
    "claude-3-7-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    
    # Anthropic Claude 3.5 family
    "claude-3-5-opus": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-3-5-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-3-5-haiku": ModelPricing(input_per_million=0.8, output_per_million=4.0),
    
    # Anthropic Claude 3 family
    "claude-3-opus": ModelPricing(input_per_million=15.0, output_per_million=75.0),
    "claude-3-sonnet": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "claude-3-haiku": ModelPricing(input_per_million=0.25, output_per_million=1.25),
    
    # OpenAI GPT-4.1 family
    "gpt-4.1": ModelPricing(input_per_million=2.0, output_per_million=8.0),
    "gpt-4.1-mini": ModelPricing(input_per_million=0.4, output_per_million=1.6),
    "gpt-4.1-nano": ModelPricing(input_per_million=0.1, output_per_million=0.4),
    
    # OpenAI GPT-4o family
    "gpt-4o": ModelPricing(input_per_million=2.5, output_per_million=10.0),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.6),
    
    # OpenAI GPT-4 family
    "gpt-4": ModelPricing(input_per_million=30.0, output_per_million=60.0),
    "gpt-4-turbo": ModelPricing(input_per_million=10.0, output_per_million=30.0),
    
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": ModelPricing(input_per_million=0.5, output_per_million=1.5),
    
    # OpenAI o1 family
    "o1": ModelPricing(input_per_million=15.0, output_per_million=60.0),
    "o1-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
    "o1-preview": ModelPricing(input_per_million=15.0, output_per_million=60.0),
    
    # OpenAI o3 family
    "o3": ModelPricing(input_per_million=2.0, output_per_million=8.0),
    "o3-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
    "o4-mini": ModelPricing(input_per_million=1.1, output_per_million=4.4),
    
    # Google Gemini
    "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.0),
    "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.3),
    "gemini-1.0-pro": ModelPricing(input_per_million=0.5, output_per_million=1.5),
    
    # DeepSeek
    "deepseek-chat": ModelPricing(input_per_million=0.14, output_per_million=0.28),
    "deepseek-coder": ModelPricing(input_per_million=0.14, output_per_million=0.28),
    "deepseek-v3": ModelPricing(input_per_million=0.14, output_per_million=0.28),
    "deepseek-r1": ModelPricing(input_per_million=0.55, output_per_million=2.19),
    
    # Mistral
    "mistral-large": ModelPricing(input_per_million=3.0, output_per_million=9.0),
    "mistral-medium": ModelPricing(input_per_million=2.7, output_per_million=8.1),
    "mistral-small": ModelPricing(input_per_million=1.0, output_per_million=3.0),
    "mistral-tiny": ModelPricing(input_per_million=0.25, output_per_million=0.75),
    "mixtral-8x7b": ModelPricing(input_per_million=0.6, output_per_million=0.6),
    "mixtral-8x22b": ModelPricing(input_per_million=1.2, output_per_million=1.2),
    
    # Cohere
    "command-r-plus": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "command-r": ModelPricing(input_per_million=0.5, output_per_million=1.5),
    "command": ModelPricing(input_per_million=1.0, output_per_million=2.0),
    
    # Groq (Llama models)
    "llama-3.1-405b": ModelPricing(input_per_million=0.9, output_per_million=0.9),
    "llama-3.1-70b": ModelPricing(input_per_million=0.6, output_per_million=0.6),
    "llama-3.1-8b": ModelPricing(input_per_million=0.05, output_per_million=0.05),
    "llama-3.2-90b": ModelPricing(input_per_million=0.6, output_per_million=0.6),
    "llama-3.2-11b": ModelPricing(input_per_million=0.1, output_per_million=0.1),
    "llama-3.2-3b": ModelPricing(input_per_million=0.05, output_per_million=0.05),
    "llama-3.2-1b": ModelPricing(input_per_million=0.02, output_per_million=0.02),
    "llama-guard-3-8b": ModelPricing(input_per_million=0.05, output_per_million=0.05),
    "gemma-7b": ModelPricing(input_per_million=0.05, output_per_million=0.05),
    "gemma-2-9b": ModelPricing(input_per_million=0.1, output_per_million=0.1),
    "gemma-2-27b": ModelPricing(input_per_million=0.2, output_per_million=0.2),
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
    model: str, pricing: Mapping[str, ModelPricing] | None = None
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
    model: str, usage: Usage, pricing: Mapping[str, ModelPricing] | None = None
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
            + _token_count(usage, "cache_write_tokens") / 1_000_000.0
            * float(entry.cache_write_per_million)
            + _token_count(usage, "cache_read_tokens") / 1_000_000.0
            * float(entry.cache_read_per_million)
        )
        if cost <= 0:
            return 0.0
        return round(cost, 6)
    except Exception:
        return 0.0


def estimate_cost_breakdown(
    model: str, usage: Usage, pricing: Mapping[str, ModelPricing] | None = None
) -> dict[str, float]:
    """Return a detailed cost breakdown for *usage* for *model*.

    Returns a dictionary with the cost components. Never raises.
    """
    try:
        entry = price_for_model(model, pricing)
        if entry is None:
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_write_cost": 0.0,
                "cache_read_cost": 0.0,
                "total_cost": 0.0,
            }
        
        input_cost = round(
            _token_count(usage, "input_tokens") / 1_000_000.0 * float(entry.input_per_million), 6
        )
        output_cost = round(
            _token_count(usage, "output_tokens") / 1_000_000.0 * float(entry.output_per_million), 6
        )
        cache_write_cost = round(
            _token_count(usage, "cache_write_tokens") / 1_000_000.0 * float(entry.cache_write_per_million), 6
        )
        cache_read_cost = round(
            _token_count(usage, "cache_read_tokens") / 1_000_000.0 * float(entry.cache_read_per_million), 6
        )
        total_cost = round(input_cost + output_cost + cache_write_cost + cache_read_cost, 6)
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_write_cost": cache_write_cost,
            "cache_read_cost": cache_read_cost,
            "total_cost": total_cost,
        }
    except Exception:
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cache_write_cost": 0.0,
            "cache_read_cost": 0.0,
            "total_cost": 0.0,
        }
