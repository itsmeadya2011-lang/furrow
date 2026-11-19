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
from dataclasses import dataclass, field
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


#: Characters that may follow a model-family key when it is used as a prefix.
_PREFIX_SEPARATORS = "-:/@"


def _has_prefix_boundary(candidate: str, key: str) -> bool:
    """True when *key* is a whole leading token of *candidate*.

    A longer key must not match merely because it is a string prefix: the
    character after the key must be a separator (or the strings must be
    equal). This keeps ``gpt-4.1`` (normalized ``gpt-4-1``) from matching
    the dated GPT-4 snapshot ``gpt-4-1106-preview``.
    """
    if candidate == key:
        return True
    return len(candidate) > len(key) and candidate[len(key)] in _PREFIX_SEPARATORS


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
                if (
                    candidate.startswith(key)
                    and _has_prefix_boundary(candidate, key)
                    and (best_key is None or len(key) > len(best_key))
                ):
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


@dataclass
class ModelCostBreakdown:
    """Cost breakdown for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def add_usage(
        self, usage: Usage, model: str, pricing: Mapping[str, ModelPricing] | None = None
    ) -> None:
        """Add usage from a call to this model."""
        input_tokens = int(_token_count(usage, "input_tokens"))
        output_tokens = int(_token_count(usage, "output_tokens"))
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        entry = price_for_model(model, pricing)
        if entry is not None:
            self.input_cost_usd += input_tokens / 1_000_000.0 * float(entry.input_per_million)
            self.output_cost_usd += output_tokens / 1_000_000.0 * float(entry.output_per_million)
        self.total_cost_usd = round(self.input_cost_usd + self.output_cost_usd, 6)


@dataclass
class CycleCostBreakdown:
    """Cost breakdown for a single cycle."""
    cycle: int
    models: dict[str, ModelCostBreakdown] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0

    def add_usage(
        self, model: str, usage: Usage, pricing: Mapping[str, ModelPricing] | None = None
    ) -> None:
        """Add usage from a call in this cycle."""
        if model not in self.models:
            self.models[model] = ModelCostBreakdown(model=model)
        self.models[model].add_usage(usage, model, pricing)
        input_tokens = int(_token_count(usage, "input_tokens"))
        output_tokens = int(_token_count(usage, "output_tokens"))
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        self.total_cost_usd = round(sum(m.total_cost_usd for m in self.models.values()), 6)


@dataclass
class RunCostSummary:
    """Complete cost summary for a run."""
    cycles: list[CycleCostBreakdown] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0
    models: dict[str, ModelCostBreakdown] = field(default_factory=dict)

    def add_cycle(self, cycle_breakdown: CycleCostBreakdown) -> None:
        """Add a cycle's cost breakdown to the run summary."""
        self.cycles.append(cycle_breakdown)
        self.total_input_tokens += cycle_breakdown.total_input_tokens
        self.total_output_tokens += cycle_breakdown.total_output_tokens
        self.total_calls += cycle_breakdown.total_calls
        self.total_cost_usd = round(self.total_cost_usd + cycle_breakdown.total_cost_usd, 6)
        for model, breakdown in cycle_breakdown.models.items():
            if model not in self.models:
                self.models[model] = ModelCostBreakdown(model=model)
            self.models[model].input_tokens += breakdown.input_tokens
            self.models[model].output_tokens += breakdown.output_tokens
            self.models[model].calls += breakdown.calls
            self.models[model].input_cost_usd += breakdown.input_cost_usd
            self.models[model].output_cost_usd += breakdown.output_cost_usd
            self.models[model].total_cost_usd = round(
                self.models[model].input_cost_usd + self.models[model].output_cost_usd, 6
            )


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
        )
        if cost <= 0:
            return 0.0
        return round(cost, 6)
    except Exception:
        return 0.0