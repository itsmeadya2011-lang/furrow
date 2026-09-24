from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import aiofiles
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import TextBlock
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMStats:
    total_calls: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost: float = 0.0
    errors: int = 0
    
    def record_call(self, usage: TokenUsage, cost: float = 0.0) -> None:
        self.total_calls += 1
        self.total_tokens.add(usage)
        self.total_cost += cost


class LLMClient:
    # Approximate pricing per 1M tokens (as of 2024)
    PRICING: ClassVar[dict[str, dict[str, float]]] = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self.stats = LLMStats()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((AnthropicRateLimitError, AnthropicAPIConnectionError, OpenAIRateLimitError, OpenAIAPIConnectionError)),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Track usage
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )
        cost = self._calculate_cost(model, usage)
        self.stats.record_call(usage, cost)
        
        # Extract text from response content (handle different block types)
        for block in response.content:
            if isinstance(block, TextBlock):
                return block.text
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        
        # Track usage
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        cost = self._calculate_cost(model, usage)
        self.stats.record_call(usage, cost)
        
        return response.choices[0].message.content or ""

    def _calculate_cost(self, model: str, usage: TokenUsage) -> float:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_calls": self.stats.total_calls,
            "total_input_tokens": self.stats.total_tokens.input_tokens,
            "total_output_tokens": self.stats.total_tokens.output_tokens,
            "total_tokens": self.stats.total_tokens.total_tokens,
            "estimated_cost_usd": round(self.stats.total_cost, 4),
            "errors": self.stats.errors,
        }

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]