from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None


class TokenUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str, provider: Provider) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += self._estimate_cost(input_tokens, output_tokens, model, provider)

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str, provider: Provider) -> float:
        # Rough cost estimates per 1M tokens (as of 2024)
        costs = {
            Provider.ANTHROPIC: {
                "claude-sonnet-4": (3.00, 15.00),
                "claude-3-5-sonnet": (3.00, 15.00),
                "claude-3-5-haiku": (1.00, 5.00),
                "claude-3-opus": (15.00, 75.00),
            },
            Provider.OPENAI: {
                "gpt-4o": (5.00, 15.00),
                "gpt-4o-mini": (0.15, 0.60),
                "gpt-4-turbo": (10.00, 30.00),
            },
        }
        provider_costs = costs.get(provider, {})
        # Find best match
        for key, (in_cost, out_cost) in provider_costs.items():
            if key in model.lower():
                return (input_tokens * in_cost + output_tokens * out_cost) / 1_000_000
        return 0.0

    def summary(self) -> str:
        return f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | Est. cost: ${self.total_cost:.4f}"


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: httpx.AsyncClient | None = None
        self.usage = TokenUsage()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            if AsyncAnthropic is None:
                raise ValueError("anthropic package not installed")
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

    @property
    def ollama(self) -> httpx.AsyncClient:
        if self._ollama_client is None:
            self._ollama_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=120.0,
            )
        return self._ollama_client

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
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
        if response.usage:
            self.usage.add(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model,
                Provider.ANTHROPIC,
            )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        # Track usage
        if response.usage:
            self.usage.add(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                model,
                Provider.OPENAI,
            )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.ollama.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        # Ollama doesn't return token counts reliably, estimate roughly
        content = data.get("message", {}).get("content", "")
        self.usage.add(len(prompt) // 4, len(content) // 4, model, Provider.OLLAMA)
        return content

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

    async def close(self) -> None:
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None