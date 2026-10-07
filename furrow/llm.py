from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import anthropic as anthropic_lib
import httpx
import openai as openai_lib
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    
    def add(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += cost


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
            self._ollama_client = httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=120.0)
        return self._ollama_client

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Rough cost estimates (per 1M tokens)
        costs = {
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
            "claude-3-5-haiku-20241022": (0.25, 1.25),
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        }
        input_cost, output_cost = costs.get(model, (0.0, 0.0))
        return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((
            anthropic_lib.RateLimitError, 
            openai_lib.RateLimitError, 
            httpx.HTTPError, 
            Exception
        ))
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm_request", model=model, provider=self.settings.provider.value, prompt_len=len(prompt))
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
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        self.usage.add(input_tokens, output_tokens, cost)
        logger.info("llm_usage", model=model, input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
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
        usage = response.usage
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            self.usage.add(input_tokens, output_tokens, cost)
            logger.info("llm_usage", model=model, input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.ollama.post("/api/chat", json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 4096}
        })
        response.raise_for_status()
        data = response.json()
        # OLLAMA doesn't provide token counts easily, estimate
        content = data.get("message", {}).get("content", "")
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
