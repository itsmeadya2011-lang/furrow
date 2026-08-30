from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

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

    def _get_timeout(self) -> float:
        return float(self.settings.request_timeout)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, temperature, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, temperature: float, max_tokens: int) -> str:
        retryable = (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            ConnectionError,
            asyncio.TimeoutError,
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.settings.retry_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self.anthropic.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system or "You are a helpful coding assistant.",
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=self._get_timeout(),
                )
                return response.content[0].text
            except retryable as exc:
                last_exc = exc
                if attempt >= self.settings.retry_attempts:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1) * self.settings.retry_backoff, 30))
        raise RuntimeError(
            f"Anthropic completion failed for model {model} after {attempt} attempts: {last_exc}"
        ) from last_exc

    async def _complete_openai(self, prompt: str, system: str, model: str, temperature: float, max_tokens: int) -> str:
        retryable = (
            openai.RateLimitError,
            openai.APIStatusError,
            ConnectionError,
            asyncio.TimeoutError,
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.settings.retry_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self.openai.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": system or "You are a helpful coding assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    ),
                    timeout=self._get_timeout(),
                )
                return response.choices[0].message.content or ""
            except retryable as exc:
                last_exc = exc
                if attempt >= self.settings.retry_attempts:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1) * self.settings.retry_backoff, 30))
        raise RuntimeError(
            f"OpenAI completion failed for model {model} after {attempt} attempts: {last_exc}"
        ) from last_exc

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
