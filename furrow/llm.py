from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import NotFoundError as AnthropicNotFoundError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import NotFoundError as OpenAINotFoundError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings

_RETRYABLE = (
    AnthropicRateLimitError,
    AnthropicAPIStatusError,
    OpenAIRateLimitError,
    OpenAIAPIStatusError,
    asyncio.TimeoutError,
    ConnectionError,
)


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

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens=max_tokens, temperature=temperature)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens=max_tokens, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0,
                **({"temperature": temperature} if temperature is not None else {}),
            )
            return response.content[0].text
        except Exception as e:
            if isinstance(e, _RETRYABLE):
                raise
            raise RuntimeError(f"Anthropic API call failed for model {model}: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=30.0,
                max_tokens=max_tokens or 4096,
                **({"temperature": temperature} if temperature is not None else {}),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if isinstance(e, _RETRYABLE):
                raise
            raise RuntimeError(f"OpenAI API call failed for model {model}: {e}") from e

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

    async def complete_with_files(
        self,
        prompt: str,
        files: list[str | Path],
        model: str | None = None,
        system: str = "",
    ) -> str:
        parts = [prompt, "\n--- Provided files ---"]
        for f in files:
            try:
                content = await self.read_file(f)
            except (FileNotFoundError, OSError):
                continue
            parts.append(f"\n---FILE: {f}---\n{content}\n---END FILE---")
        full_prompt = "\n".join(parts)
        return await self.complete(full_prompt, system=system, model=model)
