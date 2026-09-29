from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger("furrow.llm")

# Exceptions that indicate transient failures worth retrying
_RETRYABLE_EXC = (
    anthropic.AnthropicError,
    openai.APIError,
    openai.APIConnectionError,
    openai.RateLimitError,
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("LLM.complete", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def _retry_config(self):
        """Build a tenacity retry config from settings."""
        return list(
            AsyncRetrying(
                reraise=True,
                retry=retry_if_exception_type(_RETRYABLE_EXC),
                stop=stop_after_attempt(self.settings.retry_max_attempts),
                wait=wait_exponential(
                    multiplier=self.settings.retry_wait_min,
                    max=self.settings.retry_wait_max,
                ),
            )
        )

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        retry_chain = self._retry_config()
        retry_chain.append(self._do_anthropic(prompt, system, model))
        async for attempt in retry_chain:
            if isinstance(attempt, Exception):
                logger.warning("Anthropic retry exhausted", error=str(attempt))
                raise attempt
            return attempt

    async def _do_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        retry_chain = self._retry_config()
        retry_chain.append(self._do_openai(prompt, system, model))
        async for attempt in retry_chain:
            if isinstance(attempt, Exception):
                logger.warning("OpenAI retry exhausted", error=str(attempt))
                raise attempt
            return attempt

    async def _do_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

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
