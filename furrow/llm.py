from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiofiles  # type: ignore[import-untyped]
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI  # type: ignore[import-untyped]
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

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
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            AnthropicRateLimitError,
            OpenAIRateLimitError,
            asyncio.TimeoutError,
            Exception,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        # Extract text from the first content block, handling the union type
        first_block = response.content[0]
        if hasattr(first_block, "text"):
            return first_block.text  # type: ignore[no-any-return]
        return str(first_block)  # type: ignore[no-any-return]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            AnthropicRateLimitError,
            OpenAIRateLimitError,
            asyncio.TimeoutError,
            Exception,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            timeout=60,
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            AnthropicRateLimitError,
            OpenAIRateLimitError,
            asyncio.TimeoutError,
            Exception,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            timeout=60,
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