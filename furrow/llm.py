from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from structlog import get_logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = get_logger(__name__)

# Exceptions that indicate a transient failure which can be retried.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    TimeoutError,
)

# Module-level retry decorator for LLM API calls (applied below).
_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)


class LLMClient:
    """Async LLM client with provider dispatch, retry, and logging."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
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
        """Ollama is OpenAI-compatible; we use OpenAI client with custom base URL."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",  # Ollama ignores the key but the client requires one
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm_complete", provider=str(self.settings.provider), model=model)

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            client = self.openai if self.settings.provider == Provider.OPENAI else self.ollama
            return await self._complete_openai(prompt, system, model, client)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_retry
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns typed content blocks; extract text from the first one
        return cast("str", getattr(response.content[0], "text", ""))

    @_retry
    async def _complete_openai(
        self, prompt: str, system: str, model: str, client: AsyncOpenAI
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
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
