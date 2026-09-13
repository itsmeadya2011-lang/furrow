from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import openai
import tenacity
from anthropic import AsyncAnthropic
from anthropic import APIError as AnthropicAPIError
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError

from furrow.config import Provider, Settings, settings

logger = logging.getLogger(__name__)

# Exceptions that are worth retrying (transient network / rate-limit / server errors)
_RETRYABLE = (
    AnthropicAPIError,
    OpenAIAPIError,
    ConnectionError,
    TimeoutError,
)


class LLMClient:
    """Unified async client for Anthropic, OpenAI, and Ollama (OpenAI-compatible)."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    # -- Provider clients ------------------------------------------------

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
        """Ollama uses an OpenAI-compatible API, so we reuse AsyncOpenAI."""
        if self._ollama is None:
            # Ollama ignores the api_key but OpenAI client requires one
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

    # -- Completion ------------------------------------------------------

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        """Send a prompt to the configured LLM provider and return the text response.

        Retries on transient errors (network, rate limits, 5xx) up to 3 times
        with exponential backoff.
        """
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug("Completing with provider=%s model=%s", provider, model)

        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(
        self, prompt: str, system: str, model: str
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self, prompt: str, system: str, model: str
    ) -> str:
        """Ollama uses an OpenAI-compatible API but with its own base URL."""
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # -- File utilities --------------------------------------------------

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
