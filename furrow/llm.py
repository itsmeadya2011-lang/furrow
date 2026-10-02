from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

# Exceptions that warrant a retry (transient network/API issues)
_RETRYABLE_EXCEPTIONS = (
    anthropic.AnthropicError,
    openai.APIError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    ConnectionError,
    TimeoutError,
)

# Default timeout for LLM API calls (seconds)
_DEFAULT_LLM_TIMEOUT = 120


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

    @property
    def ollama(self) -> AsyncOpenAI:
        """Return an OpenAI-compatible client pointed at the Ollama server."""
        if self._openai is None:
            self._openai = AsyncOpenAI(
                api_key="ollama",
                base_url=self.settings.ollama_base_url + "/v1/",
            )
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        timeout = self.settings.llm_timeout
        max_retries = self.settings.max_retries

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                if self.settings.provider == Provider.ANTHROPIC:
                    return await self._complete_anthropic(prompt, system, model, timeout)
                elif self.settings.provider == Provider.OPENAI:
                    return await self._complete_openai(prompt, system, model, timeout)
                elif self.settings.provider == Provider.OLLAMA:
                    return await self._complete_ollama(prompt, system, model, timeout)
                else:
                    raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, timeout: int) -> str:
        response = await asyncio.wait_for(
            self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout,
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str, timeout: int) -> str:
        response = await asyncio.wait_for(
            self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str, timeout: int) -> str:
        response = await asyncio.wait_for(
            self.ollama.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=timeout,
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
