from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None

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
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

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
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((AnthropicRateLimitError, ConnectionError, TimeoutError)),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            stop=stop_after_attempt(3),
        ):
            with attempt:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system or "You are a helpful coding assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.content:
                    return response.content[0].text
                raise ValueError("Empty response from Anthropic API")
        raise RuntimeError("Failed to get response from Anthropic after retries")

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((OpenAIRateLimitError, ConnectionError, TimeoutError)),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            stop=stop_after_attempt(3),
        ):
            with attempt:
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                raise ValueError("Empty response from OpenAI API")
        raise RuntimeError("Failed to get response from OpenAI after retries")

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete using Ollama's OpenAI-compatible API."""
        base_url = self.settings.ollama_base_url.rstrip("/")
        # Use OpenAI client pointed at Ollama for compatibility
        if self._openai is None or getattr(self._openai, "_base_url", None) != base_url:
            self._openai = AsyncOpenAI(base_url=base_url, api_key="ollama")
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            stop=stop_after_attempt(3),
        ):
            with attempt:
                response = await self._openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                raise ValueError("Empty response from Ollama API")
        raise RuntimeError("Failed to get response from Ollama after retries")

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
        """Clean up HTTP clients."""
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None
        if self._openai is not None:
            await self._openai.close()
            self._openai = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
