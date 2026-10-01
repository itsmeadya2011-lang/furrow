from __future__ import annotations

import logging
import os
from pathlib import Path

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings

logger = logging.getLogger(__name__)

# Exception types that should trigger a retry on LLM calls.
_RETRYABLE_ERRORS = (
    anthropic.APIError,
    openai.APIError,
)

# Retry configuration shared across all provider implementations.
_RETRY_DECORATOR = retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)

_OLLAMA_RETRY = retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.RequestError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        from furrow.config import settings as default_settings

        self.settings = settings or default_settings
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
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_RETRY_DECORATOR
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # The first content block is typically a TextBlock with a .text attribute.
        for block in response.content:
            if block.type == "text":
                return block.text
        return response.content[0].text if hasattr(response.content[0], "text") else ""

    @_RETRY_DECORATOR
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_OLLAMA_RETRY
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt using an Ollama server.

        Ollama does not require an API key. The base URL is configurable
        via ``settings.ollama_base_url`` (default ``http://localhost:11434``).
        """
        model_name = model.replace("ollama/", "") if model.startswith("ollama/") else model
        async with httpx.AsyncClient(
            base_url=self.settings.ollama_base_url, timeout=120
        ) as http:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system or "You are a helpful coding assistant.",
                "stream": False,
            }
            resp = await http.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", ""))

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return str(await f.read())

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
