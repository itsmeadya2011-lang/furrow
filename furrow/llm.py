from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic as anthropic_sdk
import openai as openai_sdk
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings


class LLMError(Exception):
    """Raised when an LLM call fails after retries."""


# Exception types that should trigger a retry. Defined at module level so the
# @retry decorator can reference them without colliding with instance properties.
_RETRYABLE = (
    anthropic_sdk.APIStatusError,
    anthropic_sdk.APIConnectionError,
    openai_sdk.APIError,
    LLMError,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: Any = None

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
    def ollama(self) -> Any:
        if self._ollama is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("openai is required for Ollama support") from e
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key=self.settings.ollama_api_key or "ollama",
            )
        return self._ollama

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise LLMError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic_sdk.APIStatusError, anthropic_sdk.APIConnectionError) as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        if not response.content:
            raise LLMError("Anthropic returned empty content")
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except openai_sdk.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e
        if not response.choices:
            raise LLMError("OpenAI returned empty choices")
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.ollama.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except openai_sdk.APIError as e:
            raise LLMError(f"Ollama API error: {e}") from e
        if not response.choices:
            raise LLMError("Ollama returned empty choices")
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
        root = p.resolve()
        result: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            try:
                result.append(str(f.resolve().relative_to(root)))
            except ValueError:
                # File is not under the directory (symlink escape); skip it.
                continue
        return result