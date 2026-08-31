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
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings


def _safe_import_exceptions() -> tuple:
    exceptions: list = []
    try:
        from anthropic import APIConnectionError as _anth_api_conn

        exceptions.append(_anth_api_conn)
    except ImportError:
        pass
    try:
        from anthropic import RateLimitError as _anth_rate

        exceptions.append(_anth_rate)
    except ImportError:
        pass
    try:
        from anthropic import APITimeoutError as _anth_timeout

        exceptions.append(_anth_timeout)
    except ImportError:
        pass
    try:
        from openai import APIConnectionError as _openai_api_conn

        exceptions.append(_openai_api_conn)
    except ImportError:
        pass
    try:
        from openai import RateLimitError as _openai_rate

        exceptions.append(_openai_rate)
    except ImportError:
        pass
    try:
        from openai import APITimeoutError as _openai_timeout

        exceptions.append(_openai_timeout)
    except ImportError:
        pass
    return tuple(exceptions)


RETRY_EXCEPTIONS = _safe_import_exceptions()

RETRY_ATTEMPTS = getattr(settings, "retry_attempts", 3)


def _retry_decorator():
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        reraise=True,
    )


class LLMClient:
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
        if self._ollama is None:
            base_url = self.settings.ollama_base_url
            if not base_url:
                raise ValueError("ollama_base_url is not set")
            self._ollama = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama")
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

    @_retry_decorator()
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        max_tokens = getattr(self.settings, "max_tokens", 4096)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @_retry_decorator()
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        max_tokens = getattr(self.settings, "max_tokens", 4096)
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_retry_decorator()
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        max_tokens = getattr(self.settings, "max_tokens", 4096)
        response = await self.ollama.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
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
