from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

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

T = TypeVar("T")

_RETRYABLE_EXCEPTIONS = (
    anthropic.APIStatusError,
    openai.APIStatusError,
    asyncio.TimeoutError,
)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (e.g. ```json ... ```) and trim whitespace."""
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


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
            self._ollama = AsyncOpenAI(
                base_url=f"{self.settings.ollama_base_url}/v1",
                api_key="ollama",
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            text = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            text = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            text = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        text = strip_code_fences(text)
        if not text:
            raise ValueError("LLM response content is empty")
        return text

    async def _invoke(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run a provider call with tenacity retry and a per-call timeout."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            wait=wait_exponential(),
            stop=stop_after_attempt(self.settings.max_retries),
            reraise=True,
        ):
            with attempt:
                return await asyncio.wait_for(
                    call(), timeout=self.settings.request_timeout
                )

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._invoke(
            lambda: self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._invoke(
            lambda: self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self._invoke(
            lambda: self.ollama.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
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
