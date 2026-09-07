from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

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

from furrow.config import Provider, Settings, settings

_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    httpx.TimeoutException,
    ConnectionError,
    TimeoutError,
)


def _retry_complete() -> Any:
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
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

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, timeout)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, timeout)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, timeout)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_retry_complete()
    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, timeout: float
    ) -> str:
        response = await asyncio.wait_for(
            self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout,
        )
        if not response.content:
            return ""
        return response.content[0].text

    @_retry_complete()
    async def _complete_openai(
        self, prompt: str, system: str, model: str, timeout: float
    ) -> str:
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
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        return choices[0].message.content or ""

    @_retry_complete()
    async def _complete_ollama(
        self, prompt: str, system: str, model: str, timeout: float
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={"model": model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        return message.get("content") or ""

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
