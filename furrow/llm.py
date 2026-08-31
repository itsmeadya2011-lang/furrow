from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None
        self._retry_config = {
            "retry": retry_if_exception_type((openai.APITimeoutError, openai.APIError, anthropic.APITimeoutError, anthropic.APIError, httpx.HTTPError)),
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=1, max=10),
        }

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(60.0),
            )
        return self._httpx

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        log = logger.bind(provider="anthropic", model=model)
        log.debug("request_start")
        async for attempt in AsyncRetrying(**self._retry_config):
            with attempt:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system or "You are a helpful coding assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
        result = response.content[0].text
        log.debug("request_complete", response_len=len(result))
        return result

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        log = logger.bind(provider="openai", model=model)
        log.debug("request_start")
        async for attempt in AsyncRetrying(**self._retry_config):
            with attempt:
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
        result = response.choices[0].message.content or ""
        log.debug("request_complete", response_len=len(result))
        return result

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        log = logger.bind(provider="ollama", model=model)
        log.debug("request_start")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }
        async for attempt in AsyncRetrying(**self._retry_config):
            with attempt:
                response = await self.httpx_client.post(
                    "/api/generate", json=payload
                )
                response.raise_for_status()
                data = response.json()
        result = data.get("response", "")
        log.debug("request_complete", response_len=len(result))
        return result

    async def aclose(self) -> None:
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None

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
