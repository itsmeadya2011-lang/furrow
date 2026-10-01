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
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

log = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv(
                "ANTHROPIC_API_KEY"
            )
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
                timeout=60,
            )
        return self._httpx

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        log.debug(
            "llm_complete",
            provider=self.settings.provider,
            model=model,
            prompt_length=len(prompt),
        )

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def _retry_decorator(self) -> Any:
        """Build a tenacity retry decorator appropriate for LLM API calls."""
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (anthropic.APIError, openai.APIError, httpx.HTTPError, ValueError)
            ),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str
    ) -> str:
        @self._retry_decorator()
        async def _call() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return await _call()

    async def _complete_openai(
        self, prompt: str, system: str, model: str
    ) -> str:
        @self._retry_decorator()
        async def _call() -> str:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system
                        or "You are a helpful coding assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        return await _call()

    async def _complete_ollama(
        self, prompt: str, system: str, model: str
    ) -> str:
        @self._retry_decorator()
        async def _call() -> str:
            response = await self.httpx_client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system
                            or "You are a helpful coding assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                raise ValueError(
                    f"Ollama returned empty response: {data}"
                )
            return content

        try:
            return await _call()
        finally:
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
