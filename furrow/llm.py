from __future__ import annotations

import asyncio
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

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._httpx

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        timeout: float | None = None,
    ) -> str:
        model = model or self.settings.model
        timeout = timeout or self.settings.completion_timeout
        logger.debug(
            "llm_complete",
            provider=self.settings.provider.value,
            model=model,
            timeout=timeout,
        )

        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await asyncio.wait_for(
                    self._complete_anthropic(prompt, system, model),
                    timeout=timeout,
                )
            elif self.settings.provider == Provider.OPENAI:
                result = await asyncio.wait_for(
                    self._complete_openai(prompt, system, model),
                    timeout=timeout,
                )
            elif self.settings.provider == Provider.OLLAMA:
                result = await asyncio.wait_for(
                    self._complete_ollama(prompt, system, model),
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
        except asyncio.TimeoutError:
            logger.error("llm_timeout", model=model, timeout=timeout)
            raise

        logger.debug("llm_complete_done", model=model, response_len=len(result))
        return result

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.httpx.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system or DEFAULT_SYSTEM_PROMPT,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

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
        """Clean up resources, e.g. close HTTP clients."""
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None
