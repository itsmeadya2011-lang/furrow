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
from furrow.logging import get_logger

logger = get_logger(__name__)


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
            if self.settings.provider == Provider.OLLAMA:
                self._openai = AsyncOpenAI(
                    base_url=self.settings.ollama_base_url,
                    api_key="ollama",
                )
            else:
                api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not set")
                self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm.complete.started", provider=self.settings.provider.value, model=model, prompt_length=len(prompt))
        if self.settings.provider == Provider.ANTHROPIC:
            result = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            result = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            result = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        logger.debug("llm.complete.completed", provider=self.settings.provider.value, model=model, response_length=len(result))
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (anthropic.APIStatusError, openai.APIStatusError, httpx.HTTPError, asyncio.TimeoutError)
        ),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (anthropic.APIStatusError, openai.APIStatusError, httpx.HTTPError, asyncio.TimeoutError)
        ),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (anthropic.APIStatusError, openai.APIStatusError, httpx.HTTPError, asyncio.TimeoutError)
        ),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        path = Path(path)
        logger.debug("llm.file.read", path=str(path))
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        logger.debug("llm.file.read_completed", path=str(path), size=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        logger.debug("llm.file.write", path=str(p), size=len(content))
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.debug("llm.file.write_completed", path=str(p))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
