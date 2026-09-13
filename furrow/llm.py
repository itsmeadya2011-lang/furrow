from __future__ import annotations

import logging
import time
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
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import structlog

from furrow.config import Provider, Settings, settings

std_logger = logging.getLogger("furrow")
logger = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._logger = logger.bind(component="llm")

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
        self._logger.info(
            "llm_call_start",
            provider=self.settings.provider.value,
            model=model,
            prompt_length=len(prompt),
        )
        start = time.monotonic()
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
            elapsed = time.monotonic() - start
            self._logger.info(
                "llm_call_complete",
                provider=self.settings.provider.value,
                model=model,
                duration_seconds=round(elapsed, 3),
                response_length=len(result),
            )
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            self._logger.error(
                "llm_call_error",
                provider=self.settings.provider.value,
                model=model,
                duration_seconds=round(elapsed, 3),
                error=str(e),
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(std_logger, logging.INFO),
        reraise=True,
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
        before_sleep=before_sleep_log(std_logger, logging.INFO),
        reraise=True,
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
        """Close underlying LLM client connections if they exist."""
        if self._anthropic is not None:
            self._anthropic = None
            self._logger.info("client_closed", provider="anthropic")
        if self._openai is not None:
            self._openai = None
            self._logger.info("client_closed", provider="openai")
