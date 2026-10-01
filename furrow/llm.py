from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
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

logger = structlog.get_logger(__name__)


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

    def _retry_config(self) -> dict[str, Any]:
        return {
            "retry": retry_if_exception_type((anthropic.AnthropicError, openai.OpenAIError)),
            "stop": stop_after_attempt(self.settings.max_retries),
            "wait": wait_exponential(multiplier=self.settings.retry_delay, min=1, max=30),
        }

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("complete_anthropic", model=model)
        system_prompt = system or "You are a helpful coding assistant."

        async for attempt in AsyncRetrying(**self._retry_config()):
            with attempt:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=self.settings.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.settings.timeout,
                )
                return response.content[0].text
        logger.error("complete_anthropic_exhausted", model=model)
        raise RuntimeError(
            "Failed to get response from Anthropic after"
            f" {self.settings.max_retries} attempts"
        )

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("complete_openai", model=model)
        system_prompt = system or "You are a helpful coding assistant."

        async for attempt in AsyncRetrying(**self._retry_config()):
            with attempt:
                response = await self.openai.chat.completions.create(
                    model=model,
                    max_tokens=self.settings.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=self.settings.timeout,
                )
                return response.choices[0].message.content or ""
        logger.error("complete_openai_exhausted", model=model)
        raise RuntimeError(
            "Failed to get response from OpenAI after"
            f" {self.settings.max_retries} attempts"
        )

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
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

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the workspace directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.settings.workspace / p).resolve()
