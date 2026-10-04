from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from anthropic.types import Message
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

MAX_TOKENS = 8192
TEMPERATURE = 0.2

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APIStatusError,
    openai.APIConnectionError,
    openai.APIStatusError,
    httpx.HTTPError,
    asyncio.TimeoutError,
)

IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        ".tox",
        ".eggs",
        "*.egg-info",
    }
)


def _is_ignored(relative_path: Path) -> bool:
    return any(
        fnmatch.fnmatch(part, pattern)
        for part in relative_path.parts
        for pattern in IGNORED_DIRECTORIES
    )


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
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

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        client: Any = self.anthropic.messages
        response = await client.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_anthropic_text(response)

    @staticmethod
    def _extract_anthropic_text(response: Message) -> str:
        text_parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        text = "".join(text_parts)
        if not text and response.content:
            first = getattr(response.content[0], "text", "")
            return first if isinstance(first, str) else ""
        return text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            content = await f.read()
            return content if isinstance(content, str) else ""

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        root = Path(directory)
        if not root.exists():
            return []
        files: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root)
            if _is_ignored(relative_path):
                continue
            files.append(str(relative_path))
        return files
