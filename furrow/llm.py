from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import threading
from anthropic import AsyncAnthropic
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAIAPITimeoutError

from furrow.config import Provider, Settings, settings


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._anthropic_lock = threading.Lock()
        self._openai_lock = threading.Lock()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            with self._anthropic_lock:
                if self._anthropic is None:
                    try:
                        api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
                        if not api_key:
                            raise ValueError("ANTHROPIC_API_KEY is not set")
                        self._anthropic = AsyncAnthropic(api_key=api_key, timeout=120.0)
                    except Exception as exc:
                        raise RuntimeError("Failed to initialize Anthropic client") from exc
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            with self._openai_lock:
                if self._openai is None:
                    try:
                        api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            raise ValueError("OPENAI_API_KEY is not set")
                        self._openai = AsyncOpenAI(api_key=api_key, timeout=120.0)
                    except Exception as exc:
                        raise RuntimeError("Failed to initialize OpenAI client") from exc
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicAPIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc
        except AnthropicAPITimeoutError as exc:
            raise RuntimeError(f"Anthropic API timeout: {exc}") from exc
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except OpenAIAPIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc
        except OpenAIAPITimeoutError as exc:
            raise RuntimeError(f"OpenAI API timeout: {exc}") from exc
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except OSError as exc:
            raise RuntimeError(f"Failed to read file {path}: {exc}") from exc

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)
        except OSError as exc:
            raise RuntimeError(f"Failed to write file {path}: {exc}") from exc

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
