from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings as default_settings

logger = structlog.get_logger(__name__)


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

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None, max_tokens: int | None = None) -> str:
        model = model or self.settings.model
        tokens = max_tokens or self.settings.max_tokens
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        logger.info("llm_request", provider="anthropic", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        logger.info("llm_request", provider="openai", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        logger.info("llm_request", provider="ollama", model=model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=120) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def read_file_lines(self, path: str | Path, max_lines: int = 100) -> str:
        """Read a file, truncating to max_lines for large files."""
        async with aiofiles.open(path, "r") as f:
            lines = []
            async for line in f:
                lines.append(line)
                if len(lines) >= max_lines:
                    break
        return "".join(lines)

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
