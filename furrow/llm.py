from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
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

logger = structlog.get_logger("furrow")


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http = httpx.AsyncClient(base_url=self.settings.ollama_base_url)

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
        logger.info("completion_request", provider=str(self.settings.provider), model=model)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=2, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                if self.settings.provider == Provider.ANTHROPIC:
                    result = await self._complete_anthropic(prompt, system, model)
                elif self.settings.provider == Provider.OPENAI:
                    result = await self._complete_openai(prompt, system, model)
                elif self.settings.provider == Provider.OLLAMA:
                    result = await self._complete_ollama(prompt, system, model)
                else:
                    raise ValueError(f"Unsupported provider: {self.settings.provider}")

        logger.info("completion_response", provider=str(self.settings.provider), length=len(result))
        return result

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self._http.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def edit_file(self, path: str | Path, old: str, new: str) -> None:
        p = Path(path)
        content = await self.read_file(p)
        if old not in content:
            raise ValueError(f"String not found in file: {old[:80]}")
        content = content.replace(old, new, 1)
        await self.write_file(p, content)

    async def delete_file(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
