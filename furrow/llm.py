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
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
            logger.debug("anthropic_client_initialized")
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
            logger.debug("openai_client_initialized")
        return self._openai

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, openai.APIError)),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.info("llm_complete_started", provider=self.settings.provider.value, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("llm_complete_finished", provider="anthropic", model=model)
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        logger.info("llm_complete_finished", provider="openai", model=model)
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        logger.info("llm_complete_finished", provider="ollama", model=model, base_url=self.settings.ollama_base_url)
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        logger.debug("reading_file", path=str(path))
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        logger.debug("read_file_complete", path=str(path), length=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("writing_file", path=str(p), length=len(content))
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.debug("write_file_complete", path=str(p))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            logger.debug("list_files_directory_missing", directory=str(p))
            return []
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        logger.debug("list_files_complete", directory=str(p), count=len(files))
        return files
