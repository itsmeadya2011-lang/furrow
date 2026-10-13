from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import APIError as AnthropicAPIError, AsyncAnthropic
from openai import APIError as OpenAIAPIError, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((AnthropicAPIError, OpenAIAPIError, Exception)),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.info("llm_complete_start", provider=self.settings.provider.value, model=model, prompt_len=len(prompt))
        
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
            
            logger.info("llm_complete_success", model=model, response_len=len(result))
            return result
        except Exception as e:
            logger.error("llm_complete_failed", model=model, error=str(e))
            raise

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

    async def read_file(self, path: str | Path) -> str:
        logger.debug("read_file", path=str(path))
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        logger.debug("read_file_success", path=str(path), size=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        logger.debug("write_file", path=str(path), size=len(content))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.debug("write_file_success", path=str(path))

    def list_files(self, directory: str | Path) -> list[str]:
        logger.debug("list_files", directory=str(directory))
        p = Path(directory)
        if not p.exists():
            return []
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        logger.debug("list_files_success", directory=str(directory), count=len(files))
        return files
