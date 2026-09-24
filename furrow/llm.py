from __future__ import annotations

import os
import time
from pathlib import Path

import aiofiles
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

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
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        start_time = time.time()

        logger.info(
            "llm_request_start",
            provider=self.settings.provider.value,
            model=model,
            prompt_length=len(prompt),
        )

        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")

            elapsed = time.time() - start_time
            logger.info(
                "llm_request_success",
                provider=self.settings.provider.value,
                model=model,
                elapsed=elapsed,
                response_length=len(result),
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "llm_request_failed",
                provider=self.settings.provider.value,
                model=model,
                elapsed=elapsed,
                error=str(e),
            )
            raise

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of content blocks, we want the text from the first text block
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

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
        logger.debug("file_read", path=str(path))
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
            logger.debug("file_read_success", path=str(path), size=len(content))
            return content
        except FileNotFoundError:
            logger.debug("file_read_not_found", path=str(path))
            return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        logger.info("file_write", path=str(p), size=len(content))
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.info("file_write_success", path=str(p))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
