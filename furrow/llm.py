from __future__ import annotations

import logging
import os
from pathlib import Path

import aiofiles
import anthropic as _anthropic
import openai as _openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
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
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY is not set")
                raise ValueError("ANTHROPIC_API_KEY is not set")
            logger.debug("Creating AsyncAnthropic client")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY is not set")
                raise ValueError("OPENAI_API_KEY is not set")
            logger.debug("Creating AsyncOpenAI client")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                _anthropic.APIConnectionError,
                _anthropic.RateLimitError,
                _anthropic.APIStatusError,
                _openai.APIConnectionError,
                _openai.RateLimitError,
                _openai.APITimeoutError,
                _openai.APIStatusError,
                ConnectionError,
                TimeoutError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug(
            "LLM API call",
            provider=self.settings.provider.value,
            model=model,
            prompt_length=len(prompt),
            prompt_preview=prompt[:200] if prompt else "",
        )
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            logger.error("Unsupported provider", provider=self.settings.provider)
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            logger.debug("Calling Anthropic API", model=model)
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            logger.debug(
                "Anthropic API response",
                model=model,
                response_length=len(text),
                response_preview=text[:200] if text else "",
            )
            return text
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e), model=model)
            raise

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            logger.debug("Calling OpenAI API", model=model)
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                "OpenAI API response",
                model=model,
                response_length=len(text),
                response_preview=text[:200] if text else "",
            )
            return text
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e), model=model)
            raise

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        logger.debug("Reading file", path=str(path))
        p = Path(path)
        content = ""
        async with aiofiles.open(p, "r") as f:
            content = await f.read()
        logger.debug("File read completed", path=str(path), length=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        logger.debug("Writing file", path=str(path), length=len(content))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.debug("File write completed", path=str(path))

    def list_files(self, directory: str | Path) -> list[str]:
        logger.debug("Listing files", directory=str(directory))
        p = Path(directory)
        if not p.exists():
            logger.debug("Directory does not exist", directory=str(directory))
            return []
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        logger.debug("Listed files", directory=str(directory), count=len(files))
        return files
