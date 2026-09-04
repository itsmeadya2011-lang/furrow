from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import RetryError, retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.BoundLogger,
    cache_logger_on_first_use=True,
)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, RetryError):
        return False
    if isinstance(exc, anthropic.APIStatusError):
        return getattr(exc, "status_code", None) in {429, 500, 503}
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", None) in {429, 500, 503}
    if isinstance(exc, openai.APIConnectionError):
        return True
    return False


llm_retry = retry(
    retry=retry_if_exception(_should_retry),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: AsyncOpenAI | None = None

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

    @property
    def _ollama(self) -> AsyncOpenAI:
        if self._ollama_client is None:
            self._ollama_client = AsyncOpenAI(
                base_url=self.settings.ollama_base_url.rstrip("/") + "/v1",
                api_key="ollama",
            )
        return self._ollama_client

    @_ollama.setter
    def _ollama(self, value: AsyncOpenAI | None) -> None:
        self._ollama_client = value

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @llm_retry
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.info("llm_call_anthropic", model=model, prompt_length=len(prompt))
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
                timeout=self.settings.request_timeout,
            )
            return response.content[0].text
        except Exception as exc:
            logger.error(
                "llm_call_anthropic_failed", model=model, exc_type=type(exc).__name__
            )
            raise

    @llm_retry
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.info("llm_call_openai", model=model, prompt_length=len(prompt))
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.settings.request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(
                "llm_call_openai_failed", model=model, exc_type=type(exc).__name__
            )
            raise

    @llm_retry
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        logger.info("llm_call_ollama", model=model, prompt_length=len(prompt))
        try:
            response = await self._ollama.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.settings.request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(
                "llm_call_ollama_failed", model=model, exc_type=type(exc).__name__
            )
            raise

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
