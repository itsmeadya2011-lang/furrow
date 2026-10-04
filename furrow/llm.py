from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

log = structlog.get_logger()

DEFAULT_SYSTEM = "You are a helpful coding assistant."

RETRYABLE_ERRORS = (
    httpx.TransportError,
    httpx.TimeoutException,
    ConnectionError,
    TimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _log_before_sleep(retry_state: Any) -> None:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None and outcome.failed else None
    log.warning(
        "llm.retry",
        attempt=retry_state.attempt_number,
        error=str(exc) if exc is not None else "unknown",
        error_type=type(exc).__name__ if exc is not None else "unknown",
        sleep_for=round(retry_state.next_action.sleep, 2) if retry_state.next_action else None,
    )


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
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            base_url = str(self.settings.ollama_base_url).rstrip("/") + "/v1"
            self._ollama = AsyncOpenAI(base_url=base_url, api_key="ollama")
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        log.debug(
            "llm.request",
            provider=provider.value,
            model=model,
            prompt_chars=len(prompt),
            system_chars=len(system),
        )
        started = time.perf_counter()
        try:
            if provider == Provider.ANTHROPIC:
                text = await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                text = await self._complete_openai(prompt, system, model)
            elif provider == Provider.OLLAMA:
                text = await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as exc:
            log.error(
                "llm.error",
                provider=provider.value,
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed=round(time.perf_counter() - started, 3),
            )
            raise
        log.info(
            "llm.completed",
            provider=provider.value,
            model=model,
            elapsed=round(time.perf_counter() - started, 3),
            chars=len(text),
        )
        return text

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in response.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if not parts:
            raise ValueError("Anthropic returned empty content")
        return "".join(parts)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

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
