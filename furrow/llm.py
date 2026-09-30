from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

logger = structlog.get_logger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_EXC_TYPES = (
    httpx.TimeoutException,
    httpx.TransportError,
)


def _is_transient(exc: BaseException) -> bool:
    """Return True when *exc* is a transient, retryable failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        return status in _TRANSIENT_STATUS_CODES
    if isinstance(exc, _TRANSIENT_EXC_TYPES):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
    if isinstance(
        exc,
        (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.OverloadedError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
    return False


def _should_retry(retry_state: RetryCallState) -> bool:
    outcome = retry_state.outcome
    if outcome is None:
        return False
    exc = outcome.exception()
    if exc is None:
        return False
    return _is_transient(exc)


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    logger.warning(
        "llm_retry_attempt",
        attempt=retry_state.attempt_number,
        upcoming_sleep=round(retry_state.upcoming_sleep, 2),
        error=repr(exc),
    )


_retry = retry(
    retry=_should_retry,
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(60.0),
            )
        return self._httpx

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        if model is None:
            if self.settings.provider == Provider.OLLAMA:
                model = self.settings.ollama_model
            else:
                model = self.settings.model
        log = logger.bind(provider=self.settings.provider.value, model=model)
        log.info("llm_complete_start", prompt_len=len(prompt))
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            elif self.settings.provider == Provider.OLLAMA:
                result = await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
            log.info("llm_complete_success", response_len=len(result))
            return result
        except Exception:
            log.exception("llm_complete_error")
            raise

    @_retry
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        content_block = response.content[0]
        if isinstance(content_block, TextBlock):
            return content_block.text
        return ""

    @_retry
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_retry
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        response = await self.httpx_client.post("api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["response"]

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        try:
            async with aiofiles.open(p, "r") as f:
                return await f.read()
        except FileNotFoundError:
            logger.error("read_file_not_found", path=str(p))
            raise FileNotFoundError(f"File not found: {p}") from None
        except OSError as e:
            logger.error("read_file_error", path=str(p), error=str(e))
            raise

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        log = logger.bind(directory=str(directory))
        log.info("list_files_start")

        def _scan() -> list[str]:
            p = Path(directory)
            if not p.exists():
                log.warning("list_files_not_found", directory=str(p))
                return []
            return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

        try:
            return await asyncio.to_thread(_scan)
        except OSError as e:
            log.error("list_files_error", error=str(e))
            raise
