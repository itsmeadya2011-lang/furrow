from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import anthropic
import openai
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as _settings

logger = structlog.get_logger("furrow.llm")

_DEFAULT_LLM_TIMEOUT = 90.0
# Transient errors worth retrying: timeouts, network/API errors from either provider.
_TRANSIENT_ERRORS = (
    asyncio.TimeoutError,
    anthropic.APIError,
    openai.APIError,
)


def _before_sleep(retry_state) -> None:
    """Log each retry attempt (invoked by tenacity before sleeping between attempts)."""
    exc = retry_state.outcome.exception()
    logger.warning(
        "llm_retry",
        attempt=retry_state.attempt_number,
        error=repr(exc) if exc is not None else None,
    )


async def _with_timeout(awaitable, seconds: float):
    """Wrap an awaitable with a timeout guard.

    Uses ``asyncio.timeout`` when available (Python 3.11+) and falls back to
    ``asyncio.wait_for`` on older interpreters, so the guard works across the
    supported ``requires-python`` range (>=3.10).
    """
    if hasattr(asyncio, "timeout"):
        async with asyncio.timeout(seconds):
            return await awaitable
    return await asyncio.wait_for(awaitable, timeout=seconds)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _settings
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.info(
            "llm_request_start",
            provider=provider.value,
            model=model,
            prompt_length=len(prompt),
        )
        try:
            if provider == Provider.ANTHROPIC:
                text = await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                text = await self._complete_openai(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception:
            logger.exception("llm_request_failure", provider=provider.value, model=model)
            raise
        logger.info(
            "llm_request_success",
            provider=provider.value,
            model=model,
            response_length=len(text),
        )
        return text

    @tenacity.retry(
        wait=tenacity.wait_exponential(),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        retry=tenacity.retry_if_exception_type(_TRANSIENT_ERRORS),
        before_sleep=_before_sleep,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        timeout = getattr(self.settings, "timeout", _DEFAULT_LLM_TIMEOUT)
        response = await _with_timeout(
            self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout,
        )
        return response.content[0].text

    @tenacity.retry(
        wait=tenacity.wait_exponential(),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        retry=tenacity.retry_if_exception_type(_TRANSIENT_ERRORS),
        before_sleep=_before_sleep,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        timeout = getattr(self.settings, "timeout", _DEFAULT_LLM_TIMEOUT)
        response = await _with_timeout(
            self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout,
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
