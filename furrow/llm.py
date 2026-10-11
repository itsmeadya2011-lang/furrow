from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import aiofiles
import anthropic
import httpx
import openai
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int) -> None:
        self.rate = requests_per_minute / 60.0  # requests per second
        self.tokens = float(requests_per_minute)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.tokens + elapsed * self.rate, self.rate * 60)
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                logger.debug("rate_limit_wait", wait_time=wait_time)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


def _retry_if_retryable_error(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exception, (anthropic.RateLimitError, openai.RateLimitError)):
        return True
    if isinstance(exception, (anthropic.APITimeoutError, openai.APITimeoutError)):
        return True
    if isinstance(exception, (anthropic.InternalServerError, openai.InternalServerError)):
        return True
    if isinstance(exception, httpx.TimeoutException):
        return True
    return bool(isinstance(exception, httpx.ConnectError))


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_base_url: str = settings.ollama_base_url
        self._rate_limiter = RateLimiter(settings.rate_limit_requests_per_minute)

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,  # We handle retries ourselves
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,  # We handle retries ourselves
            )
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm_complete", provider=self.settings.provider.value, model=model, prompt_length=len(prompt))
        
        await self._rate_limiter.acquire()
        
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_retry(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_retry(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_with_retry(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
        retry=tenacity.retry_if_exception(_retry_if_retryable_error),
        before_sleep=lambda retry_state: logger.warning(
            "llm_retry", attempt=retry_state.attempt_number, error=str(retry_state.outcome.exception() if retry_state.outcome is not None else "")
        ),
    )
    async def _complete_anthropic_with_retry(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_anthropic(prompt, system, model)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
        retry=tenacity.retry_if_exception(_retry_if_retryable_error),
        before_sleep=lambda retry_state: logger.warning(
            "llm_retry", attempt=retry_state.attempt_number, error=str(retry_state.outcome.exception() if retry_state.outcome is not None else "")
        ),
    )
    async def _complete_openai_with_retry(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_openai(prompt, system, model)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
        retry=tenacity.retry_if_exception(_retry_if_retryable_error),
        before_sleep=lambda retry_state: logger.warning(
            "llm_retry", attempt=retry_state.attempt_number, error=str(retry_state.outcome.exception() if retry_state.outcome is not None else "")
        ),
    )
    async def _complete_ollama_with_retry(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_ollama(prompt, system, model)

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
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.settings.llm_timeout)) as client:
            response = await client.post(
                f"{self._ollama_base_url}/api/chat",
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
            data = response.json()
            return data.get("message", {}).get("content", "")

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
