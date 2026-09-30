from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AsyncOpenAI
from openai import InternalServerError as OpenAIInternalServerError
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger("furrow.llm")

RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.RequestError,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIInternalServerError,
    OpenAIRateLimitError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicInternalServerError,
    AnthropicRateLimitError,
)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Retry an HTTP status error only for transient (5xx) responses."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


RETRY_PREDICATE = retry_if_exception_type(RETRYABLE_EXCEPTIONS) | retry_if_exception(
    _is_retryable_http_error
)


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

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        model = model or self.settings.model
        logger.info(
            "completion_request",
            provider=self.settings.provider.value,
            model=model,
            max_tokens=max_tokens,
        )

        if self.settings.provider == Provider.ANTHROPIC:
            func = self._complete_anthropic
        elif self.settings.provider == Provider.OPENAI:
            func = self._complete_openai
        elif self.settings.provider == Provider.OLLAMA:
            func = self._complete_ollama
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

        try:
            return await self._run_with_retry(func, prompt, system, model, max_tokens)
        except Exception:
            logger.exception(
                "completion_error",
                provider=self.settings.provider.value,
                model=model,
            )
            raise

    async def _run_with_retry(self, func, *args, **kwargs) -> str:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2),
            retry=RETRY_PREDICATE,
            reraise=True,
        )
        return await retrying(func, *args, **kwargs)

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int = 4096
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(
        self, prompt: str, system: str, model: str, max_tokens: int = 4096
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, max_tokens: int = 4096
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=120.0,
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

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
