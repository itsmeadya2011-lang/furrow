from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

logger = structlog.get_logger(__name__)

# Default timeout (seconds) for LLM API calls.
DEFAULT_TIMEOUT = 30.0
# Default retry behaviour.
DEFAULT_MAX_RETRIES = 3


def _is_retryable(exc: BaseException) -> bool:
    """Return True for errors that are worth retrying."""
    # Network-level issues (timeouts, connection errors).
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, asyncio.TimeoutError)):
        return True
    # HTTP status errors from httpx (Ollama provider).
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response is not None:
            return exc.response.status_code in (429, 500, 502, 503, 504)
        return False
    # Anthropic rate-limit / server errors.
    if isinstance(exc, AnthropicRateLimitError):
        return True
    # OpenAI rate-limit errors.
    return isinstance(exc, OpenAIRateLimitError)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None
        self._timeout = getattr(self.settings, "request_timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT
        self._max_retries = (
            getattr(self.settings, "max_retries", DEFAULT_MAX_RETRIES) or DEFAULT_MAX_RETRIES
        )

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key, timeout=self._timeout)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, timeout=self._timeout)
        return self._openai

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=self._timeout)
        return self._httpx

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Generate a completion from the configured LLM provider.

        Retries transient errors (timeouts, rate limits, 5xx) up to
        ``_max_retries`` times with exponential back-off.
        """
        model = model or self.settings.model
        provider = self.settings.provider
        provider_str = provider.value if hasattr(provider, "value") else str(provider)
        logger.info(
            "llm_complete",
            provider=provider_str,
            model=model,
            prompt_len=len(prompt),
        )
        if provider == Provider.ANTHROPIC:
            result = await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            result = await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            result = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        logger.info("llm_complete_done", model=model, response_len=len(result))
        return result

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
        """Send a completion request to a local Ollama instance."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        resp = await self.httpx_client.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns the full response in `response` when stream=False.
        return data.get("response", "")

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

    async def close(self) -> None:
        """Close any persistent HTTP connections."""
        if self._httpx is not None:
            await self._httpx.aclose()
