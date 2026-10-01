from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

log = structlog.get_logger(__name__)

# Exceptions that should never be retried — retrying will not fix them.
_NON_RETRYABLE = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    json.JSONDecodeError,
)


def _should_retry(exc: BaseException) -> bool:
    """Return True for transient exceptions that warrant a retry."""
    return not isinstance(exc, _NON_RETRYABLE) and isinstance(exc, Exception)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
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
    def httpx(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=120)
        return self._httpx

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a completion.

        Args:
            prompt: The user prompt (used when *messages* is not provided).
            system: Optional system message.
            model: Override the model for this call.
            messages: Optional full conversation history as a list of
                ``{"role": "...", "content": "..."}`` dicts.  When provided,
                *prompt* and *system* are ignored for the conversation body.

        Returns:
            The generated text response.
        """
        model = model or self.settings.model

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, messages)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, messages)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        log.debug("anthropic.complete", model=model)
        if messages is not None:
            msgs = [m for m in messages if m.get("role") != "system"]
            response = await self._retry(
                self.anthropic.messages.create,
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
            )
        else:
            response = await self._retry(
                self.anthropic.messages.create,
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
        return response.content[0].text

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        log.debug("openai.complete", model=model)
        if messages is not None:
            msgs = list(messages)
        else:
            msgs = [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ]
        response = await self._retry(
            self.openai.chat.completions.create,
            model=model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            messages=msgs,
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
    ) -> str:
        log.debug("ollama.complete", model=model, base_url=self.settings.ollama_base_url)
        payload: dict[str, Any] = {
            "model": model or self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        response = await self._retry(
            self.httpx.post,
            f"{self.settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def _retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry an async callable on transient failures with exponential backoff.

        Args:
            fn: An async callable (e.g. ``client.messages.create``).
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            The result of ``fn(*args, **kwargs)``.
        """

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self.settings.retry_attempts),
            wait=tenacity.wait_exponential(
                multiplier=self.settings.retry_backoff, min=1, max=30
            ),
            retry=tenacity.retry_if_exception(_should_retry),
            reraise=True,
            before_sleep=tenacity.before_sleep_log(
                logging.getLogger("furrow.llm"), logging.WARNING
            ),
        )
        async def _run() -> Any:
            return await fn(*args, **kwargs)

        return await _run()

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
