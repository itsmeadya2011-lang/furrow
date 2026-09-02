from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings


# Exceptions that are worth retrying on (network/provider issues, rate limits).
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    httpx.TransportError,
    httpx.HTTPStatusError,
    TimeoutError,
)


class LLMClient:
    """Async LLM client supporting multiple providers with retry logic.

    Supports Anthropic, OpenAI, and Ollama (via OpenAI-compatible endpoint).
    All ``complete`` calls are automatically retried with exponential backoff
    on retriable errors.
    """

    DEFAULT_MAX_TOKENS = 8192

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: httpx.AsyncClient | None = None

    # -- Provider clients ----------------------------------------------------

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
    def ollama(self) -> httpx.AsyncClient:
        """HTTP client for Ollama's OpenAI-compatible API."""
        if self._ollama is None:
            self._ollama = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._ollama

    # -- Completion ----------------------------------------------------------

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _complete_anthropic(
        self, prompt: str, system: str, model: str
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _complete_openai(
        self, prompt: str, system: str, model: str
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _complete_ollama(
        self, prompt: str, system: str, model: str
    ) -> str:
        """Call Ollama via its OpenAI-compatible chat endpoint."""
        response = await self.ollama.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.DEFAULT_MAX_TOKENS,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # -- File operations -----------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        """List all files in a directory (recursive). Async version of list_files."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    # -- Chat / conversation -------------------------------------------------

    async def chat(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Send a multi-turn conversation to the LLM and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override.

        Returns:
            The LLM response text.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            # Separate system messages for Anthropic
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )
            return await self._chat_anthropic(user_messages, system_msg, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._chat_openai(messages, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._chat_ollama(messages, model)  # Ollama uses HTTP endpoint
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _chat_anthropic(
        self, messages: list[dict[str, str]], system: str, model: str
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _chat_openai(
        self, messages: list[dict[str, str]], model: str
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _chat_ollama(
        self, messages: list[dict[str, str]], model: str
    ) -> str:
        """Chat with Ollama via its OpenAI-compatible chat endpoint."""
        response = await self.ollama.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": self.DEFAULT_MAX_TOKENS,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
