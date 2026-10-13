from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import tenacity

from furrow.config import Provider, Settings, settings
from .logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http: httpx.AsyncClient | None = None

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
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.settings.ollama_base_url, timeout=60.0)
        return self._http

    def validate(self) -> None:
        if self.settings.provider == Provider.ANTHROPIC:
            if not (self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
                raise ValueError("ANTHROPIC_API_KEY is not set")
        elif self.settings.provider == Provider.OPENAI:
            if not (self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
                raise ValueError("OPENAI_API_KEY is not set")
        elif self.settings.provider == Provider.OLLAMA:
            import httpx
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{self.settings.ollama_base_url}/api/tags")
                    resp.raise_for_status()
            except Exception as e:
                raise ValueError(f"Cannot reach Ollama at {self.settings.ollama_base_url}: {e}")
        else:
            raise ValueError(f"Unknown provider: {self.settings.provider}")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("Calling LLM provider", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(
            (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(
            (openai.APIError, openai.APIConnectionError, openai.RateLimitError)
        ),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type((httpx.HTTPError, httpx.ConnectError)),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        model = model or os.getenv("FURROW_MODEL") or "llama3"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self.http.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.settings.workspace / p
        return p

    async def read_file(self, path: str | Path) -> str:
        p = await self._resolve_path(path)
        async with aiofiles.open(p, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = await self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.is_absolute():
            p = self.settings.workspace / p
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("Starting chat session", provider=self.settings.provider, model=model)
        system = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._chat_anthropic(filtered_messages, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._chat_openai(filtered_messages, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._chat_ollama(filtered_messages, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(
            (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _chat_anthropic(self, messages: list[dict[str, str]], system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
        )
        return response.content[0].text

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(
            (openai.APIError, openai.APIConnectionError, openai.RateLimitError)
        ),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _chat_openai(self, messages: list[dict[str, str]], system: str, model: str) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=msgs,
        )
        return response.choices[0].message.content or ""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type((httpx.HTTPError, httpx.ConnectError)),
        before_retry=tenacity.before_sleep_log(logger, logging.WARNING),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _chat_ollama(self, messages: list[dict[str, str]], system: str, model: str) -> str:
        model = model or os.getenv("FURROW_MODEL") or "llama3"
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        response = await self.http.post(
            "/api/chat",
            json={"model": model, "messages": msgs, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")