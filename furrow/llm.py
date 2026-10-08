from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings as default_settings


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: httpx.AsyncClient | None = None

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
    def ollama_client(self) -> httpx.AsyncClient:
        if self._ollama_client is None:
            base_url = self.settings.ollama_base_url or "http://localhost:11434"
            self._ollama_client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        return self._ollama_client

    @staticmethod
    def _retry_decorator() -> Any:
        return tenacity.retry(
            retry=tenacity.retry_if_exception_type(
                (
                    anthropic.APIError,
                    openai.APIError,
                    httpx.HTTPError,
                    asyncio.TimeoutError,
                )
            ),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
            stop=tenacity.stop_after_attempt(3),
            reraise=True,
        )

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

    @_retry_decorator()
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            raise ValueError("Anthropic returned empty response content")
        return response.content[0].text

    @_retry_decorator()
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise ValueError("OpenAI returned empty choices")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned None content")
        return content

    @_retry_decorator()
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await self.ollama_client.post(
            "/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError(f"Ollama returned unexpected response: {data}")
        return data["response"]

    async def complete_multi(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_multi(messages, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_multi(messages, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_multi(messages, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_retry_decorator()
    async def _complete_anthropic_multi(self, messages: list[dict[str, str]], model: str) -> str:
        system_parts = []
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)
        system = "\n\n".join(system_parts) if system_parts else "You are a helpful coding assistant."
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=user_messages,
        )
        if not response.content:
            raise ValueError("Anthropic returned empty response content")
        return response.content[0].text

    @_retry_decorator()
    async def _complete_openai_multi(self, messages: list[dict[str, str]], model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
        )
        if not response.choices:
            raise ValueError("OpenAI returned empty choices")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned None content")
        return content

    @_retry_decorator()
    async def _complete_ollama_multi(self, messages: list[dict[str, str]], model: str) -> str:
        last_user = None
        system_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                last_user = msg["content"]
        system = "\n\n".join(system_parts) if system_parts else ""
        prompt = f"{system}\n\n{last_user}" if system and last_user else (last_user or "")
        if not prompt:
            raise ValueError("No user message found for Ollama completion")
        return await self._complete_ollama(prompt, "", model)

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