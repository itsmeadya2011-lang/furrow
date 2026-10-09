from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: httpx.AsyncClient | None = None

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
        if self._ollama is None:
            self._ollama = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        messages = [{"role": "user", "content": prompt}]
        return await self.complete_with_messages(messages, system, model)

    async def complete_with_messages(self, messages: list[dict], system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_messages(messages, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_messages(messages, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_messages(messages, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_anthropic_messages([{"role": "user", "content": prompt}], system, model)

    async def _complete_anthropic_messages(self, messages: list[dict], system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_openai_messages([{"role": "user", "content": prompt}], system, model)

    async def _complete_openai_messages(self, messages: list[dict], system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                *messages,
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_ollama_messages([{"role": "user", "content": prompt}], system, model)

    async def _complete_ollama_messages(self, messages: list[dict], system: str, model: str) -> str:
        # Ollama doesn't natively support multi-message chat, so we concatenate
        prompt_parts = []
        if system:
            prompt_parts.append(f"System: {system}")
        for msg in messages:
            prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        prompt = "\n\n".join(prompt_parts)
        response = await self.ollama.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system or "You are a helpful coding assistant.",
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
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
        if not p.exists() or not p.is_dir():
            return []
        return sorted(str(f.relative_to(p)) for f in p.rglob("*") if f.is_file())
