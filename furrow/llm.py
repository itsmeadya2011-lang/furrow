from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


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

    async def stream(self, prompt: str, system: str = "", model: str | None = None) -> AsyncGenerator[str, None]:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            async for chunk in self._stream_anthropic(prompt, system, model):
                yield chunk
        elif self.settings.provider == Provider.OPENAI:
            async for chunk in self._stream_openai(prompt, system, model):
                yield chunk
        elif self.settings.provider == Provider.OLLAMA:
            async for chunk in self._stream_ollama(prompt, system, model):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

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
        import httpx
        base_url = self.settings.ollama_base_url.rstrip("/")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/api/generate", json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def _stream_anthropic(self, prompt: str, system: str, model: str) -> AsyncGenerator[str, None]:
        async with self.anthropic.messages.stream(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_openai(self, prompt: str, system: str, model: str) -> AsyncGenerator[str, None]:
        stream = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_ollama(self, prompt: str, system: str, model: str) -> AsyncGenerator[str, None]:
        import httpx
        base_url = self.settings.ollama_base_url.rstrip("/")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{base_url}/api/generate", json=payload, timeout=120.0
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        async with aiofiles.open(p, "r") as f:
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
