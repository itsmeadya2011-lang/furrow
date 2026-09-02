from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.post(
            "/api/chat",
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
        return response.json()["message"]["content"]

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

    async def stream(self, prompt: str, system: str = "", model: str | None = None) -> AsyncGenerator[str, None]:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            async with self.anthropic.messages.stream(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        elif self.settings.provider == Provider.OPENAI:
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
        elif self.settings.provider == Provider.OLLAMA:
            async with self.ollama.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            ) as stream:
                async for line in stream.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path, *, ignore: list[str] | None = None) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        default_ignore = {".git", "__pycache__", "node_modules", ".venv", "dist", "build"}
        ignore_set = set(ignore) if ignore else set()
        ignore_set.update(default_ignore)

        result = []
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    rel = f.relative_to(p)
                    if any(part in ignore_set for part in rel.parts):
                        continue
                    result.append(str(rel))
                except ValueError:
                    continue
        return result

    def apply_patch(self, diff: str, *, timeout: int = 30) -> bool:
        try:
            proc = subprocess.run(
                ["patch", "-p1"],
                input=diff,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
