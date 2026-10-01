from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, get_logger

log = get_logger("furrow.llm")


class LLMClient:
    def __init__(self, settings: Settings = None) -> None:
        self.settings = settings or Settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None

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
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url, timeout=120
            )
        return self._http_client

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Complete a prompt and return text response. Dispatches to provider-specific method."""
        model = model or self.settings.model
        log.info(
            "llm.complete",
            provider=self.settings.provider.value,
            model=model,
            prompt_len=len(prompt),
        )

        if self.settings.provider == Provider.ANTHROPIC:
            result = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            result = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            result = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

        log.info(
            "llm.complete.done", provider=self.settings.provider.value, response_len=len(result)
        )
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
        """Complete a prompt using an Ollama server."""
        payload: dict[str, Any] = {
            "model": model or "llama3",
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        response = await self.http_client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def read_file(self, path: str | Path) -> str:
        """Read a file's contents asynchronously."""
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write content to a file, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List all files in a directory recursively."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def aclose(self) -> None:
        """Clean up async resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
