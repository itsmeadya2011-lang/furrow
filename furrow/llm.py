from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings


class LLMClient:
    """Async LLM client supporting Anthropic, OpenAI, and Ollama providers.

    Settings can be passed explicitly or will fall back to the module-level
    singleton ``settings`` when ``None``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or Settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    #  Lazy client properties
    # ------------------------------------------------------------------ #

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._openai

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.settings.llm_timeout)
        return self._http

    # ------------------------------------------------------------------ #
    #  Completion dispatch
    # ------------------------------------------------------------------ #

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content[0] is a TextBlock (may be other types in theory)
        return str(getattr(response.content[0], "text", ""))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return str(response.choices[0].message.content or "")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = await self.http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return str(data["message"]["content"])

    # ------------------------------------------------------------------ #
    #  File helpers
    # ------------------------------------------------------------------ #

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve *path* relative to the configured workspace."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.settings.workspace / p

    async def read_file(self, path: str | Path) -> str:
        full_path = self.resolve_path(path)
        async with aiofiles.open(full_path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        full_path = self.resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(full_path, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = self.resolve_path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
