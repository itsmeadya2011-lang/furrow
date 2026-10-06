from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """Thin async wrapper around Anthropic / OpenAI / Ollama completions.

    Provider selection, model names, and ``max_tokens`` are read from the
    global :class:`~furrow.config.Settings` instance, which in turn reads
    from the ``FURROW_`` environment namespace or ``.env`` file.
    """

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

    # -- lazy client properties ----------------------------------------------

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
            base_url = self.settings.ollama_base_url
            self._httpx = httpx.AsyncClient(base_url=base_url)
        return self._httpx

    # -- public API ----------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion for *prompt* using the configured provider.

        ``model`` defaults to ``settings.model``.  ``max_tokens`` defaults to
        ``settings.max_tokens``.
        """
        model = model or self.settings.model
        max_tokens = max_tokens or self.settings.max_tokens

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # -- provider-specific implementations -----------------------------------

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        log = logger.bind(model=model, max_tokens=max_tokens)
        log.info("llm.request", provider="anthropic")
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text if response.content else ""
        log.info("llm.response", provider="anthropic", length=len(content))
        return content

    async def _complete_openai(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        log = logger.bind(model=model, max_tokens=max_tokens)
        log.info("llm.request", provider="openai")
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        log.info("llm.response", provider="openai", length=len(content))
        return content

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        log = logger.bind(model=model, max_tokens=max_tokens)
        log.info("llm.request", provider="ollama")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        response = await self.httpx.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        log.info("llm.response", provider="ollama", length=len(content))
        return content

    # -- file helpers --------------------------------------------------------

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
