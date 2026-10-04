from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles

from furrow.config import Provider, Settings, settings

try:  # the SDKs are optional at import time: this module must import without them
    from anthropic import AsyncAnthropic
except ModuleNotFoundError:  # pragma: no cover - only taken when the SDK is absent
    AsyncAnthropic = None  # type: ignore[assignment]

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - only taken when the SDK is absent
    AsyncOpenAI = None  # type: ignore[assignment]

# Ollama ignores the API key, but the OpenAI-compatible client insists on one.
OLLAMA_PLACEHOLDER_API_KEY = "ollama"


def _sdk_client_class(client_class: Any, package: str, provider: Provider) -> Any:
    """Return ``client_class``, or explain that ``package`` still has to be installed."""
    if client_class is None:
        raise ImportError(
            f"Provider.{provider.name} needs the {package!r} package, which is not installed. "
            f"Install it with: pip install {package}"
        )
    return client_class


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Export ANTHROPIC_API_KEY, or set "
                    "FURROW_ANTHROPIC_API_KEY, to use the anthropic provider."
                )
            client_class = _sdk_client_class(AsyncAnthropic, "anthropic", Provider.ANTHROPIC)
            self._anthropic = client_class(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Export OPENAI_API_KEY, or set "
                    "FURROW_OPENAI_API_KEY, to use the openai provider."
                )
            client_class = _sdk_client_class(AsyncOpenAI, "openai", Provider.OPENAI)
            self._openai = client_class(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama speaks the OpenAI API, so no API key is required."""
        if self._ollama is None:
            client_class = _sdk_client_class(AsyncOpenAI, "openai", Provider.OLLAMA)
            self._ollama = client_class(
                base_url=f"{self.settings.ollama_base_url}/v1",
                api_key=OLLAMA_PLACEHOLDER_API_KEY,
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        match self.settings.provider:
            case Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            case Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            case Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model)
            case provider:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Supported: {[p.value for p in Provider]}"
                )

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_chat_completions(self.openai, prompt, system, model)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_chat_completions(self.ollama, prompt, system, model)

    async def _complete_chat_completions(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
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
