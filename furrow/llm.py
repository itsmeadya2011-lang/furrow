from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

log = structlog.get_logger(__name__)

# Reduce noise from httpx/urllib3 in structlog output
logging.getLogger("httpx").setLevel(logging.WARNING)


def _strip_json_markers(text: str) -> str:
    """Strip markdown code-fence markers and whitespace from LLM JSON output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


class LLMClient:
    """Async LLM client supporting Anthropic, OpenAI, and Ollama providers.

    Also provides filesystem helpers (read_file, write_file, list_files)
    that agents use to persist their work.
    """

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Provider clients
    # ------------------------------------------------------------------

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
            log.debug("Initialized Anthropic client")
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
            log.debug("Initialized OpenAI client")
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama is OpenAI-compatible; reuse the AsyncOpenAI client."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                api_key="ollama",  # Ollama ignores API key
                base_url=self.settings.ollama_base_url,
            )
            log.debug("Initialized Ollama client", base_url=self.settings.ollama_base_url)
        return self._ollama

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> str:
        model = model or self.settings.model
        log.debug("LLM complete", provider=self.settings.provider, model=model)

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
        text = response.content[0].text
        log.debug("Anthropic response", length=len(text))
        return text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        log.debug("OpenAI response", length=len(text))
        return text

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        log.debug("Ollama response", length=len(text))
        return text

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Call the LLM and parse the response as JSON, handling markdown wrappers."""
        raw = await self.complete(prompt, system=system, model=model)
        cleaned = _strip_json_markers(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.warning("JSON parse failed", error=str(e), raw_preview=raw[:200])
            raise

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        log.debug("File written", path=str(p), bytes=len(content))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
