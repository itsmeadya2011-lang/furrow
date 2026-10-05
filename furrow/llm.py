from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, get_settings

_DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
_OLLAMA_TIMEOUT_SECONDS = 120.0


def _is_transient(exc: BaseException) -> bool:
    """Return ``True`` for provider errors that are worth retrying.

    Our own configuration errors (:class:`ValueError`, e.g. a missing API key)
    and control-flow exceptions are excluded: retrying them only burns the
    budget and delays the real error, and swallowing ``CancelledError`` would
    break task cancellation.
    """
    return not isinstance(
        exc, (ValueError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)
    )


# Bounded backoff: 3 attempts, waits of ~1s then ~2s (each capped at 10s).
_RETRY_TRANSIENT = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        # Resolved lazily: importing ``Settings`` must not build a config object.
        self.settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key_text or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key_text or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {provider}")

    @_RETRY_TRANSIENT
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or _DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Block 0 is not necessarily text: thinking or tool-use blocks may come
        # first, so join every block that actually carries text.
        chunks: list[str] = []
        for block in getattr(response, "content", None) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(str(text))
        return "".join(chunks)

    @_RETRY_TRANSIENT
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or _DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_RETRY_TRANSIENT
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete ``prompt`` through a local Ollama server (``/api/chat``)."""
        base_url = self.settings.ollama_base_url.rstrip("/")
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        # The client is scoped to this call: no shared/connection-pool state.
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = (message or {}).get("content") or ""
        return content if isinstance(content, str) else ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        """List project files relative to ``directory`` (default: workspace).

        Any path passing through one of ``settings.ignored_dirs`` is skipped, and
        ignored directories are pruned so ``.git``/``node_modules``/virtualenvs
        are never walked. Returns POSIX-style relative paths, sorted; ``[]`` when
        the directory does not exist.
        """
        root = Path(directory) if directory is not None else self.settings.workspace
        if not root.exists():
            return []
        ignored = set(self.settings.ignored_dirs)
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in ignored]
            for name in filenames:
                rel = (Path(dirpath) / name).relative_to(root)
                if ignored.intersection(rel.parts):
                    continue
                found.append(rel.as_posix())
        return sorted(found)
