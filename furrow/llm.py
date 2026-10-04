from __future__ import annotations

import os
from pathlib import Path

import aiofiles  # type: ignore[import-untyped]  # no stubs published (types-aiofiles)
import httpx
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

# ``furrow.llm.settings`` stays available as the process-wide default; it is
# aliased here because ``LLMClient.__init__`` needs to refer to that default
# from inside a parameter of the same name.
settings = default_settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

# Local models can be slow to load, so the overall budget is generous, but the
# connect budget stays short so a dead server fails fast instead of hanging.
OLLAMA_TIMEOUT_SECONDS = 300.0
OLLAMA_CONNECT_TIMEOUT_SECONDS = 10.0


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        # ``None`` means "use the module-level default" so callers that forward
        # an optional ``settings`` (e.g. the agents) get a working client
        # instead of one that explodes on first attribute access.
        self.settings: Settings = default_settings if settings is None else settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def http(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    OLLAMA_TIMEOUT_SECONDS, connect=OLLAMA_CONNECT_TIMEOUT_SECONDS
                )
            )
        return self._httpx

    async def aclose(self) -> None:
        """Close the lazily created HTTP client. Safe to call more than once."""
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None

    def default_model(self) -> str:
        if self.settings.provider == Provider.OLLAMA:
            return self.settings.ollama_model
        return self.settings.model

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.default_model()
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
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # ``content`` is a union of block types and only ``TextBlock`` carries
        # ``.text``, so narrow to the first text block rather than assuming the
        # very first block is one.
        for block in response.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Anthropic response contained no text block")

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Call Ollama's OpenAI-compatible chat completions endpoint."""
        url = f"{self.settings.ollama_base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            response = await self.http.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request to {url} failed: {exc}") from exc

        if not response.is_success:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} from {url}: {response.text[:500]}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Could not parse Ollama response from {url}: {response.text[:500]}"
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError(
                f"Ollama response from {url} had non-text content: {content!r}"
            )
        return content

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
