from __future__ import annotations

import os
import time
from pathlib import Path

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

log = structlog.get_logger(__name__)

DEFAULT_MAX_TOKENS = 4096

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv"}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.APIError, openai.APIError, httpx.TransportError)):
        # don't retry auth errors
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status not in (401, 403)
    return False


class LLMClient:
    def __init__(self, settings: Settings | None = None, max_tokens: int | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient()
        return self._httpx

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        start = time.monotonic()
        try:
            if provider == Provider.ANTHROPIC:
                result = await self._call_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                result = await self._call_openai(prompt, system, model)
            elif provider == Provider.OLLAMA:
                result = await self._call_ollama(prompt, system, model)
            else:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    "Supported: anthropic, openai, ollama"
                )
            log.info(
                "completion",
                provider=provider.value,
                model=model,
                prompt_len=len(prompt),
                elapsed=round(time.monotonic() - start, 3),
            )
            return result
        except Exception:
            log.exception("completion_failed", provider=provider.value, model=model)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient),
    )
    async def _call_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        texts = [
            getattr(b, "text", "")
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        joined = "".join(texts)
        if not joined:
            types = [getattr(b, "type", None) for b in response.content]
            raise ValueError(f"Anthropic response contained no text blocks; block types: {types}")
        return joined

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient),
    )
    async def _call_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient),
    )
    async def _call_ollama(self, prompt: str, system: str, model: str) -> str:
        base = self.settings.ollama_base_url.rstrip("/")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        resp = await self.httpx_client.post(f"{base}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

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
        result: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(p)
            if any(part in _SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            if f.name.startswith(".") or f.parent.name.startswith("."):
                pass
            result.append(str(f.relative_to(p)))
        return result
