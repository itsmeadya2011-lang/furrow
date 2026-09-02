from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

log = structlog.get_logger(__name__)


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


class LLMClient:
    """Async client that dispatches LLM completion requests to the configured provider.

    Supports Anthropic, OpenAI, and Ollama backends with automatic retries on
    transient failures and structured logging throughout.
    """

    def __init__(self, settings: Settings = settings) -> None:
        """Initialise the client with the given *settings*.

        Provider-specific clients are created lazily on first use.
        """
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: httpx.AsyncClient | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily-initialised Anthropic async client."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily-initialised OpenAI async client."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> httpx.AsyncClient:
        """Lazily-initialised ``httpx.AsyncClient`` for the Ollama backend."""
        if self._ollama is None:
            self._ollama = httpx.AsyncClient(timeout=120.0)
        return self._ollama

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        """Generate a completion from the configured provider.

        Retries up to 3 times with exponential backoff on any exception.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt override.
            model: Optional model name; defaults to the configured model.

        Returns:
            The generated completion string.
        """
        model = model or self.settings.model
        provider = self.settings.provider.value
        await log.ainfo("llm.complete", provider=provider, model=model)
        await log.adebug(
            "llm.complete.prompt",
            provider=provider,
            model=model,
            system=_truncate(system) if system else "",
            prompt=_truncate(prompt),
        )

        if self.settings.provider == Provider.ANTHROPIC:
            result = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            result = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            result = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

        await log.adebug("llm.complete.response", provider=provider, model=model, response=_truncate(result))
        return result

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Send a completion request to the Anthropic API."""
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        usage = getattr(response, "usage", None)
        if usage:
            await log.ainfo(
                "llm.complete.anthropic",
                model=model,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            )
        return content

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Send a completion request to the OpenAI API."""
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system or "You are a helpful coding assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage:
            await log.ainfo(
                "llm.complete.openai",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Send a completion request to the Ollama ``/api/chat`` endpoint."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = await self.ollama.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        await log.ainfo(
            "llm.complete.ollama",
            model=model,
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            prompt_eval_count=data.get("prompt_eval_count"),
        )
        return content

    async def read_file(self, path: str | Path) -> str:
        """Read and return the full text contents of *path*.

        Args:
            path: File path to read (``str`` or ``Path``).

        Returns:
            The file's text content.
        """
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write *content* to *path*, creating parent directories as needed.

        Args:
            path: Destination file path (``str`` or ``Path``).
            content: Text content to write.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path, max_depth: int = 2) -> list[str]:
        """List files under *directory* up to *max_depth* levels deep.

        Args:
            directory: Root directory to scan (``str`` or ``Path``).
            max_depth: Maximum depth relative to *directory*. ``0`` means unlimited.

        Returns:
            Sorted list of file paths relative to *directory*.
        """
        p = Path(directory)
        if not p.exists():
            return []
        results: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            if max_depth > 0:
                depth = len(f.relative_to(p).parts) - 1
                if depth > max_depth:
                    continue
            results.append(str(f.relative_to(p)))
        results.sort()
        return results

    async def close(self) -> None:
        """Close the underlying HTTP clients held by this instance."""
        if self._ollama is not None:
            await self._ollama.aclose()
            self._ollama = None
