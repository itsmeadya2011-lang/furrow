from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger("furrow.llm")


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily initialize and return the Anthropic async client."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily initialize and return the OpenAI async client."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> str:
        """Generate a completion from the configured LLM provider.

        Retries up to 3 times with exponential backoff on transient failures
        and enforces a per-attempt timeout via asyncio.wait_for.
        """
        resolved_model = model or self.settings.model
        logger.debug(
            "completion_attempt",
            provider=self.settings.provider,
            model=resolved_model,
        )
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await asyncio.wait_for(
                    self._complete_anthropic(prompt, system, resolved_model),
                    timeout=timeout,
                )
            elif self.settings.provider == Provider.OPENAI:
                result = await asyncio.wait_for(
                    self._complete_openai(prompt, system, resolved_model),
                    timeout=timeout,
                )
            elif self.settings.provider == Provider.OLLAMA:
                result = await asyncio.wait_for(
                    self._complete_ollama(prompt, system, resolved_model),
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
        except asyncio.TimeoutError:
            logger.error(
                "completion_timeout",
                provider=self.settings.provider,
                model=resolved_model,
                timeout=timeout,
            )
            raise
        logger.debug(
            "completion_success",
            provider=self.settings.provider,
            model=resolved_model,
        )
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Generate a completion using the Anthropic provider."""
        logger.debug("anthropic_request", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Generate a completion using the OpenAI provider."""
        logger.debug("openai_request", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Generate a completion using the Ollama provider via local HTTP API."""
        logger.debug("ollama_request", model=model)
        if hasattr(httpx, "AsyncClient"):
            async with httpx.AsyncClient() as http:
                response = await http.post(
                    f"{self.settings.ollama_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "system": system or "You are a helpful coding assistant.",
                        "stream": False,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        else:
            raise ValueError("httpx is required for Ollama provider")

    async def read_file(self, path: str | Path) -> str:
        """Read and return the contents of a file asynchronously."""
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write content to a file asynchronously, creating parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List all files recursively under a directory, returning relative paths."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
