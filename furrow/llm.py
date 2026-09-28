from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

RETRYABLE_EXCEPTIONS = (
    httpx.RequestError,
    httpx.HTTPStatusError,
    anthropic.AnthropicError,
    openai.APIError,
    openai.APIConnectionError,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http: httpx.AsyncClient | None = None

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
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if model is None:
            if self.settings.provider == Provider.OLLAMA:
                model = self.settings.ollama_model
            else:
                model = self.settings.model
        logger.info("complete", provider=self.settings.provider.value, model=model)
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            elif self.settings.provider == Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
        except Exception as e:
            logger.error(
                "complete_failed",
                provider=self.settings.provider.value,
                model=model,
                error=repr(e),
            )
            raise

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
        model = model or self.settings.ollama_model
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        response = await self.http.post(
            "/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return str(data["message"]["content"])

    async def read_file(self, path: str | Path) -> str:
        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except FileNotFoundError:
            logger.error("read_file_not_found", path=str(path))
            raise FileNotFoundError(f"File not found: {path}") from None

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            logger.warning("list_files_not_found", directory=str(directory))
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
