from __future__ import annotations

import os
import tiktoken
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: httpx.AsyncClient | None = None
        self._token_usage: dict[str, int] = {"input": 0, "output": 0}
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

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
    def ollama(self) -> httpx.AsyncClient:
        if self._ollama_client is None:
            self._ollama_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=300.0,
            )
        return self._ollama_client

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
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((AnthropicRateLimitError, httpx.TimeoutException)),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("Calling Anthropic", model=model, prompt_length=len(prompt))
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Anthropic response received", model=model)
        # Track token usage
        input_tokens = self._count_tokens(system + prompt)
        output_tokens = self._count_tokens(response.content[0].text)
        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens
        return response.content[0].text

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OpenAIRateLimitError, httpx.TimeoutException)),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("Calling OpenAI", model=model, prompt_length=len(prompt))
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        logger.debug("OpenAI response received", model=model)
        # Track token usage
        input_tokens = self._count_tokens(system + prompt)
        output_tokens = self._count_tokens(response.choices[0].message.content or "")
        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens
        return response.choices[0].message.content or ""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        logger.debug("Calling Ollama", model=model, prompt_length=len(prompt))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.ollama.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("Ollama response received", model=model)
        content = data.get("message", {}).get("content", "")
        # Track token usage (approximate)
        input_tokens = self._count_tokens(system + prompt)
        output_tokens = self._count_tokens(content)
        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens
        return content

    def get_token_usage(self) -> dict[str, int]:
        return self._token_usage.copy()

    def reset_token_usage(self) -> None:
        self._token_usage = {"input": 0, "output": 0}

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

    async def close(self) -> None:
        """Close any open connections."""
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
