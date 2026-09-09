from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self.last_usage: dict[str, int] | None = None

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider_name = self.settings.provider.value if isinstance(self.settings.provider, Provider) else self.settings.provider
        logger.info(
            "llm_call",
            provider=provider_name,
            model=model,
            prompt_length=len(prompt),
        )
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            elif self.settings.provider == Provider.OLLAMA:
                result = await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
            return result
        except Exception as e:
            logger.error(
                "llm_call_failed",
                provider=provider_name,
                model=model,
                error=str(e),
            )
            raise

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0,
        )
        usage = None
        if hasattr(response, "usage") and response.usage is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        self.last_usage = usage
        if usage:
            logger.info(
                "token_usage",
                provider=Provider.ANTHROPIC.value,
                model=model,
                **usage,
            )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            timeout=60.0,
        )
        usage = None
        if hasattr(response, "usage") and response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        self.last_usage = usage
        if usage:
            logger.info(
                "token_usage",
                provider=Provider.OPENAI.value,
                model=model,
                **usage,
            )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        base_url = self.settings.ollama_base_url.rstrip("/")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        content = data.get("message", {}).get("content", "")

        usage = None
        prompt_eval_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            usage = {
                "prompt_tokens": prompt_eval_count or 0,
                "completion_tokens": eval_count or 0,
            }
        self.last_usage = usage
        if usage:
            logger.info(
                "token_usage",
                provider=Provider.OLLAMA.value,
                model=model,
                **usage,
            )

        return content

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
