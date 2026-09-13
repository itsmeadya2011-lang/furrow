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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

_NETWORK_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.RequestError,
)
_DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


def _log_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception()
    client = retry_state.args[0] if retry_state.args else None
    provider = getattr(getattr(client, "settings", None), "provider", None)
    model = retry_state.kwargs.get("model")
    if model is None and len(retry_state.args) > 3:
        model = retry_state.args[3]
    logger.warning(
        "llm_retry",
        provider=getattr(provider, "value", provider),
        model=model,
        attempt=retry_state.attempt_number,
        error=str(exception),
    )


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        provider_name = getattr(provider, "value", provider)
        logger.info("llm_call", provider=provider_name, model=model)
        try:
            if provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            elif provider == Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as exc:
            logger.error("llm_error", provider=provider_name, model=model, error=str(exc))
            raise

    @retry(
        retry=retry_if_exception_type(_NETWORK_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_retry,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or _DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(
        retry=retry_if_exception_type(_NETWORK_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_retry,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or _DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_exception_type(_NETWORK_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_retry,
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                },
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            data = response.json()
        return data.get("response", "")

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
