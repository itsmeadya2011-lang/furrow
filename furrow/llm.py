from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from structlog import get_logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = get_logger(__name__)

_default_settings = settings

DEFAULT_TIMEOUT_SECONDS = 30.0


def _log_retry(retry_state: RetryCallState) -> None:
    instance: Any = retry_state.args[0] if retry_state.args else None
    provider = getattr(getattr(instance, "settings", None), "provider", None)
    model = retry_state.kwargs.get("model")
    if model is None and instance is not None:
        model = getattr(getattr(instance, "settings", None), "model", None)
    logger.warning(
        "retrying_after_failure",
        provider=provider.value if provider else None,
        model=model,
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx_client: httpx.AsyncClient | None = None

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
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=_log_retry,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug("completing", provider=provider.value, model=model, system=bool(system))
        try:
            if provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            elif provider == Provider.OLLAMA:
                result = await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception:
            logger.error("completion_failed", provider=provider.value, model=model)
            raise
        logger.debug("completed", provider=provider.value, model=model)
        return result

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content or not isinstance(response.content, list):
            raise ValueError("Anthropic returned empty or invalid content")
        first = response.content[0]
        text = getattr(first, "text", None)
        if not text:
            raise ValueError("Anthropic returned no text content")
        return text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise ValueError("OpenAI returned no choices")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content")
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        timeout = getattr(self.settings, "timeout", DEFAULT_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        data = response.json()
        if error := data.get("error"):
            raise RuntimeError(f"Ollama API error: {error}")
        generated = data.get("response")
        if not generated:
            raise ValueError("Ollama returned empty response")
        return generated

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_tool_result: Callable[[str, str], str] | None = None,
        max_iterations: int = 10,
    ) -> str:
        if not tools or not on_tool_result:
            return await self.complete(prompt, system, model)
        current_prompt = prompt
        for _ in range(max_iterations):
            response_text = await self.complete(current_prompt, system, model)
            tool_output = on_tool_result(response_text, model or self.settings.model)
            if not tool_output:
                return response_text
            current_prompt = f"{response_text}\n\nTool result:\n{tool_output}"
        return await self.complete(current_prompt, system, model)

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
