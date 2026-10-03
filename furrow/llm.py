from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
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
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings

log = structlog.get_logger(__name__)

OLLAMA_TIMEOUT_SECONDS = 300.0
_SNIPPET_LENGTH = 200
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)

_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


def _log_retry(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None else None
    sleep_for = getattr(retry_state.next_action, "sleep", None)
    log.warning(
        "llm.retry",
        attempt=retry_state.attempt_number,
        sleep=round(sleep_for, 2) if isinstance(sleep_for, (int, float)) else None,
        error=type(error).__name__ if error is not None else None,
        detail=str(error) if error is not None else None,
    )


_retry_llm = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    before_sleep=_log_retry,
    reraise=True,
)


def _iter_balanced_blocks(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` and ``[...]`` block found in ``text``."""
    for opener, closer in (("{", "}"), ("[", "]")):
        cursor = 0
        while True:
            start = text.find(opener, cursor)
            if start == -1:
                break
            depth = 0
            in_string = False
            escaped = False
            end = -1
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end == -1:
                break
            yield text[start : end + 1]
            cursor = end + 1


def _json_candidates(text: str) -> Iterator[str]:
    stripped = text.strip()
    if stripped:
        yield stripped
    for match in _FENCE_RE.finditer(text):
        block = match.group(2).strip()
        if block:
            yield block
    yield from _iter_balanced_blocks(text)


def extract_json(text: str) -> Any:
    """Extract and parse the first JSON object/array from an LLM response.

    Handles ```json / ``` fenced blocks, surrounding prose and bare payloads.
    Raises ``ValueError`` when nothing parseable is present.
    """
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except ValueError:  # json.JSONDecodeError is a ValueError
            continue
    snippet = text.strip()[:_SNIPPET_LENGTH]
    raise ValueError(f"No valid JSON object or array found in response: {snippet!r}")


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
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
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {provider}")

    @_retry_llm
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            str(getattr(block, "text", ""))
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        if not text:
            raise ValueError("Anthropic returned a response with no text block")
        return text

    @_retry_llm
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise ValueError("OpenAI returned a response with no choices")
        return response.choices[0].message.content or ""

    @_retry_llm
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        base_url = self.settings.ollama_base_url.rstrip("/")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        timeout = httpx.Timeout(OLLAMA_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        return str(content or "")

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