from __future__ import annotations

import json
import os
import re
from pathlib import Path

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings


def _is_transient(exc: BaseException) -> bool:
    """Return True if exc represents a transient error that should be retried."""
    # OpenAI errors
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, OpenAIAPITimeoutError):
        return True
    if isinstance(exc, OpenAIAPIConnectionError):
        return True
        if isinstance(exc, OpenAIAPIStatusError):
            status = getattr(exc, "status_code", None)
            return bool(status is not None and (status >= 500 or status == 429))
    # Anthropic errors
    if isinstance(exc, AnthropicAPITimeoutError):
        return True
    if isinstance(exc, AnthropicAPIConnectionError):
        return True
        if isinstance(exc, AnthropicAPIStatusError):
            status = getattr(exc, "status_code", None)
            return bool(status is not None and (status >= 500 or status == 429))
    # HTTPX errors (raw transport issues)
    return bool(isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)))


_chat_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```)."""
    m = re.search(r"```(?:[a-zA-Z]+\n)?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Unclosed fence: return everything after the opening fence marker + optional lang tag
    idx = text.find("```")
    if idx != -1:
        after = text[idx + 3:]
        lang_match = re.match(r"\s*[a-zA-Z]+\s*", after)
        if lang_match:
            after = after[lang_match.end():]
        return after
    return text


def _find_json_object(text: str) -> str | None:
    """Find the first balanced JSON object in text, returning the substring or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from text, stripping markdown fences.
    Raises ValueError with raw response if no JSON object found.
    """
    cleaned = _strip_code_fences(text)
    candidate = _find_json_object(cleaned)
    if candidate is None:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"No JSON object found in response: {text[:500]}") from e
    if not isinstance(obj, dict):
        raise TypeError(f"No JSON object found in response: {text[:500]}")
    return obj


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

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
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=f"{self.settings.ollama_base_url}/v1",
                api_key="ollama",
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        messages = [{"role": "user", "content": prompt}]
        return await self._chat_completion(messages, system, model)

    async def complete_json(self, prompt: str, system: str = "", model: str | None = None) -> dict:
        model = model or self.settings.model
        text = await self.complete(prompt, system, model)
        return extract_json(text)

    async def complete_messages(self, messages: list[dict], system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        return await self._chat_completion(messages, system, model)

    @_chat_retry
    async def _chat_completion(self, messages: list[dict], system: str, model: str) -> str:
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            text = await self._complete_anthropic(messages, system, model)
        elif provider == Provider.OPENAI:
            text = await self._complete_openai(self.openai, messages, system, model)
        elif provider == Provider.OLLAMA:
            text = await self._complete_openai(self.ollama, messages, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        if not text.strip():
            raise ValueError("LLM returned an empty response")
        return text

    async def _complete_anthropic(self, messages: list[dict], system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,  # type: ignore[arg-type]
        )
        content = response.content[0]
        if hasattr(content, "text"):
            return content.text or ""
        return str(content)

    async def _complete_openai(self, client: AsyncOpenAI, messages: list[dict], system: str, model: str) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system or "You are a helpful coding assistant."}, *messages],  # type: ignore[list-item]
        )
        return response.choices[0].message.content or ""

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