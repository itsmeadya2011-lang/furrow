from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings

try:
    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - tenacity is a declared dependency
    _TENACITY_AVAILABLE = False

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None


def _is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* is a transient provider error worth retrying."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert neutral tool schemas to Anthropic tool definitions."""
    converted: list[dict] = []
    for tool in tools:
        converted.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
        )
    return converted


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert neutral tool schemas to OpenAI function definitions."""
    converted: list[dict] = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
        )
    return converted


def _anthropic_block(block: dict) -> dict:
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block.get("text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block["id"],
            "name": block["name"],
            "input": block.get("input", {}),
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block["tool_use_id"],
            "content": block.get("content", ""),
        }
    return dict(block)


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to Anthropic message format."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if isinstance(content, list):
            converted.append(
                {"role": role, "content": [_anthropic_block(block) for block in content]}
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to OpenAI chat completions format."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
            converted.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                }
            )
        elif role == "user" and isinstance(content, list):
            user_text: list[str] = []
            for block in content:
                if block.get("type") == "text":
                    user_text.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        }
                    )
            if user_text:
                converted.append({"role": "user", "content": "\n".join(user_text)})
        else:
            converted.append({"role": role, "content": content})
    return converted


def parse_tool_arguments(raw: Any) -> dict:
    """Parse tool call arguments from a dict or JSON string ({} on failure)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key, timeout=self.settings.request_timeout
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key, timeout=self.settings.request_timeout
            )
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            base_url = self.settings.ollama_base_url.rstrip("/") + "/v1"
            self._ollama = AsyncOpenAI(
                api_key="ollama", base_url=base_url, timeout=self.settings.request_timeout
            )
        return self._ollama

    async def _retry(self, call):
        max_attempts = max(1, int(self.settings.max_retries))

        if _TENACITY_AVAILABLE:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    return await call()

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= max_attempts or not _is_retryable(exc):
                    raise
                await asyncio.sleep(min(20.0, float(2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry loop exited unexpectedly")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider

        async def call() -> str:
            if provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                return await self._complete_openai(self.openai, prompt, system, model)
            elif provider == Provider.OLLAMA:
                return await self._complete_openai(self.ollama, prompt, system, model)
            raise ValueError(f"Unsupported provider: {provider}")

        return await self._retry(call)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
    ) -> LLMResponse:
        model = model or self.settings.model
        provider = self.settings.provider

        async def call() -> LLMResponse:
            if provider == Provider.ANTHROPIC:
                return await self._chat_anthropic(messages, tools, system, model)
            elif provider == Provider.OPENAI:
                return await self._chat_openai_compatible(
                    self.openai, messages, tools, system, model
                )
            elif provider == Provider.OLLAMA:
                return await self._chat_openai_compatible(
                    self.ollama, messages, tools, system, model
                )
            raise ValueError(f"Unsupported provider: {provider}")

        return await self._retry(call)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

    async def _complete_openai(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        request_tools = to_anthropic_tools(tools or [])
        request_messages = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await self.anthropic.messages.create(**kwargs)
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
        )

    async def _chat_openai_compatible(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        request_messages = to_openai_messages(messages)
        if system:
            request_messages.insert(0, {"role": "system", "content": system})
        request_tools = to_openai_tools(tools or [])
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_arguments(tc.function.arguments),
                )
            )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            raw=response,
        )

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
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
