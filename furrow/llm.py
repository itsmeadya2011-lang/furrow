from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger("furrow")


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _log_retry(retry_state: tenacity.RetryCallState) -> None:
    ex = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "LLM call retrying",
        attempt=retry_state.attempt_number,
        error=repr(ex),
    )


def _retry() -> Any:
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=_log_retry,
    )


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
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result = await self._complete(messages, system, model)
        return result.text

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        if self.settings.provider == Provider.OLLAMA:
            raise ValueError("Tool use is not supported for the Ollama provider")
        model = model or self.settings.model
        return await self._complete(messages, system, model, tool_schemas, max_tokens)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        tool_schemas: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(messages, system, model, tool_schemas, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(
                self.openai, messages, system, model, tool_schemas, max_tokens
            )
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_openai(
                self.ollama, messages, system, model, tool_schemas, max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        tool_schemas: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> CompletionResult:
        native_messages, embedded_system = _to_anthropic_messages(messages)
        system_text = (system or "You are a helpful coding assistant.").strip()
        if embedded_system:
            system_text = f"{system_text}\n\n{embedded_system}".strip()
        tools = _to_anthropic_tools(tool_schemas) if tool_schemas else None
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_text,
            "messages": native_messages,
        }
        if tool_schemas:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"}
        raw = await self._anthropic_create(**kwargs)
        return _parse_anthropic_response(raw)

    async def _complete_openai(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        tool_schemas: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> CompletionResult:
        native_messages = _to_openai_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": native_messages,
            "max_tokens": max_tokens,
        }
        if tool_schemas:
            kwargs["tools"] = _to_openai_tools(tool_schemas)
            kwargs["tool_choice"] = "auto"
        raw = await self._openai_create(client, **kwargs)
        return _parse_openai_response(raw)

    @_retry()
    async def _anthropic_create(self, **kwargs: Any) -> Any:
        logger.debug(
            "calling anthropic", model=kwargs.get("model"), has_tools=bool(kwargs.get("tools"))
        )
        return await self.anthropic.messages.create(**kwargs)

    @_retry()
    async def _openai_create(self, client: AsyncOpenAI, **kwargs: Any) -> Any:
        logger.debug(
            "calling openai", model=kwargs.get("model"), has_tools=bool(kwargs.get("tools"))
        )
        return await client.chat.completions.create(**kwargs)

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


# ---------------------------------------------------------------------------
# Unified-message <-> provider-native converters
# ---------------------------------------------------------------------------


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    system_parts: list[str] = []
    native: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            if m.get("content"):
                system_parts.append(str(m["content"]))
            continue
        if role == "user":
            native.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            content = m.get("content", "")
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in m.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"] if isinstance(tc["args"], dict) else {},
                    }
                )
            native.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content", ""),
            }
            native.append({"role": "user", "content": [block]})
    return native, "\n\n".join(system_parts)


def _to_openai_messages(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": m.get("content", "")})
            continue
        msg: dict[str, Any] = {"role": role, "content": m.get("content", "")}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if tc["args"] else "{}",
                    },
                }
                for tc in m["tool_calls"]
            ]
        if role == "tool":
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _to_anthropic_tools(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for s in tool_schemas:
        tools.append(
            {
                "type": "tool",
                "name": s["name"],
                "description": s.get("description", ""),
                "input_schema": s.get(
                    "parameters", {"type": "object", "properties": {}, "required": []}
                ),
            }
        )
    return tools


def _to_openai_tools(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for s in tool_schemas:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get(
                        "parameters", {"type": "object", "properties": {}, "required": []}
                    ),
                },
            }
        )
    return tools


def _parse_anthropic_response(response: Any) -> CompletionResult:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text or "")
        elif getattr(block, "type", None) == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id or "",
                    name=getattr(block, "name", "") or "",
                    args=getattr(block, "input", {}) or {},
                )
            )
    return CompletionResult(text="".join(text_parts), tool_calls=tool_calls)


def _parse_openai_response(response: Any) -> CompletionResult:
    message = response.choices[0].message
    text = message.content or ""
    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except (json.JSONDecodeError, ValueError):
            args = {}
        tool_calls.append(
            ToolCall(
                id=tc.id or "",
                name=tc.function.name or "",
                args=args,
            )
        )
    return CompletionResult(text=text, tool_calls=tool_calls)
