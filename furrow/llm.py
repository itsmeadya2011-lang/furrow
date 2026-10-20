from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIStatusError as AnthropicStatusError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIStatusError as OpenAIStatusError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

DEFAULT_IGNORE = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)
DEFAULT_IGNORE_SUFFIXES = frozenset({".egg-info", ".pyc"})


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, (AnthropicConnectionError, AnthropicTimeoutError, AnthropicRateLimitError)):
        return True
    if isinstance(exc, (OpenAIConnectionError, OpenAITimeoutError, OpenAIRateLimitError)):
        return True
    if isinstance(exc, (AnthropicStatusError, OpenAIStatusError)):
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code >= 500
    return False


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            self.settings = default_settings
        else:
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
            self._ollama = AsyncOpenAI(base_url=self.settings.ollama_base_url, api_key="ollama")
        return self._ollama

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def _call_anthropic_create(self, **kwargs: Any) -> Any:
        return await self.anthropic.messages.create(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def _call_openai_create(self, **kwargs: Any) -> Any:
        provider = self.settings.provider
        if provider == Provider.OPENAI:
            return await self.openai.chat.completions.create(**kwargs)
        elif provider == Provider.OLLAMA:
            return await self.ollama.chat.completions.create(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

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

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._call_anthropic_create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts: list[str] = []
        for block in response.content:
            block_text = getattr(block, "text", "")
            if block_text:
                text_parts.append(block_text)
        return "".join(text_parts)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._call_openai_create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_openai(prompt, system, model)

    def _anthropic_tools_to_openai(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return openai_tools

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(prompt, system, model, tools)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_with_tools_openai(prompt, system, model, tools)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_with_tools_ollama(prompt, system, model, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self, prompt: str, system: str, model: str, tools: list[dict[str, Any]] | None
    ) -> LLMResponse:
        kwargs_clean = {
            "model": model,
            "max_tokens": 4096,
            "system": system or "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs_clean["tools"] = tools
        response = await self._call_anthropic_create(**kwargs_clean)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_call" or block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}) or {},
                    )
                )
            else:
                block_text = getattr(block, "text", "")
                if block_text:
                    text_parts.append(block_text)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", None),
        )

    async def _complete_with_tools_openai(
        self, prompt: str, system: str, model: str, tools: list[dict[str, Any]] | None
    ) -> LLMResponse:
        openai_tools = self._anthropic_tools_to_openai(tools)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        response = await self._call_openai_create(**kwargs)
        message = response.choices[0].message
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                args_str = tc.function.arguments if hasattr(tc.function, "arguments") else "{}"
                args = _parse_tool_arguments(args_str)
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", ""),
                        name=tc.function.name if hasattr(tc.function, "name") else "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=getattr(response.choices[0], "finish_reason", None),
        )

    async def _complete_with_tools_ollama(
        self, prompt: str, system: str, model: str, tools: list[dict[str, Any]] | None
    ) -> LLMResponse:
        return await self._complete_with_tools_openai(prompt, system, model, tools)

    async def run_tool_loop(
        self,
        prompt: str,
        execute: Callable[[ToolCall], Awaitable[str]],
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_rounds: int = 8,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._run_tool_loop_anthropic(
                prompt, execute, system, model, tools, max_rounds
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._run_tool_loop_openai(
                prompt, execute, system, model, tools, max_rounds
            )
        elif self.settings.provider == Provider.OLLAMA:
            return await self._run_tool_loop_ollama(
                prompt, execute, system, model, tools, max_rounds
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _run_tool_loop_anthropic(
        self,
        prompt: str,
        execute: Callable[[ToolCall], Awaitable[str]],
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_rounds: int,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        last_text = ""
        for _round in range(max_rounds):
            kwargs_clean: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "system": system or "You are a helpful coding assistant.",
                "messages": messages,
            }
            if tools:
                kwargs_clean["tools"] = tools
            response = await self._call_anthropic_create(**kwargs_clean)
            # Collect text and tool calls
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            assistant_content: list[dict[str, Any]] = []
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_call" or block_type == "tool_use":
                    tool_call = ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}) or {},
                    )
                    tool_calls.append(tool_call)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    )
                else:
                    block_text = getattr(block, "text", "")
                    if block_text:
                        text_parts.append(block_text)
                        assistant_content.append({"type": "text", "text": block_text})
            last_text = "".join(text_parts)
            if not tool_calls:
                return last_text
            # Add assistant message with content blocks
            messages.append({"role": "assistant", "content": assistant_content})
            # Execute tools and add tool results as user message
            tool_result_blocks: list[dict[str, Any]] = []
            for tc in tool_calls:
                result = await execute(tc)
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_result_blocks})
        return last_text

    async def _run_tool_loop_openai(
        self,
        prompt: str,
        execute: Callable[[ToolCall], Awaitable[str]],
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_rounds: int,
    ) -> str:
        openai_tools = self._anthropic_tools_to_openai(tools)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        last_text = ""
        for _round in range(max_rounds):
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if openai_tools:
                kwargs["tools"] = openai_tools
            response = await self._call_openai_create(**kwargs)
            message = response.choices[0].message
            text = message.content or ""
            last_text = text
            tool_calls: list[ToolCall] = []
            raw_tool_calls = getattr(message, "tool_calls", None)
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    args_str = tc.function.arguments if hasattr(tc.function, "arguments") else "{}"
                    args = _parse_tool_arguments(args_str)
                    tool_calls.append(
                        ToolCall(
                            id=getattr(tc, "id", ""),
                            name=tc.function.name if hasattr(tc.function, "name") else "",
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )
            if not tool_calls:
                return last_text
            # Add assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            assistant_tool_calls = []
            for tc in tool_calls:
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                )
            assistant_msg["tool_calls"] = assistant_tool_calls
            messages.append(assistant_msg)
            # Add tool results
            for tc in tool_calls:
                result = await execute(tc)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return last_text

    async def _run_tool_loop_ollama(
        self,
        prompt: str,
        execute: Callable[[ToolCall], Awaitable[str]],
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_rounds: int,
    ) -> str:
        # Same as OpenAI
        return await self._run_tool_loop_openai(prompt, execute, system, model, tools, max_rounds)

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(content)

    def list_files(self, directory: str | Path, ignore: Sequence[str] | None = None) -> list[str]:
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            return []
        ignore_names = set(DEFAULT_IGNORE)
        ignore_suffixes = set(DEFAULT_IGNORE_SUFFIXES)
        if ignore is not None:
            for item in ignore:
                ignore_names.add(item)
                ignore_suffixes.add(item)
        results: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(p)
            skip = False
            for part in rel.parts:
                if part in ignore_names:
                    skip = True
                    break
                if part.startswith("."):
                    skip = True
                    break
                if any(part.endswith(s) for s in ignore_suffixes):
                    skip = True
                    break
            if skip:
                continue
            results.append(str(rel))
        return results
