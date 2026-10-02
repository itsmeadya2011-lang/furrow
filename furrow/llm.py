from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ToolCall:
    id: str
    type: str
    name: str
    input: dict[str, Any]


@dataclass(slots=True)
class CompletionResult:
    text: str
    tool_calls: list[ToolCall]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory recursively.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list files in.",
                },
            },
            "required": ["directory"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing old_string with new_string.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "old_string": {
                    "type": "string",
                    "description": "Exact string to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]


def _on_api_failure(retry_state: RetryCallState) -> None:
    logger.warning(
        "LLM API call failed; retrying",
        attempt=retry_state.attempt_number,
        error=repr(retry_state.outcome.exception()),
    )


_llm_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    before_sleep=_on_api_failure,
    reraise=True,
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

    def _resolve_tools(self, tools: list[str] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        by_name = {t["name"]: t for t in TOOL_SCHEMAS}
        return [by_name[name] for name in tools if name in by_name]

    @staticmethod
    def _to_anthropic_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        result = await self.complete_with_tools(prompt, system, model, tools=None)
        return result.text

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[str] | None = None,
    ) -> CompletionResult:
        model = model or self.settings.model
        enabled = self._resolve_tools(tools)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, enabled)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, enabled)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, enabled)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_llm_retry
    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system or "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)
        response = await self.anthropic.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type=block.type,
                        name=block.name,
                        input=dict(block.input),
                    )
                )
        return CompletionResult(text="".join(text_parts), tool_calls=tool_calls)

    @_llm_retry
    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
    ) -> CompletionResult:
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {"model": model, "messages": msgs}
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
        response = await self.openai.chat.completions.create(**kwargs)

        message = response.choices[0].message
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                input_dict: dict[str, Any] = {}
                if tc.function.arguments:
                    input_dict = json.loads(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type=tc.type or "function",
                        name=tc.function.name,
                        input=input_dict,
                    )
                )
        return CompletionResult(text=text, tool_calls=tool_calls)

    @_llm_retry
    async def _complete_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system or "You are a helpful coding assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if tools:
            payload["tools"] = self._to_openai_tools(tools)

        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            http_response.raise_for_status()
            data = http_response.json()

        message = data.get("message", {})
        text = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for raw_tc in message.get("tool_calls") or []:
            arguments = raw_tc.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
            elif arguments is None:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=raw_tc.get("id", ""),
                    type=raw_tc.get("type", "function"),
                    name=raw_tc.get("name", ""),
                    input=arguments,
                )
            )
        return CompletionResult(text=text, tool_calls=tool_calls)

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def _edit_file(
        self, path: str, old_string: str, new_string: str
    ) -> str:
        content = await self.read_file(path)
        if old_string not in content:
            raise ValueError(f"old_string not found in {path}")
        updated = content.replace(old_string, new_string, 1)
        await self.write_file(path, updated)
        return f"Updated {path}"

    async def execute_tool(self, tool_call: ToolCall) -> str:
        logger.debug(
            "Executing tool",
            name=tool_call.name,
            input=tool_call.input,
        )
        name = tool_call.name
        args = tool_call.input
        if name == "read_file":
            return await self.read_file(args["path"])
        elif name == "write_file":
            await self.write_file(args["path"], args["content"])
            return f"Wrote content to {args['path']}"
        elif name == "list_files":
            files = await self.list_files(args["directory"])
            return json.dumps(files)
        elif name == "edit_file":
            return await self._edit_file(
                args["path"], args["old_string"], args["new_string"]
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
