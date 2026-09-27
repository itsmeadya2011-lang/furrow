from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger()

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read and return the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file at the given path. Parent directories are created if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Content to write to the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files recursively in the given directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list files in."},
            },
            "required": ["directory"],
        },
    },
]

_DEFAULT_SYSTEM = "You are a helpful coding assistant."

_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    anthropic.APIStatusError,
    openai.APIError,
)

_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
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

    def _anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for spec in TOOL_SPECS
        ]

    def _openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "tool",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in TOOL_SPECS
        ]

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("complete_start", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            text = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            text = await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        logger.debug("complete_success", provider=self.settings.provider, model=model)
        return text

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        model = model or self.settings.model
        logger.debug(
            "complete_with_tools_start",
            provider=self.settings.provider,
            model=model,
            max_iterations=max_iterations,
        )
        if self.settings.provider == Provider.ANTHROPIC:
            text = await self._complete_with_tools_anthropic(prompt, system, model, max_iterations)
        elif self.settings.provider == Provider.OPENAI:
            text = await self._complete_with_tools_openai(prompt, system, model, max_iterations)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        logger.debug(
            "complete_with_tools_success",
            provider=self.settings.provider,
            model=model,
        )
        return text

    @_retry
    async def _anthropic_create(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system or _DEFAULT_SYSTEM,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return await self.anthropic.messages.create(**kwargs)

    @_retry
    async def _openai_create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return await self.openai.chat.completions.create(**kwargs)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._anthropic_create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._openai_create(
            model=model,
            system=system,
            messages=[
                {"role": "system", "content": system or _DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_with_tools_anthropic(
        self, prompt: str, system: str, model: str, max_iterations: int
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = self._anthropic_tools()
        last_text = ""
        for iteration in range(max_iterations):
            logger.debug("anthropic_api_call_start", model=model, iteration=iteration)
            response = await self._anthropic_create(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
            )
            logger.debug("anthropic_api_call_success", iteration=iteration)

            text_parts: list[str] = []
            tool_uses: list[Any] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            last_text = "".join(text_parts)
            if not tool_uses:
                return last_text

            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                logger.debug("tool_execution", tool=tu.name, tool_id=tu.id)
                result = await self._dispatch_tool(tu.name, tu.input)
                logger.debug("tool_execution_result", tool=tu.name, result=result[:200])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})

        return last_text or "Maximum iterations reached without a final response."

    async def _complete_with_tools_openai(
        self, prompt: str, system: str, model: str, max_iterations: int
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        tools = self._openai_tools()
        last_text = ""
        for iteration in range(max_iterations):
            logger.debug("openai_api_call_start", model=model, iteration=iteration)
            response = await self._openai_create(
                model=model,
                messages=messages,
                tools=tools,
            )
            logger.debug("openai_api_call_success", iteration=iteration)

            message = response.choices[0].message
            last_text = message.content or ""
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return last_text

            serialized_tool_calls = [
                {
                    "id": tc.id,
                    "type": "tool_call",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": serialized_tool_calls,
                }
            )
            for tc in tool_calls:
                params = json.loads(tc.function.arguments)
                logger.debug("tool_execution", tool=tc.function.name, tool_id=tc.id)
                result = await self._dispatch_tool(tc.function.name, params)
                logger.debug("tool_execution_result", tool=tc.function.name, result=result[:200])
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        return last_text or "Maximum iterations reached without a final response."

    async def _dispatch_tool(self, name: str, params: dict[str, Any]) -> str:
        if name == "read_file":
            return await self.read_file(params["path"])
        elif name == "write_file":
            await self.write_file(params["path"], params["content"])
            return f"Wrote file: {params['path']}"
        elif name == "list_files":
            return "\n".join(self.list_files(params["directory"]))
        else:
            return f"Unknown tool: {name}"

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
