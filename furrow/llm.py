from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger("furrow.llm")

# Module-level default settings (captured before the `settings` parameter name
# shadows the import inside LLMClient.__init__).
_default_settings = settings

# Exceptions that are safe to retry (transient network/API failures).
RETRYABLE = (
    anthropic.APIError,
    openai.APIError,
    httpx.HTTPError,
    OSError,
    asyncio.TimeoutError,
)

# A tool that the LLM can invoke during a tool-using turn.
ToolFn = Callable[..., Awaitable[Any]]


class LLMClient:
    """Asynchronous LLM client supporting Anthropic, OpenAI, and Ollama."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or _default_settings
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
        """Ollama exposes an OpenAI-compatible API, so reuse the OpenAI client."""
        if self._ollama is None:
            base_url = self.settings.ollama_base_url
            # Ollama's OpenAI-compatible API lives under the /v1 path.
            if not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            self._ollama = AsyncOpenAI(
                base_url=base_url,
                api_key="ollama",  # value is ignored by Ollama but required by the client
            )
        return self._ollama

    @property
    def chat(self) -> AsyncOpenAI:
        """The OpenAI-compatible client used for the active provider."""
        if self.settings.provider == Provider.OPENAI:
            return self.openai
        if self.settings.provider == Provider.OLLAMA:
            return self.ollama
        # Anthropic is not OpenAI-compatible; callers should use .anthropic.
        return self.openai

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug("llm.complete", provider=provider, model=model, prompt_len=len(prompt))
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai(prompt, system, model)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        if block.type == "text":
            return block.text
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.chat.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Tool-using completion
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file's contents. Returns the text or an error note.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read."},
                    },
                    "required": ["path"],
                },
            },
            {
                "type": "function",
                "name": "write_file",
                "description": "Write content to a file, creating parent directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write."},
                        "content": {"type": "string", "description": "Content to write."},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "type": "function",
                "name": "list_files",
                "description": "Recursively list files under a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list."},
                    },
                    "required": ["directory"],
                },
            },
        ]

    def _tools(self) -> dict[str, ToolFn]:
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
        }

    def _safe_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the workspace, refusing to escape it."""
        workspace = Path(self.settings.workspace).resolve()
        p = Path(path)
        if not p.is_absolute():
            p = workspace / p
        resolved = p.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise PermissionError(f"Path '{resolved}' is outside the workspace '{workspace}'") from exc
        return resolved

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        try:
            async with aiofiles.open(p, "r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return f"Error: file not found: {p}"
        except IsADirectoryError:
            return f"Error: not a file: {p}"

    async def write_file(self, path: str | Path, content: str) -> None:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        fn = self._tools().get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await fn(**args)
        except PermissionError as exc:
            return f"Error: {exc}"
        if result is None:
            return "OK"
        if isinstance(result, list):
            return "\n".join(result)
        return str(result)

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_turns: int = 10,
    ) -> str:
        """Run a tool-using completion loop until the model stops calling tools.

        The returned string is the model's final text response. File reads/writes
        within the loop operate on the real filesystem, constrained to the workspace.
        """
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug(
            "llm.complete_with_tools",
            provider=provider,
            model=model,
            max_turns=max_turns,
        )
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(prompt, system, model, max_turns)
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai_with_tools(prompt, system, model, max_turns)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        max_turns: int,
    ) -> str:
        client = self.anthropic
        tools = self._tool_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        text_response = ""

        for _ in range(max_turns):
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=cast(Any, messages),
                tools=cast(Any, tools),
            )
            tool_calls: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text_response += block.text or ""
                elif block.type == "tool_use":
                    tool_calls.append(
                        {"name": block.name, "args": block.input, "id": block.id}
                    )

            if not tool_calls:
                return text_response

            # Append the assistant's message (with tool_use) and tool results.
            assistant_content: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text" and block.text:
                    assistant_content.append(
                        {"type": "text", "text": block.text}
                    )
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for call in tool_calls:
                result = await self._execute_tool(call["name"], call["args"])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call["id"], "content": result}
                )
            messages.append({"role": "user", "content": tool_results})

        return text_response

    async def _complete_openai_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        max_turns: int,
    ) -> str:
        client = self.chat
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tool_schemas()
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        text_response = ""

        for _ in range(max_turns):
            response = await client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="auto",
            )
            message = response.choices[0].message
            text_response = message.content or ""
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return text_response

            messages.append(
                {"role": "assistant", "content": message.content, "tool_calls": tool_calls}
            )
            for call in tool_calls:
                # `tool_calls` items may be function- or custom-tool calls;
                # only function calls carry a `.function` payload we can invoke.
                func = getattr(call, "function", None)
                if func is None:
                    continue
                arg_key = func.name
                arg_value = func.arguments
                try:
                    args = json.loads(arg_value) if arg_value else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": arg_value}
                result = await self._execute_tool(arg_key, args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return text_response
