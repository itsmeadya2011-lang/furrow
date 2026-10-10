from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiofiles
import structlog
import tenacity
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read the contents of a file at the given path. Returns the file "
        "contents as a string, or an error message if the file cannot be read."
    ),
    "write_file": (
        "Write content to a file at the given path. Creates parent "
        "directories if needed. Overwrites existing files. Returns a "
        "confirmation message."
    ),
    "list_files": (
        "List all files in a directory recursively. Returns a JSON array "
        "of file paths relative to the directory."
    ),
}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read.",
            }
        },
        "required": ["path"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
    "list_files": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "The directory path to list files in.",
            }
        },
        "required": ["directory"],
    },
}


def _anthropic_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": desc,
            "input_schema": _TOOL_SCHEMAS[name],
        }
        for name, desc in _TOOL_DESCRIPTIONS.items()
    ]


def _openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": _TOOL_SCHEMAS[name],
            },
        }
        for name, desc in _TOOL_DESCRIPTIONS.items()
    ]


def _retryable(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap an async callable with tenacity retries for transient errors."""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        retry=tenacity.retry_if_exception_type(
            (AnthropicRateLimitError, OpenAIRateLimitError, asyncio.TimeoutError)
        ),
        reraise=True,
    )
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    return wrapper


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Provider clients
    # ------------------------------------------------------------------

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
        """Ollama is OpenAI-compatible, so we reuse the OpenAI client."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",  # dummy key; Ollama doesn't require auth
            )
        return self._ollama

    # ------------------------------------------------------------------
    # Simple completion (no tool use)
    # ------------------------------------------------------------------

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm_complete", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_retryable
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @_retryable
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        client = self.ollama if self.settings.provider == Provider.OLLAMA else self.openai
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Completion with tool use
    # ------------------------------------------------------------------

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tool_calls: int | None = None,
    ) -> str:
        """Run an LLM completion that may call file system tools.

        The LLM can invoke ``read_file``, ``write_file``, and ``list_files``.
        Tool calls are executed and their results fed back to the LLM until
        it returns a final text response or ``max_tool_calls`` is reached.

        Returns the final text response.
        """
        model = model or self.settings.model
        max_tool_calls = max_tool_calls or self.settings.max_tool_calls
        logger.debug("llm_complete_with_tools", provider=self.settings.provider, model=model)

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(prompt, system, model, max_tool_calls)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_with_tools_openai(prompt, system, model, max_tool_calls)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self, prompt: str, system: str, model: str, max_tool_calls: int
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        final_text = ""

        for iteration in range(max_tool_calls):
            response = await self._anthropic_create(messages, system, model)

            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        {"id": block.id, "name": block.name, "input": block.input}
                    )
                elif block.type == "text":
                    text_parts.append(block.text)

            final_text = "".join(text_parts)

            if not tool_calls:
                return final_text

            logger.debug(
                "llm_tool_calls",
                provider="anthropic",
                iteration=iteration + 1,
                tool_count=len(tool_calls),
            )

            tool_results = []
            for tc in tool_calls:
                result = await self._execute_tool(tc["name"], tc["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    }
                )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return final_text or "Maximum tool calls reached."

    async def _complete_with_tools_openai(
        self, prompt: str, system: str, model: str, max_tool_calls: int
    ) -> str:
        client = self.ollama if self.settings.provider == Provider.OLLAMA else self.openai
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]

        for iteration in range(max_tool_calls):
            response = await self._openai_create(client, messages, model)
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    args = json.loads(tool_call.function.arguments or "{}")
                    result = await self._execute_tool(tool_call.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            else:
                return msg.content or ""

        return "Maximum tool calls reached."

    @_retryable
    async def _anthropic_create(
        self, messages: list[dict[str, Any]], system: str, model: str
    ):
        return await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
            tools=_anthropic_tools(),
        )

    @_retryable
    async def _openai_create(self, client: AsyncOpenAI, messages: list[dict[str, Any]], model: str):
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_openai_tools(),
            tool_choice="auto",
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a registered tool with the given arguments.

        Returns the tool result as a string (or error message).
        """
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await handler(**args)
            return str(result)
        except Exception as e:
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return f"Error executing {name}: {e}"

    async def _tool_read_file(self, path: str) -> str:
        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"

    async def _tool_write_file(self, path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return f"Written {len(content)} bytes to {path}"

    async def _tool_list_files(self, directory: str) -> str:
        p = Path(directory)
        if not p.exists():
            return json.dumps([])
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        return json.dumps(files)

    # ------------------------------------------------------------------
    # Legacy thin wrappers (kept for backward compatibility / direct use)
    # ------------------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        return await self._tool_read_file(str(path))

    async def write_file(self, path: str | Path, content: str) -> None:
        await self._tool_write_file(str(path), content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Sync wrapper for list_files (runs the async tool to completion)."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
