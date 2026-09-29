from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._httpx

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type((
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            ConnectionError,
            TimeoutError,
        )),
        reraise=True,
    )
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

    async def complete_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        model: str | None = None,
        max_iterations: int = 5,
    ) -> str:
        """Complete a prompt with tool-use support, iterating until no more tool calls.

        Each tool call is executed via the local file I/O helpers, and the
        results are fed back to the model until it produces a final response
        with no further tool calls (or max_iterations is reached).
        """
        model = model or self.settings.model

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(
                prompt, system, tools, model, max_iterations
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(
                prompt, system, tools, model, max_iterations
            )
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_with_tools(
                prompt, system, tools, model, max_iterations
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # ------------------------------------------------------------------ #
    # Anthropic
    # ------------------------------------------------------------------ #

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("anthropic_request", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_anthropic_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_iterations: int,
    ) -> str:
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            })

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        for iteration in range(max_iterations):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                tools=anthropic_tools,
                messages=messages,
            )

            if response.content:
                has_tool_use = any(block.type == "tool_use" for block in response.content)
                text_blocks = [block.text for block in response.content if block.type == "text"]
                assistant_text = "\n".join(text_blocks)

                if has_tool_use:
                    # Collect tool uses and add them to message history.
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                    content_parts: list[dict[str, Any]] = []
                    for block in response.content:
                        if block.type == "text":
                            content_parts.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            content_parts.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            })

                    messages.append({"role": "assistant", "content": content_parts})

                    # Execute tools and add results.
                    tool_result_parts: list[dict[str, Any]] = []
                    for block in tool_use_blocks:
                        result = await self._execute_tool(block.name, block.input)
                        tool_result_parts.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                    messages.append({"role": "user", "content": tool_result_parts})
                else:
                    return assistant_text
            else:
                return ""

        return ""

    # ------------------------------------------------------------------ #
    # OpenAI
    # ------------------------------------------------------------------ #

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("openai_request", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_openai_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_iterations: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_iterations):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Add assistant message with tool calls.
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute tools.
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = await self._execute_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                return msg.content or ""

        return ""

    # ------------------------------------------------------------------ #
    # Ollama
    # ------------------------------------------------------------------ #

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        logger.debug("ollama_request", model=model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = await self.httpx_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    async def _complete_ollama_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_iterations: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_iterations):
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "tools": tools,
            }
            resp = await self.httpx_client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data["message"]
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                for tc in tool_calls:
                    args = json.loads(tc.get("arguments", "{}") or "{}")
                    result = await self._execute_tool(tc["name"], args)
                    messages.append({
                        "role": "tool",
                        "content": result,
                    })
            else:
                return content

        return ""

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name. Currently supports file I/O tools."""
        try:
            if name == "read_file":
                path = args.get("path", "")
                content = await self.read_file(path)
                return content
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                await self.write_file(path, content)
                return f"File written: {path}"
            elif name == "list_files":
                directory = args.get("directory", "")
                files = self.list_files(directory)
                return ", ".join(files) or "(no files)"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001 — report tool errors as strings
            logger.error("tool_execution_error", tool=name, error=str(e))
            return f"Tool error ({name}): {e}"

    # ------------------------------------------------------------------ #
    # File I/O helpers
    # ------------------------------------------------------------------ #

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

    async def aclose(self) -> None:
        """Close any persistent HTTP connections."""
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None
