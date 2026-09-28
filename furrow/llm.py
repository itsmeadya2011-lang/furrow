from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

ToolFunc = Callable[..., Awaitable[str]]


@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    """Result of an LLM completion, which may include tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class Tool:
    """A tool/function that can be called by the LLM."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: ToolFunc | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def spec(self, provider: Provider) -> dict[str, Any]:
        """Return the tool spec in the format expected by the given provider."""
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
        # Anthropic
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def call(self, arguments: dict[str, Any]) -> str:
        """Execute the tool function with the given arguments."""
        if self.func is None:
            raise ValueError(f"Tool {self.name} has no function assigned")
        return await self.func(**arguments)


class LLMClient:
    def __init__(self, settings: Settings | None = settings) -> None:
        self.settings = settings if settings is not None else Settings()
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
            self._httpx = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url, timeout=60.0
            )
        return self._httpx

    # ------------------------------------------------------------------
    # Basic completion (no tools)
    # ------------------------------------------------------------------

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
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Find the first text content block
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = await self.httpx_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Tool-aware completion
    # ------------------------------------------------------------------

    async def complete_with_tools(
        self,
        prompt: str | None,
        tools: list[Tool],
        system: str = "",
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Make a single LLM call that may return tool calls.

        ``messages`` is the accumulated conversation history in the provider's
        native format. When ``None`` a fresh conversation is started using
        ``prompt`` as the first user message.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_with_tools(prompt, tools, system, model, messages)
        elif self.settings.provider == Provider.OPENAI:
            return await self._openai_with_tools(prompt, tools, system, model, messages)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._ollama_with_tools(prompt, tools, system, model, messages)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _anthropic_with_tools(
        self,
        prompt: str | None,
        tools: list[Tool],
        system: str,
        model: str,
        messages: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        if messages is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt or ""}]}]
        tool_specs = [t.spec(Provider.ANTHROPIC) for t in tools]
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,  # type: ignore[arg-type]
            tools=tool_specs,  # type: ignore[arg-type]
        )
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,
                        id=block.id,
                    )
                )
        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls)

    async def _openai_with_tools(
        self,
        prompt: str | None,
        tools: list[Tool],
        system: str,
        model: str,
        messages: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        if messages is None:
            messages = [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt or ""},
            ]
        tool_specs = [t.spec(Provider.OPENAI) for t in tools]
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tool_specs,  # type: ignore[arg-type]
        )
        msg = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            if tc.type != "function":
                continue
            args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(
                tc.function.arguments or "{}"
            )
            tool_calls.append(
                ToolCall(name=tc.function.name, arguments=args, id=tc.id or "")
            )
        return LLMResponse(text=msg.content or "", tool_calls=tool_calls)

    async def _ollama_with_tools(
        self,
        prompt: str | None,
        tools: list[Tool],
        system: str,
        model: str,
        messages: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        if messages is None:
            messages = [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt or ""},
            ]
        tool_specs = [t.spec(Provider.OLLAMA) for t in tools]
        payload: dict[str, Any] = {
            "model": model or self.settings.ollama_model,
            "messages": messages,
            "tools": tool_specs,
            "stream": False,
        }
        resp = await self.httpx_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        tool_calls: list[ToolCall] = []
        for raw in msg.get("tool_calls", []):
            func = raw.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args or "{}")
            tool_calls.append(
                ToolCall(
                    name=func.get("name", ""),
                    arguments=args,
                    id=raw.get("id", ""),
                )
            )
        return LLMResponse(text=msg.get("content", ""), tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------

    async def run_tool_loop(
        self,
        prompt: str,
        tools: list[Tool],
        system: str = "",
        model: str | None = None,
        max_iterations: int = 20,
    ) -> str:
        """Run an LLM agent loop that can call tools until it stops.

        Returns the final text response from the LLM after all tool calls
        have been resolved.
        """
        provider = self.settings.provider
        # Build the initial conversation in the provider's native format so
        # that tool results can be appended consistently across iterations.
        messages = self._build_initial_messages(prompt, system, provider)
        last_response = LLMResponse()

        for _ in range(max_iterations):
            response = await self.complete_with_tools(
                prompt=None,
                tools=tools,
                system=system,
                model=model,
                messages=messages,
            )
            if not response.has_tool_calls:
                return response.text

            # Execute tool calls and append results as new messages
            tool_map = {t.name: t for t in tools}
            for tc in response.tool_calls:
                tool = tool_map.get(tc.name)
                if tool is None:
                    result = f"Tool '{tc.name}' is not available."
                else:
                    try:
                        result = await tool.call(tc.arguments)
                    except Exception as e:  # noqa: BLE001
                        result = f"Tool '{tc.name}' failed: {e}"
                self._append_tool_result(messages, tc, result, provider)

            last_response = response

        # Exhausted iterations – return any accumulated text or a notice
        return last_response.text or "Max tool iterations reached."

    def _build_initial_messages(
        self, prompt: str, system: str, provider: Provider
    ) -> list[dict[str, Any]]:
        """Build the initial conversation messages in the provider's native format."""
        if provider == Provider.ANTHROPIC:
            # Anthropic passes system separately; messages only have turns.
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        # OpenAI / Ollama support a system message in the conversation.
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _append_tool_result(
        messages: list[dict[str, Any]],
        tool_call: ToolCall,
        result: str,
        provider: Provider,
    ) -> None:
        if provider == Provider.ANTHROPIC:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                    }
                ],
            })
        elif provider in (Provider.OPENAI, Provider.OLLAMA):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # ------------------------------------------------------------------
    # File & command helpers
    # ------------------------------------------------------------------

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

    async def run_command(self, command: str, timeout: int = 60) -> str:
        """Execute a shell command in the workspace and return combined output."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.workspace,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                if proc.returncode != 0:
                    output += f"\n[exit code: {proc.returncode}]"
                return output
            except asyncio.TimeoutError:
                proc.kill()
                return f"Command timed out after {timeout}s"
        except FileNotFoundError as e:
            return f"Command not found: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Command failed: {e}"
