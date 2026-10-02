from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings
from furrow.tools import Tool, ToolResult, get_default_tools


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
            self._httpx = httpx.AsyncClient(timeout=60.0)
        return self._httpx

    # --- Simple completion ---

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
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return response.content[0].text

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
        response = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    # --- Tool-use loop ---

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[Tool] | None = None,
        max_rounds: int | None = None,
    ) -> str:
        """
        Send a prompt to the LLM with tools available. The LLM may call tools;
        we execute them and feed results back in a loop until the LLM produces
        a final text response or we reach *max_rounds*.
        """
        model = model or self.settings.model
        tools = tools or get_default_tools()
        max_rounds = max_rounds or self.settings.max_tool_rounds

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_tool_loop(prompt, system, model, tools, max_rounds)
        elif self.settings.provider == Provider.OPENAI:
            return await self._openai_tool_loop(prompt, system, model, tools, max_rounds)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._ollama_tool_loop(prompt, system, model, tools, max_rounds)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _execute_tool(self, call: dict[str, Any], tools: list[Tool]) -> str:
        """Look up and execute a single tool call, returning a formatted result string."""
        tool_name = call.get("name", "")
        tool_input = call.get("input", {})
        for tool in tools:
            if tool.name == tool_name:
                result: ToolResult = await tool.execute(**tool_input)
                if result.success:
                    return result.output
                else:
                    return f"Error: {result.error or result.output}"
        return f"Tool not found: {tool_name}"

    async def _anthropic_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[Tool],
        max_rounds: int,
    ) -> str:
        tool_schemas = [t.to_anthropic_schema() for t in tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        text = ""

        for _round in range(max_rounds):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=8192,
                system=system or "You are a helpful coding assistant with tools.",
                messages=messages,
                tools=tool_schemas,
            )

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            text = "\n".join(text_parts)

            if not tool_calls:
                return text

            # Build assistant message with text + tool uses
            assistant_content: list[dict[str, Any]] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools and append results
            tool_result_blocks: list[dict[str, Any]] = []
            for tc in tool_calls:
                result = await self._execute_tool(tc, tools)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_result_blocks})

        return text

    async def _openai_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[Tool],
        max_rounds: int,
    ) -> str:
        tool_schemas = [t.to_openai_schema() for t in tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant with tools."},
            {"role": "user", "content": prompt},
        ]
        text = ""

        for _round in range(max_rounds):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                max_tokens=8192,
            )

            msg = response.choices[0].message
            text = msg.content or ""
            tool_calls = msg.tool_calls

            if not tool_calls:
                return text

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": tool_calls,
            })

            # Execute tools and append results
            for tc in tool_calls:
                call = {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments) if tc.function.arguments else {},
                }
                result = await self._execute_tool(call, tools)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.id,
                })

        return text

    async def _ollama_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[Tool],
        max_rounds: int,
    ) -> str:
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant with tools."},
            {"role": "user", "content": prompt},
        ]
        text = ""

        for _round in range(max_rounds):
            payload: dict[str, Any] = {
                "model": model or self.settings.ollama_model,
                "messages": messages,
                "stream": False,
            }
            if ollama_tools:
                payload["tools"] = ollama_tools

            response = await self.httpx_client.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            text = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                return text

            # Add assistant message
            messages.append({"role": "assistant", "content": text})

            # Execute tools and add results
            for tc in tool_calls:
                func = tc.get("function", {})
                call = {
                    "id": tc.get("id", f"call_{len(messages)}"),
                    "name": func.get("name", ""),
                    "input": func.get("arguments", {}) or {},
                }
                result = await self._execute_tool(call, tools)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": call["name"],
                })

        return text

    # --- File utilities (kept for backwards compatibility) ---

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
