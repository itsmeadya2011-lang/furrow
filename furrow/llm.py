from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    text: str | None
    tool_calls: list[ToolCall]


_SYSTEM = "You are a helpful coding assistant."

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at path, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a file at path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Recursively list files under a directory. Returns a JSON array of relative paths.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}},
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_file",
            "description": "Replace old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]


def _anthropic_tools() -> list[dict]:
    out: list[dict] = []
    for t in _TOOLS:
        f = t["function"]
        params = f["parameters"]
        out.append(
            {
                "type": "tool",
                "name": f["name"],
                "description": f["description"],
                "input_schema": {
                    "type": params["type"],
                    "properties": params["properties"],
                    "required": params.get("required", []),
                },
            }
        )
    return out


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
            self._httpx = httpx.AsyncClient(timeout=120.0)
        return self._httpx

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
            system=system or _SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/chat", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Tool-calling
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_system(messages: list[dict], system: str) -> tuple[str, list[dict]]:
        sp = system
        if not sp:
            for m in messages:
                if m.get("role") == "system":
                    sp = m.get("content", "") or ""
                    break
        filtered = [m for m in messages if m.get("role") != "system"]
        return sp, filtered

    async def complete_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        model: str | None = None,
    ) -> Completion:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_tools(messages, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_tools(messages, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_tools(messages, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        result: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            if role == "assistant":
                content: list[dict] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                result.append({"role": "assistant", "content": content})
            elif role == "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m.get("content", ""),
                            }
                        ],
                    }
                )
            else:
                result.append({"role": "user", "content": m.get("content", "")})
        return result

    async def _complete_anthropic_tools(
        self, messages: list[dict], system: str, model: str
    ) -> Completion:
        sp, filtered = self._resolve_system(messages, system)
        am = self._to_anthropic_messages(filtered)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=sp or _SYSTEM,
            messages=am,
            tools=_anthropic_tools(),
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        text = "".join(text_parts) if text_parts else None
        return Completion(text=text, tool_calls=tool_calls)

    @staticmethod
    def _to_openai_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                out.append({"role": "system", "content": m.get("content", "")})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m.get("content", ""),
                    }
                )
            else:
                out.append({"role": "user", "content": m.get("content", "")})
        return out

    async def _complete_openai_tools(
        self, messages: list[dict], system: str, model: str
    ) -> Completion:
        om = self._to_openai_messages(messages)
        if system and not any(m.get("role") == "system" for m in om):
            om.insert(0, {"role": "system", "content": system})
        response = await self.openai.chat.completions.create(
            model=model,
            messages=om,
            tools=_TOOLS,
        )
        msg = response.choices[0].message
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )
        text = msg.content or None
        return Completion(text=text, tool_calls=tool_calls)

    async def _complete_ollama_tools(
        self, messages: list[dict], system: str, model: str
    ) -> Completion:
        sp, filtered = self._resolve_system(messages, system)
        ollama_msgs: list[dict] = []
        if sp:
            ollama_msgs.append({"role": "system", "content": sp})
        for m in filtered:
            role = m["role"]
            if role == "assistant":
                om: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    om["tool_calls"] = [
                        {"name": tc.name, "arguments": tc.arguments} for tc in m["tool_calls"]
                    ]
                ollama_msgs.append(om)
            elif role == "tool":
                ollama_msgs.append(
                    {"role": "tool", "content": m.get("content", ""), "name": m.get("name")}
                )
            else:
                ollama_msgs.append({"role": "user", "content": m.get("content", "")})
        payload = {
            "model": model,
            "messages": ollama_msgs,
            "tools": _TOOLS,
            "stream": False,
        }
        resp = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/chat", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            try:
                args = json.loads(tc.get("arguments", "")) if tc.get("arguments") else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", "ollama"), name=tc.get("name", ""), arguments=args)
            )
        text = msg.get("content") or None
        return Completion(text=text, tool_calls=tool_calls)

    async def execute_tool(self, tool: ToolCall) -> str:
        args = tool.arguments
        if tool.name == "write_file":
            path = args["path"]
            content = args.get("content", "")
            await self.write_file(path, content)
            return f"Wrote {path} ({len(content)} chars)."
        if tool.name == "read_file":
            path = args["path"]
            return await self.read_file(path)
        if tool.name == "list_files":
            directory = args.get("directory", ".")
            return json.dumps(self.list_files(directory))
        if tool.name == "replace_file":
            path = args["path"]
            old_string = args["old_string"]
            new_string = args["new_string"]
            original = await self.read_file(path)
            if old_string not in original:
                return f"Error: old_string not found in {path}."
            updated = original.replace(old_string, new_string, 1)
            await self.write_file(path, updated)
            return f"Replaced text in {path}."
        return f"Unknown tool: {tool.name}"

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
