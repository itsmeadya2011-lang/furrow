"""LLM client with tool-use support for Anthropic, OpenAI, and Ollama."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings as _default_settings
from furrow.logging import get_logger

log = get_logger(__name__)


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResult:
    """Result of an LLM completion call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or _default_settings
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

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama-compatible client (OpenAI API format)."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "ollama"
            self._openai = AsyncOpenAI(api_key=api_key, base_url=self.settings.ollama_base_url)
        return self._openai

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Simple text completion. Returns the LLM's text response.

        Backward compatible with the original ``complete(prompt, model=...)``
        signature — extra parameters have defaults.
        """
        result = await self.complete_with_tools(
            prompt=prompt,
            system=system,
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        return result.text

    async def complete_with_tools(
        self,
        prompt: str = "",
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResult:
        """Complete with optional tool-use support.

        Parameters
        ----------
        prompt:
            Initial user prompt. If ``messages`` is ``None`` this becomes
            the first user message.
        system:
            System prompt.
        model:
            Model override (falls back to ``settings.model``).
        tools:
            Tool definitions in **Anthropic format**. If ``None`` or empty,
            no tool-use is requested.
        messages:
            Conversation history in **Anthropic format** (list of
            ``{"role", "content"}`` dicts). When provided, ``prompt`` is
            ignored unless the list is empty.
        max_tokens:
            Maximum tokens to generate.
        """
        model = model or self.settings.model
        provider = self.settings.provider

        # Build the message list
        if messages is None:
            messages = []
        if prompt and (not messages or messages[-1]["role"] != "user"):
            content: Any = [{"type": "text", "text": prompt}]
            messages.append({"role": "user", "content": content})

        log.debug("LLM complete_with_tools", model=model, provider=provider, tools=tools, messages=len(messages))

        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(system, model, tools, messages, max_tokens)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(system, model, tools, messages, max_tokens, self.openai)
        elif provider == Provider.OLLAMA:
            return await self._complete_openai(
                system, model, tools, messages, max_tokens,
                client=self._ollama_client(),
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _ollama_client(self) -> AsyncOpenAI:
        api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "ollama"
        self._openai = AsyncOpenAI(api_key=api_key, base_url=self.settings.ollama_base_url)
        return self._openai

    async def _complete_anthropic(
        self,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.anthropic.messages.create(**kwargs)

        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        result = LLMResult(text=text, tool_calls=tool_calls)
        log.debug("Anthropic response", text_len=len(text), tool_calls=len(tool_calls))
        return result

    async def _complete_openai(
        self,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        client: AsyncOpenAI,
    ) -> LLMResult:
        # Convert Anthropic-format messages to OpenAI format
        openai_messages = self._anthropic_to_openai_messages(messages)

        system_prompt = system or "You are a helpful coding assistant."
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + openai_messages,
        }
        if tools:
            kwargs["tools"] = self._anthropic_to_openai_tools(tools)

        response = await client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        text = msg.content or ""

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{len(tool_calls)}",
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments or "{}"),
                    )
                )

        result = LLMResult(text=text, tool_calls=tool_calls)
        log.debug("OpenAI response", text_len=len(text), tool_calls=len(tool_calls))
        return result

    @staticmethod
    def _anthropic_to_openai_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Anthropic-format conversation to OpenAI format."""
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                parts: list[str] = []
                tool_results: list[dict] = []
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id"),
                                "content": block.get("content"),
                            })

                if parts:
                    openai_messages.append({
                        "role": "user",
                        "content": "\n".join(parts),
                    })
                for tr in tool_results:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"] if isinstance(tr["content"], str)
                                   else json.dumps(tr["content"]),
                    })

            elif role == "assistant":
                text_parts = []
                tool_calls = []
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", f"call_{len(tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                openai_messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls if tool_calls else None,
                })
        return openai_messages

    @staticmethod
    def _anthropic_to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tool definitions to OpenAI format."""
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}, "required": []}),
                },
            })
        return openai_tools

    # ------------------------------------------------------------------
    # File / system helpers
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
