"""LLM client with tool use, retry logic, and file operations."""
from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Base error for LLM operations."""


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

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, openai.APIError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str | list[dict[str, Any]]:
        """Complete a prompt, optionally with tool use.

        Returns text content, or a list of tool_use blocks if tools are provided.
        """
        model = model or self.settings.model
        logger.debug("llm_complete", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, tools, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, tools, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> str | list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.anthropic.messages.create(**kwargs)
        if tools:
            return [block for block in response.content if block.type == "tool_use"]
        return response.content[0].text

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> str | list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        prompt: str,
        system: str,
        tool_handlers: dict[str, Any],
        model: str | None = None,
        max_rounds: int = 5,
    ) -> str:
        """Run a tool-use loop: send prompt, execute tool calls, repeat until done."""
        model = model or self.settings.model
        tool_defs = [
            {
                "name": name,
                "description": handler.get("description", ""),
                "input_schema": handler.get("input_schema", {"type": "object", "properties": {}}),
            }
            for name, handler in tool_handlers.items()
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(max_rounds):
            response_text = await self.complete(
                prompt=messages[-1]["content"],
                system=system,
                model=model,
                tools=tool_defs if self.settings.provider == Provider.ANTHROPIC else None,
            )

            if isinstance(response_text, list):
                # Anthropic tool_use blocks
                tool_results = []
                for block in response_text:
                    if block.type == "tool_use":
                        handler = tool_handlers.get(block.name)
                        if handler:
                            try:
                                result = handler["func"](**block.input)
                            except Exception as e:
                                result = f"Error: {e}"
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(result),
                                }
                            )
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": tool_results})
            else:
                return response_text
        return "Max tool rounds reached."

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