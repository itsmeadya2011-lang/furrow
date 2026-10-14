from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
)

try:
    import ollama
except Exception:  # pragma: no cover  # noqa: BLE001
    ollama = None

from furrow.config import Completion, Provider, Settings, ToolCall, settings


def _json_loads(s: str) -> Any:
    try:
        return json.loads(s) if s else {}
    except Exception:  # noqa: BLE001
        return {}


log = structlog.get_logger("furrow.llm")


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

    @property
    def ollama_client(self) -> Any:
        if ollama is None:
            raise ImportError("ollama package is not installed")
        return ollama.AsyncClient(host=self.settings.ollama_base_url)

    async def complete(self, prompt: str, system: str = "", model: str | None = None, max_tokens: int | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens=max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens=max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> Completion:
        max_tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system or "You are a helpful coding assistant.",
                "messages": [{"role": "user", "content": prompt}],
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            response = await asyncio.wait_for(
                self.anthropic.messages.create(**kwargs),
                timeout=self.settings.timeout,
            )
            tool_calls: list[ToolCall] = []
            text = ""
            for block in response.content:
                if getattr(block, "type", None) == "tool_call":
                    tc = getattr(block, "tool", None)
                    tool_calls.append(
                        ToolCall(
                            name=getattr(tc, "name", "") if tc else "",
                            arguments=getattr(tc, "input", {}) or {},
                            id=getattr(block, "id", ""),
                        )
                    )
                elif getattr(block, "type", None) == "thought":
                    pass
                else:
                    text = getattr(block, "text", "") or text
            return Completion(text=text, tool_calls=tool_calls, finish_reason=getattr(response, "stop_reason", None))
        except Exception:
            log.error("anthropic_tools_failed", model=model)
            raise

    async def _complete_openai_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> Completion:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            response = await asyncio.wait_for(
                self.openai.chat.completions.create(**kwargs),
                timeout=self.settings.timeout,
            )
            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            tool_calls: list[ToolCall] = []
            if message and getattr(message, "tool_calls", None):
                for tc in message.tool_calls:
                    func = getattr(tc, "function", None)
                    tool_calls.append(
                        ToolCall(
                            name=getattr(func, "name", "") if func else getattr(tc, "name", ""),
                            arguments=_json_loads(getattr(func, "arguments", "{}")) if func else {},
                            id=getattr(tc, "id", ""),
                        )
                    )
            text = (getattr(message, "content", None) or "") if message else ""
            return Completion(text=text, tool_calls=tool_calls, finish_reason=getattr(choice, "finish_reason", None) if choice else None)
        except Exception:
            log.error("openai_tools_failed", model=model)
            raise

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


    async def _complete_anthropic(self, prompt: str, system: str, model: str, max_tokens: int | None = None) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await asyncio.wait_for(
                        self.anthropic.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            system=system or "You are a helpful coding assistant.",
                            messages=[{"role": "user", "content": prompt}],
                        ),
                        timeout=self.settings.timeout,
                    )
                    return response.content[0].text
                except Exception:
                    log.error("anthropic_complete_failed", model=model)
                    raise

    async def _complete_openai(self, prompt: str, system: str, model: str, max_tokens: int | None = None) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await asyncio.wait_for(
                        self.openai.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system or "You are a helpful coding assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=max_tokens,
                        ),
                        timeout=self.settings.timeout,
                    )
                    return response.choices[0].message.content or ""
                except Exception:
                    log.error("openai_complete_failed", model=model)
                    raise

    async def _complete_ollama(self, prompt: str, system: str, model: str, max_tokens: int | None = None) -> str:
        if ollama is None:
            raise ImportError("ollama package is required for OLLAMA provider")
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await asyncio.wait_for(
                        self.ollama_client.chat(
                            model=model,
                            messages=[
                                {"role": "system", "content": system or "You are a helpful coding assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            options={"num_predict": max_tokens} if max_tokens else None,
                        ),
                        timeout=self.settings.timeout,
                    )
                    return response["message"]["content"] if isinstance(response, dict) else getattr(response.message, "content", "")
                except Exception:
                    log.error("ollama_complete_failed", model=model)
                    raise

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        model = model or self.settings.model
        tools = tools or []

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_tools(prompt, system, model, tools, tool_choice, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_tools(prompt, system, model, tools, tool_choice, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            # Ollama may not have full tool calling; fall back to text completion
            text = await self._complete_ollama(prompt, system, model, max_tokens)
            return Completion(text=text)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")


__all__ = ["LLMClient", "settings"]
