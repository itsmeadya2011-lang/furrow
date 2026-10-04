"""Provider-agnostic LLM access plus sandboxed workspace file helpers.

The client is responsible for four things:

* text extraction that tolerates ``thinking``/reasoning blocks,
* small retry budgets for transient provider failures,
* a tool-use loop (:meth:`LLMClient.agent_loop`) that stays provider-agnostic --
  the *caller* owns the side effects and is handed the text result of each tool
  call,
* file access confined to ``settings.workspace``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NamedTuple

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow import config
from furrow.config import Provider, Settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
DEFAULT_MAX_TOOL_TURNS = 10
MAX_ATTEMPTS = 3
MAX_TOKENS = 4096

#: Directories that are noise (or enormous) when surveying a workspace.
IGNORED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def _default_settings() -> Settings:
    """The process-wide settings singleton.

    Looked up on the module at call time so that patching ``furrow.config.settings``
    at runtime is honoured.
    """
    return config.settings


def _is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is worth another attempt.

    Retries cover rate limits, timeouts, connection failures and 5xx responses.
    Bad requests, auth failures and a missing API key are *not* transient -- the
    latter surfaces as a ``ValueError`` from the provider property below.
    """
    if isinstance(exc, (anthropic.RateLimitError, openai.RateLimitError)):
        return True
    if isinstance(exc, (anthropic.APITimeoutError, openai.APITimeoutError)):
        return True
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    if isinstance(exc, (anthropic.InternalServerError, openai.InternalServerError)):
        return True
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status >= 500
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError))


_RETRY = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an SDK content block (typed object or plain dict)."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _block_text(block: Any) -> str | None:
    value = _block_get(block, "text")
    return value if isinstance(value, str) else None


def _as_param(block: Any) -> Any:
    """Convert an SDK response block back into a request parameter."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return block


def _anthropic_text(blocks: list[Any]) -> str:
    """Join the text-bearing blocks, ignoring ``thinking``/``tool_use`` ones.

    A response whose blocks are *all* non-text yields ``""``. Falling back to
    ``str(block)`` would surface a pydantic repr such as
    ``ToolUseBlock(id=..., name=...)`` as if the model had written it, which is
    exactly what a tool-heavy turn looks like when ``max_turns`` runs out.
    Callers keep their last non-empty answer so exhaustion still returns text.
    """
    return "\n".join(text for text in (_block_text(block) for block in blocks) if text)


def _openai_text(message: Any) -> str:
    """Assistant text from a chat completion message, tolerating empty content."""
    content = _block_get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [text for text in (_block_text(part) for part in content) if text]
        if parts:
            return "\n".join(parts)
    return ""


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic tool schema into an OpenAI function tool."""
    schema = tool.get("input_schema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": schema,
        },
    }


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {"input": parsed}


def _unsupported_provider(provider: Any) -> ValueError:
    supported = ", ".join(sorted(p.value for p in (Provider.ANTHROPIC, Provider.OPENAI)))
    return ValueError(
        f"Unsupported provider: {getattr(provider, 'value', provider)!r}. "
        f"Furrow currently supports {supported} only -- Ollama support is not implemented. "
        "Set FURROW_PROVIDER=anthropic (or FURROW_PROVIDER=openai) and try again."
    )


class _ToolResult(NamedTuple):
    """Text handed back to the model, plus whether the tool itself failed."""

    text: str
    is_error: bool = False


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings()
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        raise _unsupported_provider(self.settings.provider)

    async def agent_loop(
        self,
        prompt: str,
        tools: list[dict],
        execute: Callable[[str, dict], Awaitable[str]],
        system: str = "",
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Drive a model through as many tool rounds as it asks for.

        ``tools`` are Anthropic-style schemas (``name``/``description``/
        ``input_schema``); they are translated for OpenAI automatically.
        ``execute(tool_name, tool_input)`` is awaited for every requested call and
        its returned text is fed back to the model. The loop stops when the model
        answers without requesting a tool, when ``max_turns`` rounds have run, or
        when the provider returns nothing usable -- the last text produced is
        returned instead of raising.
        """
        model = model or self.settings.model
        limit = (
            max_turns or getattr(self.settings, "max_tool_turns", None) or (DEFAULT_MAX_TOOL_TURNS)
        )
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._loop_anthropic(prompt, tools, execute, system, model, limit)
        if self.settings.provider == Provider.OPENAI:
            return await self._loop_openai(prompt, tools, execute, system, model, limit)
        raise _unsupported_provider(self.settings.provider)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._anthropic_create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(list(getattr(response, "content", None) or []))

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._openai_chat_create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            return ""
        return _openai_text(_block_get(choices[0], "message"))

    async def _loop_anthropic(
        self,
        prompt: str,
        tools: list[dict],
        execute: Callable[[str, dict], Awaitable[str]],
        system: str,
        model: str,
        max_turns: int,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        last_text = ""
        for _ in range(max(1, max_turns)):
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": system or DEFAULT_SYSTEM_PROMPT,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
            response = await self._anthropic_create(**kwargs)

            blocks = list(getattr(response, "content", None) or [])
            text = _anthropic_text(blocks)
            if text:
                # A tool-only turn carries no text, so keep the last real answer.
                last_text = text
            tool_uses = [b for b in blocks if _block_get(b, "type") == "tool_use"]
            if not tool_uses:
                return last_text

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                name = _block_get(block, "name", "") or ""
                tool_input = _block_get(block, "input") or {}
                if not isinstance(tool_input, dict):
                    tool_input = {"input": tool_input}
                outcome = await self._run_tool(execute, name, tool_input)
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": _block_get(block, "id", ""),
                    "content": outcome.text,
                }
                if outcome.is_error:
                    result["is_error"] = True
                results.append(result)

            messages.append({"role": "assistant", "content": [_as_param(b) for b in blocks]})
            messages.append({"role": "user", "content": results})
        return last_text

    async def _loop_openai(
        self,
        prompt: str,
        tools: list[dict],
        execute: Callable[[str, dict], Awaitable[str]],
        system: str,
        model: str,
        max_turns: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_specs = [_openai_tool(tool) for tool in tools]
        last_text = ""
        for _ in range(max(1, max_turns)):
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if tool_specs:
                kwargs["tools"] = tool_specs
                kwargs["tool_choice"] = "auto"
            response = await self._openai_chat_create(**kwargs)

            choices = list(getattr(response, "choices", None) or [])
            if not choices:
                return last_text
            message = _block_get(choices[0], "message")
            text = _openai_text(message)
            if text:
                # Same rule as the Anthropic loop: a tool-call turn has no content.
                last_text = text
            tool_calls = list(_block_get(message, "tool_calls") or [])
            if not tool_calls:
                return last_text

            messages.append(
                {
                    "role": "assistant",
                    "content": _block_get(message, "content"),
                    "tool_calls": [
                        {
                            "id": _block_get(call, "id", ""),
                            "type": "function",
                            "function": {
                                "name": _block_get(_block_get(call, "function"), "name", ""),
                                "arguments": _block_get(
                                    _block_get(call, "function"), "arguments", "{}"
                                ),
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                function = _block_get(call, "function")
                outcome = await self._run_tool(
                    execute,
                    _block_get(function, "name", "") or "",
                    _parse_tool_arguments(_block_get(function, "arguments")),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _block_get(call, "id", ""),
                        "content": outcome.text,
                    }
                )
        return last_text

    async def _run_tool(
        self,
        execute: Callable[[str, dict], Awaitable[str]],
        name: str,
        tool_input: dict[str, Any],
    ) -> _ToolResult:
        """Run one caller-supplied tool, turning failures into model-visible text."""
        try:
            text = await execute(name, tool_input)
        except Exception as exc:  # the model should see the failure and be able to retry
            logger.warning("Tool %s failed: %s", name, exc)
            return _ToolResult(f"Error: {type(exc).__name__}: {exc}", is_error=True)
        return _ToolResult(text if isinstance(text, str) else str(text))

    @_RETRY
    async def _anthropic_create(self, **kwargs: Any) -> Any:
        return await self.anthropic.messages.create(**kwargs)

    @_RETRY
    async def _openai_chat_create(self, **kwargs: Any) -> Any:
        return await self.openai.chat.completions.create(**kwargs)

    # -- sandboxed workspace access ---------------------------------------

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` inside the workspace, refusing anything that escapes it."""
        root = Path(self.settings.workspace).expanduser().resolve()
        candidate = Path(path).expanduser()
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not candidate.is_absolute() and not target.exists():
            # ``settings.workspace`` may itself be relative; honour CWD-relative input.
            fallback = candidate.resolve()
            if fallback.exists():
                target = fallback
        if target != root and root not in target.parents:
            raise ValueError(f"Refusing to access {str(path)!r}: outside workspace {root}")
        return target

    async def read_file(self, path: str | Path) -> str:
        target = self.resolve_path(path)
        async with aiofiles.open(target, encoding="utf-8", errors="replace") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        target = self.resolve_path(path)
        # ``target`` is never a directory from here on, so ``target.parent`` is
        # strictly inside the workspace root and the temp file below cannot be
        # created outside the sandbox.
        if target.is_dir():
            raise ValueError(f"Refusing to overwrite directory {str(path)!r} with a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target then swap, so readers never see a partial file.
        tmp = target.with_name(f".{target.name}.furrow-tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(content)
        os.replace(tmp, target)

    def list_files(self, directory: str | Path = ".") -> list[str]:
        root = self.resolve_path(directory)
        if not root.is_dir():
            return []
        found: list[str] = []
        for entry in sorted(root.rglob("*")):
            parts = entry.relative_to(root).parts
            if any(part in IGNORED_DIRS for part in parts):
                continue
            if entry.is_file():
                found.append(str(Path(*parts)))
        return sorted(found)
