from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import AsyncAnthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import AsyncOpenAI

try:
    from google import genai
    from google.genai import types as gemini_types

    _GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - google-genai is a declared dependency
    genai = None  # type: ignore[assignment]
    gemini_types = None  # type: ignore[assignment]
    _GEMINI_AVAILABLE = False

from furrow.config import Provider, Settings, Usage
from furrow.config import settings as global_settings
from furrow.costs import estimate_cost

try:
    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - tenacity is a declared dependency
    _TENACITY_AVAILABLE = False

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Provider-specific retryable exceptions for Gemini
try:
    from google.api_core.exceptions import (
        DeadlineExceeded as GeminiDeadlineExceeded,
    )
    from google.api_core.exceptions import (
        InternalServerError as GeminiInternalError,
    )
    from google.api_core.exceptions import (
        ResourceExhausted as GeminiResourceExhausted,
    )
    from google.api_core.exceptions import (
        ServiceUnavailable as GeminiServiceUnavailable,
    )

    _GEMINI_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        GeminiResourceExhausted,
        GeminiInternalError,
        GeminiServiceUnavailable,
        GeminiDeadlineExceeded,
    )
except ImportError:  # pragma: no cover - google-api-core may not be available
    _GEMINI_RETRYABLE_EXCEPTIONS = ()

#: Directories that never hold first-party source and would only flood a
#: listing: version-control internals, caches, virtualenvs, dependency
#: installs, build output and editor metadata. ``node_modules`` or ``.git``
#: alone can hold tens of thousands of files, so they are pruned before the
#: tree is walked.
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
        ".furrow",
    }
)

#: Result type of a single LLM request (a plain string for `complete`, an
#: :class:`LLMResponse` for `chat`).
T = TypeVar("T")


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None
    usage: Usage = field(default_factory=Usage)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* is a transient provider error worth retrying."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    # Provider SDKs wrap transport failures in their own APIConnectionError
    # (APITimeoutError subclasses it), which carries no status code.
    if isinstance(exc, (AnthropicAPIConnectionError, OpenAIAPIConnectionError)):
        return True
    if _GEMINI_RETRYABLE_EXCEPTIONS and isinstance(exc, _GEMINI_RETRYABLE_EXCEPTIONS):
        return True
    status_code = None
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES


def _extract_anthropic_usage(response: Any) -> Usage:
    """Pull token usage out of an Anthropic messages response (graceful if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, calls=1)


def _extract_openai_usage(response: Any) -> Usage:
    """Pull token usage out of an OpenAI-compatible response (graceful if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens, calls=1)


def _extract_gemini_usage(response: Any) -> Usage:
    """Pull token usage out of a Gemini response (graceful if absent)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return Usage(calls=1)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, calls=1)


def to_gemini_tools(tools: list[dict]) -> list[dict]:
    """Convert neutral tool schemas to Gemini function definitions."""
    if not _GEMINI_AVAILABLE or gemini_types is None:
        return []
    converted: list[dict] = []
    for tool in tools:
        converted.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
        )
    return converted


def to_gemini_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to Gemini format."""
    if not _GEMINI_AVAILABLE or gemini_types is None:
        return []
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            # For assistant messages with tool calls, we need to handle both text and tool calls
            text_parts: list[str] = []
            function_calls: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    function_calls.append(
                        {
                            "name": block["name"],
                            "args": block.get("input", {}),
                            "id": block["id"],
                        }
                    )
            # Add text part if present
            if text_parts:
                converted.append({"role": "model", "parts": [{"text": "\n".join(text_parts)}]})
            # Add function calls if present
            for fc in function_calls:
                converted.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": fc["name"],
                                    "args": fc["args"],
                                    "id": fc["id"],
                                }
                            }
                        ],
                    }
                )
        elif role == "user" and isinstance(content, list):
            # For user messages with tool results
            for block in content:
                if block.get("type") == "text":
                    converted.append({"role": "user", "parts": [{"text": block.get("text", "")}]})
                elif block.get("type") == "tool_result":
                    converted.append(
                        {
                            "role": "function",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": block.get("name", ""),
                                        "response": {"content": block.get("content", "")},
                                        "id": block.get("tool_use_id", ""),
                                    }
                                }
                            ],
                        }
                    )
        else:
            # Simple text message
            role_map = {"user": "user", "assistant": "model", "system": "user"}
            gemini_role = role_map.get(role, "user")
            converted.append({"role": gemini_role, "parts": [{"text": str(content or "")}]})
    return converted


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert neutral tool schemas to Anthropic tool definitions."""
    converted: list[dict] = []
    for tool in tools:
        converted.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
        )
    return converted


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert neutral tool schemas to OpenAI function definitions."""
    converted: list[dict] = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
        )
    return converted


def _anthropic_block(block: dict) -> dict:
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block.get("text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block["id"],
            "name": block["name"],
            "input": block.get("input", {}),
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block["tool_use_id"],
            "content": block.get("content", ""),
        }
    return dict(block)


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to Anthropic message format."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if isinstance(content, list):
            converted.append(
                {"role": role, "content": [_anthropic_block(block) for block in content]}
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to OpenAI chat completions format."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
            converted.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                }
            )
        elif role == "user" and isinstance(content, list):
            user_text: list[str] = []
            for block in content:
                if block.get("type") == "text":
                    user_text.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        }
                    )
            if user_text:
                converted.append({"role": "user", "content": "\n".join(user_text)})
        else:
            converted.append({"role": role, "content": content})
    return converted


def parse_tool_arguments(raw: Any) -> dict:
    """Parse tool call arguments from a dict or JSON string ({} on failure)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None
        self._gemini: genai.Client | None = None
        self.total_usage: Usage = Usage()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key, timeout=self.settings.request_timeout
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key, timeout=self.settings.request_timeout
            )
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            base_url = self.settings.ollama_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self._ollama = AsyncOpenAI(
                api_key="ollama", base_url=base_url, timeout=self.settings.request_timeout
            )
        return self._ollama

    @property
    def gemini(self) -> genai.Client:
        if not _GEMINI_AVAILABLE or genai is None:
            raise ValueError("google-genai package is not installed")
        if self._gemini is None:
            api_key = (
                self.settings.gemini_api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            if not api_key:
                raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
            self._gemini = genai.Client(api_key=api_key)
        return self._gemini

    async def aclose(self) -> None:
        """Release any lazily-created provider clients.

        Reusable: cached references are reset to ``None`` so a later request
        creates a fresh client. Never raises.
        """
        for attr in ("_anthropic", "_openai", "_ollama", "_gemini"):
            client = getattr(self, attr)
            if client is None:
                continue
            try:
                close = getattr(client, "aclose", None)
                if close is None:
                    close = getattr(client, "close", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass
            setattr(self, attr, None)
    async def _retry(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run ``call`` and retry it while the error looks transient.

        The call is retried up to ``settings.max_retries`` attempts. Errors that
        are not retryable (see :func:`_is_retryable`) are raised immediately.
        """
        max_attempts = max(1, int(self.settings.max_retries))

        if _TENACITY_AVAILABLE:
            async for attempt_manager in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt_manager:
                    return await call()

        last_exc: Exception | None = None
        for attempt_index in range(max_attempts):
            try:
                return await call()
            except Exception as exc:
                last_exc = exc
                if attempt_index + 1 >= max_attempts or not _is_retryable(exc):
                    raise
                await asyncio.sleep(min(20.0, float(2 ** attempt_index)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry loop exited unexpectedly")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider

        async def call() -> str:
            if provider == Provider.ANTHROPIC:
                response = await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                response = await self._complete_openai(self.openai, prompt, system, model)
            elif provider == Provider.OLLAMA:
                response = await self._complete_openai(self.ollama, prompt, system, model)
            elif provider == Provider.GEMINI:
                response = await self._complete_gemini(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            return response.text

        return await self._retry(call)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
    ) -> LLMResponse:
        model = model or self.settings.model
        provider = self.settings.provider

        async def call() -> LLMResponse:
            if provider == Provider.ANTHROPIC:
                return await self._chat_anthropic(messages, tools, system, model)
            elif provider == Provider.OPENAI:
                return await self._chat_openai_compatible(
                    self.openai, messages, tools, system, model
                )
            elif provider == Provider.OLLAMA:
                return await self._chat_openai_compatible(
                    self.ollama, messages, tools, system, model
                )
            elif provider == Provider.GEMINI:
                return await self._chat_gemini(messages, tools, system, model)
            raise ValueError(f"Unsupported provider: {provider}")

        return await self._retry(call)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> LLMResponse:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                t = getattr(block, "text", "")
                if isinstance(t, str):
                    text += t
                elif t is not None:
                    text += str(t)
        usage = _extract_anthropic_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(text=text, raw=response, usage=usage)

    async def _complete_openai(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> LLMResponse:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("provider response contained no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ValueError("provider response choice has no message")
        _content = getattr(message, "content", None)
        if isinstance(_content, str):
            text = _content
        elif _content is None:
            text = ""
        else:
            text = str(_content)
        usage = _extract_openai_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(text=text, raw=response, usage=usage)

    async def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        request_tools = to_anthropic_tools(tools or [])
        request_messages = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await self.anthropic.messages.create(**kwargs)
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                t = getattr(block, "text", "")
                if isinstance(t, str):
                    text += t
                elif t is not None:
                    text += str(t)
            elif btype == "tool_use":
                tid = getattr(block, "id", None)
                name = getattr(block, "name", None)
                if not (isinstance(tid, str) and tid):
                    continue
                if not (isinstance(name, str) and name):
                    continue
                args = getattr(block, "input", None)
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(
                    ToolCall(id=tid, name=name, arguments=args)
                )
        usage = _extract_anthropic_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
            usage=usage,
        )

    async def _chat_openai_compatible(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        request_messages = to_openai_messages(messages)
        if system:
            request_messages.insert(0, {"role": "system", "content": system})
        request_tools = to_openai_tools(tools or [])
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("provider response contained no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ValueError("provider response choice has no message")
        _content = getattr(message, "content", None)
        if isinstance(_content, str):
            text = _content
        elif _content is None:
            text = ""
        else:
            text = str(_content)
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or "",
                    name=getattr(fn, "name", "") or "",
                    arguments=parse_tool_arguments(getattr(fn, "arguments", None)),
                )
            )
        usage = _extract_openai_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=getattr(choice, "finish_reason", None),
            raw=response,
            usage=usage,
        )

    async def _complete_gemini(self, prompt: str, system: str, model: str) -> LLMResponse:
        if not _GEMINI_AVAILABLE or genai is None or gemini_types is None:
            raise ValueError("google-genai package is not installed")
        contents = []
        if system:
            contents.append(
                gemini_types.Content(role="user", parts=[gemini_types.Part(text=system)])
            )
            contents.append(
                gemini_types.Content(role="model", parts=[gemini_types.Part(text="Understood.")])
            )
        contents.append(gemini_types.Content(role="user", parts=[gemini_types.Part(text=prompt)]))
        response = await self.gemini.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gemini_types.GenerateContentConfig(
                max_output_tokens=self.settings.max_tokens,
            ),
        )
        text = response.text or ""
        usage = _extract_gemini_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(text=text, raw=response, usage=usage)

    async def _chat_gemini(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        if not _GEMINI_AVAILABLE or genai is None or gemini_types is None:
            raise ValueError("google-genai package is not installed")
        request_messages = to_gemini_messages(messages)
        request_tools = to_gemini_tools(tools or [])
        config: dict[str, Any] = {
            "max_output_tokens": self.settings.max_tokens,
        }
        if system:
            config["system_instruction"] = system
        if request_tools:
            config["tools"] = [{"function_declarations": request_tools}]
        response = await self.gemini.aio.models.generate_content(
            model=model,
            contents=request_messages,
            config=gemini_types.GenerateContentConfig(**config),
        )
        text = response.text or ""
        tool_calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if parts is None:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                name = getattr(fc, "name", None)
                args = getattr(fc, "args", None)
                call_id = getattr(fc, "id", None)
                if not name:
                    continue
                tool_calls.append(
                    ToolCall(
                        id=call_id or "",
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        usage = _extract_gemini_usage(response)
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=None,
            raw=response,
            usage=usage,
        )

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List files under *directory*, as paths relative to it.

        Version-control internals, caches, virtualenvs, dependency installs,
        build output and editor metadata are skipped (see
        :data:`IGNORED_DIRECTORIES`): they are not first-party source and
        would otherwise bury the real files. A ``*.egg-info`` directory is
        skipped too. The result is sorted so the listing is deterministic.
        """
        p = Path(directory)
        if not p.exists():
            return []
        results: list[str] = []
        for root, dirs, files in os.walk(p):
            # Prune ignored directories in place so os.walk never descends.
            dirs[:] = [
                d
                for d in dirs
                if d not in IGNORED_DIRECTORIES and not d.endswith(".egg-info")
            ]
            for name in files:
                results.append(os.path.relpath(os.path.join(root, name), p))
        results.sort()
        return results
