from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, Usage
from furrow.config import settings as global_settings
from furrow.costs import estimate_cost

try:
    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - tenacity is a declared dependency
    _TENACITY_AVAILABLE = False

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

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


class _ProviderClient(ABC):
    """Abstract base class for provider-specific client logic."""

    @abstractmethod
    async def complete(self, prompt: str, system: str, model: str) -> LLMResponse:
        """Execute a completion request."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
    ) -> LLMResponse:
        """Execute a chat completion request with optional tools."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        ...


class _AnthropicClient(_ProviderClient):
    """Anthropic provider client implementation."""

    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def complete(self, prompt: str, system: str, model: str) -> LLMResponse:
        response = await self._client.messages.create(
            model=model,
            max_tokens=self._settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content or []:
            if block.type == "text":
                text += block.text
        usage = _extract_anthropic_usage(response)
        self._apply_cost(model, usage)
        return LLMResponse(text=text, raw=response, usage=usage)

    async def chat(
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
            "max_tokens": self._settings.max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await self._client.messages.create(**kwargs)
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content or []:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        usage = _extract_anthropic_usage(response)
        self._apply_cost(model, usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
            usage=usage,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is None:
            close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def _apply_cost(self, model: str, usage: Usage) -> None:
        try:
            usage.cost_usd = estimate_cost(model, usage, self._settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0


class _OpenAICompatibleClient(_ProviderClient):
    """OpenAI-compatible provider client implementation (OpenAI, Ollama)."""

    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def complete(self, prompt: str, system: str, model: str) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=model,
            max_tokens=self._settings.max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return self._process_openai_response(response, model)

    async def chat(
        self,
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
            "max_tokens": self._settings.max_tokens,
            "messages": request_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools
        response = await self._client.chat.completions.create(**kwargs)
        return self._process_openai_response(response, model)

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is None:
            close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def _process_openai_response(self, response: Any, model: str) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("provider response contained no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ValueError("provider response choice has no message")
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_arguments(tc.function.arguments),
                )
            )
        usage = _extract_openai_usage(response)
        self._apply_cost(model, usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            raw=response,
            usage=usage,
        )

    def _apply_cost(self, model: str, usage: Usage) -> None:
        try:
            usage.cost_usd = estimate_cost(model, usage, self._settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0


class LLMClient:
    """Unified LLM client supporting Anthropic, OpenAI, and Ollama providers.

    Provides both simple completion (`complete`) and tool-calling chat (`chat`)
    interfaces with automatic retry on transient errors, usage tracking, and
    cost estimation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else global_settings
        self._clients: dict[Provider, _ProviderClient] = {}
        self.total_usage: Usage = Usage()
        # Compatibility shims for tests and external code
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def settings(self) -> Settings:
        """Settings used by this client (for backward compatibility)."""
        return self._settings

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily-created Anthropic client (for backward compatibility)."""
        if self._anthropic is None:
            api_key = self._settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key, timeout=self._settings.request_timeout
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily-created OpenAI client (for backward compatibility)."""
        if self._openai is None:
            api_key = self._settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key, timeout=self._settings.request_timeout
            )
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Lazily-created Ollama client (for backward compatibility)."""
        if self._ollama is None:
            base_url = self._settings.ollama_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self._ollama = AsyncOpenAI(
                api_key="ollama", base_url=base_url, timeout=self._settings.request_timeout
            )
        return self._ollama

    def _get_client(self, provider: Provider) -> _ProviderClient:
        """Get or create a provider-specific client (new internal API)."""
        if provider in self._clients:
            return self._clients[provider]

        if provider == Provider.ANTHROPIC:
            # Use the existing lazy client if available, otherwise create new
            if self._anthropic is not None:
                provider_client = _AnthropicClient(self._anthropic, self._settings)
            else:
                api_key = self._settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY is not set")
                client = AsyncAnthropic(
                    api_key=api_key, timeout=self._settings.request_timeout
                )
                self._anthropic = client
                provider_client = _AnthropicClient(client, self._settings)
        elif provider == Provider.OPENAI:
            if self._openai is not None:
                provider_client = _OpenAICompatibleClient(self._openai, self._settings)
            else:
                api_key = self._settings.openai_api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not set")
                client = AsyncOpenAI(
                    api_key=api_key, timeout=self._settings.request_timeout
                )
                self._openai = client
                provider_client = _OpenAICompatibleClient(client, self._settings)
        elif provider == Provider.OLLAMA:
            if self._ollama is not None:
                provider_client = _OpenAICompatibleClient(self._ollama, self._settings)
            else:
                base_url = self._settings.ollama_base_url.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = base_url + "/v1"
                client = AsyncOpenAI(
                    api_key="ollama", base_url=base_url, timeout=self._settings.request_timeout
                )
                self._ollama = client
                provider_client = _OpenAICompatibleClient(client, self._settings)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self._clients[provider] = provider_client
        return provider_client

    async def aclose(self) -> None:
        """Release any lazily-created provider clients.

        Reusable: cached references are reset to ``None`` so a later request
        creates a fresh client. Never raises.
        """
        # Close new internal clients
        for provider_client in self._clients.values():
            try:
                await provider_client.aclose()
            except Exception:
                pass
        self._clients.clear()

        # Close compatibility shim clients
        for attr in ("_anthropic", "_openai", "_ollama"):
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
        max_attempts = max(1, int(self._settings.max_retries))

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
        """Simple text completion without tool calling.

        Args:
            prompt: The user prompt to complete.
            system: Optional system prompt.
            model: Optional model override (defaults to settings.model).

        Returns:
            The completion text.
        """
        model = model or self._settings.model
        provider = self._settings.provider
        client = self._get_client(provider)

        async def call() -> str:
            response = await client.complete(prompt, system, model)
            self.total_usage.add(response.usage)
            return response.text

        return await self._retry(call)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
    ) -> LLMResponse:
        """Chat completion with optional tool calling.

        Args:
            messages: Conversation history in neutral format.
            tools: Optional tool schemas in neutral format.
            system: Optional system prompt.
            model: Optional model override (defaults to settings.model).

        Returns:
            LLMResponse with text, tool_calls, stop_reason, and usage.
        """
        model = model or self._settings.model
        provider = self._settings.provider
        client = self._get_client(provider)

        async def call() -> LLMResponse:
            response = await client.chat(messages, tools, system, model)
            self.total_usage.add(response.usage)
            return response

        return await self._retry(call)

    async def read_file(self, path: str | Path) -> str:
        """Read a file asynchronously."""
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write a file asynchronously, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List all files in a directory recursively, relative to the directory."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


__all__ = [
    "ToolCall",
    "LLMResponse",
    "LLMClient",
    "to_anthropic_tools",
    "to_openai_tools",
    "to_anthropic_messages",
    "to_openai_messages",
    "parse_tool_arguments",
    "_is_retryable",
    "_extract_anthropic_usage",
    "_extract_openai_usage",
]