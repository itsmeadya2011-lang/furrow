from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, Usage
from furrow.config import settings as global_settings
from furrow.costs import estimate_cost
from furrow.logging import get_logger

try:
    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential
    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - tenacity is a declared dependency
    _TENACITY_AVAILABLE = False

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# Circuit breaker states
_CIRCUIT_CLOSED = "closed"
_CIRCUIT_OPEN = "open"
_CIRCUIT_HALF_OPEN = "half_open"

_logger = get_logger(__name__)

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


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for provider failures."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    state: str = _CIRCUIT_CLOSED
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds to wait before trying half-open

    def record_success(self) -> None:
        self.failure_count = 0
        self.success_count += 1
        if self.state == _CIRCUIT_HALF_OPEN and self.success_count >= self.success_threshold:
            self.state = _CIRCUIT_CLOSED
            _logger.info("circuit_breaker_closed", provider=self.__class__.__name__)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()
        if self.state == _CIRCUIT_CLOSED and self.failure_count >= self.failure_threshold:
            self.state = _CIRCUIT_OPEN
            _logger.warning("circuit_breaker_opened", provider=self.__class__.__name__, failures=self.failure_count)
        elif self.state == _CIRCUIT_HALF_OPEN:
            self.state = _CIRCUIT_OPEN
            _logger.warning("circuit_breaker_reopened", provider=self.__class__.__name__)

    def can_execute(self) -> bool:
        if self.state == _CIRCUIT_CLOSED:
            return True
        if self.state == _CIRCUIT_OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = _CIRCUIT_HALF_OPEN
                _logger.info("circuit_breaker_half_open", provider=self.__class__.__name__)
                return True
            return False
        # HALF_OPEN
        return True


def _is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* is a transient provider error worth retrying."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
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
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        calls=1,
    )


def _extract_openai_usage(response: Any) -> Usage:
    """Pull token usage out of an OpenAI-compatible response (graceful if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    # OpenAI doesn't currently report cache tokens separately
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        calls=1,
    )


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
        self.total_usage: Usage = Usage()
        
        # Circuit breakers for each provider
        self._anthropic_circuit = CircuitBreaker()
        self._openai_circuit = CircuitBreaker()
        self._ollama_circuit = CircuitBreaker()

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
            base_url = self.settings.ollama_base_url.rstrip("/") + "/v1"
            self._ollama = AsyncOpenAI(
                api_key="ollama", base_url=base_url, timeout=self.settings.request_timeout
            )
        return self._ollama

    def _get_circuit_breaker(self, provider: Provider) -> CircuitBreaker:
        """Get the circuit breaker for the given provider."""
        if provider == Provider.ANTHROPIC:
            return self._anthropic_circuit
        elif provider == Provider.OPENAI:
            return self._openai_circuit
        elif provider == Provider.OLLAMA:
            return self._ollama_circuit
        return self._anthropic_circuit  # default

    async def aclose(self) -> None:
        """Release any lazily-created provider clients.

        Reusable: cached references are reset to ``None`` so a later request
        creates a fresh client. Never raises.
        """
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
    
    async def _retry_with_circuit(
        self, call: Callable[[], Awaitable[T]], provider: Provider
    ) -> T:
        """Run ``call`` with circuit breaker and retry logic."""
        circuit = self._get_circuit_breaker(provider)
        
        if not circuit.can_execute():
            raise RuntimeError(f"Circuit breaker open for {provider.value}")
        
        max_attempts = max(1, int(self.settings.max_retries))
        last_exc: Exception | None = None
        
        for attempt_index in range(max_attempts):
            try:
                result = await call()
                circuit.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                circuit.record_failure()
                
                if attempt_index + 1 >= max_attempts or not _is_retryable(exc):
                    raise
                
                # Exponential backoff with jitter
                wait_time = min(20.0, float(2 ** attempt_index)) + (attempt_index * 0.1)
                _logger.warning(
                    "llm_retry_attempt",
                    provider=provider.value,
                    attempt=attempt_index + 1,
                    max_attempts=max_attempts,
                    wait_time=wait_time,
                    error=str(exc),
                )
                await asyncio.sleep(wait_time)
        
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
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            return response.text

        return await self._retry_with_circuit(call, provider)

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
            raise ValueError(f"Unsupported provider: {provider}")

        return await self._retry_with_circuit(call, provider)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> LLMResponse:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content or []:
            if block.type == "text":
                text += block.text
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
        text = message.content or ""
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
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
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
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
        self.total_usage.add(usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            raw=response,
            usage=usage,
        )

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
