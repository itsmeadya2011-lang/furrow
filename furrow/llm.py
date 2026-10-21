from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, Usage
from furrow.config import settings as global_settings
from furrow.pricing import compute_cost_usd, price_for

try:
    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - tenacity is a declared dependency
    _TENACITY_AVAILABLE = False

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


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
    status_code = getattr(exc, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES


def _extract_anthropic_usage(response: Any, model: str = "",
                                settings: Settings | None = None) -> Usage:
    """Pull token usage out of an Anthropic messages response (graceful if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cost = _compute_cost("anthropic", model, input_tokens, output_tokens, settings)
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, calls=1, cost_usd=cost)


def _extract_openai_usage(response: Any, model: str = "",
                             settings: Settings | None = None) -> Usage:
    """Pull token usage out of an OpenAI-compatible response (graceful if absent)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = _compute_cost("openai", model, prompt_tokens, completion_tokens, settings)
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        calls=1,
        cost_usd=cost,
    )


def _compute_cost(provider: str, model: str, input_tokens: int, output_tokens: int,
                      settings: Settings | None = None) -> float:
    """Compute USD cost for the given tokens using *settings* pricing.

    Returns 0.0 when cost tracking is disabled or the model is unknown and the
    default price is free. When *settings* is None the global settings are used.
    """
    if settings is None:
        settings = global_settings
    if not settings.track_cost:
        return 0.0
    input_per_1k, output_per_1k = price_for(
        provider,
        model,
        custom=settings.model_pricing,
        default_per_1k=settings.default_cost_per_1k,
    )
    return compute_cost_usd(input_tokens, output_tokens, input_per_1k, output_per_1k)


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
        # Per-role API-key overrides so different roles can use different keys.
        self._role_keys: dict[str, str | None] = {}

    def _role_provider(self, role: str) -> Provider:
        """Return the provider to use for *role* (planner/worker/tester/default)."""
        override = getattr(self.settings, f"{role}_provider", None)
        if override is not None:
            return Provider(override) if not isinstance(override, Provider) else override
        return self.settings.provider

    def _role_model(self, role: str) -> str:
        """Return the model to use for *role*."""
        return getattr(self.settings, f"{role}_model", None) or self.settings.model

    def _role_api_key(self, role: str) -> str | None:
        """Return the API key to use for *role*, or None to use the default."""
        override = getattr(self.settings, f"{role}_api_key", None)
        if override:
            return override
        return None

    def _client_for_provider(self, provider: Provider, api_key: str | None = None) -> Any:
        """Return the underlying SDK client for *provider*, creating it lazily."""
        if provider == Provider.ANTHROPIC:
            if self._anthropic is None:
                key = api_key or self.settings.anthropic_api_key or os.getenv(
                    "ANTHROPIC_API_KEY"
                )
                if not key:
                    raise ValueError("ANTHROPIC_API_KEY is not set")
                self._anthropic = AsyncAnthropic(
                    api_key=key, timeout=self.settings.request_timeout
                )
            return self._anthropic
        if provider == Provider.OPENAI:
            if self._openai is None:
                key = api_key or self.settings.openai_api_key or os.getenv(
                    "OPENAI_API_KEY"
                )
                if not key:
                    raise ValueError("OPENAI_API_KEY is not set")
                self._openai = AsyncOpenAI(
                    api_key=key, timeout=self.settings.request_timeout
                )
            return self._openai
        if provider == Provider.OLLAMA:
            if self._ollama is None:
                base_url = self.settings.ollama_base_url.rstrip("/") + "/v1"
                self._ollama = AsyncOpenAI(
                    api_key="ollama", base_url=base_url, timeout=self.settings.request_timeout
                )
            return self._ollama
        raise ValueError(f"Unsupported provider: {provider}")

    # Backwards-compatible properties for direct client access.
    @property
    def anthropic(self) -> AsyncAnthropic:
        return self._client_for_provider(Provider.ANTHROPIC)

    @property
    def openai(self) -> AsyncOpenAI:
        return self._client_for_provider(Provider.OPENAI)

    @property
    def ollama(self) -> AsyncOpenAI:
        return self._client_for_provider(Provider.OLLAMA)

    def _provider_for(self, provider: Provider | str) -> Provider:
        if isinstance(provider, Provider):
            return provider
        return Provider(provider)

    def _fallback_chain(self, role: str, model: str | None = None) -> list[tuple[Provider, str]]:
        """Return the ordered (provider, model) list to try for *role*.

        The primary (role provider, role model or *model* override) is first;
        the configured ``fallback_chain`` follows; the global default is last.
        """
        chain: list[tuple[Provider, str]] = []
        primary_provider = self._role_provider(role)
        primary_model = model or self._role_model(role)
        chain.append((primary_provider, primary_model))
        for provider_str, fallback_model in self.settings.fallback_chain:
            chain.append((self._provider_for(provider_str), fallback_model))
        if (self.settings.provider, self.settings.model) not in chain:
            chain.append((self.settings.provider, self.settings.model))
        return chain

    async def _retry_with_fallback(self, call, role: str = "default", model: str | None = None):
        """Run *call* with retry, then walk the fallback chain on retryable errors.

        *call* must be a zero-arg callable returning an awaitable. Each entry in
        the fallback chain is tried with a fresh retry budget; only retryable
        errors trigger the next entry.
        """
        chain = self._fallback_chain(role, model)
        last_exc: Exception | None = None
        for provider, model in chain:
            try:
                return await self._retry(lambda p=provider, m=model: call(p, m))
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM fallback chain exited unexpectedly")

    async def _retry(self, call):
        max_attempts = max(1, int(self.settings.max_retries))

        if _TENACITY_AVAILABLE:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    return await call()

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= max_attempts or not _is_retryable(exc):
                    raise
                await asyncio.sleep(min(20.0, float(2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry loop exited unexpectedly")

    async def complete(self, prompt: str, system: str = "", model: str | None = None,
                       role: str = "default") -> str:
        """Run a completion for *role* (planner/worker/tester/default).

        When *model* is given it overrides the role's configured model. The
        fallback chain is walked on retryable errors.
        """
        api_key = self._role_api_key(role)

        async def call(prov: Provider, mod: str) -> str:
            if prov == Provider.ANTHROPIC:
                response = await self._complete_anthropic(prompt, system, mod, api_key)
            elif prov == Provider.OPENAI or prov == Provider.OLLAMA:
                client = self._client_for_provider(prov, api_key)
                response = await self._complete_openai(client, prompt, system, mod)
            else:
                raise ValueError(f"Unsupported provider: {prov}")
            return response.text

        return await self._retry_with_fallback(call, role, model)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
        role: str = "default",
    ) -> LLMResponse:
        """Run a chat completion for *role* with tool calling and fallback."""
        api_key = self._role_api_key(role)

        async def call(prov: Provider, mod: str) -> LLMResponse:
            if prov == Provider.ANTHROPIC:
                return await self._chat_anthropic(messages, tools, system, mod, api_key)
            elif prov == Provider.OPENAI or prov == Provider.OLLAMA:
                client = self._client_for_provider(prov, api_key)
                return await self._chat_openai_compatible(
                    client, messages, tools, system, mod
                )
            raise ValueError(f"Unsupported provider: {prov}")

        return await self._retry_with_fallback(call, role, model)

    async def _complete_anthropic(self, prompt: str, system: str, model: str,
                                  api_key: str | None = None) -> LLMResponse:
        client = self._client_for_provider(Provider.ANTHROPIC, api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        usage = _extract_anthropic_usage(response, model, self.settings)
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
        text = response.choices[0].message.content or ""
        usage = _extract_openai_usage(response, model, self.settings)
        self.total_usage.add(usage)
        return LLMResponse(text=text, raw=response, usage=usage)

    async def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
        model: str,
        api_key: str | None = None,
    ) -> LLMResponse:
        client = self._client_for_provider(Provider.ANTHROPIC, api_key)
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
        response = await client.messages.create(**kwargs)
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        usage = _extract_anthropic_usage(response, model, self.settings)
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
        choice = response.choices[0]
        message = choice.message
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
        usage = _extract_openai_usage(response, model, self.settings)
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
