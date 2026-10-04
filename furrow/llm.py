"""Provider-neutral LLM access for furrow.

`LLMClient` wraps the Anthropic, OpenAI and Ollama (OpenAI-compatible)
backends behind a single async surface: plain completion, JSON completion
with self-repair, and a bounded tool-use loop that surfaces every
requested tool call to the caller.  The client never executes tools
itself; the worker's outer loop (furrow/agents/worker.py) owns tool
execution and feeds results back.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, TypeVar

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicTimeoutError,
)
from anthropic import (
    AsyncAnthropic,
)
from anthropic import (
    InternalServerError as AnthropicServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.types import Message, TextBlock, ToolUseBlock
from openai import (
    APIConnectionError as OpenAIConnectionError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    AsyncOpenAI,
)
from openai import (
    InternalServerError as OpenAIServerError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from furrow.config import Provider, Settings, get_settings

__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMResponseError",
    "LLMRateLimitError",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
]

_TRUNCATION_MARKER = "\n[warning: response truncated at max_tokens]"
_OLLAMA_DEFAULT_MODEL = "llama3.1"

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicServerError,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIServerError,
)
_RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (
    AnthropicRateLimitError,
    OpenAIRateLimitError,
)


class LLMError(RuntimeError):
    """Base class for every LLM failure surfaced by this module."""


class LLMConfigurationError(LLMError):
    """A required API key is missing or the provider is unknown."""


class LLMResponseError(LLMError):
    """The model produced malformed or unparseable output."""


class LLMRateLimitError(LLMError):
    """Retries were exhausted while the provider kept returning HTTP 429."""


class ToolSpec(TypedDict):
    """Provider-neutral description of a tool the model may request."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """Outcome of a `complete_with_tools` call.

    `tool_calls` is the ordered log of every tool the model requested;
    each entry is {"name": str, "arguments": dict, "result": str} with
    an empty "result" because this module never executes tools.
    """

    text: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0


_T = TypeVar("_T")


def _parse_json_payload(raw: str) -> Any:
    """Parse model output that may be fenced and/or wrapped in prose."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence is not None:
        text = fence.group(1).strip()
    for candidate in (text, _json_fragment(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMResponseError(f"Model output was not valid JSON: {raw[:200]!r}")


def _json_fragment(text: str) -> str:
    """Return the outermost {...} or [...] block of `text` ("" when absent)."""
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    ends = [index for index in (text.rfind("}"), text.rfind("]")) if index != -1]
    if not starts or not ends:
        return ""
    return text[min(starts) : max(ends) + 1]


def _anthropic_text(response: Message) -> str:
    """Concatenate every text block; flag truncation instead of failing silently."""
    text = "".join(block.text for block in response.content if isinstance(block, TextBlock))
    if response.stop_reason == "max_tokens":
        return text + _TRUNCATION_MARKER
    return text


def _openai_text(response: ChatCompletion) -> str:
    """Return the message content; flag truncation instead of failing silently."""
    if not response.choices:
        return ""
    choice = response.choices[0]
    text = choice.message.content or ""
    if choice.finish_reason == "length":
        text += _TRUNCATION_MARKER
    return text


def _anthropic_usage(response: Message) -> tuple[int, int]:
    return response.usage.input_tokens, response.usage.output_tokens


def _openai_usage(response: ChatCompletion) -> tuple[int, int]:
    usage = response.usage
    if usage is None:
        return 0, 0
    return usage.prompt_tokens, usage.completion_tokens


class LLMClient:
    """Async facade over the Anthropic, OpenAI and Ollama model APIs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None
        self._usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
        }

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built Anthropic client; raises LLMConfigurationError without a key."""
        if self._anthropic is None:
            api_key = self.settings.api_key_for(Provider.ANTHROPIC)
            if not api_key:
                raise LLMConfigurationError(
                    "No Anthropic API key configured "
                    "(set FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)"
                )
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built OpenAI client; raises LLMConfigurationError without a key."""
        if self._openai is None:
            api_key = self.settings.api_key_for(Provider.OPENAI)
            if not api_key:
                raise LLMConfigurationError(
                    "No OpenAI API key configured "
                    "(set FURROW_OPENAI_API_KEY or OPENAI_API_KEY)"
                )
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._openai

    def _ollama_client(self) -> AsyncOpenAI:
        """Lazily build an OpenAI-compatible client pointed at the Ollama server.

        Ollama needs no real key, so a placeholder is supplied and the
        SDK's own retry loop is disabled - tenacity owns all retry policy.
        """
        if self._ollama is None:
            base_url = self.settings.ollama_base_url.rstrip("/")
            self._ollama = AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama",
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._ollama

    @property
    def usage(self) -> dict[str, int]:
        """Cumulative counters: {"input_tokens": int, "output_tokens": int, "requests": int}."""
        return dict(self._usage)

    def reset_usage(self) -> None:
        """Zero the cumulative token and request counters."""
        self._usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
        }

    def _record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._usage["input_tokens"] += input_tokens
        self._usage["output_tokens"] += output_tokens
        self._usage["requests"] += 1

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a plain text completion for `prompt`.

        `model`, `max_tokens` and `temperature` fall back to the matching
        `Settings` field.  `system` is passed through verbatim and omitted
        entirely when empty - no generic persona is injected here.
        """
        settings = self.settings
        resolved_model = self._resolve_model(model)
        resolved_max_tokens = (
            max_tokens if max_tokens is not None else settings.max_tokens
        )
        resolved_temperature = (
            temperature if temperature is not None else settings.temperature
        )
        provider = settings.provider
        if provider == Provider.ANTHROPIC:
            anthropic_reply = await self._retry(
                lambda: self._anthropic_request(
                    prompt,
                    system=system,
                    model=resolved_model,
                    max_tokens=resolved_max_tokens,
                    temperature=resolved_temperature,
                )
            )
            # FIX (test suite): plain completions never recorded their token usage, so
            # `usage` only ever counted tool-loop rounds. Planner and tester traffic is
            # the bulk of a run, and `Orchestrator.snapshot()` surfaces these counters.
            self._record_usage(*_anthropic_usage(anthropic_reply))
            return _anthropic_text(anthropic_reply)
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            if provider == Provider.OPENAI:
                client = self.openai
            else:
                client = self._ollama_client()
            chat_reply = await self._retry(
                lambda: self._openai_request(
                    prompt,
                    client=client,
                    system=system,
                    model=resolved_model,
                    max_tokens=resolved_max_tokens,
                    temperature=resolved_temperature,
                )
            )
            self._record_usage(*_openai_usage(chat_reply))
            return _openai_text(chat_reply)
        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        """Generate a completion and return its parsed JSON value.

        Markdown ```json fences are stripped and leading/trailing prose
        is tolerated.  When parsing fails, the prompt is re-issued with
        an explicit "reply with JSON only" repair instruction, up to
        `max_attempts` times, before raising LLMResponseError.
        """
        attempts = max(1, max_attempts)
        current = prompt
        for attempt in range(attempts):
            raw = await self.complete(current, system=system, model=model)
            try:
                return _parse_json_payload(raw)
            except LLMResponseError:
                if attempt + 1 >= attempts:
                    raise
                current = (
                    f"{prompt}\n\nYour previous reply was not valid JSON. "
                    "Reply with JSON only."
                )
        raise LLMResponseError("JSON completion produced no parseable response")

    async def complete_with_tools(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        *,
        system: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
    ) -> ToolResult:
        """Drive a bounded tool-use conversation and surface requested tool calls.

        HAND-OFF CONTRACT: this method does NOT execute tools and does NOT
        feed tool results back into the conversation.  Every round-trip
        calls the model, converts its tool_use / function_call blocks into
        ToolCall objects and awaits `on_tool_call(call)` for each one.
        Because no results are fed back, a model that keeps requesting
        tools repeats its request until the iteration cap is reached -
        that is by design: the worker's outer loop (furrow/agents/worker.py)
        executes the tools and feeds results back, e.g. by re-invoking
        with an augmented prompt.

        stop_reason is "end_turn" when the model asked for no tools,
        "max_iterations" when the cap was reached, and "error" (returned,
        never raised) when the model call failed after retries - in that
        case `text` carries a one-line error description for the caller
        to surface.
        """
        if max_iterations is None:
            cap = max(1, self.settings.max_tool_iterations)
        else:
            cap = max(1, max_iterations)
        resolved_model = self._resolve_model(model)
        tool_log: list[dict[str, Any]] = []
        text = ""
        iterations = 0
        input_tokens = 0
        output_tokens = 0

        def finish(stop_reason: str, final_text: str) -> ToolResult:
            return ToolResult(
                text=final_text,
                tool_calls=tool_log,
                iterations=iterations,
                stop_reason=stop_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        while iterations < cap:
            iterations += 1
            try:
                text, requested, round_input, round_output = await self._tool_round_trip(
                    prompt, tools, system=system, model=resolved_model
                )
            except Exception as err:
                return finish("error", f"[error] {type(err).__name__}: {err}")
            input_tokens += round_input
            output_tokens += round_output
            self._record_usage(round_input, round_output)
            for call in requested:
                tool_log.append(
                    {"name": call.name, "arguments": call.arguments, "result": ""}
                )
                if on_tool_call is not None:
                    await on_tool_call(call)
            if not requested:
                return finish("end_turn", text)
        return finish("max_iterations", text)

    def _resolve_model(self, model: str | None) -> str:
        """Apply the per-call override, falling back to a provider-appropriate default."""
        if model is not None:
            return model
        if (
            self.settings.provider == Provider.OLLAMA
            and self.settings.model.lower().startswith("claude")
        ):
            # The configured default names an Anthropic model Ollama cannot serve.
            return _OLLAMA_DEFAULT_MODEL
        return self.settings.model

    async def _retry(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Await `operation`, retrying transient failures with exponential backoff.

        Rate limits, connection errors, timeouts and 5xx are retried up
        to `settings.max_retries` times (plus the first attempt);
        authentication and other client errors propagate immediately.
        Exhausted rate-limit retries are converted to LLMRateLimitError.

        POLISH (wave 5b): the attempt budget was floored at `max(1, max_retries) + 1`,
        so `FURROW_MAX_RETRIES=0` silently bought one extra attempt. Total attempts are
        now `max(0, max_retries) + 1`: 0 -> 1 attempt, 1 -> 2, 3 -> 4.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max(0, self.settings.max_retries) + 1),
                wait=wait_exponential_jitter(initial=self.settings.retry_base_delay),
                # FIX (test suite): rate limits were only in the conversion clause, never in
                # the retry predicate, so a 429 was converted on the FIRST attempt and the
                # documented "retried, then converted on exhaustion" behaviour never ran.
                retry=retry_if_exception_type(_TRANSIENT_ERRORS + _RATE_LIMIT_ERRORS),
                reraise=True,
            ):
                with attempt:
                    return await operation()
        except _RATE_LIMIT_ERRORS as err:
            raise LLMRateLimitError(f"Rate limited after retries: {err}") from err
        raise LLMError("Retry loop exited without a result")

    async def _anthropic_request(
        self,
        prompt: str,
        *,
        system: str | None,
        model: str,
        max_tokens: int,
        temperature: float | None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Message:
        """Issue an Anthropic messages.create call, passing only set parameters."""
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]
        return await self.anthropic.messages.create(**payload)

    async def _openai_request(
        self,
        prompt: str,
        *,
        client: AsyncOpenAI,
        system: str | None,
        model: str,
        max_tokens: int,
        temperature: float | None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> ChatCompletion:
        """Issue a chat.completions.create call (OpenAI or Ollama's compatible API)."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]
        return await client.chat.completions.create(**payload)

    async def _tool_round_trip(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        model: str,
    ) -> tuple[str, list[ToolCall], int, int]:
        """Run one model round-trip: final text, requested ToolCalls, token usage."""
        if self.settings.provider == Provider.ANTHROPIC:
            anthropic_reply = await self._retry(
                lambda: self._anthropic_request(
                    prompt,
                    system=system,
                    model=model,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    tools=tools,
                )
            )
            requested = [
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                for block in anthropic_reply.content
                if isinstance(block, ToolUseBlock)
            ]
            round_input, round_output = _anthropic_usage(anthropic_reply)
            return _anthropic_text(anthropic_reply), requested, round_input, round_output
        if self.settings.provider == Provider.OPENAI:
            client = self.openai
        else:
            client = self._ollama_client()
        chat_reply = await self._retry(
            lambda: self._openai_request(
                prompt,
                client=client,
                system=system,
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                tools=tools,
            )
        )
        message = chat_reply.choices[0].message if chat_reply.choices else None
        tool_calls = (message.tool_calls or []) if message is not None else []
        requested = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in tool_calls
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        ]
        round_input, round_output = _openai_usage(chat_reply)
        return _openai_text(chat_reply), requested, round_input, round_output
