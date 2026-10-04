"""LLM provider client for Furrow.

Wraps the Anthropic and OpenAI async SDKs behind a single
:class:`LLMClient`. Ollama is reached through its OpenAI-compatible
endpoint. Provider calls are retried with exponential backoff, usage is
accumulated per client, and ``complete_with_tools`` implements the
agentic tool-calling loop used by the worker agent.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import ToolUseBlock
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as config_settings
from furrow.errors import (
    ConfigurationError,
    LLMError,
    LLMResponseError,
    MissingCredentialsError,
)
from furrow.tools import ToolResult, WorkspaceTools

_LOGGER = logging.getLogger(__name__)

#: Default cap on tokens generated per completion.
DEFAULT_MAX_TOKENS = 4096

#: Fallback system prompt used when a caller passes an empty one.
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

#: Characters of a tool result fed back into the conversation.
TOOL_RESULT_CHARS = 8_000

#: Retry tuning: 4 attempts, waiting 2s then doubling up to 30s.
_RETRY_ATTEMPTS = 4
_RETRY_WAIT_MULTIPLIER = 2
_RETRY_WAIT_MAX = 30

#: Transient SDK failures worth retrying. Authentication and
#: validation errors are deliberately absent: they will not heal.
_RETRYABLE_ERRORS = (
    AnthropicRateLimitError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    OpenAIRateLimitError,
    OpenAIConnectionError,
    OpenAITimeoutError,
)


@dataclass(slots=True)
class Usage:
    """Token usage for a single call (or an accumulation of them)."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Element-wise addition, for accumulating across calls."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(slots=True)
class ToolCall:
    """One tool invocation performed during an agentic loop."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the call and its outcome."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict() if self.result is not None else None,
        }


@dataclass(slots=True)
class CompletionResult:
    """Full outcome of a completion: text, tools used, usage, metadata."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    iterations: int = 1
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the completion."""
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "model": self.model,
            "provider": self.provider,
            "iterations": self.iterations,
            "truncated": self.truncated,
        }


def _is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is a transient failure worth retrying."""
    return isinstance(exc, _RETRYABLE_ERRORS)


async def _retry_call(
    provider: str,
    model: str,
    call: Callable[[], Awaitable[Any]],
) -> Any:
    """Run one SDK call, retrying transient failures with backoff.

    Rate limits, connection errors, and timeouts are retried up to
    ``_RETRY_ATTEMPTS`` times with exponential backoff. Any other
    exception (authentication, bad request, ...) fails immediately, and
    a final failure of any kind is translated into :class:`LLMError`
    carrying the provider and model. ``LLMResponseError`` never reaches
    this layer, so malformed responses are never retried.
    """

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=_RETRY_WAIT_MULTIPLIER,
            max=_RETRY_WAIT_MAX,
        ),
        reraise=True,
    )
    async def _attempt() -> Any:
        return await call()

    try:
        return await _attempt()
    except Exception as e:
        raise LLMError(
            f"{provider} provider call failed",
            provider=provider,
            model=model,
            error=f"{type(e).__name__}: {e}",
        ) from e


def _extract_anthropic_text(response: Any) -> tuple[str, bool]:
    """Concatenate every text block in an Anthropic response.

    Tool use and thinking blocks have no ``text`` attribute and are
    skipped. Returns ``(text, truncated)`` where ``truncated`` reports
    whether the model hit ``max_tokens``. Raises
    :class:`LLMResponseError` when the response has no text at all.
    """
    parts = [
        block_text
        for block_text in (getattr(block, "text", None) for block in response.content)
        if isinstance(block_text, str)
    ]
    if not parts:
        raise LLMResponseError(
            "Anthropic response contained no text blocks",
            model=getattr(response, "model", "") or "",
            stop_reason=getattr(response, "stop_reason", None) or "",
        )
    truncated = response.stop_reason == "max_tokens"
    return "".join(parts), truncated


def _usage_from_anthropic(response: Any) -> Usage:
    """Read token usage from an Anthropic response, if present."""
    raw = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
    )


def _usage_from_openai(response: Any) -> Usage:
    """Read token usage from an OpenAI-compatible response, if present."""
    raw = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
    )


def _truncate_context(text: str, limit: int = TOOL_RESULT_CHARS) -> str:
    """Cap tool output fed back into the conversation."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _iteration_budget_note(max_iterations: int) -> str:
    """Note returned when the tool loop exhausts its iteration budget."""
    return (
        f"Stopped: the tool-calling iteration budget of {max_iterations} was "
        "exhausted before the model produced a final answer."
    )


def _decode_tool_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """JSON-decode OpenAI tool arguments into a dict.

    Returns ``(arguments, error)``; ``error`` is set when the raw
    string is not valid JSON or not a JSON object, so the loop can
    turn the failure into a ``ToolResult`` error.
    """
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"Malformed tool arguments: {e}"
    if not isinstance(parsed, dict):
        return {}, "Malformed tool arguments: not a JSON object"
    return parsed, None


class LLMClient:
    """Async client for the configured LLM provider.

    Construction never touches the network: SDK clients are built
    lazily on first use so that a missing API key only fails the calls
    that would need it. Usage is accumulated across every call made
    through this instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tools: WorkspaceTools | None = None,
    ) -> None:
        self._settings = settings if settings is not None else config_settings
        self._tools = tools
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None
        self._usage = Usage()
        self._handlers: dict[
            Provider,
            Callable[[str, str, str, int], Awaitable[CompletionResult]],
        ] = {
            Provider.ANTHROPIC: self._complete_anthropic,
            Provider.OPENAI: self._complete_openai,
            Provider.OLLAMA: self._complete_openai,
        }
        self._tool_handlers: dict[
            Provider,
            Callable[
                [str, str, str, WorkspaceTools, int],
                Awaitable[CompletionResult],
            ],
        ] = {
            Provider.ANTHROPIC: self._tools_loop_anthropic,
            Provider.OPENAI: self._tools_loop_openai,
            Provider.OLLAMA: self._tools_loop_openai,
        }

    @property
    def settings(self) -> Settings:
        """The settings this client was built from."""
        return self._settings

    @property
    def tools(self) -> WorkspaceTools | None:
        """Workspace tools attached at construction time, if any."""
        return self._tools

    @property
    def usage(self) -> Usage:
        """Cumulative token usage across all calls on this client."""
        return Usage(self._usage.input_tokens, self._usage.output_tokens)

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built Anthropic client."""
        if self._anthropic is None:
            api_key = self._settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise MissingCredentialsError(
                    "No Anthropic API key configured",
                    provider=Provider.ANTHROPIC.value,
                    hint="set FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY",
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built OpenAI client (or Ollama-compatible client)."""
        if self._settings.provider == Provider.OLLAMA:
            if self._ollama is None:
                self._ollama = AsyncOpenAI(
                    api_key="ollama",
                    base_url=self._settings.ollama_base_url,
                )
            return self._ollama
        if self._openai is None:
            api_key = self._settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise MissingCredentialsError(
                    "No OpenAI API key configured",
                    provider=Provider.OPENAI.value,
                    hint="set FURROW_OPENAI_API_KEY or OPENAI_API_KEY",
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    def reset_usage(self) -> None:
        """Zero the cumulative usage counter."""
        self._usage = Usage()

    def _accumulate_usage(self, usage: Usage) -> None:
        """Add a single call's usage into the cumulative counter."""
        self._usage = self._usage + usage

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Complete a prompt and return the response text."""
        result = await self.complete_detailed(
            prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
        )
        return result.text

    async def complete_detailed(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> CompletionResult:
        """Complete a prompt and return the full completion metadata."""
        resolved_model = model or self._settings.model
        handler = self._handlers.get(self._settings.provider)
        if handler is None:
            raise ConfigurationError(
                f"Unsupported provider: {self._settings.provider}",
                provider=str(self._settings.provider),
            )
        return await handler(prompt, system, resolved_model, max_tokens)

    async def complete_with_tools(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        tools: WorkspaceTools | None = None,
        max_iterations: int | None = None,
    ) -> CompletionResult:
        """Run the agentic tool-calling loop until a final answer.

        Repeatedly invokes the model with the workspace tools attached,
        executes any requested tool calls, and feeds the results back
        until the model answers without calling tools or the iteration
        budget is spent. Without tools attached this degrades to a
        plain detailed completion.
        """
        if tools is None:
            tools = self.tools
        if tools is None:
            return await self.complete_detailed(prompt, system=system, model=model)

        resolved_model = model or self._settings.model
        if max_iterations is None:
            max_iterations = self._settings.worker_max_iterations

        handler = self._tool_handlers.get(self._settings.provider)
        if handler is None:
            raise ConfigurationError(
                f"Unsupported provider: {self._settings.provider}",
                provider=str(self._settings.provider),
            )
        return await handler(prompt, system, resolved_model, tools, max_iterations)

    async def aclose(self) -> None:
        """Close any SDK clients that were instantiated.

        Safe to call more than once; a failure closing one client does
        not prevent the others from being closed.
        """
        for attr in ("_anthropic", "_openai", "_ollama"):
            client = getattr(self, attr)
            if client is None:
                continue
            setattr(self, attr, None)
            try:
                await client.close()
            # Best-effort teardown: one failing close must not block the others.
            except Exception:
                _LOGGER.debug("failed to close %s client: %s", attr, "client")

    # --- one-shot completions ---

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
    ) -> CompletionResult:
        """Single-turn completion via the Anthropic Messages API."""
        client = self.anthropic
        provider = self._settings.provider.value
        response = await _retry_call(
            provider,
            model,
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        text, truncated = _extract_anthropic_text(response)
        usage = _usage_from_anthropic(response)
        self._accumulate_usage(usage)
        return CompletionResult(
            text=text,
            usage=usage,
            model=getattr(response, "model", "") or model,
            provider=provider,
            truncated=truncated,
        )

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
    ) -> CompletionResult:
        """Single-turn completion via the OpenAI chat API (or Ollama)."""
        client = self.openai
        provider = self._settings.provider.value
        response = await _retry_call(
            provider,
            model,
            lambda: client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            ),
        )
        if not response.choices:
            raise LLMResponseError(
                "OpenAI response contained no choices",
                provider=provider,
                model=model,
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        truncated = choice.finish_reason == "length"
        usage = _usage_from_openai(response)
        self._accumulate_usage(usage)
        return CompletionResult(
            text=text,
            usage=usage,
            model=getattr(response, "model", "") or model,
            provider=provider,
            truncated=truncated,
        )

    # --- agentic tool loops ---

    async def _tools_loop_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: WorkspaceTools,
        max_iterations: int,
    ) -> CompletionResult:
        """Agentic loop against the Anthropic Messages API."""
        client = self.anthropic
        provider = self._settings.provider.value
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_calls: list[ToolCall] = []
        usage = Usage()
        text = ""
        truncated = False
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            response = await _retry_call(
                provider,
                model,
                lambda: client.messages.create(
                    model=model,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    system=system or DEFAULT_SYSTEM_PROMPT,
                    tools=tools.anthropic_tools(),
                    messages=messages,
                ),
            )
            call_usage = _usage_from_anthropic(response)
            usage = usage + call_usage
            self._accumulate_usage(call_usage)
            truncated = response.stop_reason == "max_tokens"

            # Replay the assistant turn with its raw content blocks.
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump(exclude_none=True) for block in response.content
                    ],
                }
            )

            use_blocks = [
                block for block in response.content if isinstance(block, ToolUseBlock)
            ]
            text_parts = [
                block_text
                for block_text in (getattr(b, "text", None) for b in response.content)
                if isinstance(block_text, str)
            ]
            if text_parts:
                text = "".join(text_parts)

            if not use_blocks:
                return CompletionResult(
                    text=text,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=getattr(response, "model", "") or model,
                    provider=provider,
                    iterations=iterations,
                    truncated=truncated,
                )

            tool_results: list[dict[str, Any]] = []
            for block in use_blocks:
                arguments = dict(block.input)
                result = await tools.invoke(block.name, arguments)
                tool_calls.append(
                    ToolCall(name=block.name, arguments=arguments, result=result)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _truncate_context(result.output or result.error or ""),
                        "is_error": not result.ok,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return CompletionResult(
            text=_iteration_budget_note(max_iterations),
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider=provider,
            iterations=iterations,
            truncated=truncated,
        )

    async def _tools_loop_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: WorkspaceTools,
        max_iterations: int,
    ) -> CompletionResult:
        """Agentic loop against the OpenAI chat API (or Ollama)."""
        client = self.openai
        provider = self._settings.provider.value
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_calls: list[ToolCall] = []
        usage = Usage()
        truncated = False
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            response = await _retry_call(
                provider,
                model,
                lambda: client.chat.completions.create(
                    model=model,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    tools=tools.openai_tools(),
                    messages=messages,
                ),
            )
            call_usage = _usage_from_openai(response)
            usage = usage + call_usage
            self._accumulate_usage(call_usage)

            if not response.choices:
                raise LLMResponseError(
                    "OpenAI response contained no choices",
                    provider=provider,
                    model=model,
                )
            choice = response.choices[0]
            message = choice.message
            truncated = choice.finish_reason == "length"

            # Replay the assistant turn, including any tool calls.
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                return CompletionResult(
                    text=message.content or "",
                    tool_calls=tool_calls,
                    usage=usage,
                    model=getattr(response, "model", "") or model,
                    provider=provider,
                    iterations=iterations,
                    truncated=truncated,
                )

            for call in message.tool_calls:
                name = call.function.name
                arguments, arguments_error = _decode_tool_arguments(
                    call.function.arguments
                )
                if arguments_error is not None:
                    result = ToolResult(name=name, ok=False, error=arguments_error)
                else:
                    result = await tools.invoke(name, arguments)
                tool_calls.append(
                    ToolCall(name=name, arguments=arguments, result=result)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _truncate_context(result.output or result.error or ""),
                    }
                )

        return CompletionResult(
            text=_iteration_budget_note(max_iterations),
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider=provider,
            iterations=iterations,
            truncated=truncated,
        )
