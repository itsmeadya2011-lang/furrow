"""Provider transport: prompt in, text out, typed failures out.

This module owns exactly one concern -- talking to whichever provider
:class:`furrow.config.Settings` selects -- and translating every failure into a
member of the :class:`LLMError` taxonomy so callers can tell "the network
hiccupped" (retried here) from "the model was cut off" or "the model said
nothing usable" (never retried here).

Filesystem access used to live here too. It now lives in :mod:`furrow.tools.fs`,
where it can be confined to the configured workspace; keeping unrestricted
``open()`` calls on the transport class made the sandbox boundary impossible to
enforce.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, get_settings

_T = TypeVar("_T")

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

# Provider SDKs name their exceptions differently per provider and per release,
# and the names we care about also arrive from ``httpx`` and the stdlib socket
# layer underneath them. Matching on the class *name* keeps this module working
# across anthropic/openai versions without importing one exception class per
# dependency version (and without pretending an import is available when it is
# not). Names not in this set -- auth failures, bad requests, our own LLMError --
# are never retried.
TRANSIENT_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "TimeoutError",
    }
)


def is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a provider failure worth retrying.

    Matched against the whole MRO so a provider's own subclass of
    ``RateLimitError`` is still recognised.
    """
    return any(cls.__name__ in TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__)


class LLMError(RuntimeError):
    """Base class for every failure raised by :class:`LLMClient`."""


class LLMTruncatedError(LLMError):
    """The provider stopped mid-answer because the token budget ran out.

    Distinct from :class:`LLMResponseError` because the fix differs: the caller
    should ask for a shorter answer or raise ``settings.llm_max_tokens``, whereas
    a response error means the request itself needs rethinking. Silently returning
    half a JSON document is what corrupted planner output before this existed.
    """


class LLMResponseError(LLMError):
    """The provider answered, but the answer carried no usable text.

    Raised for empty content blocks, OpenAI refusals (``content is None``) and
    replies with no choices -- all of which used to arrive as ``""`` and surface
    much later as a baffling JSON parse error in an unrelated module.
    """


class LLMClient:
    """Thin async wrapper over the supported LLM providers.

    The SDK clients are created lazily (so importing this module never requires
    credentials) and own ``httpx`` connection pools, so they must be closed --
    see :meth:`aclose`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # `settings or get_settings()` also covers the explicit `LLMClient(settings=None)`
        # call that PlannerAgent/WorkerAgent make, which used to leave `self.settings`
        # as None and crash on the very next attribute access.
        self.settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return the model's text answer to *prompt*.

        Raises:
            LLMError: provider is unsupported or not configured, the request timed
                out, or the provider failed (transient failures are retried first;
                the SDK exception is kept as ``__cause__``).
            LLMTruncatedError: the answer was cut off by the token limit.
            LLMResponseError: the reply contained no usable text.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        # Defensive: furrow.config.Provider has exactly two members today, so this
        # only fires if a provider is added without a backend here.
        raise LLMError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        # Resolve the client (and therefore the API key) before entering the retry
        # machinery: a missing key is a config error, not something to retry.
        messages = self.anthropic.messages
        max_tokens = self.settings.llm_max_tokens

        async def call() -> Any:
            return await asyncio.wait_for(
                messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system or DEFAULT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self.settings.llm_timeout,
            )

        # Only the provider call is retried; parsing below runs once, on the answer
        # we actually got, so a truncated reply is never mistaken for a transient one.
        return _anthropic_text(await self._with_retries(call, model), model, max_tokens)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        completions = self.openai.chat.completions
        max_tokens = self.settings.llm_max_tokens

        async def call() -> Any:
            return await asyncio.wait_for(
                completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                ),
                timeout=self.settings.llm_timeout,
            )

        return _openai_text(await self._with_retries(call, model), model, max_tokens)

    async def _with_retries(self, call: Callable[[], Awaitable[_T]], model: str) -> _T:
        """Await ``call()``, retrying transient provider failures.

        Each attempt carries its own ``settings.llm_timeout`` budget, so a hung
        provider is retried and only surfaces as a single :class:`LLMError` once
        the attempts are spent. ``llm_max_retries == 0`` means exactly one attempt.

        Whatever the provider finally raises is re-raised as an :class:`LLMError`
        chained to it, so callers never have to import an SDK to classify a
        failure; the SDK exception stays reachable as ``__cause__``.
        """
        with_retries = retry(
            stop=stop_after_attempt(self.settings.llm_max_retries + 1),
            wait=wait_exponential(multiplier=0.5, max=8, min=0.5),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        try:
            return await with_retries(call)()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            # asyncio.TimeoutError is the builtin TimeoutError on 3.11+ and a plain
            # Exception subclass on 3.10, so name both to be explicit about either.
            raise LLMError(
                f"LLM request to {model} timed out after {self.settings.llm_timeout}s"
            ) from exc
        except LLMError:
            raise  # our own taxonomy already; never re-wrap it
        except Exception as exc:
            raise LLMError(f"LLM request to {model} failed: {type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        """Close the provider SDK clients and their connection pools.

        Both providers are attempted even if the first one fails, so a single bad
        close cannot leak the other pool; the failures are then reported together.
        """
        errors: list[BaseException] = []
        for name in ("_anthropic", "_openai"):
            client: Any = getattr(self, name)
            if client is None:
                continue
            setattr(self, name, None)
            try:
                await client.close()
            except Exception as exc:  # noqa: PERF203 - both providers must be tried
                errors.append(exc)
        if errors:
            detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            raise LLMError(f"failed to close provider client(s): {detail}") from errors[0]

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _anthropic_text(response: Any, model: str, max_tokens: int) -> str:
    """Extract the answer text from an Anthropic message response."""
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise LLMTruncatedError(
            f"{model} hit the {max_tokens}-token output limit and was cut off; "
            "raise settings.llm_max_tokens or ask for a shorter answer"
        )
    blocks = getattr(response, "content", None) or []
    text = "".join(
        getattr(block, "text", "") or ""
        for block in blocks
        if getattr(block, "type", None) == "text"
    )
    if not text:
        # `content[0].text` used to raise AttributeError on a leading thinking /
        # tool_use block and IndexError on empty content; neither told the caller
        # anything useful.
        kinds = [getattr(block, "type", None) for block in blocks]
        raise LLMResponseError(
            f"{model} returned no text content (stop_reason={stop_reason!r}, blocks={kinds})"
        )
    return text


def _openai_text(response: Any, model: str, max_tokens: int) -> str:
    """Extract the answer text from an OpenAI chat completion response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise LLMResponseError(f"{model} returned no choices")
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise LLMTruncatedError(
            f"{model} hit the {max_tokens}-token output limit and was cut off; "
            "raise settings.llm_max_tokens or ask for a shorter answer"
        )
    content = getattr(getattr(choice, "message", None), "content", None)
    if content is None:
        # A refusal or a tool call: there is no text to return, and coercing this to
        # "" only moved the failure into the caller's JSON parser.
        raise LLMResponseError(
            f"{model} returned no message content (finish_reason={finish_reason!r}); "
            "the model refused or requested a tool call"
        )
    if not content:
        raise LLMResponseError(f"{model} returned empty message content")
    return content
