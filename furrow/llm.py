"""One async client in front of the three LLM backends furrow supports, plus file helpers.

A single :class:`LLMClient` hides whether a completion goes to Anthropic, OpenAI or a local
Ollama daemon, and carries the small filesystem surface the agents use to inspect and write
the workspace.

Why the error hierarchy matters
-------------------------------
A provider failure is either *transient* or *permanent*, and the two demand opposite
responses. A refused connection, a socket timeout, HTTP 429 or a 5xx clears up by itself
within seconds, so replaying the request is right and killing a whole run over one blip is
wasteful. A missing or rejected credential, or a request the provider refuses to accept,
never heals itself: retrying a bad API key burns the entire retry budget (and the cycle's
wall-clock) to arrive at exactly the same failure, while hiding the one fact the operator
actually needs - which environment variable to set. So provider failures are translated into
:class:`ProviderUnavailableError` (transient, retried) or :class:`ProviderAuthError` /
:class:`LLMError` (permanent, surfaced immediately), and only the first is retried.

Why nothing is imported for its side effect
-------------------------------------------
Both SDK clients are constructed lazily. Importing this module - which ``import furrow``
does - must never need an API key, must never open a socket, and must not freeze
configuration into a default argument: the old ``def __init__(self, settings: Settings =
settings)`` bound the process settings at import time, and every caller that passed
``settings=None`` stored ``None`` and crashed on the first attribute access.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from furrow.config import Provider, Settings, get_settings

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponseTruncated",
    "ProviderAuthError",
    "ProviderUnavailableError",
]


class LLMError(RuntimeError):
    """Base class for every failure raised by :mod:`furrow.llm`.

    A single base type lets the orchestrator catch "anything the provider did" with one
    ``except`` and still distinguish the retryable subclass when it wants to.
    """


class ProviderAuthError(LLMError):
    """The credential is missing, unreadable or rejected by the provider.

    Permanent, and deliberately never retried: the message names the environment variable to
    set, so the fastest route back to a working run is for a human to read it and act.
    """


class ProviderUnavailableError(LLMError):
    """The provider could not be reached, or answered with a transient failure.

    Transient by definition (connection refused, timeout, HTTP 429/5xx) and therefore the
    only class :meth:`LLMClient.complete` retries.
    """


class LLMResponseTruncated(LLMError):
    """The model ran out of completion tokens before producing the content we needed.

    Not retried: the request succeeded, so replaying it unchanged truncates identically. The
    fix is a bigger ``max_tokens`` or a smaller ask, both of which are the caller's decision.
    """


#: Environment variable consulted for each provider's credential. Named explicitly because the
#: auth error tells the user exactly which variable to set.
_PROVIDER_ENV_VARS: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.OLLAMA: "OLLAMA_API_KEY",
}

#: Statuses that mean "the credential is wrong", never "try again".
_AUTH_STATUSES = frozenset({401, 403})

#: Statuses that mean "the request is fine, the moment is not": the standard "come back in a
#: bit" set. Everything >= 500 is treated the same way.
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})

#: Failures that mean the request never got an answer, so replaying it is always correct. The
#: SDKs already wrap their transport errors in ``APIConnectionError``/``APITimeoutError``; the
#: ``httpx`` and timeout entries catch the ones that escape unwrapped. Both spellings of
#: "timed out" are listed because on Python 3.10 ``asyncio.TimeoutError`` is a distinct class
#: from the builtin ``TimeoutError``, and on 3.11+ they are the same one.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.TransportError,
    httpx.HTTPStatusError,
    asyncio.TimeoutError,
    TimeoutError,
)

#: The SDK clients get zero of their own retries. Each SDK ships its own `max_retries=2`, and
#: leaving it on would nest three HTTP attempts inside each of ours - up to sixteen requests
#: for one logical call, with four different backoff schedules and a wall-clock blow-up that
#: makes the outer deadline meaningless. Tenacity is the single retry authority here.
_SDK_MAX_RETRIES = 0

#: Backoff bounds, in seconds, for the retry loop.
_BACKOFF_MIN_SECONDS = 1
_BACKOFF_MAX_SECONDS = 30

#: Directories :meth:`LLMClient.list_files` never descends into, whatever the pattern is:
#: VCS metadata, interpreter/build caches, dependency trees and virtualenvs. Walking them is
#: pure prompt noise, and in a checked-out repository it is also most of the wall-clock time.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)


def _list_files(root: Path, pattern: str) -> list[str]:
    """Return the sorted relative paths under ``root`` matching ``pattern``.

    ``os.walk`` is used rather than ``Path.rglob`` for two reasons: pruning ``dirnames`` in
    place skips whole junk trees instead of filtering their contents afterwards, and its
    default ``onerror=None`` silently ignores directories that cannot be read, which is what
    keeps one permission-denied subdirectory from aborting an agent's view of the project.

    The result is sorted because it feeds prompts and diffs: an unsorted listing changes
    order between two identical runs of the same repository, which makes every downstream
    completion non-reproducible and every diff noisy.

    ``pattern`` is an :func:`fnmatch` glob matched against the POSIX-style path relative to
    ``root``. Note that ``fnmatch``'s ``*`` spans ``/``, so the default ``"*"`` returns every
    file at any depth and ``"*.py"`` every Python file in the tree - the same reach the old
    ``rglob("*")`` had.
    """
    if not root.is_dir():
        return []
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRECTORIES)
        base = Path(dirpath)
        for name in filenames:
            try:
                relative = (base / name).relative_to(root).as_posix()
            except (OSError, ValueError):  # pragma: no cover - defensive, not seen in practice
                continue
            if fnmatch(relative, pattern):
                found.append(relative)
    return sorted(found)


class LLMClient:
    """Async facade over the Anthropic Messages, OpenAI chat and Ollama chat APIs.

    Construction is cheap and side-effect free: settings are read at call time and the SDK
    clients are built on first use, so creating a client in a module body, a test or a CLI
    that is never asked for a completion costs nothing and needs no credentials.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Bind configuration, falling back to the process-wide settings.

        ``settings is not None`` rather than a truthiness test, and ``None`` rather than an
        import-time default. Two separate bugs hid in the old signature
        ``def __init__(self, settings: Settings = settings)``: the default was the module-level
        singleton, evaluated once at import and therefore frozen before any ``--flag`` or
        ``.env`` edit could apply, and passing ``settings=None`` - which every agent did -
        stored ``None`` verbatim so the next ``self.settings.model`` raised ``AttributeError``
        in the middle of a run. ``None`` means "not supplied", and nothing else.
        """
        self.settings: Settings = settings if settings is not None else get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Credentials and lazily built SDK clients
    # ------------------------------------------------------------------

    def _api_key(self, provider: Provider) -> str:
        """Return the raw credential for ``provider`` or raise :class:`ProviderAuthError`.

        The configured :class:`~pydantic.SecretStr` is read first (its ``.get_secret_value()``
        is the only sanctioned way to unwrap it), then the process environment. The fallback
        matters because :class:`Settings` snapshots the environment when it is built: a client
        holding an older settings object, or one built with ``_env_file=None``, would
        otherwise miss a key that was exported afterwards.

        The error message names the *environment variable*, never the value - and no value is
        available to leak, because the only branch that raises has already established there
        isn't one.
        """
        if provider is Provider.OLLAMA:
            # Ollama ignores the Authorization header entirely, but the OpenAI SDK refuses to
            # build a client without a non-empty key, so a placeholder satisfies both.
            return "ollama"
        if provider is Provider.ANTHROPIC:
            configured: Any = self.settings.anthropic_api_key
        elif provider is Provider.OPENAI:
            configured = self.settings.openai_api_key
        else:  # pragma: no cover - Provider has exactly three members
            raise LLMError(f"unsupported provider {provider!r}")
        key = configured.get_secret_value() if configured is not None else ""
        env_var = _PROVIDER_ENV_VARS[provider]
        if not key:
            key = os.getenv(env_var, "")
        if not key:
            raise ProviderAuthError(
                f"no API key for provider {provider.value!r}: set {env_var} in the "
                f"environment, or FURROW_{env_var} in a .env file"
            )
        return key

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Cached ``AsyncAnthropic`` client, built on first use.

        Raises:
            ProviderAuthError: when no Anthropic credential is configured.
        """
        if self._anthropic is None:
            self._anthropic = AsyncAnthropic(
                api_key=self._api_key(Provider.ANTHROPIC),
                timeout=self.settings.request_timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Cached ``AsyncOpenAI`` client for ``provider=openai``, built on first use.

        Raises:
            ProviderAuthError: when no OpenAI credential is configured.
        """
        if self._openai is None:
            self._openai = AsyncOpenAI(
                api_key=self._api_key(Provider.OPENAI),
                timeout=self.settings.request_timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Cached ``AsyncOpenAI`` client pointed at the local Ollama daemon.

        Ollama speaks the OpenAI chat protocol, so it reuses that SDK rather than growing a
        hand-rolled ``httpx`` branch; only ``base_url`` (and the dummy key) differ. Raises
        nothing: the key is a placeholder and the daemon may not be running yet.
        """
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key=self._api_key(Provider.OLLAMA),
                timeout=self.settings.request_timeout,
                max_retries=_SDK_MAX_RETRIES,
            )
        return self._ollama

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _retrying(self) -> AsyncRetrying:
        """Build the retry policy for one :meth:`complete` call.

        Built per call rather than applied as a ``@retry`` decorator at import time because
        the attempt budget comes from ``settings.llm_max_retries``, which is per instance and
        legitimately different between a test, a worker agent and the CLI. A decorator would
        freeze whatever the first-imported settings said, for every instance, forever.

        Only transient classes are retried. Retrying :class:`ProviderAuthError`,
        :class:`LLMResponseTruncated` or a plain :class:`LLMError` would spend the full
        exponential backoff - and the whole cycle - to fail in exactly the same way, while
        burying the one actionable message. ``reraise=True`` then hands the caller the
        original typed exception instead of tenacity's opaque :class:`RetryError`.

        ``stop_after_attempt(llm_max_retries + 1)`` because the setting counts *retries*, and
        a configured ``0`` has to mean exactly one attempt rather than none.
        """
        return AsyncRetrying(
            stop=stop_after_attempt(self.settings.llm_max_retries + 1),
            wait=wait_random_exponential(
                min=_BACKOFF_MIN_SECONDS, max=_BACKOFF_MAX_SECONDS
            ),
            retry=retry_if_exception_type((ProviderUnavailableError, *_TRANSIENT_EXCEPTIONS)),
            reraise=True,
        )

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one completion through the configured provider and return its text.

        Transient provider failures are retried with jittered exponential backoff; permanent
        ones are raised immediately.

        Args:
            prompt: The user message.
            system: Optional system prompt. Omitted from the request entirely when falsy -
                the old code invented a "You are a helpful coding assistant." filler that no
                caller asked for, which the Anthropic SDK rejects when empty and newer OpenAI
                models reject or ignore outright.
            model: Model id override; defaults to ``settings.model`` (or
                ``settings.ollama_model`` when the provider is Ollama, which carries its own
                default).
            max_tokens: Output token budget; defaults to ``settings.max_tokens`` instead of
                the hardcoded 4096 that used to override configuration entirely.

        Returns:
            The model's text. May be empty when the provider returns an empty message; that
            is not diagnosed here because every caller feeds the result to a JSON parser,
            which fails loudly and specifically on empty input.

        Raises:
            ProviderAuthError: The credential is missing or rejected (HTTP 401/403).
            ProviderUnavailableError: Unreachable, timed out, HTTP 429 or 5xx. Retried.
            LLMResponseTruncated: Anthropic stopped at ``max_tokens`` with no usable text.
            LLMError: Any other provider, transport or protocol failure.
        """
        provider = self.settings.provider
        target = self._resolve_model(provider, model)
        budget = max_tokens if max_tokens is not None else self.settings.max_tokens

        async for attempt in self._retrying():
            with attempt:
                if provider is Provider.ANTHROPIC:
                    return await self._complete_anthropic(prompt, system, target, budget)
                if provider is Provider.OPENAI:
                    return await self._complete_openai(prompt, system, target, budget)
                return await self._complete_chat(
                    Provider.OLLAMA, self.ollama, prompt, system, target, budget
                )
        # Unreachable: the loop either returns a result or propagates the last exception.
        raise LLMError(f"provider {provider!r} produced no result for model {target!r}")

    def _resolve_model(self, provider: Provider, model: str | None) -> str:
        """Return the model id to request, or raise for a provider we cannot dispatch."""
        if provider in (Provider.ANTHROPIC, Provider.OPENAI):
            return model or self.settings.model
        if provider is Provider.OLLAMA:
            return model or self.settings.ollama_model
        raise LLMError(
            f"unsupported provider {provider!r}: expected one of "
            f"{', '.join(sorted(item.value for item in Provider))}"
        )

    async def _complete_anthropic(
        self, prompt: str, system: str | None, model: str, max_tokens: int
    ) -> str:
        """One Anthropic Messages call, returning every text block concatenated.

        The content list is a union of block types - text, thinking, redacted thinking, tool
        use, server tool results - and only some of them carry ``.text``. The old code read
        ``response.content[0].text``, which raises ``AttributeError`` the moment a thinking
        model puts a ``ThinkingBlock`` first, so the blocks are walked and anything without
        text is skipped.

        Raises:
            ProviderAuthError: HTTP 401/403.
            ProviderUnavailableError: Connection error, timeout, HTTP 429 or 5xx.
            LLMResponseTruncated: Stopped at ``max_tokens`` with no usable text.
            LLMError: Any other Anthropic failure.
        """
        # Resolved before the `try` so a missing key propagates as ProviderAuthError instead
        # of being caught and re-wrapped as a generic failure.
        client = self.anthropic
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # The SDK rejects an empty `system`, so the key is omitted rather than passed "".
            request["system"] = system
        try:
            response = await client.messages.create(**request)
        except Exception as exc:
            raise self._translate(Provider.ANTHROPIC, model, exc) from exc

        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        text = "".join(parts)
        if not text and response.stop_reason == "max_tokens":
            raise LLMResponseTruncated(
                f"anthropic produced no text for model {model!r} within max_tokens="
                f"{max_tokens}; raise max_tokens or ask for less output"
            )
        return text

    async def _complete_openai(
        self, prompt: str, system: str | None, model: str, max_tokens: int
    ) -> str:
        """One OpenAI chat-completions call against the hosted OpenAI API."""
        return await self._complete_chat(
            Provider.OPENAI, self.openai, prompt, system, model, max_tokens
        )

    async def _complete_chat(
        self,
        provider: Provider,
        client: AsyncOpenAI,
        prompt: str,
        system: str | None,
        model: str,
        max_tokens: int,
    ) -> str:
        """One OpenAI-protocol chat completion, shared by OpenAI and Ollama.

        The two differ only in which client they were handed, so they share this code: Ollama
        is an OpenAI-compatible server and a second copy of the message assembly would only
        give the two branches room to drift.

        Raises:
            ProviderAuthError: HTTP 401/403.
            ProviderUnavailableError: Connection error, timeout, HTTP 429 or 5xx.
            LLMError: Any other failure. An empty or absent message is *not* one of those: it
                is returned as "" so the contract stays total.
        """
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise self._translate(provider, model, exc) from exc
        if not response.choices:
            return ""
        # `content` is None for a refusal or a tool call with no text; returning "" keeps the
        # contract total, and the caller's JSON parse then fails loudly with the real reason.
        return response.choices[0].message.content or ""

    def _translate(self, provider: Provider, model: str, exc: Exception) -> LLMError:
        """Map a raw SDK/httpx exception onto the typed hierarchy.

        The status code, when there is one, decides: 401/403 are a credential the operator has
        to fix, the standard transient set plus every 5xx is worth replaying, and any other
        4xx means the request itself is wrong so replaying it just burns quota. A missing
        status means it never reached the provider (refused connection, DNS failure, timeout),
        which is transient by definition.

        The message carries the HTTP status and the SDK's own text, both of which are safe;
        the configured credentials are scrubbed out of it defensively, because an exception
        message is the one string in this project most likely to end up in a log file.
        """
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        detail = self._scrub(str(exc) or type(exc).__name__)
        if isinstance(status, int):
            if status in _AUTH_STATUSES:
                return ProviderAuthError(
                    f"{provider.value} rejected the credential for model {model!r} "
                    f"(HTTP {status}): {detail}"
                )
            if status in _TRANSIENT_STATUSES or status >= 500:
                return ProviderUnavailableError(
                    f"{provider.value} returned HTTP {status} for model {model!r}: {detail}"
                )
            return LLMError(
                f"{provider.value} rejected the request for model {model!r} "
                f"(HTTP {status}): {detail}"
            )
        if isinstance(exc, _TRANSIENT_EXCEPTIONS):
            return ProviderUnavailableError(
                f"cannot reach {provider.value} for model {model!r}: {detail}"
            )
        return LLMError(f"{provider.value} call for model {model!r} failed: {detail}")

    def _scrub(self, message: str) -> str:
        """Replace any configured credential appearing in ``message`` with ``***``."""
        for provider in (Provider.ANTHROPIC, Provider.OPENAI):
            secret: Any = getattr(self.settings, f"{provider.value}_api_key", None)
            if secret is None:
                continue
            raw = secret.get_secret_value()
            if raw:
                message = message.replace(raw, "***")
        return message

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        """Resolve a relative path against ``settings.workspace``.

        Convenience, not security: agents run inside a process whose working directory is not
        guaranteed to be the project (the web server starts in one place and serves another),
        so "notes.md" resolving against an arbitrary CWD writes to the wrong tree. Guaranteeing
        that model-supplied paths stay inside the workspace is the worker agent's job, because
        only it knows what it is being asked to write.
        """
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.settings.workspace / candidate

    async def read_file(self, path: str | Path) -> str:
        """Return the UTF-8 contents of ``path``, resolved against the workspace.

        Uses :func:`asyncio.to_thread` so the blocking read happens off the event loop; the
        agents that call this are running many tasks concurrently on one loop, and a blocking
        read there stalls every one of them.

        Raises:
            OSError: The file is missing or unreadable.
        """
        return await asyncio.to_thread(self._resolve(path).read_text, encoding="utf-8")

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write ``content`` as UTF-8 to ``path``, creating missing parent directories.

        ``content`` is written exactly as given - no trailing newline is added, because a
        prompt that appends one produces a diff the agent never intended.

        Raises:
            OSError: The file could not be created or written.
        """
        target = self._resolve(path)
        await asyncio.to_thread(self._write_text, target, content)

    @staticmethod
    def _write_text(target: Path, content: str) -> None:
        """Blocking body of :meth:`write_file`."""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_files(self, directory: str | Path, pattern: str = "*") -> list[str]:
        """Return sorted paths under ``directory`` matching ``pattern``, relative to it.

        Args:
            directory: Root to walk, resolved against the workspace when relative. A missing
                or non-directory path yields an empty list rather than raising, because
                "this directory does not exist yet" is a normal answer for an agent.
            pattern: ``fnmatch`` glob applied to the relative POSIX path. ``*`` spans ``/``,
                so ``"*`` is every file at any depth and ``"*.py"`` every Python file.

        Returns:
            Relative POSIX paths, sorted, excluding :data:`SKIP_DIRECTORIES`. Sorting is not
            cosmetic: these strings go into prompts and diffs, and filesystem iteration order
            varies between machines and runs, which would make identical repositories produce
            different completions.
        """
        return _list_files(self._resolve(directory), pattern)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close whichever SDK clients were built. Idempotent, and safe with none.

        The cached references are cleared before closing so a second call is a no-op even if a
        close raised, and one failing connection pool cannot leak the other two.
        """
        for attribute in ("_anthropic", "_openai", "_ollama"):
            client = getattr(self, attribute)
            setattr(self, attribute, None)
            if client is None:
                continue
            with suppress(Exception):
                await client.close()

    async def __aenter__(self) -> LLMClient:
        """Enter an async context so :meth:`aclose` runs on exit."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release the SDK clients; never swallows the in-flight exception."""
        await self.aclose()
