"""Provider-agnostic async LLM client plus small filesystem helpers.

One :class:`LLMClient` fronts the three chat backends furrow supports (Anthropic, OpenAI
and a local Ollama daemon) behind a single :meth:`LLMClient.complete` call, and carries
the handful of UTF-8 filesystem helpers the agents use to inspect and write the
workspace.

Retries
-------
:meth:`LLMClient.complete` retries **only** transient failures — a refused connection, a
socket timeout, HTTP 429 or a 5xx — with exponential backoff, because those clear up on
their own within a few seconds and killing an entire run over one blip is wasteful. It
deliberately does *not* retry:

* :class:`ProviderAuthError` — a missing or wrong credential cannot heal itself, and the
  correct response is to stop and tell the operator which variable to set.
* any other 4xx — the request itself is wrong, so replaying it wastes time and quota.
* :class:`LLMResponseTruncated` — the request succeeded; the *output budget* was too
  small. Retrying the identical request truncates identically, so the caller has to raise
  ``max_tokens`` or ask for less output.

Non-transient failures are therefore wrapped in :class:`LLMError` (or one of its
subclasses) and surface immediately, which keeps a broken configuration obvious instead
of hiding it behind four identical stack traces.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from fnmatch import fnmatch
from pathlib import Path

import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, get_settings

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_SYSTEM",
    "LLMClient",
    "LLMError",
    "LLMResponseTruncated",
    "ProviderAuthError",
    "ProviderUnavailableError",
]

#: System prompt used when a caller does not supply one. A single constant keeps the two
#: provider branches from drifting apart on the wording.
DEFAULT_SYSTEM = "You are a helpful coding assistant."

#: Per-request HTTP timeout, in seconds, for the hand-rolled Ollama path. The Anthropic and
#: OpenAI SDK clients carry their own timeout configuration; ``Settings`` has no timeout
#: field for a raw ``httpx`` call, so the module owns this constant instead.
DEFAULT_REQUEST_TIMEOUT = 120.0

#: Directories :meth:`LLMClient.list_files` never descends into, whatever the pattern is:
#: interpreter/build caches, VCS metadata, dependency trees and virtualenvs. Walking them is
#: pure noise for an agent and very slow in a checked-out repository.
SKIP_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "node_modules",
        ".venv",
        "venv",
    }
)


class LLMError(RuntimeError):
    """Base class for every failure raised by :mod:`furrow.llm`."""


class ProviderAuthError(LLMError):
    """The provider credential is missing, unreadable or rejected.

    Never retried: the fix is to set the environment variable named in the message.
    """


class ProviderUnavailableError(LLMError):
    """The provider could not be reached, or answered with a transient failure.

    Retried by :meth:`LLMClient.complete` with exponential backoff.
    """


class LLMResponseTruncated(LLMError):
    """The provider stopped generating because the completion token budget ran out.

    Never retried: the caller must raise ``max_tokens`` or request a shorter answer.
    """


def _missing_key_message(provider: Provider) -> str:
    """Actionable message naming `provider` and the exact environment variables to set."""
    env_var = provider.env_var or f"{provider.value.upper()}_API_KEY"
    return (
        f"No API key configured for provider {provider.value!r}. "
        f"Set {env_var} or FURROW_{env_var} in the environment or in a .env file."
    )


def _read_text(path: Path, max_bytes: int) -> str:
    """Blocking body of :meth:`LLMClient.read_file`: read at most `max_bytes` of UTF-8 text."""
    data = path.read_bytes()
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")
    kept = data[:max_bytes].decode("utf-8", errors="replace")
    omitted = len(data) - max_bytes
    return f"{kept}\n\n[... furrow read_file truncated {omitted} of {len(data)} bytes ...]"


def _write_text(path: Path, content: str) -> Path:
    """Blocking body of :meth:`LLMClient.write_file`: create parents, write UTF-8, return path."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"failed to write {path}: {exc}") from exc
    return path


def _list_files(root: Path, pattern: str, max_entries: int, skip_hidden: bool) -> list[str]:
    """Blocking body of :meth:`LLMClient.list_files`: pruned, sorted, capped relative paths."""
    if not root.is_dir():
        return []
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place: os.walk never descends into a name removed from dirnames, which is
        # what keeps node_modules/.git checkouts from being traversed at all.
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIP_DIRECTORIES and not (skip_hidden and name.startswith("."))
        )
        base = Path(dirpath)
        for name in filenames:
            if skip_hidden and name.startswith("."):
                continue
            rel = (base / name).relative_to(root).as_posix()
            if fnmatch(rel, pattern):
                found.append(rel)
    found.sort()
    return found[:max_entries]


class LLMClient:
    """Async facade over the Anthropic Messages, OpenAI chat and Ollama chat APIs.

    The client is cheap to construct and does no I/O until :meth:`complete` runs: provider
    SDK clients are built lazily and cached, so an :class:`LLMClient` created at module
    scope or in a test without credentials is harmless until it is actually used.

    Settings are read through :meth:`Settings.resolved_model` / ``Settings.max_tokens`` at
    call time, and provider clients are cached for the lifetime of the instance — call
    :meth:`aclose` (or use the instance as an async context manager) to release their
    connection pools when a run finishes.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Lazy provider client. `settings=None` falls back to the process-wide
        `get_settings()` singleton rather than storing None."""
        self.settings: Settings = settings if settings is not None else get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Cached ``AsyncAnthropic`` client, built on first use.

        Raises:
            ProviderAuthError: if no Anthropic key is configured.
        """
        if self._anthropic is None:
            self._anthropic = AsyncAnthropic(api_key=self._api_key(Provider.ANTHROPIC))
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Cached ``AsyncOpenAI`` client, built on first use.

        Raises:
            ProviderAuthError: if no OpenAI key is configured.
        """
        if self._openai is None:
            self._openai = AsyncOpenAI(api_key=self._api_key(Provider.OPENAI))
        return self._openai

    def _api_key(self, provider: Provider) -> str:
        """Credential for `provider`, or raise :class:`ProviderAuthError` naming the env vars.

        The provider-specific settings field is consulted first so that ``client.anthropic``
        stays correct even when ``settings.provider`` selects a different backend;
        :meth:`Settings.ensure_api_key` is the fallback and its ``ValueError`` (which names
        the missing variable) is what the auth error is built from.
        """
        key = getattr(self.settings, f"{provider.value}_api_key", None)
        if not key:
            try:
                key = self.settings.ensure_api_key()
            except ValueError as exc:
                raise ProviderAuthError(_missing_key_message(provider)) from exc
        if not key:
            raise ProviderAuthError(_missing_key_message(provider))
        return str(key)

    @retry(
        retry=retry_if_exception_type((ProviderUnavailableError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one completion through the configured provider.

        `system` is sent as a real system prompt (it used to be silently ignored).
        `max_tokens` defaults to `settings.max_tokens`. Retries transient failures with
        exponential backoff. Raises ProviderAuthError, ProviderUnavailableError,
        LLMResponseTruncated or LLMError.

        Args:
            prompt: User message content.
            system: System prompt; when empty, `DEFAULT_SYSTEM` is sent instead.
            model: Model id override; defaults to `settings.resolved_model()` for the hosted
                providers and `settings.ollama_model` for Ollama.
            max_tokens: Completion token budget; defaults to `settings.max_tokens`.

        Raises:
            ProviderAuthError: The provider credential is missing or rejected.
            ProviderUnavailableError: The provider was unreachable or returned 429/5xx.
            LLMResponseTruncated: The token budget ran out before the answer was finished.
            LLMError: Any other provider or transport failure.
        """
        provider = self.settings.provider
        if provider is Provider.OLLAMA:
            # Ollama carries its own default model id instead of the generic Settings.model.
            target = model or self.settings.ollama_model
        else:
            target = model or self.settings.resolved_model()
        budget = max_tokens or self.settings.max_tokens

        if provider is Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, target, budget)
        if provider is Provider.OPENAI:
            return await self._complete_openai(prompt, system, target, budget)
        if provider is Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, target, budget)
        raise LLMError(f"Unsupported provider: {provider!r}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        """One Anthropic Messages call; concatenate every text block of the response.

        The content list is a union of block types (text, thinking, tool use, server tool
        results) and only the text blocks carry `.text`, so the blocks are filtered by their
        `type` rather than indexed positionally.

        Raises:
            ProviderAuthError: Bad credential.
            ProviderUnavailableError: Connection error, 429 or 5xx.
            LLMResponseTruncated: `stop_reason` is ``max_tokens``.
            LLMError: Any other Anthropic error, or a response with no text block.
        """
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or DEFAULT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(f"Anthropic rejected the API key: {exc}") from exc
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            raise ProviderUnavailableError(f"Anthropic request to {model} failed: {exc}") from exc
        except anthropic.APIStatusError as exc:
            # Any other status code is a client error: not worth replaying.
            raise LLMError(f"Anthropic rejected the request ({exc.status_code}): {exc}") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc

        parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        if not parts:
            kinds = [str(getattr(block, "type", "unknown")) for block in response.content]
            raise LLMError(
                f"Anthropic returned no text content for model {model} "
                f"(stop_reason={response.stop_reason!r}, blocks={kinds or ['<none>']})"
            )
        if response.stop_reason == "max_tokens":
            raise LLMResponseTruncated(
                f"Anthropic hit the max_tokens limit ({max_tokens}) for model {model}. "
                "Raise max_tokens or ask for a shorter response."
            )
        return "".join(parts)

    async def _complete_openai(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        """One OpenAI chat-completions call, returning the single choice's message content.

        The system message is omitted entirely when `system` is empty: newer OpenAI models
        reject a `system` role, and sending a filler one would break them.

        Raises:
            ProviderAuthError: Bad credential.
            ProviderUnavailableError: Connection error, 429 or 5xx.
            LLMResponseTruncated: `finish_reason` is ``length``.
            LLMError: Any other OpenAI error, or a response with no choices/content.
        """
        messages: list[ChatCompletionMessageParam] = (
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            if system
            else [{"role": "user", "content": prompt}]
        )
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(f"OpenAI rejected the API key: {exc}") from exc
        except (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as exc:
            raise ProviderUnavailableError(f"OpenAI request to {model} failed: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI rejected the request ({exc.status_code}): {exc}") from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc

        if not response.choices:
            raise LLMError(f"OpenAI returned no choices for model {model}")
        choice = response.choices[0]
        text = choice.message.content or ""
        if not text:
            raise LLMError(
                f"OpenAI returned empty content for model {model} "
                f"(finish_reason={choice.finish_reason!r})"
            )
        if choice.finish_reason == "length":
            raise LLMResponseTruncated(
                f"OpenAI hit the max_tokens limit ({max_tokens}) for model {model}. "
                "Raise max_tokens or ask for a shorter response."
            )
        return text

    async def _complete_ollama(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        """POST to {ollama_base_url}/api/chat with stream=false and return the message content.

        Ollama is a plain local HTTP daemon rather than an SDK, so this is the one branch
        built on `httpx` directly. ``max_tokens`` is passed through as Ollama's ``num_predict``.

        Raises:
            ProviderUnavailableError: The daemon is unreachable, or answered 429/5xx.
            LLMResponseTruncated: `done_reason` is ``length``.
            LLMError: A 4xx reply, malformed JSON, or a response with no message content.
        """
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        messages = (
            [
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            if system
            else [{"role": "user", "content": prompt}]
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {url}: {exc}. "
                "Start it with `ollama serve` or set FURROW_OLLAMA_BASE_URL."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Ollama request to {url} failed: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Ollama at {url} answered HTTP {response.status_code} for model {model}"
            )
        if response.is_error:
            raise LLMError(
                f"Ollama at {url} rejected the request (HTTP {response.status_code}): "
                f"{response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama at {url} returned non-JSON content: {exc}") from exc
        if not isinstance(body, dict):
            raise LLMError(f"Ollama at {url} returned a {type(body).__name__}, expected an object")

        text = str((body.get("message") or {}).get("content") or "")
        if not text:
            raise LLMError(
                f"Ollama returned no message content for model {model} "
                f"(done={body.get('done')!r}, response={json.dumps(body)[:200]})"
            )
        if body.get("done_reason") == "length":
            raise LLMResponseTruncated(
                f"Ollama hit the num_predict limit ({max_tokens}) for model {model}. "
                "Raise max_tokens or ask for a shorter response."
            )
        return text

    async def read_file(self, path: str | Path, *, max_bytes: int = 200_000) -> str:
        """Read a UTF-8 text file, truncating at `max_bytes` and appending a marker.

        Uses a thread executor (`asyncio.to_thread`) rather than aiofiles so a missing
        aiofiles install cannot break `import furrow`.

        Args:
            path: File to read; `~` is expanded.
            max_bytes: Byte budget for the returned text.

        Returns:
            The file's contents, or the leading `max_bytes` of them with a truncation
            marker appended.

        Raises:
            OSError: The file is missing or unreadable (the message names the path).
        """
        return await asyncio.to_thread(_read_text, Path(path).expanduser(), max_bytes)

    async def write_file(self, path: str | Path, content: str) -> Path:
        """Write `content` as UTF-8, creating parent directories. Returns the written path.

        Args:
            path: Target file; `~` is expanded and missing parents are created.
            content: Text to write, written as-is with no trailing newline added.

        Returns:
            The path that was written.

        Raises:
            OSError: The file could not be created or written.
        """
        return await asyncio.to_thread(_write_text, Path(path).expanduser(), content)

    def list_files(
        self,
        directory: str | Path,
        *,
        pattern: str = "*",
        max_entries: int = 200,
        skip_hidden: bool = True,
    ) -> list[str]:
        """Return sorted relative paths under `directory` matching `glob(pattern)`,
        capped at `max_entries`. Skips `__pycache__`, `.git`, `node_modules` and virtualenv dirs
        unconditionally, plus dotfiles when `skip_hidden`. Pure stdlib, no I/O surprises.

        Args:
            directory: Root to walk; a missing or non-directory path yields an empty list.
            pattern: :func:`fnmatch.fnmatch` pattern applied to the POSIX-style path relative
                to `directory`. `fnmatch`'s `*` spans `/`, so the default `"*"` returns every
                non-skipped file at any depth and `"*.py"` every Python file in the tree.
            max_entries: Upper bound on the returned list; entries past it are dropped after
                sorting, so the result is stable for a given tree.
            skip_hidden: Also skip dot-directories and dotfiles.

        Returns:
            At most `max_entries` relative paths, sorted lexicographically.
        """
        return _list_files(Path(directory), pattern, max_entries, skip_hidden)

    async def aclose(self) -> None:
        """Close cached provider clients. Safe to call more than once."""
        pending = (self._anthropic, self._openai)
        self._anthropic = None
        self._openai = None
        for client in pending:
            if client is None:
                continue
            # One failing pool must not leak the other one.
            with suppress(Exception):
                await client.close()

    async def __aenter__(self) -> LLMClient:
        """Enter an async context, guaranteeing :meth:`aclose` runs on exit."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the cached provider clients; never suppresses the in-flight exception."""
        await self.aclose()
