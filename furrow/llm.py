"""LLM provider clients, JSON recovery helpers and small workspace filesystem utilities."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import aiofiles
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
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    BadRequestError as AnthropicBadRequestError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    PermissionDeniedError as AnthropicPermissionDeniedError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
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
    AuthenticationError as OpenAIAuthenticationError,
)
from openai import (
    BadRequestError as OpenAIBadRequestError,
)
from openai import (
    InternalServerError as OpenAIInternalServerError,
)
from openai import (
    PermissionDeniedError as OpenAIPermissionDeniedError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import RetryCallState, retry, retry_if_exception, wait_exponential_jitter

# Imported under an unambiguous alias: the module level ``settings`` singleton must not be
# shadowed by the ``settings`` parameter of ``LLMClient.__init__``.
from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

#: Fallback used when Ollama's base URL is missing or blank in the settings.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

#: Directory names skipped by :meth:`LLMClient.list_files`. These are VCS metadata,
#: dependency trees, virtualenvs and build/test caches; walking into them adds thousands of
#: irrelevant paths to planner prompts while never yielding useful project files.
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        # Version control / editor metadata
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        # Python environments and caches
        ".venv",
        "venv",
        ".tox",
        ".nox",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
        "site-packages",
        # JavaScript / frontend dependencies and build output
        "node_modules",
        ".next",
        ".nuxt",
        # Generic build and packaging output
        "build",
        "dist",
    }
)

# Errors that are permanent: retrying cannot help, so they must surface immediately.
_PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAuthenticationError,  # HTTP 401
    OpenAIAuthenticationError,  # HTTP 401
    AnthropicPermissionDeniedError,  # HTTP 403
    OpenAIPermissionDeniedError,  # HTTP 403
    AnthropicBadRequestError,  # HTTP 400
    OpenAIBadRequestError,  # HTTP 400
)

# Errors that are usually transient: rate limits, timeouts, connection failures, 5xx.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicConnectionError,
    AnthropicTimeoutError,  # subclass of AnthropicConnectionError
    OpenAIConnectionError,
    OpenAITimeoutError,  # subclass of OpenAIConnectionError
    AnthropicRateLimitError,  # HTTP 429
    OpenAIRateLimitError,  # HTTP 429
    AnthropicInternalServerError,  # HTTP 5xx
    OpenAIInternalServerError,  # HTTP 5xx
)

# Number of retries (i.e. attempts after the first one) used when ``Settings`` does not
# provide ``max_retries``. See ``LLMClient.max_retries``.
DEFAULT_MAX_RETRIES = 3

# Retry budget for the request currently in flight. A module level ``@retry`` decorator is
# evaluated at import time and therefore cannot read per-instance settings, so
# ``LLMClient.complete`` publishes the client's own budget here for the duration of the
# call. The default keeps direct calls to the provider helpers (tests, for example) sane.
_MAX_RETRIES: ContextVar[int] = ContextVar(
    "furrow_llm_max_retries", default=DEFAULT_MAX_RETRIES
)


def is_transient_error(exc: BaseException) -> bool:
    """Return whether a provider error is worth retrying.

    Transient means rate limits (HTTP 429), timeouts, connection failures and server
    errors (HTTP 5xx). Authentication (401), permission (403), invalid request (400) and
    every other non-5xx status error are permanent and are never retried.
    """
    if isinstance(exc, _PERMANENT_ERRORS):
        return False
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    # The SDKs only map a handful of statuses onto dedicated classes, so treat any
    # remaining 5xx (502, 503, 504, 529, ...) as transient as well.
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


class _StopAfterMaxRetries:
    """``tenacity`` stop policy honouring the per-call retry budget in ``_MAX_RETRIES``."""

    def __call__(self, retry_state: RetryCallState) -> bool:
        # ``attempt_number`` is 1 for the initial call, so stopping after
        # ``max_retries + 1`` attempts means "max_retries retries".
        return retry_state.attempt_number > max(0, _MAX_RETRIES.get())


def _transient_retry() -> Any:
    """Build the retry decorator for provider calls.

    Exponential backoff with jitter, so that concurrent agents hammering a rate-limited
    endpoint do not resynchronise onto the same retry window. ``reraise`` makes the final
    failure surface as the original provider error rather than a ``RetryError``.
    """
    return retry(
        retry=retry_if_exception(is_transient_error),
        wait=wait_exponential_jitter(initial=1.0, max=30.0),
        stop=_StopAfterMaxRetries(),
        reraise=True,
    )


def _preview(text: str, limit: int = 200) -> str:
    """Return a short, single line, quoted excerpt of ``text`` for error messages."""
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return repr(collapsed)


def _strip_code_fence(text: str) -> str | None:
    """Return the body of a surrounding markdown code fence, or ``None`` if there is none."""
    if not text.startswith("```"):
        return None
    line_break = text.find("\n")
    if line_break == -1:
        return None
    body = text[line_break + 1 :]
    end = body.rfind("```")
    if end == -1:
        return None
    return body[:end]


def _balanced_values(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` / ``[...]`` span found in ``text``, outermost first.

    Brackets inside JSON string literals are ignored and backslash escapes are honoured,
    so ``{"note": "a } b"}`` is extracted whole.
    """
    for start, opener in enumerate(text):
        if opener not in "{[":
            continue
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break
                if depth < 0:
                    # Unbalanced closer: this candidate is not a JSON value.
                    break


def _json_candidates(text: str) -> Iterator[str]:
    """Yield the snippets of ``text`` that may contain a JSON value, most likely first."""
    stripped = text.strip()
    if not stripped:
        return
    # Bare JSON output needs no surgery at all.
    yield stripped
    fenced = _strip_code_fence(stripped)
    if fenced is not None and fenced.strip():
        # ```json fenced output, possibly with the fence closing line still attached.
        yield fenced.strip()
    # Prose around the payload, or prose inside the fence.
    yield from _balanced_values(stripped)
    if fenced is not None:
        yield from _balanced_values(fenced)


def extract_json(text: str) -> Any:
    """Recover a JSON value from raw LLM output.

    Handles bare JSON, ```/```json fenced blocks and JSON embedded in prose by scanning
    for the first balanced ``{...}`` or ``[...]`` block, ignoring brackets that appear
    inside JSON strings.

    Args:
        text: The raw model response.

    Returns:
        The decoded JSON value (object, array, or scalar).

    Raises:
        ValueError: If no JSON value could be recovered.
    """
    if not isinstance(text, str):
        # ValueError, not TypeError: callers only ever catch ValueError from this helper
        # (a model returned something that is not text at all is still a decode failure).
        raise ValueError(  # noqa: TRY004
            f"Cannot extract JSON from {type(text).__name__}: {_preview(str(text))}"
        )
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise ValueError(f"Could not extract JSON from model output: {_preview(text)}")


class LLMClient:
    """Thin async wrapper around the supported LLM providers plus workspace file helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        # Callers legitimately pass ``settings=None`` to mean "use the configured
        # defaults", so the fallback has to happen here rather than in the signature.
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def max_retries(self) -> int:
        """How many times a transient provider failure is retried before giving up.

        NOTE: ``Settings`` does not declare ``max_retries`` yet. A
        ``max_retries: int = 3`` field should be added to ``furrow/config.py`` (env var
        ``FURROW_MAX_RETRIES``); it is read defensively with ``getattr`` so that this
        module works with or without that field.
        """
        value = getattr(self.settings, "max_retries", DEFAULT_MAX_RETRIES)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return DEFAULT_MAX_RETRIES

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            # Settings uses env_prefix="FURROW_", so the plain ANTHROPIC_API_KEY that the
            # README tells users to export has to be read from the environment as well.
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            # Same as above: plain OPENAI_API_KEY must keep working.
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """OpenAI-compatible client for a local Ollama server.

        Ollama exposes an OpenAI compatible API under ``<base_url>/v1`` and ignores the
        API key, but the OpenAI client insists on a non-empty one.
        """
        if self._ollama is None:
            base_url = self.settings.ollama_base_url or DEFAULT_OLLAMA_BASE_URL
            self._ollama = AsyncOpenAI(
                base_url=f"{base_url.rstrip('/')}/v1", api_key="ollama"
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        token = _MAX_RETRIES.set(self.max_retries)
        try:
            if provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            if provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model, client=self.openai)
            if provider == Provider.OLLAMA:
                return await self._complete_openai(prompt, system, model, client=self.ollama)
        finally:
            _MAX_RETRIES.reset(token)
        raise ValueError(f"Unsupported provider: {provider}")

    @_transient_retry()
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._anthropic_text(response)

    @staticmethod
    def _anthropic_text(response: Any) -> str:
        """Join the text of every text block of an Anthropic message response.

        ``response.content`` is a heterogeneous list (text, tool_use, thinking blocks) and
        can be empty, so ``response.content[0].text`` is unsafe: it raises ``IndexError``
        on empty content and silently drops text when a non-text block comes first.
        """
        blocks = getattr(response, "content", None) or []
        texts = [
            block_text
            for block_text in (getattr(block, "text", None) for block in blocks)
            if isinstance(block_text, str) and block_text.strip()
        ]
        text = "\n".join(texts).strip()
        if not text:
            raise ValueError(f"Anthropic returned no text content: {response!r}")
        return text

    @_transient_retry()
    async def _complete_openai(
        self, prompt: str, system: str, model: str, client: AsyncOpenAI | None = None
    ) -> str:
        client = client or self.openai
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Return the sorted, de-duplicated relative paths of the files under ``directory``.

        Directories named in :data:`IGNORED_DIRECTORIES` are pruned in place, so the walk
        never descends into VCS metadata, dependency trees or build caches.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        files: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(root):
            # Mutating dirnames in place stops os.walk from descending into the ignored
            # directories at all, rather than filtering their contents afterwards.
            dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRECTORIES]
            base = Path(dirpath)
            for name in filenames:
                files.add(str((base / name).relative_to(root)))
        return sorted(files)
