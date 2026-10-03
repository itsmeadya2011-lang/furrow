from __future__ import annotations

import os
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIStatusError as AnthropicStatusError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic
from anthropic import AuthenticationError as AnthropicAuthError
from anthropic import BadRequestError as AnthropicBadRequest
from anthropic import InternalServerError as AnthropicServerError
from anthropic import PermissionDeniedError as AnthropicPermissionDenied
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIStatusError as OpenAIStatusError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthError
from openai import BadRequestError as OpenAIBadRequest
from openai import InternalServerError as OpenAIServerError
from openai import PermissionDeniedError as OpenAIPermissionDenied
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

# Directories that are pure noise (or enormous) inside an LLM prompt: VCS metadata,
# virtualenvs, dependency trees, caches and build output.
_IGNORED_DIR_NAMES = frozenset({"node_modules", "__pycache__", "dist", "build", ".git", ".venv"})
_IGNORED_DIR_SUFFIXES = (".egg-info",)
_IGNORED_FILE_SUFFIXES = (".pyc", ".pyo", ".pyd")


class LLMError(RuntimeError):
    """Base class for failures raised by :class:`LLMClient` itself."""


class LLMTransientError(LLMError):
    """A failure a retry may fix: rate limit, 5xx, timeout or connection reset."""


class LLMAuthError(LLMError, ValueError):
    """Missing or rejected credentials.

    Also a ``ValueError`` so callers written against the old
    ``ValueError("ANTHROPIC_API_KEY is not set")`` behaviour keep working.
    """


class LLMProtocolError(LLMError, ValueError):
    """Unusable payload (bad JSON, no text block); retrying would return the same junk."""


class LLMRequestError(LLMError):
    """The provider rejected the request itself (e.g. HTTP 400)."""


# Retrying only helps for failures that are plausibly temporary...
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.TransportError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicRateLimitError,
    AnthropicServerError,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIRateLimitError,
    OpenAIServerError,
)

# ...and never for credential or request-shape problems, which fail identically forever.
_NON_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAuthError,
    AnthropicPermissionDenied,
    AnthropicBadRequest,
    OpenAIAuthError,
    OpenAIPermissionDenied,
    OpenAIBadRequest,
)


def _is_transient(exc: BaseException) -> bool:
    """Decide whether ``exc`` is worth another attempt."""
    if isinstance(exc, LLMTransientError):
        return True
    if isinstance(exc, _NON_RETRYABLE_ERRORS):
        return False
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    # Last resort for anything status-carrying we do not know about by type.
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 429) or status >= 500)


def _snippet(text: str | None, limit: int = 300) -> str:
    """One-line, length-capped excerpt of a provider error body."""
    flat = (text or "").strip().replace("\n", " ")
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _status_error(
    exc: BaseException,
    provider: str,
    key_env: str | None = None,
    legacy_env: str | None = None,
) -> LLMError:
    """Turn an HTTP-level provider failure into a typed, actionable error."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    detail = _snippet(str(exc))
    if status in (401, 403):
        hint = ""
        if key_env:
            hint = f" Set {key_env}" + (f" (or {legacy_env})" if legacy_env else "") + "."
        return LLMAuthError(f"{provider} rejected the credentials (HTTP {status}).{hint} Details: {detail}")
    if status == 429 or (isinstance(status, int) and status >= 500):
        return LLMTransientError(f"{provider} request failed with HTTP {status}: {detail}")
    return LLMRequestError(f"{provider} rejected the request (HTTP {status}): {detail}")


def _anthropic_text(response: Any, model: str) -> str:
    """Join the text blocks of an Anthropic message.

    ``content`` is a list of blocks that may mix text with non-text types (tool_use,
    thinking) and may be empty, so never assume ``content[0]`` exists.
    """
    blocks = getattr(response, "content", None) or []
    parts = [
        block.text
        for block in blocks
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    text = "".join(parts).strip()
    if not text:
        raise LLMProtocolError(
            f"Anthropic returned no text content for model {model!r} "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r})."
        )
    return text


def _openai_text(response: Any, model: str) -> str:
    """Read the assistant text of an OpenAI chat completion."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise LLMProtocolError(f"OpenAI returned no choices for model {model!r}.")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    # content is None when the model emits tool calls or is cut off by max_tokens;
    # say so instead of returning an empty string that fails to parse downstream.
    if not isinstance(content, str) or not content.strip():
        raise LLMProtocolError(
            f"OpenAI returned no text content for model {model!r} "
            f"(finish_reason={getattr(choices[0], 'finish_reason', None)!r})."
        )
    return content


def _ollama_text(data: Any, model: str, url: str) -> str:
    """Validate an Ollama ``/api/chat`` payload and return the assistant message."""
    if not isinstance(data, dict):
        raise LLMProtocolError(f"Ollama at {url} returned a non-object payload: {_snippet(str(data))}")
    message = data.get("message")
    if not isinstance(message, dict):
        raise LLMProtocolError(f"Ollama at {url} returned no 'message' object for model {model!r}.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProtocolError(
            f"Ollama at {url} returned empty message content for model {model!r} "
            f"(done_reason={data.get('done_reason')!r}); the model may not be pulled yet."
        )
    return content


def _http_status_error(response: httpx.Response, url: str) -> LLMError:
    """Classify a non-2xx Ollama response the same way as the SDK errors."""
    status = response.status_code
    detail = _snippet(response.text)
    if status in (401, 403):
        return LLMAuthError(f"Ollama at {url} rejected the request (HTTP {status}): {detail}")
    if status == 429 or status >= 500:
        return LLMTransientError(f"Ollama at {url} failed with HTTP {status}: {detail}")
    return LLMRequestError(f"Ollama at {url} rejected the request (HTTP {status}): {detail}")


class LLMClient:
    """Async wrapper around the supported chat-completion providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        # ``None`` falls back to the module-level singleton *at construction time*
        # rather than binding it as a default argument at import time, so settings
        # mutated later (e.g. by the CLI) are picked up and nothing is shared implicitly.
        self.settings = settings if settings is not None else _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMAuthError(
                    "Anthropic API key is not set. Set FURROW_ANTHROPIC_API_KEY "
                    "(or ANTHROPIC_API_KEY) in the environment or in a .env file."
                )
            # max_retries=0: retry policy lives here, so the SDK must not retry behind
            # our back and multiply the configured attempt count.
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMAuthError(
                    "OpenAI API key is not set. Set FURROW_OPENAI_API_KEY "
                    "(or OPENAI_API_KEY) in the environment or in a .env file."
                )
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._openai

    def _http_client(self) -> httpx.AsyncClient:
        """Ollama transport seam: kept separate so tests can inject a mock transport."""
        return httpx.AsyncClient(timeout=self.settings.request_timeout)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Run one chat completion through the configured provider, with retries."""
        model = model or self.settings.model
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_transient),
            # ``max_retries`` counts retries, so attempts = retries + 1.
            stop=stop_after_attempt(max(1, int(self.settings.max_retries) + 1)),
            # Exponential backoff + jitter, hard-capped so a run can never stall.
            wait=wait_exponential_jitter(
                initial=min(2.0, self.settings.max_retry_wait),
                max=self.settings.max_retry_wait,
                exp_base=2.0,
                jitter=1.0,
            ),
            reraise=True,
        )
        return await retrying(self._complete, prompt, system, model)

    async def _complete(self, prompt: str, system: str, model: str) -> str:
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise LLMRequestError(f"Unsupported provider: {provider!r}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicStatusError as exc:
            raise _status_error(
                exc, "Anthropic", "FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"
            ) from exc
        except (AnthropicConnectionError, AnthropicTimeoutError) as exc:
            raise LLMTransientError(
                f"Anthropic request failed ({type(exc).__name__}): {_snippet(str(exc))}. "
                f"Check network access and FURROW_REQUEST_TIMEOUT "
                f"(currently {self.settings.request_timeout}s)."
            )
        return _anthropic_text(response, model)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except OpenAIStatusError as exc:
            raise _status_error(exc, "OpenAI", "FURROW_OPENAI_API_KEY", "OPENAI_API_KEY") from exc
        except (OpenAIConnectionError, OpenAITimeoutError) as exc:
            raise LLMTransientError(
                f"OpenAI request failed ({type(exc).__name__}): {_snippet(str(exc))}. "
                f"Check network access and FURROW_REQUEST_TIMEOUT "
                f"(currently {self.settings.request_timeout}s)."
            )
        return _openai_text(response, model)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            async with self._http_client() as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(
                f"Ollama request to {url} timed out after {self.settings.request_timeout}s."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(
                f"Could not reach Ollama at {url} ({_snippet(str(exc))}). Is `ollama serve` "
                f"running? Set FURROW_OLLAMA_BASE_URL to a reachable instance."
            ) from exc
        if response.status_code >= 400:
            raise _http_status_error(response, url)
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMProtocolError(
                f"Ollama at {url} returned a non-JSON response: {_snippet(response.text)}"
            ) from exc
        return _ollama_text(data, model, url)

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        try:
            async with aiofiles.open(p, "r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No such file: {p}") from exc
        except IsADirectoryError as exc:
            raise IsADirectoryError(f"Expected a file but found a directory: {p}") from exc
        except OSError as exc:
            raise OSError(f"Failed to read {p} ({type(exc).__name__}): {exc}") from exc

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as exc:
            raise OSError(f"Failed to write {p} ({type(exc).__name__}): {exc}") from exc

    def list_files(
        self,
        directory: str | Path,
        *,
        max_depth: int | None = None,
        include: Sequence[str] | None = None,
    ) -> list[str]:
        """List repo files as sorted, relative POSIX paths for prompt context.

        VCS/venv/cache/build directories and bytecode are pruned *before* descending
        (in-place ``dirnames`` filtering) so they never reach the LLM prompt.
        ``include`` is a list of fnmatch patterns matched against the relative path.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            depth = len(current.relative_to(root).parts)
            dirnames[:] = sorted(d for d in dirnames if not _is_ignored_dir(d))
            if max_depth is not None and depth >= max_depth:
                dirnames[:] = []
            for name in sorted(filenames):
                if name.endswith(_IGNORED_FILE_SUFFIXES):
                    continue
                rel = (current / name).relative_to(root).as_posix()
                if include and not any(fnmatch(rel, pattern) for pattern in include):
                    continue
                files.append(rel)
        return sorted(files)


def _is_ignored_dir(name: str) -> bool:
    """True for directories that must stay out of the prompt (hidden or well-known junk)."""
    return name.startswith(".") or name in _IGNORED_DIR_NAMES or name.endswith(_IGNORED_DIR_SUFFIXES)