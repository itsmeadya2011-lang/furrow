"""Provider-agnostic facade over the language models that drive Furrow.

:class:`LLMClient` hides the differences between the three supported back ends
(Anthropic, OpenAI and a local Ollama server) behind a single small async API,
adds retry with exponential backoff for transient provider failures, and owns
the sandboxed filesystem helpers that agents are allowed to use.

Every filesystem entry point resolves its argument against `Settings.workspace`
and refuses to leave it, so an LLM-authored path such as ``../../etc/passwd`` is
rejected rather than written.

Example:
    ```python
    client = LLMClient()
    try:
        answer = await client.complete("Summarise the repository layout.")
    finally:
        client.release()
    ```
"""

from __future__ import annotations

import os
from collections.abc import Coroutine
from pathlib import Path

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, get_settings

DEFAULT_SYSTEM_PROMPT: str = "You are a helpful coding assistant."

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".tox",
        ".idea",
        ".DS_Store",
    }
)

MAX_LIST_FILES: int = 2000

_SERVER_ERROR_FLOOR = 500
_RATE_LIMIT_STATUS = 429


class LLMError(RuntimeError):
    """Raised when an LLM request fails after exhausting retries."""


def _is_transient(exc: BaseException) -> bool:
    """Return whether an exception represents a retryable provider failure.

    Only infrastructure problems qualify: rate limits, timeouts, dropped
    connections and 5xx responses. Client-side 4xx problems (authentication,
    validation, not-found) are deterministic, so retrying them only wastes time
    and tokens.

    Args:
        exc: The exception raised by a provider call or by the transport.

    Returns:
        True if the request is worth retrying, False otherwise.
    """
    if isinstance(exc, (AnthropicRateLimitError, OpenAIRateLimitError)):
        return True
    if isinstance(exc, (AnthropicAPITimeoutError, OpenAIAPITimeoutError)):
        return True
    if isinstance(exc, (AnthropicAPIConnectionError, OpenAIAPIConnectionError)):
        return True
    if isinstance(exc, (AnthropicAPIStatusError, OpenAIAPIStatusError)):
        return getattr(exc, "status_code", 0) >= _SERVER_ERROR_FLOOR
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= _SERVER_ERROR_FLOOR
    return isinstance(exc, httpx.TransportError)


def _error_snippet(response: httpx.Response, limit: int = 200) -> str:
    """Return a short, safe excerpt of a response body for error messages.

    Args:
        response: The HTTP response whose body should be excerpted.
        limit: Maximum number of characters to include.

    Returns:
        The leading characters of the body, or a placeholder if unavailable.
    """
    try:
        return response.text[:limit]
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        return "<unreadable response body>"


class LLMClient:
    """Async facade for text completion plus sandboxed workspace file access.

    Attributes:
        settings: Resolved configuration; never a process-global default when an
            explicit instance is passed, so callers cannot leak configuration
            into one another.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the client.

        Args:
            settings: Configuration to use. When omitted the shared
                :func:`furrow.config.get_settings` singleton is resolved lazily
                at construction time rather than being bound as a default
                argument, which previously made per-client mutation global.
        """
        self.settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Return the lazily constructed Anthropic client.

        Returns:
            A configured :class:`anthropic.AsyncAnthropic` client.

        Raises:
            LLMError: If no Anthropic API key is configured.
        """
        if self._anthropic is None:
            api_key = self.settings.api_key_for(Provider.ANTHROPIC)
            if not api_key:
                raise LLMError(
                    "Anthropic API key is not configured; set 'anthropic_api_key' in "
                    ".env or export ANTHROPIC_API_KEY."
                )
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Return the lazily constructed OpenAI client.

        Returns:
            A configured :class:`openai.AsyncOpenAI` client.

        Raises:
            LLMError: If no OpenAI API key is configured.
        """
        if self._openai is None:
            api_key = self.settings.api_key_for(Provider.OPENAI)
            if not api_key:
                raise LLMError(
                    "OpenAI API key is not configured; set 'openai_api_key' in .env "
                    "or export OPENAI_API_KEY."
                )
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return a single text completion for `prompt`.

        Args:
            prompt: The user message to send to the model.
            system: Optional system prompt. Falls back to
                :data:`DEFAULT_SYSTEM_PROMPT` when empty.
            model: Optional per-role model name. Ignored when
                ``settings.override_model`` is set, which is how the CLI
                ``--model`` flag takes effect.

        Returns:
            The concatenated text content of the model response.

        Raises:
            LLMError: If the provider is unknown, the request fails after the
                configured retries, or the response carries no text.
        """
        model = self.settings.override_model or model or self.settings.model
        system = system or DEFAULT_SYSTEM_PROMPT
        try:
            async for attempt in self._retrying():
                with attempt:
                    return await self._complete(prompt, system, model)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"{self.settings.provider} request for model {model!r} failed: {exc}"
            ) from exc
        # Unreachable: AsyncRetrying re-raises once the stop condition trips.
        raise LLMError(f"{self.settings.provider} request for model {model!r} made no attempt")

    async def _complete(self, prompt: str, system: str, model: str) -> str:
        """Dispatch a completion request to the configured provider.

        Args:
            prompt: The user message to send to the model.
            system: The system prompt to send with the request.
            model: The resolved model name.

        Returns:
            The text content of the response.

        Raises:
            LLMError: If the configured provider has no implementation.
        """
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise LLMError(f"Unsupported provider: {provider!r}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through the Anthropic Messages API.

        Args:
            prompt: The user message to send to the model.
            system: The system prompt to send with the request.
            model: The resolved model name.

        Returns:
            Every text block of the response, concatenated.

        Raises:
            LLMError: If the response contains no textual content block.
        """
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        chunks: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        if not chunks:
            raise LLMError(
                "Anthropic returned no text content "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
            )
        return "".join(chunks)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through the OpenAI chat completions API.

        Args:
            prompt: The user message to send to the model.
            system: The system prompt to send with the request.
            model: The resolved model name.

        Returns:
            The content of the first choice.

        Raises:
            LLMError: If the response has no choices or no textual content.
        """
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise LLMError("OpenAI returned no choices in the response")
        choice = response.choices[0]
        content = choice.message.content
        text = content if isinstance(content, str) else ""
        if not text.strip():
            raise LLMError(
                "OpenAI returned empty content "
                f"(finish_reason={choice.finish_reason!r})"
            )
        return text

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through a local Ollama server.

        Args:
            prompt: The user message to send to the model.
            system: The system prompt to send with the request.
            model: The resolved model name, expected to be pulled locally.

        Returns:
            The generated response text.

        Raises:
            LLMError: If the server answers with a non-200 status, a body that
                is not JSON, or an empty response field.
        """
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {"model": model, "prompt": prompt, "system": system, "stream": False}
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as http:
            response = await http.post(url, json=payload)
        if response.status_code != httpx.codes.OK:
            if response.status_code >= _SERVER_ERROR_FLOOR or (
                response.status_code == _RATE_LIMIT_STATUS
            ):
                response.raise_for_status()
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}: {_error_snippet(response)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                f"Ollama returned a malformed (non-JSON) response: {_error_snippet(response)}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMError("Ollama returned a JSON payload that is not an object")
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            detail = data.get("error") or data.get("done_reason") or "no 'response' field"
            raise LLMError(f"Ollama returned no text ({detail})")
        return text

    def _retrying(self) -> AsyncRetrying:
        """Build a tenacity retry controller for a single completion call.

        Returns:
            An :class:`tenacity.AsyncRetrying` configured with
            ``settings.max_retries`` attempts, exponential backoff capped at 30
            seconds, and a predicate that only retries transient failures.
        """
        return AsyncRetrying(
            stop=stop_after_attempt(max(1, self.settings.max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )

    async def read_file(self, path: str | Path) -> str:
        """Read a UTF-8 text file from inside the workspace.

        Args:
            path: Workspace-relative (or absolute, if inside) file path.

        Returns:
            The full contents of the file.

        Raises:
            ValueError: If the resolved path escapes the workspace.
            OSError: If the file does not exist or cannot be read.
        """
        target = self.resolve_path(path)
        async with aiofiles.open(target, encoding="utf-8") as handle:
            return str(await handle.read())

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write UTF-8 text to a file inside the workspace.

        Missing parent directories are created. Paths that resolve outside the
        workspace are rejected before anything is written, so an LLM cannot use
        a crafted path to modify the rest of the filesystem.

        Args:
            path: Workspace-relative (or absolute, if inside) file path.
            content: The text to write.

        Raises:
            ValueError: If the resolved path escapes the workspace.
            OSError: If the file cannot be created or written.
        """
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        """List regular files under `directory`, relative to `directory`.

        Directories named in :data:`DEFAULT_IGNORED_DIRS` are pruned at any
        depth, symlinked directories are never descended into, unreadable
        entries are skipped, and the walk is capped at
        :data:`MAX_LIST_FILES` entries.

        Args:
            directory: Directory to walk. Defaults to the configured workspace.

        Returns:
            Sorted POSIX-style relative paths, empty if the directory is absent
            or is not a directory.
        """
        base = self._workspace_root() if directory is None else self.resolve_path(directory)
        if not base.is_dir():
            return []
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
            current = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in DEFAULT_IGNORED_DIRS and not (current / name).is_symlink()
            )
            for name in sorted(filenames):
                if name in DEFAULT_IGNORED_DIRS:
                    continue
                entry = current / name
                try:
                    if not entry.is_file():
                        continue
                    found.append(entry.relative_to(base).as_posix())
                except OSError:
                    continue
                if len(found) >= MAX_LIST_FILES:
                    return sorted(found)
        return sorted(found)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve `path` against settings.workspace, rejecting escapes.

        Relative paths are joined onto the workspace root. Both sides are fully
        resolved, so symlinks and ``..`` segments cannot be used to leave the
        sandbox. A filesystem root is refused as a workspace, since it would
        make the containment check vacuous.

        Args:
            path: Workspace-relative or absolute path to resolve.

        Returns:
            The absolute, normalised path inside the workspace.

        Raises:
            ValueError: If the path resolves outside the workspace root, or if
                the configured workspace is the filesystem root.
        """
        root = self._workspace_root()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Path {str(candidate)!r} resolves to {str(resolved)!r}, which is outside "
                f"the workspace {str(root)!r}"
            )
        return resolved

    def _workspace_root(self) -> Path:
        """Return the resolved workspace root directory.

        Returns:
            The absolute workspace path with symlinks and ``..`` removed.

        Raises:
            ValueError: If the configured workspace is the filesystem root,
                which would disable path containment.
        """
        root = Path(self.settings.workspace or ".").expanduser().resolve()
        if root.parent == root:
            raise ValueError(
                f"Workspace {str(root)!r} is the filesystem root and cannot be used as a "
                "sandbox; set 'workspace' to a dedicated directory."
            )
        return root

    def release(self) -> None:
        """Close the SDK clients and drop the cached instances.

        The SDKs only expose asynchronous ``close()`` coroutines, so any
        pending close is discarded rather than awaited; the underlying HTTP
        connections are released when the client is garbage collected. Safe to
        call more than once.
        """
        for attribute in ("_anthropic", "_openai"):
            client: AsyncAnthropic | AsyncOpenAI | None = getattr(self, attribute)
            setattr(self, attribute, None)
            if client is None:
                continue
            closer = getattr(client, "close", None)
            if closer is None:
                continue
            pending = closer()
            if isinstance(pending, Coroutine):
                pending.close()
