from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings

try:  # pragma: no cover - trivial import guard, exercised indirectly
    # Anthropic's SDK moves its block classes between versions and does NOT
    # re-export them at the package root (``from anthropic import TextBlock``
    # raises ImportError on 1.11.0).  Import defensively so a future SDK layout
    # can never make ``import furrow`` fail: block handling below is duck-typed.
    from anthropic.types import TextBlock as AnthropicTextBlock
except ImportError:  # pragma: no cover - depends on the installed SDK version
    AnthropicTextBlock = None  # type: ignore[assignment, misc]

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

#: Directories that only ever add noise to a workspace listing.
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)

#: Generous ceiling for a single non-streaming HTTP completion.
HTTP_TIMEOUT_SECONDS = 120.0

#: Substrings of provider error messages that mean "retry me" when the SDK
#: did not expose a status code (SDKs are not consistent about ``status_code``).
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "server_error",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "timeout",
    "timed out",
    "connection reset",
    "connection error",
    "temporarily unavailable",
)


def _content_block_text(block: object) -> str | None:
    """Return the text carried by a response content block, or ``None`` if it has none.

    Deliberately duck-typed rather than ``isinstance(block, TextBlock)``: a response
    mixes text blocks with ``thinking`` / ``tool_use`` / ``redacted_thinking``
    blocks, and the concrete classes are SDK-version specific.  Anything exposing a
    ``text`` string is kept; everything else is skipped instead of raising.
    """
    if AnthropicTextBlock is not None and isinstance(block, AnthropicTextBlock):
        # Fast path: ``text`` is a required str on the SDK's own text block class.
        return block.text or None
    # Duck-typed path: thinking / tool_use / redacted_thinking blocks, blocks from a
    # different SDK version, or test doubles only count when they expose real text.
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else None


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code from an SDK or transport exception."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient_error(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying: throttling, overload, 5xx, transport, timeouts.

    Authentication/permission problems and bad configuration are never retried.
    """
    if isinstance(exc, ValueError):
        # Raised locally for missing API keys or unsupported providers.
        return False
    if isinstance(exc, (asyncio.TimeoutError, httpx.TransportError)):
        return True
    status = _status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or global_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        provider = self.settings.provider
        request: Callable[[], Awaitable[str]]
        if provider == Provider.ANTHROPIC:
            model = model or self.settings.model
            request = partial(self._complete_anthropic, prompt, system, model)
        elif provider == Provider.OPENAI:
            model = model or self.settings.model
            request = partial(self._complete_openai, prompt, system, model)
        elif provider == Provider.OLLAMA:
            model = model or self.settings.ollama_model or self.settings.model
            request = partial(self._complete_ollama, prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        return await self._with_retries(request)

    async def _with_retries(self, request: Callable[[], Awaitable[str]]) -> str:
        """Run a provider request, retrying transient failures with exponential backoff."""
        attempts = max(1, self.settings.llm_max_retries + 1)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception(_is_transient_error),
            reraise=True,
        ):
            with attempt:
                return await request()
        raise RuntimeError("retry loop exited without a result")  # pragma: no cover

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.llm_max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Content blocks mix text with tool_use/thinking blocks; keep the text only.
        return "".join(
            text for text in (_content_block_text(block) for block in response.content) if text
        )

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            return ""
        return str(message.get("content") or "")

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        try:
            async with aiofiles.open(p) as f:
                return await f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {p}") from exc

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Every file under ``directory``, as sorted paths relative to it."""
        root = Path(directory)
        if not root.is_dir():
            return []
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune in place so we never descend into noisy directories.
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRECTORIES]
            base = Path(dirpath)
            for name in filenames:
                candidate = base / name
                try:
                    files.append(str(candidate.relative_to(root)))
                except ValueError:  # pragma: no cover - root spelling differs
                    files.append(os.path.relpath(candidate, root))
        return sorted(files)
