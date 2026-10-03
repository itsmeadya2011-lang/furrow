from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import aiofiles
from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
"""Used whenever a caller does not supply an explicit system prompt."""

DEFAULT_HTTP_TIMEOUT = 120.0
"""Socket timeout (seconds) for the ollama HTTP transport."""

SNIPPET_LIMIT = 200
"""Maximum number of characters echoed back inside :class:`JSONParseError`."""

RETRY_ATTEMPTS = 3
"""Total number of attempts for a completion before the original error is re-raised."""

IGNORED_NAMES = frozenset(
    {
        ".git",
        ".furrow",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
"""Directory names (and dot-names generally) skipped by :meth:`LLMClient.list_files`."""

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Base class for every failure raised by the LLM layer."""


class MissingAPIKeyError(LLMError):
    """No API key is configured for the selected provider."""


class JSONParseError(LLMError):
    """A model reply did not contain a usable JSON document."""


class UnsupportedProviderError(LLMError):
    """The configured provider is unknown or its optional dependency is missing."""


def _truncate(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Shorten ``text`` so error messages never echo a whole model reply."""
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def _iter_json_spans(text: str) -> Iterator[str]:
    """Yield every balanced top-level ``{...}``/``[...]`` span found in ``text``.

    The scanner tracks string literals and backslash escapes so braces inside JSON
    strings never unbalance the result.
    """
    length = len(text)
    for start, first in enumerate(text):
        if first not in "{[":
            continue
        closing = "}" if first == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, length):
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
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    if char == closing:
                        yield text[start : index + 1]
                    break


def _iter_candidates(text: str) -> Iterator[str]:
    """Yield the plausible JSON payloads of ``text``, most likely first."""
    stripped = text.strip()
    if stripped:
        yield stripped
    for match in _FENCE_RE.finditer(text):
        inner = match.group(1).strip()
        if inner:
            yield inner
    yield from _iter_json_spans(text)


def extract_json(text: str) -> Any:
    """Parse a JSON document out of arbitrary model output.

    Handles bare JSON, ``` fenced blocks, and prose wrapped around either. The
    first balanced top-level object/array that parses wins.

    Raises:
        JSONParseError: if no JSON document can be recovered.
    """
    if not isinstance(text, str):
        raise JSONParseError(f"Expected text to parse as JSON, got {type(text).__name__}")
    for candidate in _iter_candidates(text):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    raise JSONParseError(f"No JSON object or array found in model output: {_truncate(text)!r}")


def _as_int(value: object) -> int:
    """Best-effort coercion of a provider usage field to ``int``."""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


class LLMClient:
    """Thin async wrapper over the supported LLM providers.

    The provider SDKs (``anthropic``, ``openai``) are imported lazily so the
    package imports cleanly when only one of them is installed.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings if settings is not None else global_settings
        self._anthropic: Any = None
        self._openai: Any = None
        self._httpx: Any = None
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def provider(self) -> Provider:
        return Provider(self.settings.provider)

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    # ------------------------------------------------------------------ clients

    @property
    def anthropic(self) -> Any:
        """Lazily built ``anthropic.AsyncAnthropic`` client."""
        if self._anthropic is None:
            import anthropic

            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set. Export ANTHROPIC_API_KEY or set "
                    "FURROW_ANTHROPIC_API_KEY to use the 'anthropic' provider."
                )
            self._anthropic = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> Any:
        """Lazily built ``openai.AsyncOpenAI`` client."""
        if self._openai is None:
            import openai

            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set. Export OPENAI_API_KEY or set "
                    "FURROW_OPENAI_API_KEY to use the 'openai' provider."
                )
            self._openai = openai.AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def httpx(self) -> Any:
        """Lazily built ``httpx.AsyncClient`` used for the ollama provider."""
        if self._httpx is None:
            try:
                import httpx
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise UnsupportedProviderError(
                    "The 'ollama' provider needs httpx. Install it with: pip install httpx"
                ) from error
            self._httpx = httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT)
        return self._httpx

    # --------------------------------------------------------------- completion

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the raw assistant text for ``prompt``.

        Transient provider/network errors are retried; configuration errors
        (missing key, unknown provider) fail fast.
        """
        provider = self.provider
        target_model = self._resolve_model(provider, model)
        system_prompt = system or DEFAULT_SYSTEM_PROMPT
        resolved_temperature = (
            self.settings.temperature if temperature is None else float(temperature)
        )
        resolved_max_tokens = int(
            self.settings.max_tokens if max_tokens is None else max_tokens
        )

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            # tenacity names these bounds ``min``/``max`` (initial 1s, ceiling 8s).
            wait=wait_exponential(multiplier=1, min=1, max=8),
            # Every LLMError (and so every subclass: missing key, unparseable
            # reply, unknown provider, empty response) is deterministic, so
            # retrying only burns attempts and the real backoff.  Transient
            # SDK/network failures are not LLMError and still retry.
            retry=retry_if_not_exception_type(LLMError),
        ):
            with attempt:
                if provider is Provider.ANTHROPIC:
                    return await self._complete_anthropic(
                        prompt,
                        system_prompt,
                        target_model,
                        resolved_temperature,
                        resolved_max_tokens,
                    )
                if provider is Provider.OPENAI:
                    return await self._complete_openai(
                        prompt,
                        system_prompt,
                        target_model,
                        resolved_temperature,
                        resolved_max_tokens,
                    )
                if provider is Provider.OLLAMA:
                    return await self._complete_ollama(
                        prompt,
                        system_prompt,
                        target_model,
                        resolved_temperature,
                        resolved_max_tokens,
                    )
                raise UnsupportedProviderError(
                    f"Unsupported provider: {provider!r}. Use one of: "
                    f"{', '.join(item.value for item in Provider)}."
                )
        raise LLMError("LLM completion did not produce a response")  # pragma: no cover

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        repair_attempts: int = 1,
    ) -> Any:
        """Complete ``prompt`` and parse the reply as JSON.

        A reply that is fenced, wrapped in prose, or otherwise unparsable is
        re-requested up to ``repair_attempts`` times with a repair instruction
        appended to the original prompt.

        Raises:
            JSONParseError: when every attempt was unparsable; the message carries
                the last raw response.
        """
        response = await self.complete(prompt, system=system, model=model)
        attempts = max(0, int(repair_attempts))
        error: JSONParseError | None = None
        for index in range(attempts + 1):
            try:
                return extract_json(response)
            except JSONParseError as parse_error:
                error = parse_error
            if index < attempts:
                repair_prompt = (
                    f"{prompt}\n\n"
                    "Your previous reply could not be parsed as JSON. Reply again with the "
                    "JSON document only: no markdown code fences, no prose before or after it."
                )
                response = await self.complete(repair_prompt, system=system, model=model)
        raise JSONParseError(
            f"Model did not return valid JSON after {attempts + 1} attempt(s): {error}. "
            f"Last response: {_truncate(response)!r}"
        )

    def _resolve_model(self, provider: Provider, model: str | None) -> str:
        if model:
            return model
        if provider is Provider.OLLAMA:
            return self.settings.ollama_model
        return self.settings.model

    def _record_usage(self, input_tokens: object = None, output_tokens: object = None) -> None:
        self._input_tokens += _as_int(input_tokens)
        self._output_tokens += _as_int(output_tokens)

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
            )
        blocks = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                parts.append(block_text)
        if not parts:
            raise LLMError("Anthropic returned an empty response")
        return "".join(parts)

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("OpenAI returned no choices")
        return getattr(choices[0].message, "content", None) or ""

    async def _complete_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self.httpx
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Ollama returned no content: {_truncate(str(data))!r}")
        self._record_usage(
            data.get("prompt_eval_count") if isinstance(data, dict) else None,
            data.get("eval_count") if isinstance(data, dict) else None,
        )
        return content

    # -------------------------------------------------------------------- files

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as handle:
            return str(await handle.read())

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w") as handle:
            await handle.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Return workspace-relative file paths, sorted, with noise directories pruned.

        Symlinked directories are never followed and dot entries are skipped.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        collected: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not self._is_ignored(name) and not (base / name).is_symlink()
            )
            for name in filenames:
                if self._is_ignored(name):
                    continue
                candidate = base / name
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                collected.append(candidate.relative_to(root).as_posix())
        return sorted(collected)

    @staticmethod
    def _is_ignored(name: str) -> bool:
        return name in IGNORED_NAMES or name.startswith(".")

    # --------------------------------------------------------------- lifecycle

    async def close(self) -> None:
        """Close any provider client that was created. Safe to call repeatedly."""
        for attribute in ("_anthropic", "_openai", "_httpx"):
            client = getattr(self, attribute, None)
            setattr(self, attribute, None)
            if client is None:
                continue
            close = getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # shutdown must never raise
                continue

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()