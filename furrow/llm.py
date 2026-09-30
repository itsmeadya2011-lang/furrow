from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
MAX_ATTEMPTS = 3


class LLMError(Exception):
    """Raised when an LLM request fails at runtime (after retries are exhausted)."""


class LLMConfigError(Exception):
    """Raised when the LLM client is misconfigured (missing API key, bad provider, etc.)."""


def _log_level_value(level_name: str) -> int:
    value = logging.getLevelName(level_name.upper())
    return value if isinstance(value, int) else logging.INFO


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
        wrapper_class=structlog.make_filtering_bound_logger(_log_level_value(_settings.log_level)),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("furrow.llm")


def _get_status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that are safe to retry (network, rate limits, 5xx)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return True
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        code = _get_status_code(exc)
        if code is not None and 500 <= code < 600:
            return True
    return False


def _on_retry(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    if outcome is None:
        return
    exc = outcome.exception()
    if exc is None:
        return
    logger.warning(
        "llm_retry_scheduled",
        attempt=retry_state.attempt_number,
        error=type(exc).__name__,
        error_message=str(exc),
    )


def build_retry_config() -> dict[str, Any]:
    """Build the tenacity retry configuration shared by all LLM requests."""
    return {
        "stop": stop_after_attempt(MAX_ATTEMPTS),
        "wait": wait_exponential(multiplier=1, min=2, max=10),
        "retry": retry_if_exception(_is_transient),
        "before_sleep": _on_retry,
        "reraise": True,
    }


def _user_message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or _settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMConfigError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        return await self._invoke(
            [_user_message(prompt)],
            system=system,
            model=model or self.settings.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def complete_with_messages(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        return await self._invoke(
            messages,
            system=system,
            model=model or self.settings.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _invoke(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        provider = self.settings.provider
        input_length = sum(
            len(m["content"]) for m in messages if isinstance(m.get("content"), str)
        )
        self._log_call(provider, model, system, input_length)
        try:
            if provider == Provider.ANTHROPIC:
                text = await self._complete_anthropic(
                    messages, system, model, max_tokens, temperature
                )
            elif provider == Provider.OPENAI:
                text = await self._complete_openai(
                    messages, system, model, max_tokens, temperature
                )
            else:
                raise LLMConfigError(f"Unsupported provider: {provider}")
        except LLMError:
            raise
        except (openai.OpenAIError, anthropic.AnthropicError) as exc:
            self._log_error(provider, model, exc)
            raise LLMError(
                f"Provider {provider.value} model {model} failed: {exc}"
            ) from exc
        self._log_success(
            provider, model, input_length=input_length, output_length=len(text)
        )
        return text

    @retry(**build_retry_config())
    async def _complete_anthropic(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system or DEFAULT_SYSTEM_PROMPT,
            "messages": messages,
        }
        response = await self.anthropic.messages.create(**kwargs)
        return self._extract_text(response)

    @retry(**build_retry_config())
    async def _complete_openai(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
        }
        response = await self.openai.chat.completions.create(**kwargs)
        return self._extract_text(response)

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from either an Anthropic or OpenAI response."""
        content = getattr(response, "content", None)
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if getattr(block, "type", None) == "text":
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
            return "".join(texts)
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                return getattr(message, "content", None) or ""
        return ""

    def _log_call(
        self, provider: Provider, model: str, system: str, input_length: int
    ) -> None:
        logger.info(
            "llm_call",
            provider=provider.value,
            model=model,
            input_length=input_length,
            has_system=bool(system),
        )

    def _log_success(
        self,
        provider: Provider,
        model: str,
        input_length: int,
        output_length: int,
    ) -> None:
        logger.info(
            "llm_response",
            provider=provider.value,
            model=model,
            input_length=input_length,
            output_length=output_length,
        )

    def _log_error(self, provider: Provider, model: str, exc: BaseException) -> None:
        logger.error(
            "llm_error",
            provider=provider.value,
            model=model,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
