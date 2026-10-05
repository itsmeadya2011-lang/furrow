from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Iterator, Optional

import pytest

from furrow.config import Provider, Settings

# NOTE: furrow.config.settings is a process-wide singleton built at import time.
# Tests that need hermetic configuration should build their own Settings instance
# via the fixtures below rather than mutating the singleton.

_ENV_PREFIXES = ("FURROW_", "ANTHROPIC_", "OPENAI_", "OLLAMA_", "KILO_")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Give every test a clean environment and a throwaway working directory."""
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture()
def settings(tmp_path: Any) -> Settings:
    """A hermetic Anthropic Settings instance with short timeouts."""
    return Settings.model_validate(
        {
            "provider": Provider.ANTHROPIC,
            "model": "claude-sonnet-4-20250514",
            "planner_model": "claude-3-5-haiku-20241022",
            "worker_model": "claude-3-5-sonnet-20241022",
            "tester_model": "claude-3-5-sonnet-20241022",
            "anthropic_api_key": None,
            "openai_api_key": None,
            "workspace": tmp_path,
            "state_dir": tmp_path / ".furrow",
            "max_parallel_tasks": 2,
            "max_cycles": 3,
            "llm_timeout_seconds": 1.0,
            "test_timeout_seconds": 5.0,
            "task_timeout_seconds": 5.0,
            "llm_max_retries": 1,
        }
    )


@pytest.fixture()
def settings_openai(tmp_path: Any) -> Settings:
    """A hermetic OpenAI Settings instance with short timeouts."""
    return Settings.model_validate(
        {
            "provider": Provider.OPENAI,
            "model": "gpt-4o",
            "planner_model": "gpt-4o-mini",
            "worker_model": "gpt-4o",
            "tester_model": "gpt-4o",
            "anthropic_api_key": None,
            "openai_api_key": None,
            "workspace": tmp_path,
            "state_dir": tmp_path / ".furrow",
            "max_parallel_tasks": 2,
            "max_cycles": 3,
            "llm_timeout_seconds": 1.0,
            "test_timeout_seconds": 5.0,
            "task_timeout_seconds": 5.0,
            "llm_max_retries": 1,
        }
    )


class StubLLMClient:
    """Deterministic LLM client for tests.

    Responses are popped from a queue in order; the last one repeats when the
    queue is exhausted.  Every call is recorded in ``self.calls``.
    """

    def __init__(
        self,
        settings: Any,
        responses: Optional[list[str]] = None,
        default_response: str = "",
    ) -> None:
        self.settings = settings
        self._responses: list[str] = list(responses) if responses else []
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []
        self._raise_next: Optional[BaseException] = None
        self.response_hook: Optional[Callable[[str], str]] = None
        self._last_response: Optional[str] = None

    def raise_next(self, exc: BaseException) -> None:
        """Make the next ``complete`` call raise ``exc``."""
        self._raise_next = exc

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
    ) -> str:
        """Return the next queued response, or raise if one was queued."""
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self._raise_next is not None:
            exc = self._raise_next
            self._raise_next = None
            raise exc
        if self.response_hook is not None:
            response = self.response_hook(prompt)
        elif self._responses:
            response = self._responses.pop(0)
        elif self._last_response is not None:
            response = self._last_response
        else:
            response = self.default_response
        self._last_response = response
        return response

    async def aclose(self) -> None:
        """No-op close for interface parity with :class:`LLMClient`."""


@pytest.fixture()
def stub_client(settings: Settings) -> AsyncIterator[StubLLMClient]:
    """Yield a StubLLMClient bound to the hermetic ``settings`` fixture."""
    client = StubLLMClient(settings)
    yield client


async def wait_for(
    predicate: Callable[[], Awaitable[bool]],
    timeout: float = 2.0,
    interval: float = 0.01,
) -> bool:
    """Poll ``predicate`` until it returns True or ``timeout`` elapses."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return await predicate()