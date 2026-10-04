"""Shared fixtures and test doubles for the Furrow test suite.

Three hazards are neutralised here so the individual test modules can stay
readable and order-independent:

* **The ``get_settings()`` singleton.** ``furrow.config.get_settings`` is
  ``functools.lru_cache``-backed and the module-level ``settings`` alias points at
  whatever it returned on first import. The CLI mutates that object in place, so
  a test that leaves ``override_model`` or ``workspace`` set would leak into
  every later test. The autouse :func:`_isolate_settings` fixture snapshots the
  singleton's ``__dict__`` and clears the cache around every test.
* **Ambient environment.** ``FURROW_*`` variables and provider keys are read by
  ``pydantic-settings`` and by ``Settings.api_key_for``. The autouse
  :func:`_clean_env` fixture deletes them so a developer's shell cannot change a
  result.
* **The filesystem and the network.** ``settings_factory`` always points
  ``Settings.workspace`` at ``tmp_path``, and :class:`FakeLLMClient` replaces the
  single ``complete`` entry point while inheriting the *real* sandboxed
  ``write_file`` / ``read_file`` / ``resolve_path`` / ``list_files``. No test ever
  contacts a provider.

Example:
    ```python
    async def test_something(settings_factory):
        client = FakeLLMClient(settings=settings_factory(), responses=["hi"])
        assert await client.complete("prompt") == "hi"
    ```
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import pytest

import furrow.config as config_module
from furrow.config import Settings
from furrow.core.events import FurrowEvent
from furrow.llm import LLMClient

_ENV_PREFIX = "FURROW_"

#: Environment variables that must not influence any test, because
#: ``Settings.api_key_for`` and the provider SDKs consult them directly.
_EXTRA_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OLLAMA_HOST",
    "FURROW_OLLAMA_BASE_URL",
)


class LlmCall(NamedTuple):
    """One recorded ``LLMClient.complete`` invocation.

    Attributes:
        prompt: The user message the caller passed.
        system: The system prompt the caller passed.
        model: The per-role model the caller passed, before override resolution.
    """

    prompt: str
    system: str
    model: str | None


class FakeLLMClient(LLMClient):
    """An :class:`~furrow.llm.LLMClient` whose only networked method is stubbed.

    ``complete`` is replaced so no provider SDK is ever constructed or called.
    Every other method — ``write_file``, ``read_file``, ``resolve_path``,
    ``list_files``, ``release`` — is the production implementation, so the
    workspace sandbox is genuinely exercised by the agent tests.

    Attributes:
        responses: Queue of canned replies consumed one per ``complete`` call. A
            queued item may be an exception instance (raised) or a callable
            (invoked with the prompt and its result returned).
        calls: Every ``(prompt, system, model)`` triple seen, in order.
        released: How many times :meth:`release` was called.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        responses: list[Any] | None = None,
    ) -> None:
        """Initialise the double.

        Args:
            settings: Configuration to expose through ``.settings``. Callers
                should always pass an instance built by ``settings_factory``.
            responses: Canned replies for :meth:`complete`.
        """
        super().__init__(settings=settings)
        self.responses: list[Any] = list(responses or [])
        self.calls: list[LlmCall] = []
        self.released = 0

    def queue(self, *responses: Any) -> None:
        """Append canned replies to the queue.

        Args:
            *responses: Replies, exception instances or prompt-aware callables.
        """
        self.responses.extend(responses)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return the next queued reply, recording the call.

        Args:
            prompt: Recorded verbatim.
            system: Recorded verbatim.
            model: Recorded verbatim.

        Returns:
            The next queued reply, coerced to ``str``.

        Raises:
            AssertionError: If the queue is empty, which always means a test
                under-specified how many completions its code path makes.
            BaseException: Whatever exception instance was queued.
        """
        self.calls.append(LlmCall(prompt=prompt, system=system, model=model))
        if not self.responses:
            raise AssertionError(
                f"FakeLLMClient ran out of queued responses on call {len(self.calls)}"
            )
        reply = self.responses.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(prompt)
        return str(reply)

    def release(self) -> None:
        """Record the call, then run the production cleanup."""
        self.released += 1
        super().release()

    @property
    def prompts(self) -> list[str]:
        """Return every recorded prompt, in call order."""
        return [call.prompt for call in self.calls]

    @property
    def models(self) -> list[str | None]:
        """Return every recorded per-role model, in call order."""
        return [call.model for call in self.calls]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every ambient variable that could change a test result.

    Args:
        monkeypatch: Pytest's environment patcher.
    """
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIX) or name in _EXTRA_ENV_KEYS:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_settings() -> Iterator[Settings]:
    """Snapshot and restore the process-wide ``get_settings()`` singleton.

    The cached instance's ``__dict__`` is deep-copied on entry and written back
    on exit, so a test that mutates the CLI's settings object cannot affect any
    other test. The ``lru_cache`` is cleared as well: that way a test that
    constructs its own ``Settings`` never becomes the process-wide default for
    the next test, and ``get_settings()`` re-resolves from a known-clean
    environment.

    Yields:
        The live singleton, for tests that need to inspect it.
    """
    singleton = config_module.settings
    snapshot = copy.deepcopy(dict(singleton.__dict__))
    config_module.get_settings.cache_clear()
    try:
        yield singleton
    finally:
        config_module.get_settings.cache_clear()
        singleton.__dict__.clear()
        singleton.__dict__.update(snapshot)


@pytest.fixture
def settings_factory(tmp_path: Any) -> Callable[..., Settings]:
    """Return a builder for test-friendly, tmp-path-scoped settings.

    Args:
        tmp_path: Per-test directory used as the sandbox workspace.

    Returns:
        A callable accepting any :class:`~furrow.config.Settings` keyword and
        returning a configured instance with a fast timeout, one retry and
        throwaway API keys.
    """

    def build(**overrides: Any) -> Settings:
        """Build one :class:`~furrow.config.Settings` for a test.

        Args:
            **overrides: Fields to override on top of the test defaults.

        Returns:
            The configured settings instance.
        """
        defaults: dict[str, Any] = {
            "workspace": tmp_path,
            "max_parallel_tasks": 5,
            "max_cycles": 2,
            "max_tokens": 64,
            "request_timeout": 1.0,
            "max_retries": 1,
            "test_timeout": 10.0,
            "test_command": None,
            "anthropic_api_key": "test-anthropic-key",
            "openai_api_key": "test-openai-key",
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return build


@pytest.fixture
def event_collector() -> tuple[list[FurrowEvent], Callable[[FurrowEvent], None]]:
    """Return a growing event list paired with an ``on_event`` callback.

    Returns:
        A ``(events, on_event)`` pair for use as
        ``Orchestrator(..., on_event=on_event)``.
    """
    events: list[FurrowEvent] = []
    return events, events.append
