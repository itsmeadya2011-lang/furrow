"""Shared pytest fixtures for the Furrow test suite.

Two stand-ins do the heavy lifting:

* :class:`FakeLLMClient` mirrors :class:`furrow.llm.LLMClient` but makes no
  network calls. Every request is appended to ``.calls`` as a
  :class:`RecordedCall` so a test can assert on the prompt, system prompt,
  model and tools that were actually sent.
* :class:`EventCollector` subscribes to an :class:`~furrow.core.events.EventBus`
  and records every :class:`~furrow.core.events.Event` it receives, which makes
  loop behaviour assertable without reaching into the orchestrator.

Both are plain classes, importable outside pytest, so a test module can build
them directly when a fixture is not wanted.

The ``make_settings`` factory never mutates the module-level
``furrow.config.settings`` singleton: every call returns a brand-new
:class:`~furrow.config.Settings` built from hermetic defaults.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Settings
from furrow.config import TestResult as TestResultModel
from furrow.core.events import Event, EventBus
from furrow.llm import CompletionResult, ToolCall, Usage
from furrow.tools import ToolResult

# ``furrow.config.TestResult`` is a pydantic model whose name starts with
# ``Test``, so pytest tries to collect it as a test class and warns. Marking it
# non-collectable here (conftest is imported before any test module) silences
# the warning process-wide; pytest's PyCollector.isnosetest() checks exactly
# this attribute. This mutates the imported class object, not furrow/config.py.
TestResultModel.__test__ = False


#: A response that validates as BOTH ``Plan`` (planner) and ``TestResult``
#: (tester), so ``FakeLLMClient()`` is usable with zero configuration by either
#: agent. Pydantic ignores the fields each model does not declare.
DEFAULT_TEXT: str = (
    '{"tasks": [], "rationale": "Fake planner: no further work to do.", '
    '"passed": true, "summary": "Fake tester: the run passed.", "failures": []}'
)

#: Summary returned by the default tool-loop completion (worker agent).
DEFAULT_TOOL_TEXT: str = "Fake worker: the requested change was applied."


@dataclass
class RecordedCall:
    """One recorded LLM request, for assertions on what was actually sent."""

    prompt: str
    system: str
    model: str
    tools: object
    max_iterations: int | None
    kwargs: dict[str, object]


def _default_detailed() -> CompletionResult:
    """Default payload for :meth:`FakeLLMClient.complete_detailed`."""
    return CompletionResult(
        text=DEFAULT_TEXT,
        tool_calls=[],
        usage=Usage(input_tokens=0, output_tokens=0),
        model="fake-model",
        provider="fake",
        iterations=1,
        truncated=False,
    )


def _default_tool_result() -> CompletionResult:
    """Default payload for :meth:`FakeLLMClient.complete_with_tools`."""
    return CompletionResult(
        text=DEFAULT_TOOL_TEXT,
        tool_calls=[],
        usage=Usage(input_tokens=0, output_tokens=0),
        model="fake-model",
        provider="fake",
        iterations=1,
        truncated=False,
    )


class FakeLLMClient:
    """Stand-in for :class:`furrow.llm.LLMClient`. Makes NO network calls.

    Configure the responses before use::

        client = FakeLLMClient()
        client.text = '{"tasks": [], "rationale": "done"}'   # returned by .complete()
        client.detailed = CompletionResult(text=..., ...)    # returned by .complete_detailed()
        client.tool_result = CompletionResult(...)           # returned by .complete_with_tools()

    The defaults already satisfy every agent: ``.text`` parses as both a plan
    and a test verdict, ``.detailed`` wraps it, and ``.tool_result`` is a
    successful, untruncated worker summary with no tool calls.

    Every call is recorded in :attr:`calls` (a list of :class:`RecordedCall`)
    with the prompt, system, model and tools that were passed. ``model`` is
    recorded resolved (an explicit argument, else ``settings.model``) and
    ``max_iterations`` likewise defaults to ``settings.worker_max_iterations``.
    Set :attr:`raises` to an exception instance to make every call raise it
    instead — the call is still recorded first, so the request is assertable.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.tools: Any = None
        self.text: str = DEFAULT_TEXT
        self.detailed: CompletionResult = _default_detailed()
        self.tool_result: CompletionResult = _default_tool_result()
        self.raises: BaseException | None = None
        self.calls: list[RecordedCall] = []
        self.usage: Usage = Usage(input_tokens=0, output_tokens=0)
        self.closed: bool = False

    # -- recording --

    def _record(
        self,
        prompt: str,
        system: str,
        model: str | None,
        tools: object,
        max_iterations: int | None,
        kwargs: dict[str, object],
    ) -> RecordedCall:
        """Append a :class:`RecordedCall` and return it."""
        call = RecordedCall(
            prompt=prompt,
            system=system,
            model=model if model is not None else getattr(self.settings, "model", ""),
            tools=tools,
            max_iterations=max_iterations,
            kwargs=kwargs,
        )
        self.calls.append(call)
        return call

    def _fail(self) -> None:
        """Raise :attr:`raises` when it is set."""
        if self.raises is not None:
            raise self.raises

    def _accumulate(self, result: CompletionResult) -> CompletionResult:
        """Add a completion's usage to the cumulative counter."""
        self.usage = self.usage + result.usage
        return result

    # -- LLMClient surface --

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None, **kwargs: object
    ) -> str:
        """Return :attr:`text`, as :meth:`LLMClient.complete` would."""
        self._record(prompt, system, model, None, None, kwargs)
        self._fail()
        return self.text

    async def complete_detailed(
        self, prompt: str, system: str = "", model: str | None = None, **kwargs: object
    ) -> CompletionResult:
        """Return :attr:`detailed`, as :meth:`LLMClient.complete_detailed` would."""
        self._record(prompt, system, model, None, None, kwargs)
        self._fail()
        return self._accumulate(self.detailed)

    async def complete_with_tools(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        tools: Any = None,
        max_iterations: int | None = None,
    ) -> CompletionResult:
        """Return :attr:`tool_result`, as :meth:`LLMClient.complete_with_tools` would."""
        resolved_iterations = max_iterations
        if resolved_iterations is None:
            resolved_iterations = getattr(self.settings, "worker_max_iterations", None)
        self._record(prompt, system, model, tools, resolved_iterations, {})
        self._fail()
        return self._accumulate(self.tool_result)

    async def aclose(self) -> None:
        """Mark the client closed. Safe to call more than once."""
        self.closed = True

    def reset_usage(self) -> None:
        """Zero the cumulative usage counter, as :meth:`LLMClient.reset_usage` does."""
        self.usage = Usage()


class EventCollector:
    """Subscribes to an :class:`~furrow.core.events.EventBus` and records
    every :class:`~furrow.core.events.Event` it receives."""

    def __init__(self, bus: EventBus) -> None:
        self._events: list[Event] = []
        self._bus = bus
        self._unsubscribe: Callable[[], None] = bus.subscribe(self._handle)

    def _handle(self, event: Event) -> None:
        """Bus handler: append the event. Sync, so ``emit_nowait`` also works."""
        self._events.append(event)

    @property
    def events(self) -> list[Event]:
        """Every event received so far, in delivery order."""
        return self._events

    def types(self) -> list[str]:
        """The ``EventType.value`` of every recorded event, in order."""
        return [event.type.value for event in self._events]

    def of_type(self, event_type: Any) -> list[Event]:
        """Every recorded event whose ``type`` equals ``event_type``."""
        return [event for event in self._events if event.type == event_type]

    def clear(self) -> None:
        """Forget every recorded event."""
        self._events.clear()

    def unsubscribe(self) -> None:
        """Detach from the bus. Called automatically at collector teardown."""
        self._unsubscribe()


# --- pytest fixtures ---


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A Path to an empty temporary workspace directory."""
    path = tmp_path / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def make_settings(tmp_workspace: Path) -> Callable[..., Settings]:
    """Factory returning Settings instances that NEVER mutate the global
    ``furrow.config.settings`` singleton.

    ``make_settings(**overrides) -> Settings``. Defaults every test needs to be
    hermetic: ``workspace=<tmp_workspace>``, ``max_parallel_tasks=2``,
    ``max_cycles=1``, ``allow_shell=False`` and NO api keys. A ``.env`` file is
    never read, and the explicit values above win over any ``FURROW_*`` env var.
    """

    def _make_settings(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "workspace": tmp_workspace,
            "max_parallel_tasks": 2,
            "max_cycles": 1,
            "allow_shell": False,
            "anthropic_api_key": None,
            "openai_api_key": None,
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return _make_settings


@pytest.fixture
def bus() -> EventBus:
    """A fresh :class:`furrow.core.events.EventBus`."""
    return EventBus()


@pytest.fixture
def collector(bus: EventBus) -> EventCollector:
    """An :class:`EventCollector` already subscribed to ``bus``."""
    return EventCollector(bus)


@pytest.fixture
def fake_client(make_settings: Callable[..., Settings]) -> Callable[..., FakeLLMClient]:
    """Factory: ``fake_client(settings=None, **attrs) -> FakeLLMClient``.

    Builds a :class:`~furrow.config.Settings` via ``make_settings`` when none is
    given, then applies any ``attrs`` onto the returned client (e.g.
    ``fake_client(text='{"tasks": []}', raises=RuntimeError("boom"))``).
    """

    def _fake_client(settings: Settings | None = None, **attrs: Any) -> FakeLLMClient:
        client = FakeLLMClient(settings if settings is not None else make_settings())
        for name, value in attrs.items():
            setattr(client, name, value)
        return client

    return _fake_client


@pytest.fixture
def agent_env(monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path) -> Path:
    """Guarantee a hermetic environment and return ``tmp_workspace``.

    Clears ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY`` and every ``FURROW_*`` env
    var via ``monkeypatch.delenv(..., raising=False)``. Use this in any test that
    constructs a real :class:`~furrow.config.Settings` or
    :class:`~furrow.llm.LLMClient` so ambient env cannot leak in.
    """
    for name in list(os.environ):
        if name.startswith("FURROW_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_workspace


# Re-exported for convenience so a test can build expectations from one place.
__all__ = [
    "DEFAULT_TEXT",
    "DEFAULT_TOOL_TEXT",
    "CompletionResult",
    "Event",
    "EventBus",
    "EventCollector",
    "FakeLLMClient",
    "RecordedCall",
    "Settings",
    "ToolCall",
    "ToolResult",
    "Usage",
    "agent_env",
    "bus",
    "collector",
    "fake_client",
    "make_settings",
    "tmp_workspace",
]
