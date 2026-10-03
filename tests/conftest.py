"""Shared, fully offline fixtures for the furrow test-suite.

Everything here is deliberately generic so that any test module can use it:

* :func:`tmp_settings` -- a factory for per-test ``Settings`` **copies**. The
  global ``furrow.config.settings`` singleton is never mutated by a test.
* :func:`fake_llm_client` -- a factory for :class:`FakeLLMClient`, an
  ``LLMClient`` subclass that answers from canned JSON/text. It needs no API
  key and opens no socket, so no test can accidentally reach a provider.
* :func:`event_recorder` -- a list plus an ``on_event`` callback that appends to
  it, for wiring ``Orchestrator.on_event``.

Test-module-specific helpers (planner payloads, scripted tester doubles, worker
probes, ...) belong in the individual test modules rather than here.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Settings
from furrow.config import settings as global_settings
from furrow.core.events import Event
from furrow.llm import LLMClient

#: The repository listing the offline client reports instead of scanning disk.
REPO_FILES: list[str] = ["README.md", "pyproject.toml", "furrow/config.py", "furrow/llm.py"]

#: A hook receives the prompt that was sent and may await or raise.
PromptHook = Callable[[str], Awaitable[None] | None]


class FakeLLMClient(LLMClient):
    """An ``LLMClient`` double that answers from canned data and never hits the network.

    ``complete`` records the prompt, runs the optional ``hook`` (which is how a
    test injects worker failures, delays or call accounting) and returns ``text``.
    ``complete_json`` replays ``json_payload`` / ``json_payloads`` -- the last
    entry repeating once the list is exhausted -- as a fresh deep copy so callers
    cannot mutate the canned data.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        json_payload: Any = None,
        json_payloads: Sequence[Any] | None = None,
        text: str = "ok",
        hook: PromptHook | None = None,
    ) -> None:
        super().__init__(settings=settings)
        payloads = list(json_payloads) if json_payloads else [json_payload]
        self._payloads: list[Any] = payloads or [{}]
        self.text = text
        self.hook = hook
        self.prompts: list[str] = []
        self.models: list[str | None] = []
        self.json_calls = 0
        self.text_calls = 0
        self.list_files_calls = 0

    def _next_payload(self) -> Any:
        payload = self._payloads[min(self.json_calls, len(self._payloads) - 1)]
        self.json_calls += 1
        return json.loads(json.dumps(payload))

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.text_calls += 1
        self.prompts.append(prompt)
        self.models.append(model)
        if self.hook is not None:
            outcome = self.hook(prompt)
            if inspect.isawaitable(outcome):
                await outcome
        return self.text

    async def complete_json(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        self.models.append(model)
        return self._next_payload()

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        """Report a fixed file list instead of walking the filesystem."""
        self.list_files_calls += 1
        return list(REPO_FILES)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Return a factory building per-test ``Settings`` copies.

    Copies are derived from the global singleton with ``model_copy`` so nothing
    a test does can leak into ``furrow.config.settings``. Test-safe defaults
    (short timeouts, no retries, no API keys) are applied and can be overridden.
    """

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "workspace": tmp_path,
            "max_parallel_tasks": 5,
            "max_cycles": 0,
            "request_timeout": 1.0,
            "llm_max_retries": 1,
            "anthropic_api_key": None,
            "openai_api_key": None,
        }
        base.update(overrides)
        return global_settings.model_copy(update=base)

    return _make


@pytest.fixture
def fake_llm_client() -> Callable[..., FakeLLMClient]:
    """Return a factory building :class:`FakeLLMClient` doubles."""

    def _make(**kwargs: Any) -> FakeLLMClient:
        return FakeLLMClient(**kwargs)

    return _make


@pytest.fixture
def event_recorder() -> tuple[list[Event], Callable[[Event], None]]:
    """Return ``(events, callback)``: a list and an ``on_event`` callback filling it."""
    events: list[Event] = []

    def _record(event: Event) -> None:
        events.append(event)

    return events, _record
