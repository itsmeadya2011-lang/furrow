"""Shared fixtures for the furrow test suite.

CONTRACT: this file is owned by the test-suite agent and is consumed, unmodified, by
``test_core.py``, ``test_config.py``, ``test_events.py``, ``test_llm.py``,
``test_orchestrator.py`` and by the next wave's ``test_agents.py``, ``test_web.py`` and
``test_cli.py``. Ask for a new fixture instead of editing this file.

Public names other test modules may rely on:

* :class:`ScriptedLLMClient` - a no-network ``LLMClient`` double.
* :func:`as_llm_client` - type-erase a :class:`ScriptedLLMClient` to ``LLMClient`` for
  constructors that are annotated with the real class (agents, ``Orchestrator``).
* fixtures ``workspace``, ``settings``, ``bus``, ``fake_client``, ``make_client``.

Isolation rules baked in here: no network, no real provider client, nothing written
inside the repository. Every fixture that can run a command or touch a file is scoped to
pytest's ``tmp_path``, and ``Settings`` is always built through ``reload_settings`` so the
process-wide singleton is never mutated.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT
from furrow.config import Settings, TaskModel, reload_settings
from furrow.events import EventBus
from furrow.llm import LLMClient, ToolCall, ToolResult

#: Canned planner payload used by the ``fake_client`` fixture and as the fallback plan.
DEFAULT_PLAN: dict[str, Any] = {
    "tasks": [
        {
            "id": "1",
            "description": "Add a greeting module",
            "files": ["greeting.py"],
            "dependencies": [],
        }
    ],
    "rationale": "One small, self-contained task.",
}

#: Canned tester payload: the ground truth still gets the final word (see TesterAgent).
DEFAULT_TEST: dict[str, Any] = {
    "passed": True,
    "summary": "Scripted verification passed.",
    "failures": [],
}

# Fixed per-call token cost, so `usage` is deterministic and assertable.
INPUT_TOKENS_PER_CALL = 11
OUTPUT_TOKENS_PER_CALL = 7

# Environment variables that would otherwise leak a real provider key into a test.
_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def _render(payload: Any) -> str:
    """Return the raw text a model would have produced for `payload`."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


class ScriptedLLMClient:
    """A no-network LLMClient double.

    Doubles as the `settings` provider the agents read from, exactly like the real
    LLMClient does, so it can be passed straight to agents and the Orchestrator.

    `plan` / `test` accept a mapping, a list or raw text; raw text is returned verbatim so
    a test can inject malformed JSON. `queue_plan` / `queue_test` append EXTRA payloads that
    are served after the constructor default, in order, and the last one then repeats. `error`
    is raised by every entrypoint (and counted in `error_calls`).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        plan: dict[str, Any] | str | list[Any] | None = None,
        test: dict[str, Any] | str | None = None,
        tool_rounds: list[list[ToolCall]] | None = None,
        error: BaseException | None = None,
        default_tasks: list[TaskModel] | None = None,
        text: str = "",
    ) -> None:
        self.settings = settings
        self.default_tasks = list(default_tasks) if default_tasks else []
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.error_calls = 0
        self._plans: list[Any] = [plan] if plan is not None else []
        self._tests: list[Any] = [test] if test is not None else []
        self._last_plan: Any = None
        self._last_test: Any = None
        self._text = text
        self._tool_rounds = list(tool_rounds) if tool_rounds else []
        self._tool_round = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    # ------------------------------------------------------------------ queues
    def queue_plan(self, plan: dict[str, Any] | str | list[Any]) -> None:
        """Append one more planner payload; served after the current one, in order."""
        self._plans.append(plan)

    def queue_test(self, test: dict[str, Any] | str) -> None:
        """Append one more tester payload; served after the current one, in order."""
        self._tests.append(test)

    @property
    def usage(self) -> dict[str, int]:
        """Cumulative counters, shaped exactly like ``LLMClient.usage``."""
        return dict(self._usage)

    def reset_usage(self) -> None:
        """Zero the cumulative counters, like ``LLMClient.reset_usage``."""
        self._usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    # ------------------------------------------------------------ LLM surface
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the canned payload for this role as raw text, else ``text``."""
        self._record(
            "complete",
            prompt,
            system,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._fail()
        payload = self._payload(system, prompt)
        return self._text if payload is None else _render(payload)

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        """Return the canned payload for this role as a Python value.

        A raw string is returned unparsed, which is how a test feeds malformed JSON.
        """
        self._record("complete_json", prompt, system, model, max_attempts=max_attempts)
        self._fail()
        payload = self._payload(system, prompt)
        if payload is None:
            return self._text
        return payload if isinstance(payload, (dict, list)) else _render(payload)

    async def complete_with_tools(
        self,
        prompt: str,
        tools: Sequence[Any],
        *,
        system: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
    ) -> ToolResult:
        """Emit the next scripted tool round, mirroring the real hand-off contract.

        An empty round ends the conversation (``stop_reason="end_turn"``); a non-empty one
        reports ``"max_iterations"`` because this double never feeds results back.
        """
        self._record(
            "complete_with_tools",
            prompt,
            system,
            model,
            tools=[tool["name"] for tool in tools],
            max_iterations=max_iterations,
        )
        self._fail()
        # Yield to the loop while a slot is held: without this, concurrent workers would
        # never interleave and a peak-concurrency assertion could not see a regression.
        await asyncio.sleep(0)
        rounds = self._tool_rounds
        requested = list(rounds[min(self._tool_round, len(rounds) - 1)]) if rounds else []
        self._tool_round += 1
        log = [
            {"name": call.name, "arguments": dict(call.arguments), "result": ""}
            for call in requested
        ]
        for call in requested:
            if on_tool_call is not None:
                await on_tool_call(call)
        return ToolResult(
            text=self._text,
            tool_calls=log,
            iterations=1,
            stop_reason="end_turn" if not requested else "max_iterations",
        )

    # --------------------------------------------------------------- internals
    def _payload(self, system: str | None, prompt: str) -> Any:
        """Return the canned payload for the role named by `system` (or `prompt`)."""
        haystack = f"{system or ''}\n{prompt}"
        if PLANNER_PROMPT in haystack:
            return self._plan_value()
        if TESTER_PROMPT in haystack:
            return self._test_value()
        return None

    def _plan_value(self) -> Any:
        if self._plans:
            self._last_plan = self._plans.pop(0)
        if self._last_plan is not None:
            return self._last_plan
        return {
            "tasks": [task.model_dump() for task in self.default_tasks],
            "rationale": "Derived from default_tasks."
            if self.default_tasks
            else "Nothing to do.",
        }

    def _test_value(self) -> Any:
        if self._tests:
            self._last_test = self._tests.pop(0)
        return DEFAULT_TEST if self._last_test is None else self._last_test

    def _record(
        self,
        method: str,
        prompt: str,
        system: str | None,
        model: str | None,
        **extra: Any,
    ) -> None:
        self.calls.append(
            {"method": method, "prompt": prompt, "system": system, "model": model, **extra}
        )
        self._usage["input_tokens"] += INPUT_TOKENS_PER_CALL
        self._usage["output_tokens"] += OUTPUT_TOKENS_PER_CALL
        self._usage["requests"] += 1

    def _fail(self) -> None:
        if self.error is not None:
            self.error_calls += 1
            raise self.error


def as_llm_client(client: ScriptedLLMClient) -> LLMClient:
    """Type-erase `client` where a constructor demands the real ``LLMClient``."""
    return cast("LLMClient", client)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited FURROW_*/provider variables so config reads are deterministic."""
    for name in list(os.environ):
        if name.startswith("FURROW_") or name in _PROVIDER_ENV_VARS:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty absolute directory that is the only place a test may write."""
    return tmp_path.resolve()


@pytest.fixture
def settings(workspace: Path) -> Settings:
    """Test settings: tmp workspace, short bounds, no state file."""
    return reload_settings(
        workspace=workspace,
        max_cycles=2,
        max_stalled_cycles=2,
        test_timeout=60,
        state_file=None,
    )


@pytest.fixture
def bus(settings: Settings) -> EventBus:
    """An event bus sized from the settings under test."""
    return EventBus(history=settings.max_event_history)


@pytest.fixture
def fake_client(settings: Settings) -> ScriptedLLMClient:
    """A scripted client with the default one-task plan and a passing verdict."""
    return ScriptedLLMClient(settings, plan=DEFAULT_PLAN, test=DEFAULT_TEST)


@pytest.fixture
def make_client(settings: Settings) -> Callable[..., ScriptedLLMClient]:
    """Factory building scripted clients that all share the `settings` fixture."""

    def factory(**kwargs: Any) -> ScriptedLLMClient:
        return ScriptedLLMClient(settings, **kwargs)

    return factory
