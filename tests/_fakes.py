from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from furrow.config import Provider


class FakeSettings:
    """Minimal stand-in for ``Settings`` used by the agent/clients in tests."""

    provider: Provider = Provider.ANTHROPIC
    model: str = "fake-model"
    planner_model: str = "fake-planner"
    worker_model: str = "fake-worker"
    tester_model: str = "fake-tester"
    max_cycles: int = 0
    max_parallel_tasks: int = 5
    log_level: str = "ERROR"


def make_plan_json(tasks: Sequence[dict] | None = None) -> str:
    if tasks is None:
        tasks = [
            {"id": "1", "description": "do a thing", "files": [], "dependencies": []}
        ]
    return json.dumps({"tasks": [dict(t) for t in tasks], "rationale": "ok"})


def make_test_result(
    passed: bool,
    summary: str = "ok",
    failures: Sequence[str] | None = None,
) -> str:
    return json.dumps(
        {"passed": passed, "summary": summary, "failures": list(failures or [])}
    )


class FakeLLMClient:
    """Duck-typed replacement for ``LLMClient`` that returns canned completions.

    Responses are dispatched by the ``model`` argument to ``complete`` so a
    single client can drive the planner, worker and tester in the same run.
    """

    def __init__(
        self,
        plans: Sequence[str] | None = None,
        workers: Sequence[str] | None = None,
        test_results: Sequence[str] | None = None,
        settings: Any = None,
        worker_exc: BaseException | None = None,
    ) -> None:
        self.settings = settings or FakeSettings()
        self._plans = iter(list(plans or []))
        self._workers = iter(list(workers or []))
        self._tests = iter(list(test_results or []))
        self._worker_exc = worker_exc
        self.complete_calls: list[str | None] = []

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        self.complete_calls.append(model)
        if model == self.settings.planner_model:
            return next(self._plans)
        if model == self.settings.worker_model:
            if self._worker_exc is not None:
                raise self._worker_exc
            return next(self._workers)
        if model == self.settings.tester_model:
            return next(self._tests)
        raise AssertionError(f"unexpected model {model!r} for prompt: {prompt[:80]}")


async def fake_run_tests_ok(_self: Any) -> str:
    """Replacement for ``TesterAgent._run_tests`` that returns success output."""
    return "All tests passed."


async def fake_run_tests_raise(_self: Any) -> str:
    """Replacement for ``TesterAgent._run_tests`` that raises."""
    raise RuntimeError("simulated test runner failure")
