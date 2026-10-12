from __future__ import annotations

from typing import Any

import pytest

from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _FakeClient:
    """Minimal LLMClient stand-in for orchestrator tests."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.planner_calls = 0
        self.tester_result: TestResult | None = None

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.planner_calls += 1
        if self.planner_calls == 1:
            return (
                '{"tasks": [{"id": "1", "description": "do thing", "files": ["a.py"]}], '
                '"rationale": "first slice"}'
            )
        if self.planner_calls == 2:
            return '{"tasks": [], "rationale": "done"}'
        return '{"tasks": [], "rationale": ""}'


def _make_orchestrator(client: _FakeClient) -> Orchestrator:
    return Orchestrator(goal="test", client=client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_is_done_false_when_no_tasks() -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_true_when_all_completed() -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)
    orch._tasks = [TaskModel(id="1", description="x", status="completed")]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_false_when_any_failed() -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)
    orch._tasks = [
        TaskModel(id="1", description="x", status="completed"),
        TaskModel(id="2", description="y", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_false_when_pending_remaining() -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)
    orch._tasks = [
        TaskModel(id="1", description="x", status="completed"),
        TaskModel(id="2", description="y", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_cycle_merges_planned_tasks_into_persisted_list(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)

    class _FakeTester:
        def __init__(self, client: Any) -> None:
            pass

        async def run(self, goal: str, tasks: list[Any]) -> TestResult:
            return TestResult(passed=True, summary="ok", failures=[])

    import furrow.core.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "TesterAgent", _FakeTester)
    await orch._cycle()
    assert len(orch._tasks) == 1
    assert orch._tasks[0].id == "1"
    assert orch._tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_max_cycles_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(max_cycles=2)
    client = _FakeClient(settings=settings)
    orch = _make_orchestrator(client)

    class _FakeTester:
        def __init__(self, client: Any) -> None:
            pass

        async def run(self, goal: str, tasks: list[Any]) -> TestResult:
            return TestResult(passed=True, summary="ok", failures=[])

    import furrow.core.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "TesterAgent", _FakeTester)
    max_cycles = client.settings.max_cycles
    assert max_cycles == 2
    await orch._cycle()
    assert orch.cycles == 1
    await orch._cycle()
    assert orch.cycles == 2
    assert orch.cycles >= max_cycles


@pytest.mark.asyncio
async def test_goal_updated_on_test_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    client.tester_result = TestResult(passed=False, summary="boom", failures=["fix me"])
    orch = _make_orchestrator(client)

    class _FakeTester:
        def __init__(self, client: Any) -> None:
            pass

        async def run(self, goal: str, tasks: list[Any]) -> TestResult:
            return client.tester_result  # type: ignore[attr-defined]

    import furrow.core.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "TesterAgent", _FakeTester)
    await orch._cycle()
    assert "Fix failing tests" in orch.goal
    assert "fix me" in orch.goal


def test_get_tasks_returns_persisted_list() -> None:
    client = _FakeClient()
    orch = _make_orchestrator(client)
    tasks = [TaskModel(id="1", description="x")]
    orch._tasks = tasks
    assert orch._get_tasks() is tasks