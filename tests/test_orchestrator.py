from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "planning agent" in prompt.lower():
            return '{"tasks": [{"id": "1", "description": "Task 1"}, {"id": "2", "description": "Task 2"}], "rationale": "Test plan"}'
        if "tester agent" in prompt.lower():
            return '{"passed": true, "summary": "ok", "failures": []}'
        return "worker done"


class FailingWorkerLLMClient(FakeLLMClient):
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "planning agent" in prompt.lower():
            return '{"tasks": [{"id": "1", "description": "Task 1"}], "rationale": "Test plan"}'
        if "tester agent" in prompt.lower():
            return '{"passed": true, "summary": "ok", "failures": []}'
        if "worker" in prompt.lower() or "task" in prompt.lower():
            raise RuntimeError("Worker failed")
        return super().complete(prompt, system, model)


class MaxCyclesLLMClient(FakeLLMClient):
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "planning agent" in prompt.lower():
            return '{"tasks": [{"id": "1", "description": "Task 1"}], "rationale": "Test plan"}'
        if "tester agent" in prompt.lower():
            return '{"passed": false, "summary": "bad", "failures": ["f1"]}'
        return "worker done"


class NoTasksLLMClient(FakeLLMClient):
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "planning agent" in prompt.lower():
            return '{"tasks": [], "rationale": "nothing left to do"}'
        return await super().complete(prompt, system, model=model)


@pytest.fixture
def fake_settings(tmp_path) -> Settings:
    return Settings(
        provider=Provider.OLLAMA,
        workspace=tmp_path,
        max_parallel_tasks=3,
        max_cycles=0,
        model="test-model",
        planner_model="test-planner",
        worker_model="test-worker",
        tester_model="test-tester",
    )


@pytest.mark.asyncio
async def test_orchestrator_emits_events_in_order(fake_settings):
    fake_client = FakeLLMClient(fake_settings)
    events = []

    def collector(event):
        events.append(event)

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=collector)

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "worker done"
        await orchestrator.run()

    event_types = [e["type"] for e in events]
    expected = [
        "cycle",
        "plan",
        "task_start",
        "task_start",
        "task_done",
        "task_done",
        "test",
        "done",
    ]
    assert event_types == expected


@pytest.mark.asyncio
async def test_orchestrator_cycles_and_task_statuses(fake_settings):
    fake_client = FakeLLMClient(fake_settings)
    events = []

    def collector(event):
        events.append(event)

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=collector)

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "worker done"
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert all(task.status == "completed" for task in orchestrator._tasks)


@pytest.mark.asyncio
async def test_orchestrator_max_cycles(fake_settings):
    fake_settings.max_cycles = 1
    fake_client = MaxCyclesLLMClient(fake_settings)
    events = []

    def collector(event):
        events.append(event)

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=collector)

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "worker done"
        await orchestrator.run()

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["reason"] == "max_cycles reached"
    assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_failing_worker(fake_settings):
    fake_client = FailingWorkerLLMClient(fake_settings)
    events = []

    def collector(event):
        events.append(event)

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=collector)

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = RuntimeError("Worker failed")
        await orchestrator.run()

    task_done_events = [e for e in events if e["type"] == "task_done"]
    assert len(task_done_events) == 1
    assert task_done_events[0]["status"] == "failed"
    assert "Worker failed" in task_done_events[0]["result"]


@pytest.mark.asyncio
async def test_orchestrator_halts_when_no_tasks_planned(fake_settings):
    fake_client = NoTasksLLMClient(fake_settings)
    events = []

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=events.append)

    await orchestrator.run()

    assert orchestrator.cycles == 1
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_orchestrator_monkeypatch_tester_run(fake_settings, monkeypatch):
    fake_client = FakeLLMClient(fake_settings)
    events = []

    def collector(event):
        events.append(event)

    async def fake_run_tests(self, goal, tasks):
        return TestResult(passed=True, summary="fake test output", failures=[])

    monkeypatch.setattr("furrow.agents.tester.TesterAgent.run", fake_run_tests)

    orchestrator = Orchestrator(goal="test goal", client=fake_client, on_event=collector)

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "worker done"
        await orchestrator.run()

    test_events = [e for e in events if e["type"] == "test"]
    assert len(test_events) == 1
    assert test_events[0]["summary"] == "fake test output"


def test_plan_rationale_optional():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")])
    assert plan.rationale == ""