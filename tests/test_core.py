import pytest
from unittest.mock import MagicMock, patch
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    def __init__(self, responses: list[str], settings: Settings):
        self.settings = settings
        self._responses = list(responses)
        self.complete_calls = 0
        self.anthropic = MagicMock()
        self.openai = MagicMock()

    async def complete(self, prompt: str, system: str = "", model=None) -> str:
        self.complete_calls += 1
        if self._responses:
            return self._responses.pop(0)
        return ""

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        pass

    def list_files(self, directory):
        return []


def test_settings_defaults():
    assert Settings().provider == Provider.ANTHROPIC


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_planner_retries_on_bad_json():
    plan_json = '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    fake = FakeLLMClient(["not json", plan_json], Settings())
    plan = await PlannerAgent(client=fake).plan("goal")
    assert isinstance(plan, Plan)
    assert fake.complete_calls == 2


@pytest.mark.asyncio
async def test_tester_retries_on_bad_json():
    testresult_json = '{"passed": true, "summary": "ok", "failures": []}'
    fake = FakeLLMClient(["bad", testresult_json], Settings())
    agent = TesterAgent(client=fake)

    async def mock_run_tests():
        return "ok"

    agent._run_tests = mock_run_tests
    result = await agent.run("goal", [])
    assert result.passed is True
    assert fake.complete_calls == 2


@pytest.mark.asyncio
async def test_orchestrator_done_after_pass():
    plan_json = '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    testresult_json = '{"passed": true, "summary": "ok", "failures": []}'
    settings = Settings(max_cycles=5, max_parallel_tasks=2)
    fake = FakeLLMClient([plan_json, testresult_json], settings)

    orchestrator = Orchestrator("goal", client=fake)

    async def mock_run():
        return "ok"

    with patch.object(WorkerAgent, "run", mock_run):
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator._last_passed is True


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    plan_json = '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    testresult_json = '{"passed": false, "summary": "fail", "failures": ["fail"]}'
    settings = Settings(max_cycles=2, max_parallel_tasks=2)
    fake = FakeLLMClient(
        [plan_json, testresult_json, plan_json, testresult_json], settings
    )

    orchestrator = Orchestrator("goal", client=fake)

    events = []

    async def collect_event(event):
        events.append(event)

    orchestrator.on_event = collect_event

    async def mock_run():
        return "ok"

    with patch.object(WorkerAgent, "run", mock_run):
        await orchestrator.run()

    assert orchestrator.cycles == 2
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1
    assert done_events[0]["reason"] == "max_cycles_reached"


@pytest.mark.asyncio
async def test_orchestrator_emits_events():
    plan_json = '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    testresult_json = '{"passed": true, "summary": "ok", "failures": []}'
    settings = Settings(max_cycles=5, max_parallel_tasks=2)
    fake = FakeLLMClient([plan_json, testresult_json], settings)

    events = []

    async def collect_event(event):
        events.append(event["type"])

    orchestrator = Orchestrator("goal", client=fake, on_event=collect_event)

    async def mock_run():
        return "ok"

    with patch.object(WorkerAgent, "run", mock_run):
        await orchestrator.run()

    expected = {"cycle_start", "plan", "task_started", "task_completed", "test_result", "cycle_end", "done"}
    assert expected.issubset(set(events))
