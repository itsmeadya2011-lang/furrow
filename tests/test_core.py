import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def _settings(**overrides):
    base = {"provider": Provider.ANTHROPIC, "anthropic_api_key": "test-key", "max_parallel_tasks": 2}
    base.update(overrides)
    return Settings(**base)


# --- Orchestrator completion logic -----------------------------------------

def _orchestrator_with(tasks, test_result):
    o = Orchestrator(goal="x")
    o._last_tasks = tasks
    o.last_test_result = test_result
    return o


def test_not_done_without_results():
    assert Orchestrator(goal="x")._is_done() is False


def test_done_when_all_complete_and_tests_pass():
    tasks = [TaskModel(id="1", description="a", status="completed")]
    result = TestResult(passed=True, summary="ok", failures=[])
    assert _orchestrator_with(tasks, result)._is_done() is True


def test_not_done_when_a_task_failed():
    tasks = [TaskModel(id="1", description="a", status="failed")]
    result = TestResult(passed=True, summary="ok", failures=[])
    assert _orchestrator_with(tasks, result)._is_done() is False


def test_not_done_when_tests_fail():
    tasks = [TaskModel(id="1", description="a", status="completed")]
    result = TestResult(passed=False, summary="bad", failures=["bad"])
    assert _orchestrator_with(tasks, result)._is_done() is False


def test_done_when_no_tasks_planned_and_tests_pass():
    result = TestResult(passed=True, summary="ok", failures=[])
    assert _orchestrator_with([], result)._is_done() is True


# --- LLMClient file tool execution -----------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_write_read_list(tmp_path):
    client = LLMClient(settings=_settings(workspace=tmp_path))
    target = tmp_path / "a.txt"
    wrote = await client._execute_tool("write_file", {"path": str(target), "content": "hello"})
    assert "Wrote" in wrote
    read = await client._execute_tool("read_file", {"path": str(target)})
    assert read == "hello"
    listed = await client._execute_tool("list_files", {"directory": str(tmp_path)})
    assert "a.txt" in listed
    unknown = await client._execute_tool("nope", {})
    assert "Unknown tool" in unknown


# --- Worker delegates to tool completion -----------------------------------

class _FakeToolClient:
    def __init__(self, settings):
        self.settings = settings
        self.called_with = None

    async def complete_with_tools(self, prompt, system="", model=None, max_rounds=8):
        self.called_with = (prompt, system, model)
        return "implemented task"


@pytest.mark.asyncio
async def test_worker_delegates_to_tool_completion(tmp_path):
    client = _FakeToolClient(_settings(workspace=tmp_path))
    worker = WorkerAgent(task=TaskModel(id="1", description="do the thing"), client=client)
    result = await worker.run()
    assert result == "implemented task"
    assert client.called_with is not None


# --- Orchestrator full cycle with a fake client ----------------------------

class _FakeCycleClient:
    def __init__(self, settings, *, test_passed=True):
        self.settings = settings
        self.test_passed = test_passed

    async def complete(self, prompt, system="", model=None):
        if "Test output" in prompt:
            return json.dumps(
                {
                    "passed": self.test_passed,
                    "summary": "ok" if self.test_passed else "bad",
                    "failures": [] if self.test_passed else ["a failure"],
                }
            )
        return json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "do the thing", "files": [], "dependencies": []}
                ],
                "rationale": "ok",
            }
        )

    async def complete_with_tools(self, prompt, system="", model=None, max_rounds=8):
        return "implemented"


async def _all_pass_run_tests(self):
    return "all green"


async def _failing_run_tests(self):
    return "still failing"


@pytest.mark.asyncio
async def test_cycle_completes_when_tests_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(TesterAgent, "_run_tests", _all_pass_run_tests)
    client = _FakeCycleClient(_settings(max_cycles=0, max_parallel_tasks=2, workspace=tmp_path))
    orch = Orchestrator(goal="build something", client=client)
    await orch.run()
    assert orch.cycles == 1
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed is True


@pytest.mark.asyncio
async def test_loop_stops_at_max_cycles_when_tests_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(TesterAgent, "_run_tests", _failing_run_tests)
    client = _FakeCycleClient(
        _settings(max_cycles=2, max_parallel_tasks=2, workspace=tmp_path),
        test_passed=False,
    )
    orch = Orchestrator(goal="fix tests", client=client)
    await orch.run()
    assert orch.cycles == 2
    assert orch.last_test_result.passed is False


@pytest.mark.asyncio
async def test_loop_continues_while_tests_fail_without_max_cycles(monkeypatch, tmp_path):
    monkeypatch.setattr(TesterAgent, "_run_tests", _failing_run_tests)
    client = _FakeCycleClient(
        _settings(max_cycles=3, max_parallel_tasks=2, workspace=tmp_path),
        test_passed=False,
    )
    orch = Orchestrator(goal="fix tests", client=client)
    await orch.run()
    assert orch.cycles == 3
