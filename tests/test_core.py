from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client for testing that returns scripted responses."""

    def __init__(
        self, settings: Settings | None = None, responses: list[str] | None = None
    ) -> None:
        super().__init__(settings=settings or Settings())
        self.responses = responses or []
        self._call_count = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
            self._call_count += 1
            return response
        return "{}"


def test_plan_parse() -> None:
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_test_result() -> None:
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.failures == []


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_plan_serialization() -> None:
    p = Plan(
        tasks=[TaskModel(id="1", description="task one", files=["a.py"])],
        rationale="test plan",
    )
    data = p.model_dump()
    assert data["rationale"] == "test plan"
    assert data["tasks"][0]["id"] == "1"
    assert data["tasks"][0]["files"] == ["a.py"]


def test_test_result_with_failures() -> None:
    t = TestResult(
        passed=False,
        summary="2 tests failed",
        failures=["test_a.py::test_one", "test_b.py::test_two"],
    )
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_defaults() -> None:
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


@pytest.mark.asyncio
async def test_planner_agent_parses_plan() -> None:
    mock_response = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "Create config", "files": ["config.py"]},
                {"id": "2", "description": "Create tests", "files": ["test_config.py"]},
            ],
            "rationale": "Split into implementation and tests",
        }
    )
    client = MockLLMClient(responses=[mock_response])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Add configuration module")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].files == ["test_config.py"]
    assert plan.rationale == "Split into implementation and tests"


@pytest.mark.asyncio
async def test_planner_agent_handles_invalid_json() -> None:
    client = MockLLMClient(responses=["not valid json"])
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("test goal")


@pytest.mark.asyncio
async def test_tester_agent_parses_passed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_tests(self: TesterAgent) -> str:
        return "All tests passed"

    monkeypatch.setattr(TesterAgent, "_run_tests", mock_run_tests)

    mock_response = json.dumps(
        {
            "passed": True,
            "summary": "All 5 tests passed",
            "failures": [],
        }
    )
    client = MockLLMClient(responses=[mock_response])
    tester = TesterAgent(client=client)
    result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.summary == "All 5 tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_agent_parses_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_tests(self: TesterAgent) -> str:
        return "3 tests failed"

    monkeypatch.setattr(TesterAgent, "_run_tests", mock_run_tests)

    mock_response = json.dumps(
        {
            "passed": False,
            "summary": "3 tests failed",
            "failures": ["test_a.py::test_x", "test_b.py::test_y"],
        }
    )
    client = MockLLMClient(responses=[mock_response])
    tester = TesterAgent(client=client)
    result = await tester.run("fix tests", [])
    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_agent_handles_test_runner_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_tests(self: TesterAgent) -> str:
        raise RuntimeError("No test runner available")

    monkeypatch.setattr(TesterAgent, "_run_tests", mock_run_tests)

    client = MockLLMClient()
    tester = TesterAgent(client=client)
    result = await tester.run("test goal", [])
    assert result.passed is False
    assert "No test runner available" in result.summary


@pytest.mark.asyncio
async def test_worker_agent_runs() -> None:
    mock_response = "Implemented the task successfully"
    client = MockLLMClient(responses=[mock_response])
    task = TaskModel(id="1", description="Add feature X", files=["feature.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Implemented the task successfully"


@pytest.mark.asyncio
async def test_worker_agent_writes_files(tmp_path: Path) -> None:
    code_block = '```python\nFILE: feature.py\nprint("hello")\n```'
    client = MockLLMClient(responses=[code_block])
    task = TaskModel(id="1", description="Create feature", files=["feature.py"])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    assert client._call_count == 1


def test_llm_client_list_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.py").write_text("print('b')")
    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files


def test_llm_client_list_files_nonexistent() -> None:
    client = LLMClient()
    assert client.list_files("/nonexistent/path") == []
