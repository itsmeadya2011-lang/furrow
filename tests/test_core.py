from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import FILE_EDIT_PATTERN, WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class FakeLLMClient:
    def __init__(self, responses=None):
        self.settings = Settings(provider=Provider.ANTHROPIC)
        self._responses = responses or {}
        self.writes: list[tuple[str, str]] = []
        self.reads: list[str] = []

    def validate(self):
        pass

    async def complete(self, prompt="", system="", model=None):
        if model in self._responses:
            return self._responses[model]
        return self._responses.get("default", "")

    async def read_file(self, path):
        self.reads.append(str(path))
        raise FileNotFoundError

    async def write_file(self, path, content):
        self.writes.append((str(path), content))

    def list_files(self, directory):
        return []


def _task(task_id: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=f"task {task_id}", status=status)


# ---------------------------------------------------------------------------
# Models & Parsing
# ---------------------------------------------------------------------------
def test_plan_parse_valid():
    data = {
        "tasks": [
            {
                "id": "1",
                "description": "Implement auth",
                "files": ["src/auth.py", "tests/test_auth.py"],
                "dependencies": [],
            },
            {"id": "2", "description": "Add tests", "files": ["tests/test_auth.py"], "dependencies": ["1"]},
        ],
        "rationale": "Split auth and tests for parallel work",
    }
    plan = Plan(**data)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Split auth and tests for parallel work"


def test_plan_parse_missing_required_fields():
    with pytest.raises((ValidationError, ValueError)):
        Plan(rationale="missing tasks")
    with pytest.raises((ValidationError, ValueError)):
        Plan(tasks=[TaskModel(id="1", description="x")])


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_test_result_parse_valid():
    data = json.loads('{"passed": true, "summary": "all good", "failures": []}')
    tr = TestResult(**data)
    assert tr.passed is True
    assert tr.summary == "all good"
    assert tr.failures == []


def test_test_result_defaults():
    tr = TestResult(passed=False, summary="boom")
    assert tr.failures == []


async def test_plan_parse_invalid_json():
    client = FakeLLMClient(responses={"default": "this is not json {"})
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("some goal")


async def test_tester_non_json_fallback():
    response = "the tests overall passed and look good"
    client = FakeLLMClient(responses={"default": response})
    tester = TesterAgent(client=client)
    with patch.object(tester, "_run_tests", new=AsyncMock(return_value="sample test output")):
        result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == response
    assert result.failures == []


# ---------------------------------------------------------------------------
# Orchestrator _is_done() logic
# ---------------------------------------------------------------------------
def _make_orchestrator(max_cycles: int = 0) -> Orchestrator:
    client = FakeLLMClient()
    client.settings = Settings(provider=Provider.ANTHROPIC, max_cycles=max_cycles)
    return Orchestrator(goal="g", client=client)


def test_is_done_no_tasks():
    orch = _make_orchestrator()
    orch._tasks = []
    orch._last_test_passed = None
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = _make_orchestrator()
    orch._tasks = [_task("1", status="completed"), _task("2", status="completed")]
    orch._last_test_passed = True
    assert orch._is_done() is True


def test_is_done_some_pending():
    orch = _make_orchestrator()
    orch._tasks = [_task("1", status="completed"), _task("2", status="pending")]
    orch._last_test_passed = True
    assert orch._is_done() is False


def test_is_done_has_failed():
    orch = _make_orchestrator()
    orch._tasks = [_task("1", status="completed"), _task("2", status="failed")]
    orch._last_test_passed = True
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = _make_orchestrator()
    orch._tasks = [_task("1", status="completed")]
    orch._last_test_passed = False
    assert orch._is_done() is False


def test_is_done_tests_not_run():
    orch = _make_orchestrator()
    orch._tasks = [_task("1", status="completed")]
    orch._last_test_passed = None
    assert orch._is_done() is True


# ---------------------------------------------------------------------------
# Orchestrator cycle logic (with patched agent classes)
# ---------------------------------------------------------------------------
async def test_orchestrator_single_cycle_complete():
    task = TaskModel(id="1", description="t1")
    plan = Plan(tasks=[task], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent") as PlannerMock, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as WorkerMock, patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = AsyncMock(return_value=plan)
        WorkerMock.return_value.run = AsyncMock(return_value="completed t1")
        TesterMock.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orch = Orchestrator(goal="g", client=FakeLLMClient())
        await orch.run()
    assert orch.cycles == 1
    assert orch._tasks[0].status == "completed"
    assert orch._last_test_passed is True


async def test_orchestrator_retries_on_test_failure():
    task = TaskModel(id="1", description="t1")
    plan = Plan(tasks=[task], rationale="ok")
    failed = TestResult(passed=False, summary="fail", failures=["fix me"])
    passed = TestResult(passed=True, summary="ok", failures=[])
    with patch("furrow.core.orchestrator.PlannerAgent") as PlannerMock, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as WorkerMock, patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = AsyncMock(return_value=plan)
        WorkerMock.return_value.run = AsyncMock(return_value="completed t1")
        TesterMock.return_value.run = AsyncMock(side_effect=[failed, passed])
        orch = Orchestrator(goal="g", client=FakeLLMClient())
        await orch.run()
    assert orch.cycles == 2
    assert orch._last_test_passed is True
    assert TesterMock.return_value.run.await_count == 2


async def test_orchestrator_max_cycles_limit():
    task = TaskModel(id="1", description="t1")
    plan = Plan(tasks=[task], rationale="ok")
    failed = TestResult(passed=False, summary="fail", failures=["x"])
    with patch("furrow.core.orchestrator.PlannerAgent") as PlannerMock, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as WorkerMock, patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = AsyncMock(return_value=plan)
        WorkerMock.return_value.run = AsyncMock(return_value="completed t1")
        TesterMock.return_value.run = AsyncMock(return_value=failed)
        client = FakeLLMClient()
        client.settings = Settings(provider=Provider.ANTHROPIC, max_cycles=1)
        orch = Orchestrator(goal="g", client=client)
        await orch.run()
    assert orch.cycles == 1
    assert orch._last_test_passed is False


async def test_orchestrator_task_tracking():
    task = TaskModel(id="1", description="t1")
    plan = Plan(tasks=[task], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent") as PlannerMock, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as WorkerMock, patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = AsyncMock(return_value=plan)
        WorkerMock.return_value.run = AsyncMock(return_value="completed t1")
        TesterMock.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orch = Orchestrator(goal="g", client=FakeLLMClient())
        assert orch._tasks == []
        await orch._cycle()
        assert len(orch._tasks) == 1
        assert orch._tasks[0].id == "1"
        assert orch._tasks[0].status == "completed"


# ---------------------------------------------------------------------------
# WorkerAgent file edit parsing
# ---------------------------------------------------------------------------
def test_worker_file_edit_pattern_single():
    text = "FILE: foo.py\n```python\nprint('hi')\n```"
    m = FILE_EDIT_PATTERN.search(text)
    assert m is not None
    assert m.group("path") == "foo.py"
    assert m.group("lang") == "python"
    assert m.group("content") == "print('hi')\n"


def test_worker_file_edit_pattern_multiple():
    text = (
        "FILE: foo.py\n```python\nprint('a')\n```\n"
        "FILE: bar.js\n```javascript\nconsole.log(1)\n```"
    )
    matches = list(FILE_EDIT_PATTERN.finditer(text))
    assert len(matches) == 2
    assert matches[0].group("path") == "foo.py"
    assert matches[1].group("path") == "bar.js"
    assert matches[0].group("lang") == "python"
    assert matches[1].group("lang") == "javascript"


def test_worker_file_edit_pattern_none():
    text = "No file edits here, just prose."
    matches = list(FILE_EDIT_PATTERN.finditer(text))
    assert matches == []


def test_worker_file_edit_pattern_no_lang():
    text = "FILE: foo.txt\n```\nplain text content\n```"
    m = FILE_EDIT_PATTERN.search(text)
    assert m is not None
    assert m.group("path") == "foo.txt"
    assert m.group("lang") is None
    assert m.group("content") == "plain text content\n"


async def test_worker_writes_files():
    task = TaskModel(id="1", description="write foo", files=["foo.py"])
    content = "FILE: foo.py\n```python\nprint('hi')\n```"
    client = FakeLLMClient(responses={"default": content})
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert len(client.writes) == 1
    assert client.writes[0][0] == "foo.py"
    assert "print('hi')" in client.writes[0][1]
    assert "foo.py" in result


async def test_worker_no_files_to_read():
    task = TaskModel(id="1", description="no files")
    client = FakeLLMClient(responses={"default": "nothing to write"})
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert "No files written" in result
    assert client.writes == []


# ---------------------------------------------------------------------------
# LLMClient validation
# ---------------------------------------------------------------------------
def test_validate_anthropic_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        client.validate()


def test_validate_openai_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    s = Settings(provider=Provider.OPENAI, openai_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        client.validate()


def test_validate_ollama_ok():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    client.validate()


# ---------------------------------------------------------------------------
# TesterAgent test runner
# ---------------------------------------------------------------------------
async def test_run_tests_no_runner_found():
    tester = TesterAgent(client=FakeLLMClient())
    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError),
    ):
        result = await tester._run_tests()
    assert result == "No test runner found."
