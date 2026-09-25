import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockClient:
    """A mock LLM client that returns predetermined responses for testing."""

    def __init__(self, responses=None, settings=None):
        self.settings = settings or Settings()
        self._responses = list(responses or [])
        self._call_index = 0
        self.write_file = AsyncMock()
        self.read_file = AsyncMock(return_value="file content")
        self.list_files = MagicMock(return_value=[])

    async def complete(self, prompt, system="", model=None):
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
        elif self._responses:
            response = self._responses[-1]
        else:
            response = ""
        self._call_index += 1
        return response


# --------------------------------------------------------------------------- #
# Existing model construction tests (kept)
# --------------------------------------------------------------------------- #

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --------------------------------------------------------------------------- #
# PlannerAgent tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    response = json.dumps({
        "tasks": [
            {"id": "1", "description": "Create auth module", "files": ["src/auth.py"], "dependencies": []}
        ],
        "rationale": "Setup authentication",
    })
    client = MockClient(responses=[response])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Add authentication")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Create auth module"
    assert plan.rationale == "Setup authentication"


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    client = MockClient(responses=["not valid json at all"])
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Do something")


@pytest.mark.asyncio
async def test_planner_handles_empty_tasks():
    response = json.dumps({"tasks": [], "rationale": "Nothing to do"})
    client = MockClient(responses=[response])
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Do nothing")
    assert plan.tasks == []
    assert plan.rationale == "Nothing to do"


# --------------------------------------------------------------------------- #
# WorkerAgent tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_worker_applies_file_changes():
    response = json.dumps({
        "changes": [
            {"file": "test.txt", "content": "hello world"}
        ],
        "summary": "Created test.txt",
    })
    client = MockClient(responses=[response])
    task = TaskModel(id="1", description="Create a file", files=["test.txt"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Created test.txt"
    assert client.write_file.call_count == 1
    written_path = client.write_file.call_args[0][0]
    written_content = client.write_file.call_args[0][1]
    assert written_path.name == "test.txt"
    assert written_content == "hello world"


@pytest.mark.asyncio
async def test_worker_reads_task_files():
    response = json.dumps({"changes": [], "summary": "No changes needed"})
    client = MockClient(responses=[response])
    task = TaskModel(id="1", description="Review code", files=["src/main.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "No changes needed"
    assert client.read_file.call_count == 1
    # write_file should NOT have been called since changes is empty
    assert client.write_file.call_count == 0


@pytest.mark.asyncio
async def test_worker_handles_non_json_response():
    client = MockClient(responses=["I did not return JSON as asked"])
    task = TaskModel(id="1", description="Do something")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "I did not return JSON as asked"
    assert client.write_file.call_count == 0


@pytest.mark.asyncio
async def test_worker_no_changes_returns_summary():
    response = json.dumps({"changes": [], "summary": "Nothing to do here"})
    client = MockClient(responses=[response])
    task = TaskModel(id="1", description="Review", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Nothing to do here"
    assert client.write_file.call_count == 0


# --------------------------------------------------------------------------- #
# TesterAgent tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tester_parses_passed_result():
    response = json.dumps({"passed": True, "summary": "All 5 tests passed", "failures": []})
    client = MockClient(responses=[response])
    tester = TesterAgent(client=client)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="5 passed"):
        result = await tester.run("Do something", [])
    assert result.passed is True
    assert result.summary == "All 5 tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_parses_failed_result():
    response = json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_one failed", "test_two failed"],
    })
    client = MockClient(responses=[response])
    tester = TesterAgent(client=client)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="2 failed"):
        result = await tester.run("Do something", [])
    assert result.passed is False
    assert result.summary == "2 tests failed"
    assert len(result.failures) == 2
    assert "test_one failed" in result.failures


@pytest.mark.asyncio
async def test_tester_handles_non_json_response():
    # The fallback checks if "passed" is in the response text
    client = MockClient(responses=["passed all tests"])
    tester = TesterAgent(client=client)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="OK"):
        result = await tester.run("Do something", [])
    assert result.passed is True  # "passed" is in the response
    assert result.summary == "passed all tests"


@pytest.mark.asyncio
async def test_tester_handles_non_json_response_fail():
    client = MockClient(responses=["tests ran but had errors"])
    tester = TesterAgent(client=client)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="some output"):
        result = await tester.run("Do something", [])
    assert result.passed is False  # "passed" not in response
    assert result.summary == "tests ran but had errors"


@pytest.mark.asyncio
async def test_tester_handles_test_exception():
    client = MockClient(responses=[])
    tester = TesterAgent(client=client)
    with patch.object(
        TesterAgent, "_run_tests",
        new_callable=AsyncMock,
        side_effect=Exception("No test runner found"),
    ):
        result = await tester.run("Do something", [])
    assert result.passed is False
    assert "No test runner" in result.summary
    assert len(result.failures) == 1


# --------------------------------------------------------------------------- #
# Orchestrator tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_orchestrator_stores_tasks_and_completes():
    """Verify the halt bug is fixed: tasks are stored and completion detected from test results."""
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "Create test.txt", "files": ["test.txt"], "dependencies": []}],
        "rationale": "Create a test file",
    })
    worker_json = json.dumps({"changes": [{"file": "test.txt", "content": "hello"}], "summary": "Created test.txt"})
    tester_json = json.dumps({"passed": True, "summary": "All tests passed", "failures": []})

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    client = MockClient(responses=[plan_json, worker_json, tester_json])
    orch = Orchestrator(goal="Create a test file", client=client, event_callback=capture_event)

    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="1 passed"):
        await orch.run()

    # Tasks are stored
    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].status == "completed"
    assert orch.all_tasks[0].result == "Created test.txt"

    # Test result is stored
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed is True

    # Orchestrator halted
    assert orch.done is True
    assert orch.cycles == 1

    # Events were emitted
    event_types = [e["type"] for e in captured_events]
    assert "goal" in event_types
    assert "cycle_start" in event_types
    assert "plan" in event_types
    assert "task_completed" in event_types
    assert "test_result" in event_types
    assert "done" in event_types


@pytest.mark.asyncio
async def test_orchestrator_continues_when_tests_fail():
    """When tests fail, the orchestrator should continue to the next cycle."""
    plan_1 = json.dumps({"tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": []}], "rationale": "First attempt"})
    worker_1 = json.dumps({"changes": [], "summary": "Done"})
    tester_1 = json.dumps({"passed": False, "summary": "Failing", "failures": ["AssertionError in test_foo"]})
    plan_2 = json.dumps({"tasks": [{"id": "1", "description": "Fix test_foo", "files": [], "dependencies": []}], "rationale": "Fix failing test"})
    tester_2 = json.dumps({"passed": True, "summary": "All tests passed", "failures": []})

    # Responses: cycle 1 needs plan + worker + tester, cycle 2 needs plan + worker + tester
    client = MockClient(responses=[plan_1, worker_1, tester_1, plan_2, worker_1, tester_2])
    orch = Orchestrator(goal="Fix the bug", client=client)

    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="FAIL"):
        await orch.run()

    assert orch.done is True
    assert orch.last_test_result.passed is True
    assert orch.cycles == 2
    assert len(orch.all_tasks) == 2


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    """When max_cycles is set and reached before tests pass, the orchestrator stops."""
    settings = Settings(max_cycles=1)
    plan_json = json.dumps({"tasks": [{"id": "1", "description": "Task", "files": [], "dependencies": []}], "rationale": "test"})
    worker_json = json.dumps({"changes": [], "summary": "Done"})
    tester_json = json.dumps({"passed": False, "summary": "Failing", "failures": ["error"]})

    client = MockClient(responses=[plan_json, worker_json, tester_json], settings=settings)
    orch = Orchestrator(goal="Test", client=client)

    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="FAIL"):
        await orch.run()

    assert orch.cycles == 1  # Only 1 cycle ran
    assert orch.done is False  # Did not complete


@pytest.mark.asyncio
async def test_orchestrator_stops_when_no_tasks():
    """When planner returns no tasks, the orchestrator should stop immediately."""
    plan_json = json.dumps({"tasks": [], "rationale": "Goal is already complete"})

    client = MockClient(responses=[plan_json])
    orch = Orchestrator(goal="Do nothing", client=client)
    await orch.run()

    assert orch.done is True
    assert orch.cycles == 1
    assert len(orch.all_tasks) == 0
