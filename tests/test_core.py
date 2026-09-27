from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import FurrowState

# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_all_fields():
    t = TaskModel(
        id="42",
        description="implement feature",
        files=["src/main.py", "tests/test_main.py"],
        dependencies=["dep-a", "dep-b"],
        status="completed",
        result="done",
    )
    assert t.id == "42"
    assert t.description == "implement feature"
    assert t.files == ["src/main.py", "tests/test_main.py"]
    assert t.dependencies == ["dep-a", "dep-b"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_from_dict():
    data = {
        "tasks": [
            {"id": "1", "description": "task 1", "files": ["a.py"]},
            {"id": "2", "description": "task 2"},
        ],
        "rationale": "breaking it down",
    }
    p = Plan(**data)
    assert len(p.tasks) == 2
    assert p.tasks[0].id == "1"
    assert p.tasks[0].files == ["a.py"]
    assert p.tasks[1].id == "2"
    assert p.tasks[1].dependencies == []
    assert p.rationale == "breaking it down"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.passed is True
    assert t.summary == "all good"
    assert t.failures == []


def test_test_result_default_failures():
    t = TestResult(passed=False, summary="failed")
    assert t.passed is False
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["f1", "f2"])
    assert t.failures == ["f1", "f2"]


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    s = Settings()
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 3


# ---------------------------------------------------------------------------
# FurrowState tests
# ---------------------------------------------------------------------------

def test_state_new_is_empty(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    assert state.goal == ""
    assert state.cycles == 0
    assert state.tasks == []
    assert state.test_results == []
    assert state.status == "running"


def test_state_save_and_load(tmp_path):
    state_path = tmp_path / "state.json"
    state = FurrowState(path=state_path)
    state.set_goal("build a house")
    state.set_status("running")
    state.add_task(TaskModel(id="1", description="foundation", status="completed", result="done"))
    state.add_task(TaskModel(id="2", description="walls"))
    state.add_test_result(TestResult(passed=True, summary="ok", failures=[]))
    state.cycles = 3
    state.save()

    loaded = FurrowState.load(path=state_path)
    assert loaded is not None
    assert loaded.goal == "build a house"
    assert loaded.status == "running"
    assert loaded.cycles == 3
    assert len(loaded.tasks) == 2
    assert loaded.tasks[0].id == "1"
    assert loaded.tasks[0].status == "completed"
    assert loaded.tasks[0].result == "done"
    assert loaded.tasks[1].id == "2"
    assert loaded.tasks[1].status == "pending"
    assert len(loaded.test_results) == 1
    assert loaded.test_results[0].passed is True


def test_state_load_missing_file(tmp_path):
    result = FurrowState.load(path=tmp_path / "does-not-exist.json")
    assert result is None


def test_state_load_corrupt_json(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{ this is not valid json }")
    result = FurrowState.load(path=state_path)
    assert result is None


def test_state_init_corrupt_json(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{ corrupt")
    state = FurrowState(path=state_path)
    assert state.goal == ""
    assert state.tasks == []


def test_state_add_task(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    task = TaskModel(id="1", description="do something")
    state.add_task(task)
    assert len(state.tasks) == 1
    assert state.tasks[0] is task


def test_state_update_task(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    state.add_task(TaskModel(id="1", description="do something"))
    state.update_task(task_id="1", status="completed", result="finished")
    assert state.tasks[0].status == "completed"
    assert state.tasks[0].result == "finished"


def test_state_update_task_not_found(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    state.update_task(task_id="999", status="completed", result="none")
    assert state.tasks == []


def test_state_add_test_result(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    r = TestResult(passed=False, summary="bad", failures=["x"])
    state.add_test_result(r)
    assert len(state.test_results) == 1


def test_state_set_status(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    state.set_status("failed")
    assert state.status == "failed"


def test_state_set_goal(tmp_path):
    state = FurrowState(path=tmp_path / "state.json")
    state.set_goal("new goal")
    assert state.goal == "new goal"


# ---------------------------------------------------------------------------
# Orchestrator _is_done tests
# ---------------------------------------------------------------------------

def _make_orchestrator(max_cycles: int = 0) -> Orchestrator:
    settings = Settings(max_cycles=max_cycles)
    mock_client = MagicMock()
    mock_client.settings = settings
    return Orchestrator(goal="test goal", client=mock_client, enable_state=False)


def test_is_done_all_completed_tests_passed():
    orch = _make_orchestrator()
    orch.current_plan_tasks = [TaskModel(id="1", description="task", status="completed", result="done")]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_pending_tasks():
    orch = _make_orchestrator()
    orch.current_plan_tasks = [TaskModel(id="1", description="task", status="pending")]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = _make_orchestrator()
    orch.current_plan_tasks = [TaskModel(id="1", description="task", status="completed", result="done")]
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["f1"])
    assert orch._is_done() is False


def test_is_done_no_tasks():
    orch = _make_orchestrator()
    orch.current_plan_tasks = []
    orch.last_test_result = None
    assert orch._is_done() is True


def test_is_done_no_test_result():
    orch = _make_orchestrator()
    orch.current_plan_tasks = [TaskModel(id="1", description="task", status="completed")]
    orch.last_test_result = None
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_enforcement():
    settings = Settings(max_cycles=1)
    mock_client = MagicMock()
    mock_client.settings = settings

    plan = Plan(
        tasks=[TaskModel(id="1", description="test task", status="pending")],
        rationale="ok",
    )

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as MockWorker, patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="task result")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["t1"])
        )

        orch = Orchestrator(goal="test goal", client=mock_client, enable_state=False)
        await orch.run()

        assert orch.cycles == 1


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_calls_complete_with_tools_not_complete():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete_with_tools = AsyncMock(return_value="task summary")
    mock_client.complete = AsyncMock(return_value="should not be used")

    task = TaskModel(id="1", description="write code")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run(max_iterations=5)

    assert result == "task summary"
    mock_client.complete_with_tools.assert_awaited_once()
    mock_client.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_max_iterations_passed_through():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete_with_tools = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="write code")
    worker = WorkerAgent(task=task, client=mock_client)
    await worker.run(max_iterations=7)

    call_kwargs = mock_client.complete_with_tools.call_args
    assert call_kwargs.kwargs["max_iterations"] == 7


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tester_json_parsing():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    json_response = '{"passed": true, "summary": "all tests passed", "failures": []}'
    mock_client.complete = AsyncMock(return_value=json_response)

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output")

    result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "all tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_fallback_on_invalid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="Tests failed badly")

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output")

    result = await tester.run("test goal", [])

    assert result.passed is False
    assert result.summary == "Tests failed badly"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_fallback_with_passed_keyword():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="all passed")

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output")

    result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "all passed"
