import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def make_task(task_id, description, files=None, dependencies=None, status="pending", result=None):
    return TaskModel(
        id=task_id,
        description=description,
        files=files or [],
        dependencies=dependencies or [],
        status=status,
        result=result,
    )


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    client.settings.planner_model = "planner-model"
    client.settings.worker_model = "worker-model"
    client.settings.tester_model = "tester-model"
    client.complete = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        make_task("1", "task one", status="completed"),
        make_task("2", "task two", status="completed"),
    ]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_one_failed():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        make_task("1", "task one", status="completed"),
        make_task("2", "task two", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_pending_tasks():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        make_task("1", "task one", status="completed"),
        make_task("2", "task two", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_get_tasks_returns_all_tasks():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [make_task("1", "task one")]
    assert len(orch._get_tasks()) == 1


@pytest.mark.asyncio
async def test_execute_plan_no_dependencies_runs_all_in_parallel(mock_client):
    orch = Orchestrator(goal="test", client=mock_client)
    plan = Plan(
        tasks=[
            make_task("1", "task one", files=["a.py"]),
            make_task("2", "task two", files=["b.py"]),
        ],
        rationale="test",
    )
    mock_client.complete.return_value = '{"summary": "done", "operations": [], "issues": []}'

    await orch._execute_plan(plan)

    # Both tasks should have been executed (worker called twice)
    assert mock_client.complete.call_count == 2
    assert len(orch.all_tasks) == 2
    for t in orch.all_tasks:
        assert t.status == "completed"


@pytest.mark.asyncio
async def test_execute_plan_respects_dependencies(mock_client):
    orch = Orchestrator(goal="test", client=mock_client)
    plan = Plan(
        tasks=[
            make_task("1", "task one", dependencies=[]),
            make_task("2", "task two", dependencies=["1"]),
        ],
        rationale="test",
    )
    mock_client.complete.return_value = '{"summary": "done", "operations": [], "issues": []}'

    await orch._execute_plan(plan)

    # Both tasks should complete
    assert mock_client.complete.call_count == 2
    assert all(t.status == "completed" for t in orch.all_tasks)


@pytest.mark.asyncio
async def test_execute_plan_handles_failed_task(mock_client):
    orch = Orchestrator(goal="test", client=mock_client)
    plan = Plan(
        tasks=[
            make_task("1", "task one"),
            make_task("2", "task two", dependencies=["1"]),
        ],
        rationale="test",
    )

    # First call raises exception, second succeeds
    mock_client.complete.side_effect = [
        Exception("boom"),
        '{"summary": "done", "operations": [], "issues": []}',
    ]

    await orch._execute_plan(plan)

    assert orch.all_tasks[0].status == "failed"
    assert orch.all_tasks[0].result == "boom"
    assert orch.all_tasks[1].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_respected(mock_client):
    orch = Orchestrator(goal="test", client=mock_client)
    orch.settings.max_cycles = 2
    mock_client.complete.return_value = '{"tasks": [], "rationale": "done"}'

    await orch.run()

    assert orch.cycles == 2
    assert orch.completed is True


@pytest.mark.asyncio
async def test_orchestrator_empty_plan_marks_complete(mock_client):
    orch = Orchestrator(goal="test", client=mock_client)
    mock_client.complete.return_value = '{"tasks": [], "rationale": "nothing to do"}'

    await orch.run()

    assert orch.completed is True


@pytest.mark.asyncio
async def test_worker_executes_write_operation(mock_client):
    from furrow.agents.worker import WorkerAgent

    task = make_task("1", "create file", files=["test.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    mock_client.complete.return_value = json_dumps = (
        '{"summary": "created test.py", '
        '"operations": [{"op": "write", "path": "test.py", "content": "print(1)"}], '
        '"issues": []}'
    )
    mock_client.write_file = AsyncMock()

    result = await worker.run()

    mock_client.write_file.assert_called_once_with("test.py", "print(1)")
    assert "created test.py" in result
    assert "write test.py" in result


@pytest.mark.asyncio
async def test_worker_executes_read_operation(mock_client):
    from furrow.agents.worker import WorkerAgent

    task = make_task("1", "read file", files=["existing.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    mock_client.complete.return_value = (
        '{"summary": "read file", '
        '"operations": [{"op": "read", "path": "existing.py"}], '
        '"issues": []}'
    )
    mock_client.read_file = AsyncMock(return_value="print('hello')")

    result = await worker.run()

    mock_client.read_file.assert_called_once_with("existing.py")
    assert "read existing.py" in result


@pytest.mark.asyncio
async def test_worker_handles_invalid_json(mock_client):
    from furrow.agents.worker import WorkerAgent

    task = make_task("1", "task")
    worker = WorkerAgent(task=task, client=mock_client)
    mock_client.complete.return_value = "not json at all"

    result = await worker.run()
    assert result == "not json at all"


@pytest.mark.asyncio
async def test_tester_parses_passed_json(mock_client):
    from furrow.agents.tester import TesterAgent

    tester = TesterAgent(client=mock_client)
    mock_client._run_tests = AsyncMock(return_value="1 passed")
    mock_client.complete.return_value = (
        '{"passed": true, "summary": "All tests passed", "failures": []}'
    )

    result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_parses_failed_json(mock_client):
    from furrow.agents.tester import TesterAgent

    tester = TesterAgent(client=mock_client)
    mock_client._run_tests = AsyncMock(return_value="FAILED test_foo.py")
    mock_client.complete.return_value = (
        '{"passed": false, "summary": "Tests failed", "failures": ["test_foo.py: assertion error"]}'
    )

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "test_foo.py" in result.failures[0]


@pytest.mark.asyncio
async def test_tester_handles_invalid_json(mock_client):
    from furrow.agents.tester import TesterAgent

    tester = TesterAgent(client=mock_client)
    mock_client._run_tests = AsyncMock(return_value="all passed")
    mock_client.complete.return_value = "not valid json but says passed"

    result = await tester.run("goal", [])

    assert result.passed is True  # "passed" in response, "failed" not in response


def test_planner_parses_valid_plan(mock_client):
    from furrow.agents.planner import PlannerAgent

    planner = PlannerAgent(client=mock_client)
    mock_client.complete.return_value = (
        '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], '
        '"rationale": "test plan"}'
    )

    plan = asyncio.get_event_loop().run_until_complete(planner.plan("test goal"))

    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "test plan"
