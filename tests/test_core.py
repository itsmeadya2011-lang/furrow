import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import CompletionResult, ToolCall
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
from furrow.config import Settings


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_all_fields():
    p = Plan(
        tasks=[
            TaskModel(
                id="1",
                description="do thing",
                files=["a.py", "b.py"],
                dependencies=["0"],
                status="completed",
                result="done",
            )
        ],
        rationale="full test",
    )
    assert p.tasks[0].id == "1"
    assert p.tasks[0].description == "do thing"
    assert p.tasks[0].files == ["a.py", "b.py"]
    assert p.tasks[0].dependencies == ["0"]
    assert p.tasks[0].status == "completed"
    assert p.tasks[0].result == "done"
    assert p.rationale == "full test"


def test_orchestrator_task_tracking():
    mock_client = MagicMock()
    orch = Orchestrator(goal="test goal", client=mock_client)

    mock_plan = MagicMock()
    mock_plan.tasks = [TaskModel(id="1", description="task 1")]
    orch.plan = mock_plan

    tasks = orch._get_tasks()
    assert tasks == mock_plan.tasks
    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_planner_json_parsing():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value='{"tasks": [{"id": "1", "description": "task 1"}], "rationale": "ok"}'
    )

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("test goal")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "task 1"
    assert plan.rationale == "ok"


@pytest.mark.asyncio
async def test_worker_tool_usage():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=CompletionResult(
            text="Used tools",
            tool_calls=[
                ToolCall(name="read_file", arguments={"path": "foo.py"}),
                ToolCall(
                    name="write_file",
                    arguments={"path": "bar.py", "content": "hello"},
                ),
            ],
        )
    )
    mock_client.read_file = AsyncMock(return_value="file contents")
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="do work", files=["foo.py", "bar.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    mock_client.read_file.assert_called_once_with("foo.py")
    mock_client.write_file.assert_called_once_with("bar.py", "hello")
    assert result == "Used tools"


@pytest.mark.asyncio
async def test_tester_runner_detection():
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"fake test output\n", b""))

    with patch(
        "asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_subprocess:
        mock_client = MagicMock()
        tester = TesterAgent(client=mock_client)
        result = await tester._run_tests()

        mock_subprocess.assert_called_once_with(
            "pytest",
            "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert "fake test output" in result


@pytest.mark.asyncio
async def test_max_cycles_enforcement():
    mock_plan = MagicMock()
    mock_plan.tasks = [TaskModel(id="1", description="never completes")]
    mock_plan.model_dump.return_value = {
        "tasks": [{"id": "1", "description": "never completes"}],
        "rationale": "test",
    }

    settings = Settings(max_cycles=2)
    mock_client = MagicMock()
    orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
    orch.plan = mock_plan

    with patch.object(
        orch.planner, "plan", new_callable=AsyncMock, return_value=mock_plan
    ), patch(
        "furrow.core.orchestrator.WorkerAgent.run",
        new_callable=AsyncMock,
        side_effect=RuntimeError("task failed"),
    ), patch.object(
        TesterAgent,
        "run",
        new_callable=AsyncMock,
        return_value=TestResult(passed=False, summary="fail", failures=["fail"]),
    ):
        await orch.run()

    assert orch.cycles == 2
