import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.tasks[0].status == "pending"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="desc")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


# ---------------------------------------------------------------------------
# Planner agent tests
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


@pytest.mark.asyncio
async def test_planner_returns_valid_plan():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "tasks": [
                {"id": "1", "description": "Implement foo", "files": ["foo.py"], "dependencies": []}
            ],
            "rationale": "Single task",
        })
    )
    fake_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=fake_client)
    plan = await planner.plan("Implement foo")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Implement foo"
    assert plan.tasks[0].files == ["foo.py"]
    assert plan.rationale == "Single task"
    fake_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_planner_includes_previous_task_context():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "tasks": [{"id": "2", "description": "Implement bar", "files": ["bar.py"], "dependencies": []}],
            "rationale": "Next task",
        })
    )
    fake_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=fake_client)
    previous = [
        TaskModel(id="1", description="Implement foo", files=["foo.py"], status="completed", result="done"),
        TaskModel(id="2", description="Implement baz", files=["baz.py"], status="failed", result="timeout"),
    ]
    await planner.plan("Implement bar", previous_tasks=previous)

    call_args = fake_client.complete.call_args
    prompt = call_args[0][0]
    assert "Previously completed tasks" in prompt
    assert "Previously failed or blocked tasks" in prompt
    assert "Implement foo" in prompt
    assert "Implement baz" in prompt


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(return_value="not json at all")
    fake_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=fake_client)
    with pytest.raises(ValueError):
        await planner.plan("Do something")


# ---------------------------------------------------------------------------
# Worker agent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_writes_files():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "files": [
                {"path": "src/foo.py", "content": "print('hello')\n"},
                {"path": "tests/test_foo.py", "content": "def test_foo(): assert True\n"},
            ],
            "summary": "Created foo module with tests",
        })
    )
    fake_client.settings.worker_model = "test-model"
    fake_client.read_file = AsyncMock(side_effect=FileNotFoundError("no such file"))
    fake_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Create foo module", files=["src/foo.py", "tests/test_foo.py"])
    worker = WorkerAgent(task=task, client=fake_client)
    result = await worker.run()

    assert "Created foo module" in result
    fake_client.write_file.assert_any_call("src/foo.py", "print('hello')\n")
    fake_client.write_file.assert_any_call("tests/test_foo.py", "def test_foo(): assert True\n")


@pytest.mark.asyncio
async def test_worker_reads_existing_files_before_writing():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "files": [{"path": "src/foo.py", "content": "print('updated')\n"}],
            "summary": "Updated foo",
        })
    )
    fake_client.settings.worker_model = "test-model"
    fake_client.read_file = AsyncMock(return_value="print('original')\n")
    fake_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Update foo", files=["src/foo.py"])
    worker = WorkerAgent(task=task, client=fake_client)
    await worker.run()

    # Verify read_file was called before write_file
    assert fake_client.read_file.call_count >= 1
    assert fake_client.write_file.call_count >= 1
    # read_file should be called before write_file
    read_call = fake_client.read_file.call_args_list[0]
    write_call = fake_client.write_file.call_args_list[0]
    assert read_call[0][0] == "src/foo.py"


@pytest.mark.asyncio
async def test_worker_falls_back_to_raw_response_on_bad_json():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(return_value="The worker did some work.")
    fake_client.settings.worker_model = "test-model"
    fake_client.read_file = AsyncMock(return_value="")
    fake_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=fake_client)
    result = await worker.run()

    assert result == "The worker did some work."


# ---------------------------------------------------------------------------
# Tester agent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tester_passes_when_all_checks_pass():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "passed": True,
            "summary": "All checks passed",
            "failures": [],
        })
    )
    fake_client.settings.tester_model = "test-model"

    with patch.object(TesterAgent, "_try_commands", new_callable=AsyncMock) as mock_try:
        mock_try.side_effect = [
            "5 passed in 0.1s\n",   # test runner
            "",                      # lint (clean)
            "",                      # type check (clean)
        ]
        tester = TesterAgent(client=fake_client)
        result = await tester.run("Goal", [])

    assert result.passed is True
    assert "All checks passed" in result.summary


@pytest.mark.asyncio
async def test_tester_fails_when_checks_fail():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "passed": False,
            "summary": "Some tests failed",
            "failures": ["test_foo.py::test_bar - assertion error"],
        })
    )
    fake_client.settings.tester_model = "test-model"

    with patch.object(TesterAgent, "_try_commands", new_callable=AsyncMock) as mock_try:
        mock_try.side_effect = [
            "FAILED test_foo.py::test_bar\n",   # test runner
            "",                                  # lint
            "",                                  # type
        ]
        tester = TesterAgent(client=fake_client)
        result = await tester.run("Goal", [])

    assert result.passed is False
    assert len(result.failures) == 1


@pytest.mark.asyncio
async def test_tester_handles_no_runner_found():
    fake_client = MagicMock(spec=LLMClient)
    fake_client.complete = AsyncMock(
        return_value=json.dumps({
            "passed": True,
            "summary": "No tests found",
            "failures": [],
        })
    )
    fake_client.settings.tester_model = "test-model"

    with patch.object(TesterAgent, "_try_commands", new_callable=AsyncMock) as mock_try:
        mock_try.side_effect = ["No runner found.", "No runner found.", "No runner found."]
        tester = TesterAgent(client=fake_client)
        result = await tester.run("Goal", [])

    assert result.passed is True


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_tracks_tasks_across_cycle():
    """Tasks planned in a cycle should persist in all_tasks."""
    fake_client = MagicMock(spec=LLMClient)

    # Planner returns one task, then no tasks on second call
    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(side_effect=[
        Plan(tasks=[TaskModel(id="1", description="Task one")], rationale="First"),
        Plan(tasks=[], rationale="Done"),
    ])

    worker = MagicMock(spec=WorkerAgent)
    worker.run = AsyncMock(return_value="done")

    tester = MagicMock(spec=TesterAgent)
    tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    emitted = []
    orch = Orchestrator(goal="Test goal", client=fake_client, emit=lambda m: emitted.append(m))
    orch.planner = planner
    # Patch WorkerAgent constructor to return our mock
    with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker), \
         patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
        await orch.run()

    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].id == "1"
    assert orch.all_tasks[0].status == "completed"
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_stops_after_max_cycles():
    fake_client = MagicMock(spec=LLMClient)
    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(tasks=[TaskModel(id="1", description="Task")], rationale="r"))
    worker = MagicMock(spec=WorkerAgent)
    worker.run = AsyncMock(return_value="done")
    tester = MagicMock(spec=TesterAgent)
    tester.run = AsyncMock(return_value=TestResult(passed=False, summary="fail", failures=["x"]))

    orch = Orchestrator(goal="Test goal", client=fake_client)
    orch.settings.max_cycles = 1
    orch.planner = planner
    with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker), \
         patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
        await orch.run()

    assert orch.cycles == 1  # Stopped by max_cycles, not by completion


@pytest.mark.asyncio
async def test_orchestrator_respects_dependencies():
    """Tasks with uncompleted dependencies should not run before their deps."""
    fake_client = MagicMock(spec=LLMClient)
    execution_order = []

    async def fake_worker_run(self):
        execution_order.append(self.task.id)
        return f"result for {self.task.id}"

    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(
        tasks=[
            TaskModel(id="1", description="First", dependencies=[]),
            TaskModel(id="2", description="Second", dependencies=["1"]),
            TaskModel(id="3", description="Third", dependencies=["1"]),
        ],
        rationale="Deps",
    ))

    with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls:
        mock_worker_cls.return_value.run = fake_worker_run
        tester = MagicMock(spec=TesterAgent)
        tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

        orch = Orchestrator(goal="Test goal", client=fake_client)
        orch.planner = planner
        with patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
            await orch.run()

    # Task 1 must complete before tasks 2 and 3 start
    assert execution_order[0] == "1"
    assert "2" in execution_order and "3" in execution_order
    assert execution_order.index("1") < execution_order.index("2")
    assert execution_order.index("1") < execution_order.index("3")


@pytest.mark.asyncio
async def test_orchestrator_emits_messages():
    fake_client = MagicMock(spec=LLMClient)
    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="empty"))
    tester = MagicMock(spec=TesterAgent)
    tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    emitted = []
    orch = Orchestrator(goal="Test goal", client=fake_client, emit=lambda m: emitted.append(m))
    orch.planner = planner
    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
        await orch.run()

    assert len(emitted) > 0
    assert any("Goal complete" in m or "No tasks planned" in m for m in emitted)


@pytest.mark.asyncio
async def test_orchestrator_rewrites_goal_on_test_failure():
    fake_client = MagicMock(spec=LLMClient)
    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(side_effect=[
        Plan(tasks=[TaskModel(id="1", description="Task")], rationale="First"),
        Plan(tasks=[], rationale="Done after fix"),
    ])
    worker = MagicMock(spec=WorkerAgent)
    worker.run = AsyncMock(return_value="done")
    tester = MagicMock(spec=TesterAgent)
    tester.run = AsyncMock(return_value=TestResult(
        passed=False, summary="failures", failures=["error in foo.py", "error in bar.py"]
    ))

    orch = Orchestrator(goal="Original goal", client=fake_client)
    orch.planner = planner
    with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker), \
         patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
        await orch.run()

    assert "error in foo.py" in orch.goal
    assert "error in bar.py" in orch.goal


# ---------------------------------------------------------------------------
# LLM client retry tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_client_retries_on_transient_failure():
    client = LLMClient()
    client.settings.provider = Provider.ANTHROPIC

    # First two calls fail, third succeeds
    call_count = 0

    async def flaky_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise anthropic.APIError("transient", request=httpx.Request("POST", "http://x"))
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "success"
        return mock_response

    with patch.object(client.anthropic.__class__, "messages", new_callable=MagicMock) as mock_msgs:
        mock_msgs.create = flaky_create
        result = await client.complete("Hello")
        assert result == "success"
        assert call_count == 3


# Import Provider for the retry test
from furrow.config import Provider  # noqa: E402
import anthropic  # noqa: E402
import httpx  # noqa: E402
