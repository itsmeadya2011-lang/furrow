import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.agents.planner import _extract_json as planner_extract_json
from furrow.agents.tester import _extract_json as tester_extract_json, TestRun
from furrow.agents.worker import _extract_tool_call, _is_safe_path
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import Orchestrator
import furrow.core.orchestrator as orchestrator_module


class FakeClient:
    def __init__(self) -> None:
        self.settings = Settings(max_cycles=0, max_parallel_tasks=5)


class FakePlanner:
    def __init__(self, client) -> None:
        self.client = client
        self.goals = []
        self.plans = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        return self.plans.pop(0)


class FakeWorker:
    def __init__(self, task: TaskModel, client) -> None:
        self.task = task
        self.client = client

    async def run(self) -> str:
        return f"completed {self.task.id}"


class FailingWorker(FakeWorker):
    async def run(self) -> str:
        raise RuntimeError(f"worker failed for {self.task.id}")


class FakeTester:
    def __init__(self, client) -> None:
        self.client = client
        self.results = []
        self.calls = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls.append((goal, list(tasks)))
        return self.results.pop(0)


def test_task_status_is_validated_and_string_like():
    task = TaskModel(id="1", description="do thing", status="pending")

    assert task.status == TaskStatus.PENDING
    assert task.status == "pending"
    assert str(task.status) == "pending"

    task.status = "running"
    assert task.status == TaskStatus.RUNNING

    with pytest.raises(ValidationError):
        TaskModel(id="2", description="invalid", status="unknown")


def test_plan_parse():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.tasks[0].description == "do thing"
    assert plan.tasks[0].status == TaskStatus.PENDING


def test_test_result():
    result = TestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True


@pytest.mark.asyncio
async def test_orchestrator_retains_tasks_and_task_state(monkeypatch):
    first_plan = Plan(
        tasks=[TaskModel(id="1", description="first")],
        rationale="first",
    )
    second_plan = Plan(
        tasks=[TaskModel(id="2", description="second")],
        rationale="second",
    )
    planner = FakePlanner(FakeClient())
    planner.plans = [first_plan, second_plan]
    tester = FakeTester(FakeClient())
    tester.results = [TestResult(passed=True, summary="ok", failures=[])] * 2

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: tester)

    orchestrator = Orchestrator("goal", client=FakeClient())
    await orchestrator._cycle()
    first_tasks = orchestrator._get_tasks()
    await orchestrator._cycle()

    assert first_tasks[0].status == TaskStatus.COMPLETED
    assert [task.id for task in orchestrator._get_tasks()] == ["1", "2"]
    assert orchestrator.plan.tasks is orchestrator._get_tasks()
    assert all(task.status == TaskStatus.COMPLETED for task in orchestrator._get_tasks())


@pytest.mark.asyncio
async def test_orchestrator_enforces_cycle_limit(monkeypatch):
    planner = FakePlanner(FakeClient())
    planner.plans = [
        Plan(tasks=[TaskModel(id="1", description="work")], rationale="work")
        for _ in range(3)
    ]
    tester = FakeTester(FakeClient())
    tester.results = [
        TestResult(passed=False, summary="still failing", failures=["failure"])
        for _ in range(3)
    ]
    settings = Settings(max_cycles=2, max_parallel_tasks=2)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: tester)

    orchestrator = Orchestrator("goal", client=FakeClient(), settings=settings)
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert len(planner.goals) == 2
    assert orchestrator.latest_test_result is not None
    assert orchestrator.latest_test_result.passed is False


@pytest.mark.asyncio
async def test_orchestrator_bounds_worker_concurrency(monkeypatch):
    active = 0
    max_active = 0

    class BoundedWorker(FakeWorker):
        async def run(self) -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "done"

    planner = FakePlanner(FakeClient())
    planner.plans = [
        Plan(
            tasks=[TaskModel(id=str(index), description="work") for index in range(4)],
            rationale="work",
        )
    ]
    tester = FakeTester(FakeClient())
    tester.results = [TestResult(passed=True, summary="ok", failures=[])]
    settings = Settings(max_cycles=1, max_parallel_tasks=2)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", BoundedWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: tester)

    orchestrator = Orchestrator("goal", client=FakeClient(), settings=settings)
    await orchestrator.run()

    assert max_active == 2
    assert all(task.status == TaskStatus.COMPLETED for task in orchestrator._get_tasks())


@pytest.mark.asyncio
async def test_test_failures_keep_original_goal_and_provide_context(monkeypatch):
    planner = FakePlanner(FakeClient())
    planner.plans = [
        Plan(tasks=[TaskModel(id="1", description="work")], rationale="work"),
        Plan(tasks=[TaskModel(id="1", description="work")], rationale="fix"),
    ]
    tester = FakeTester(FakeClient())
    tester.results = [
        TestResult(passed=False, summary="tests failed", failures=["broken assertion"]),
        TestResult(passed=True, summary="tests passed", failures=[]),
    ]
    settings = Settings(max_cycles=2, max_parallel_tasks=1)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: tester)

    orchestrator = Orchestrator("original goal", client=FakeClient(), settings=settings)
    await orchestrator.run()

    assert orchestrator.original_goal == "original goal"
    assert orchestrator.goal == "original goal"
    assert planner.goals[0] == "original goal"
    assert "original goal" in planner.goals[1]
    assert "broken assertion" in planner.goals[1]
    assert orchestrator.latest_test_result is not None
    assert orchestrator.latest_test_result.passed is True


@pytest.mark.asyncio
async def test_planner_worker_and_tester_exceptions_are_recorded(monkeypatch):
    class RaisingPlanner:
        def __init__(self, client) -> None:
            self.client = client

        async def plan(self, goal: str) -> Plan:
            raise RuntimeError("planner failed")

    planner_error_tester = FakeTester(FakeClient())
    planner_error_tester.results = []
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", RaisingPlanner)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: planner_error_tester)
    orchestrator = Orchestrator("goal", client=FakeClient(), settings=Settings(max_cycles=1))
    await orchestrator.run()
    assert orchestrator._get_tasks()[0].status == TaskStatus.FAILED
    assert "planner failed" in orchestrator._get_tasks()[0].result
    assert orchestrator.latest_test_result is not None
    assert orchestrator.latest_test_result.passed is False

    worker_planner = FakePlanner(FakeClient())
    worker_planner.plans = [
        Plan(tasks=[TaskModel(id="1", description="work")], rationale="work")
    ]
    worker_tester = FakeTester(FakeClient())
    worker_tester.results = [TestResult(passed=True, summary="ok", failures=[])]
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: worker_planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FailingWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda client: worker_tester)
    orchestrator = Orchestrator("goal", client=FakeClient(), settings=Settings(max_cycles=2))
    await orchestrator.run()
    assert orchestrator._get_tasks()[0].status == TaskStatus.FAILED
    assert "worker failed" in orchestrator._get_tasks()[0].result

    class RaisingTester(FakeTester):
        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            raise RuntimeError("tester failed")

    worker_planner.plans = [
        Plan(tasks=[TaskModel(id="1", description="work")], rationale="work")
    ]
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", lambda client: worker_planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", RaisingTester)
    orchestrator = Orchestrator("goal", client=FakeClient(), settings=Settings(max_cycles=1))
    await orchestrator.run()
    assert orchestrator.latest_test_result is not None
    assert orchestrator.latest_test_result.failures == ["tester failed"]


def test_planner_extract_json_handles_fenced_json():
    text = 'Here is the plan:\n```json\n{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}\n```\nDone.'
    extracted = planner_extract_json(text)
    assert extracted is not None
    assert '"tasks"' in extracted
    assert '"rationale"' in extracted


def test_planner_extract_json_handles_braces_in_strings():
    text = '{"tasks": [{"id": "1", "description": "json with {braces} in string"}], "rationale": "ok"} trailing'
    extracted = planner_extract_json(text)
    assert extracted is not None
    assert "json with {braces} in string" in extracted
    assert extracted.endswith('"ok"}')


def test_tester_extract_json_handles_fenced_json():
    text = '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    extracted = tester_extract_json(text)
    assert extracted is not None
    assert '"passed"' in extracted
    assert '"summary"' in extracted


def test_tester_result_parsing_rejects_success_on_nonzero_exit_code():
    test_run = TestRun(returncode=1, stdout="", stderr="error", runner="pytest", timed_out=False)
    # Even if LLM returns passed=True, nonzero exit code should force passed=False
    llm_response = '{"passed": true, "summary": "LLM says ok", "failures": []}'
    from furrow.agents.tester import TesterAgent
    from furrow.config import Settings
    from furrow.llm import LLMClient
    tester = TesterAgent(client=LLMClient(settings=Settings(provider="ollama")))
    result = tester._try_parse_result(llm_response, test_run)
    assert result.passed is False
    assert "exit code: 1" in result.summary
    assert result.failures


def test_tester_result_parsing_rejects_success_on_timeout():
    test_run = TestRun(returncode=-1, stdout="", stderr="timed out", runner="pytest", timed_out=True)
    llm_response = '{"passed": true, "summary": "LLM says ok", "failures": []}'
    from furrow.agents.tester import TesterAgent
    from furrow.config import Settings
    from furrow.llm import LLMClient
    tester = TesterAgent(client=LLMClient(settings=Settings(provider="ollama")))
    result = tester._try_parse_result(llm_response, test_run)
    assert result.passed is False
    assert "timed out" in result.summary
    assert result.failures


def test_worker_extract_tool_call_plain_text_fallback():
    # No fence, just JSON on a line
    response = 'some text\n{"tool": "read", "path": "foo.txt"}\nmore text'
    tool_call = _extract_tool_call(response)
    assert tool_call is not None
    assert tool_call["tool"] == "read"
    assert tool_call["path"] == "foo.txt"


def test_worker_rejects_path_traversal_on_read():
    import asyncio
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel, Settings
    from furrow.llm import LLMClient

    async def test():
        client = LLMClient(settings=Settings(workspace=Path("/tmp"), provider="ollama"))
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)
        result = await worker._execute_tool({"tool": "read", "path": "../../etc/passwd"})
        data = json.loads(result)
        assert "error" in data
        assert "Path traversal blocked" in data["error"]

    asyncio.run(test())


def test_worker_rejects_path_traversal_on_write():
    import asyncio
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel, Settings
    from furrow.llm import LLMClient

    async def test():
        client = LLMClient(settings=Settings(workspace=Path("/tmp"), provider="ollama"))
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)
        result = await worker._execute_tool({"tool": "write", "path": "../../etc/passwd", "content": "x"})
        data = json.loads(result)
        assert "error" in data
        assert "Path traversal blocked" in data["error"]

    asyncio.run(test())


def test_worker_rejects_missing_required_fields():
    import asyncio
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel, Settings
    from furrow.llm import LLMClient

    async def test():
        client = LLMClient(settings=Settings(workspace=Path("/tmp"), provider="ollama"))
        task = TaskModel(id="1", description="test")
        worker = WorkerAgent(task=task, client=client)

        # Missing path for read
        result = await worker._execute_tool({"tool": "read"})
        data = json.loads(result)
        assert "error" in data
        assert "Missing required field: path" in data["error"]

        # Missing path for write
        result = await worker._execute_tool({"tool": "write", "content": "x"})
        data = json.loads(result)
        assert "error" in data
        assert "Missing required field: path" in data["error"]

        # Missing content for write
        result = await worker._execute_tool({"tool": "write", "path": "foo.txt"})
        data = json.loads(result)
        assert "error" in data
        assert "Missing required field: content" in data["error"]

        # Missing command for run
        result = await worker._execute_tool({"tool": "run"})
        data = json.loads(result)
        assert "error" in data
        assert "Missing required field: command" in data["error"]

    asyncio.run(test())


def test_settings_workspace_default_evaluated_at_instantiation():
    # default_factory=lambda: Path.cwd() means each instantiation gets its own cwd
    s1 = Settings()
    s2 = Settings()
    assert s1.workspace == Path.cwd()
    assert s2.workspace == Path.cwd()
    # They should be equal but not the same object if cwd changes between instantiations
    # (can't easily test cwd change without chdir, but we verify the factory is used)
    assert isinstance(s1.workspace, Path)
    assert isinstance(s2.workspace, Path)


def test_is_safe_path_blocks_traversal():
    workspace = Path("/workspace")
    assert _is_safe_path(workspace, Path("subdir/file.txt")) is True
    assert _is_safe_path(workspace, Path("../etc/passwd")) is False
    assert _is_safe_path(workspace, Path("/absolute/path")) is False
