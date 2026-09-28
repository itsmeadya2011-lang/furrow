import json
from unittest.mock import AsyncMock, patch

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    def __init__(self, responses=None, settings=None):
        self.responses = list(responses) if responses else []
        self._call_index = 0
        self.settings = settings or Settings()
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
            return resp
        return ""

    async def complete_with_tools(
        self,
        prompt: str,
        tools=None,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "tools": tools, "system": system, "model": model}
        )
        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
            return resp
        return ""

    async def aclose(self) -> None:
        pass

    def list_files(self, directory) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Config / Model tests
# ---------------------------------------------------------------------------


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test task")
    assert t.id == "1"
    assert t.description == "test task"
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_creation():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    p = Plan(tasks=tasks, rationale="test rationale")
    assert len(p.tasks) == 2
    assert p.rationale == "test rationale"
    assert p.tasks[0].id == "1"
    assert p.tasks[1].description == "task 2"


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.passed is True
    assert t.summary == "all good"
    assert t.failures == []


def test_settings_ollama_field():
    s = Settings()
    assert s.ollama_model == "llama3.1"
    s2 = Settings(ollama_model="custom-ollama-model")
    assert s2.ollama_model == "custom-ollama-model"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


async def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orch._is_done() is True


async def test_orchestrator_is_done_all_failed():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="failed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is True


async def test_orchestrator_is_done_with_no_tasks():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    orch.all_tasks = []
    assert orch._is_done() is False


async def test_orchestrator_is_done_tests_passed():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


async def test_orchestrator_is_done_tests_failed():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    orch.all_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False


async def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=1)
    plan_json = json.dumps(
        {
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "reason",
        }
    )
    mock_client = MockLLMClient(responses=[plan_json], settings=settings)
    orch = Orchestrator(goal="test", client=mock_client, settings=settings)

    with patch.object(
        WorkerAgent, "run", new_callable=AsyncMock, return_value="completed"
    ), patch.object(
        TesterAgent,
        "run",
        new_callable=AsyncMock,
        return_value=TestResult(passed=False, summary="fail", failures=["err"]),
    ):
        await orch.run()

    assert orch.cycles == 1


async def test_orchestrator_get_tasks_returns_all_tasks():
    orch = Orchestrator(goal="test", client=MockLLMClient(), settings=Settings())
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    orch.all_tasks.extend(tasks)
    assert orch._get_tasks() == tasks
    assert orch._get_tasks() is orch.all_tasks


async def test_orchestrator_stores_plan():
    plan_json = json.dumps(
        {
            "tasks": [{"id": "1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "reason",
        }
    )
    mock_client = MockLLMClient(responses=[plan_json], settings=Settings())
    orch = Orchestrator(goal="test", client=mock_client, settings=Settings())

    with patch.object(
        WorkerAgent, "run", new_callable=AsyncMock, return_value="done"
    ), patch.object(
        TesterAgent,
        "run",
        new_callable=AsyncMock,
        return_value=TestResult(passed=True, summary="ok", failures=[]),
    ):
        await orch._cycle()

    assert orch.current_plan is not None
    assert isinstance(orch.current_plan, Plan)
    assert len(orch.current_plan.tasks) == 1
    assert orch.current_plan.tasks[0].description == "task"


async def test_orchestrator_cycle_sets_task_status():
    plan_json = json.dumps(
        {
            "tasks": [{"id": "t1", "description": "task", "files": [], "dependencies": []}],
            "rationale": "reason",
        }
    )
    mock_client = MockLLMClient(responses=[plan_json], settings=Settings())
    orch = Orchestrator(goal="test", client=mock_client, settings=Settings())

    with patch.object(
        WorkerAgent, "run", new_callable=AsyncMock, return_value="done"
    ), patch.object(
        TesterAgent,
        "run",
        new_callable=AsyncMock,
        return_value=TestResult(passed=True, summary="ok", failures=[]),
    ):
        await orch._cycle()

    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].status == "completed"
    assert orch.all_tasks[0].result == "done"


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


async def test_planner_parses_valid_json():
    plan_json = json.dumps(
        {
            "tasks": [{"id": "1", "description": "task 1", "files": [], "dependencies": []}],
            "rationale": "a valid plan",
        }
    )
    mock_client = MockLLMClient(responses=[plan_json], settings=Settings())
    planner = PlannerAgent(client=mock_client, settings=Settings())
    plan = await planner.plan("build a feature")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "a valid plan"


async def test_planner_fallback_on_bad_json():
    mock_client = MockLLMClient(
        responses=["this is not json at all", "not json either", "still not json"],
        settings=Settings(),
    )
    planner = PlannerAgent(client=mock_client, settings=Settings())
    plan = await planner.plan("build a feature")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description  # contains error message
    assert "fallback" in plan.rationale.lower()


async def test_planner_json_recovery_from_markdown():
    raw = '{"tasks": [{"id": "1", "description": "recovered"}], "rationale": "ok"}'
    response = f"Here is the plan:\n```json\n{raw}\n```"
    mock_client = MockLLMClient(responses=[response], settings=Settings())
    planner = PlannerAgent(client=mock_client, settings=Settings())
    plan = await planner.plan("build a feature")

    assert isinstance(plan, Plan)
    assert plan.tasks[0].description == "recovered"


# ---------------------------------------------------------------------------
# Worker tests
# ---------------------------------------------------------------------------


async def test_worker_basic():
    mock_client = MockLLMClient(responses=["Worker completed the task."], settings=Settings())
    task = TaskModel(id="1", description="do something", files=["file.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert isinstance(result, str)
    assert result == "Worker completed the task."


async def test_worker_with_no_files():
    mock_client = MockLLMClient(responses=["Done."], settings=Settings())
    task = TaskModel(id="1", description="implement feature")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tester tests
# ---------------------------------------------------------------------------


def test_tester_default_commands():
    assert len(TesterAgent.DEFAULT_TEST_COMMANDS) > 0
    assert ["pytest", "-q"] in TesterAgent.DEFAULT_TEST_COMMANDS
    for cmd in TesterAgent.DEFAULT_TEST_COMMANDS:
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)


def test_tester_configurable_commands():
    custom = [["echo", "hello"], ["make", "test"]]
    tester = TesterAgent(client=MockLLMClient(), test_commands=custom)
    assert tester.test_commands == custom


def test_tester_custom_timeout():
    tester = TesterAgent(client=MockLLMClient(), timeout=30)
    assert tester.timeout == 30
    default_tester = TesterAgent(client=MockLLMClient())
    assert default_tester.timeout == 120


async def test_tester_returns_test_result():
    result_json = json.dumps(
        {"passed": True, "summary": "All tests passed", "failures": []}
    )
    mock_client = MockLLMClient(responses=[result_json], settings=Settings())
    tester = TesterAgent(client=mock_client)
    tasks = [TaskModel(id="1", description="task")]
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value=""):
        result = await tester.run("test goal", tasks)

    assert isinstance(result, TestResult)
    assert result.passed is True
