import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings


class MockLLMClient:
    def __init__(self, response: str = ""):
        self.response = response
        self.settings = Settings()
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self.response
    async def read_file(self, path):
        return ""
    async def write_file(self, path, content):
        pass
    def list_files(self, directory):
        return []


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Orchestrator logic ---

def test_orchestrator_get_tasks():
    from furrow.core.orchestrator import Orchestrator
    orchestrator = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="test", status="completed")]
    orchestrator._current_tasks = tasks
    assert orchestrator._get_tasks() is tasks
    assert len(orchestrator._get_tasks()) > 0


def test_orchestrator_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failed():
    from furrow.core.orchestrator import Orchestrator
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_with_pending():
    from furrow.core.orchestrator import Orchestrator
    orchestrator = Orchestrator(goal="test")
    orchestrator._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


# --- WorkerAgent ---

async def test_worker_reads_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from furrow.agents.worker import WorkerAgent
    mock_client = MockLLMClient(
        response='{"files": [{"path": "test.txt", "content": "hello"}]}'
    )
    worker = WorkerAgent(
        task=TaskModel(id="1", description="write test.txt", files=["test.txt"]),
        client=mock_client,
    )
    result = await worker.run()
    assert "completed" in result
    assert (tmp_path / "test.txt").read_text() == "hello"


# --- TesterAgent ---

async def test_tester_no_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from furrow.agents.tester import TesterAgent
    tester = TesterAgent(client=MockLLMClient())
    output = await tester._run_tests()
    assert output == "No test runner found."


async def test_tester_detects_pytest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    from furrow.agents.tester import TesterAgent
    tester = TesterAgent(client=MockLLMClient())
    output = await tester._run_tests()
    assert "No test runner found." not in output


# --- PlannerAgent ---

async def test_planner_valid_json():
    from furrow.agents.planner import PlannerAgent
    mock_client = MockLLMClient(
        response='{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    )
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("do something")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"


async def test_planner_invalid_json():
    from furrow.agents.planner import PlannerAgent
    mock_client = MockLLMClient(response="not json")
    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError):
        await planner.plan("do something")