import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator, STATE_FILE
from furrow.llm import LLMClient
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model():
    task = TaskModel(id="1", description="test", files=["a.py"], dependencies=["2"])
    assert task.id == "1"
    assert task.files == ["a.py"]
    assert task.dependencies == ["2"]
    assert task.status == "pending"


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1", files=["a.py"]),
        TaskModel(id="2", description="task 2", files=["b.py"]),
    ]
    plan = Plan(tasks=tasks, rationale="test")
    assert len(plan.tasks) == 2


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


@pytest.mark.asyncio
async def test_llm_client_initialization():
    client = LLMClient()
    assert client.settings is not None


def test_orchestrator_state_file_path():
    assert STATE_FILE.name == ".furrow_state.json"


def test_task_model_serialization():
    task = TaskModel(id="1", description="test", files=["a.py"], status="completed", result="done")
    data = task.model_dump()
    assert data["id"] == "1"
    assert data["status"] == "completed"
    assert data["result"] == "done"


def test_test_result_serialization():
    tr = TestResult(passed=False, summary="failed", failures=["error 1", "error 2"])
    data = tr.model_dump()
    assert data["passed"] is False
    assert len(data["failures"]) == 2