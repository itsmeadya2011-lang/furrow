from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ---------- Config tests ----------

def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_plan_creation():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert len(p.tasks) == 1
    assert p.tasks[0].description == "do thing"


# ---------- Orchestrator _is_done tests ----------

def _make_orch() -> Orchestrator:
    return Orchestrator(goal="test")


def test_is_done_no_tasks():
    orch = _make_orch()
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed_passing_tests():
    orch = _make_orch()
    orch.tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_is_done_failed_task():
    orch = _make_orch()
    orch.tasks = [TaskModel(id="1", description="x", status="failed")]
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = _make_orch()
    orch.tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.last_test_result = TestResult(passed=False, summary="bad", failures=["err"])
    assert orch._is_done() is False


def test_is_done_partial_completion():
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="x", status="completed"),
        TaskModel(id="2", description="y", status="pending"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_pending_no_test_result():
    orch = _make_orch()
    orch.tasks = [TaskModel(id="1", description="x", status="pending")]
    orch.last_test_result = None
    assert orch._is_done() is False


def test_get_tasks_returns_stored():
    orch = _make_orch()
    tasks = [TaskModel(id="1", description="x")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


def test_orchestrator_accepts_settings():
    s = Settings(max_cycles=3)
    orch = Orchestrator(goal="test", settings=s)
    assert orch.settings.max_cycles == 3


# ---------- Orchestrator run() tests ----------

@patch("furrow.core.orchestrator.TesterAgent")
@patch("furrow.core.orchestrator.WorkerAgent")
async def test_max_cycles_stops(mock_worker_cls, mock_tester_cls):
    settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="test", settings=settings)
    orch.planner.plan = AsyncMock(return_value=Plan(
        tasks=[TaskModel(id="1", description="do something")], rationale="ok"
    ))
    mock_worker_cls.return_value.run = AsyncMock(return_value="done")
    mock_tester_cls.return_value.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["err"])
    )
    await orch.run()
    assert orch.cycles == 2


async def test_run_stops_when_no_tasks():
    orch = Orchestrator(goal="test")
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="nothing"))
    await orch.run()
    assert orch.cycles == 1
    assert orch.tasks == []


@patch("furrow.core.orchestrator.TesterAgent")
@patch("furrow.core.orchestrator.WorkerAgent")
async def test_run_completes_when_all_pass(mock_worker_cls, mock_tester_cls):
    orch = Orchestrator(goal="test")
    orch.planner.plan = AsyncMock(return_value=Plan(
        tasks=[TaskModel(id="1", description="do something")], rationale="ok"
    ))
    mock_worker_cls.return_value.run = AsyncMock(return_value="done")
    mock_tester_cls.return_value.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch.tasks[0].status == "completed"
    assert orch.last_test_result.passed is True


# ---------- LLM OLLAMA routing test ----------

@patch("furrow.llm.httpx.AsyncClient")
async def test_complete_ollama(mock_client_cls):
    settings = Settings(provider=Provider.OLLAMA, model="llama2")
    client = LLMClient(settings=settings)

    mock_http = AsyncMock()
    mock_http.__aenter__.return_value = mock_http
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ollama response"}}
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_client_cls.return_value = mock_http

    result = await client.complete("hello", system="be helpful", model="llama2")
    assert result == "ollama response"

    call_args = mock_http.post.call_args
    assert call_args.args[0] == "http://localhost:11434/api/chat"
    assert call_args.kwargs["json"]["model"] == "llama2"
    assert call_args.kwargs["json"]["stream"] is False
    messages = call_args.kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "hello"}
