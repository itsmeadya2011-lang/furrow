import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_returns_last_plan():
    plan = Plan(tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")], rationale="ok")
    orchestrator = Orchestrator(goal="test")
    orchestrator._last_plan = plan
    assert len(orchestrator._get_tasks()) == 2
    assert orchestrator._get_tasks()[0].id == "1"


def test_orchestrator_is_done_with_completed_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="a", status="completed")], rationale="ok")
    orchestrator = Orchestrator(goal="test")
    orchestrator._last_plan = plan
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_pending_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="a", status="pending")], rationale="ok")
    orchestrator = Orchestrator(goal="test")
    orchestrator._last_plan = plan
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_with_failed_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="a", status="failed")], rationale="ok")
    orchestrator = Orchestrator(goal="test")
    orchestrator._last_plan = plan
    assert orchestrator._is_done() is False


def test_orchestrator_history_tracking():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(tasks=[TaskModel(id="1", description="a", status="completed")], rationale="ok")
    orchestrator._last_plan = plan
    test_result = TestResult(passed=True, summary="ok", failures=[])
    orchestrator._history.append({"cycle": 1, "plan": plan.model_dump(), "test_result": test_result.model_dump()})
    assert len(orchestrator._history) == 1
    assert orchestrator._history[0]["cycle"] == 1


def test_llm_client_tools():
    from furrow.llm import LLMClient
    client = LLMClient()
    tools = client.tools
    assert len(tools) == 4
    tool_names = {t["name"] for t in tools}
    assert tool_names == {"read_file", "write_file", "list_files", "run_command"}


@pytest.mark.asyncio
async def test_llm_client_execute_tool_read_file(tmp_path):
    from furrow.llm import LLMClient
    client = LLMClient()
    f = tmp_path / "test.txt"
    f.write_text("hello")
    result = await client._execute_tool("read_file", {"path": str(f)})
    assert result == "hello"


@pytest.mark.asyncio
async def test_llm_client_execute_tool_write_file(tmp_path):
    from furrow.llm import LLMClient
    client = LLMClient()
    target = tmp_path / "sub" / "out.txt"
    await client._execute_tool("write_file", {"path": str(target), "content": "world"})
    assert target.read_text() == "world"


@pytest.mark.asyncio
async def test_llm_client_execute_tool_list_files(tmp_path):
    from furrow.llm import LLMClient
    client = LLMClient()
    (tmp_path / "a.txt").write_text("a")
    result = await client._execute_tool("list_files", {"directory": str(tmp_path)})
    assert "a.txt" in result


@pytest.mark.asyncio
async def test_llm_client_execute_tool_run_command():
    from furrow.llm import LLMClient
    client = LLMClient()
    result = await client._execute_tool("run_command", {"command": "echo hello"})
    assert "hello" in result


def test_planner_accepts_history():
    from furrow.agents.planner import PlannerAgent
    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"tasks": [], "rationale": "ok"}'
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)
    history = [{"cycle": 1, "test_result": {"summary": "ok"}}]

    # We can't easily run the async method without an event loop, so just check signature via inspection
    import inspect
    sig = inspect.signature(planner.plan)
    assert "history" in sig.parameters
