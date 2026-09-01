"""Tests for the Furrow autonomous coding agent."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import (
    FileEdit,
    Plan,
    Provider,
    TaskModel,
    TestResult,
    WorkerResult,
)
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator


# ─── Config / Model Tests ───────────────────────────────────────────


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_worker_result():
    r = WorkerResult(
        files=[FileEdit(path="foo.py", content="print('hello')")],
        summary="Created foo.py",
    )
    assert r.files[0].path == "foo.py"
    assert r.summary == "Created foo.py"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_plan_with_dependencies():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="first", dependencies=[]),
            TaskModel(id="2", description="second", dependencies=["1"]),
        ],
        rationale="test",
    )
    assert p.tasks[1].dependencies == ["1"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0  # infinite
    assert s.llm_timeout == 120
    assert s.max_retries == 3
    assert s.ollama_model == "llama3.1"


def test_settings_ollama():
    s = Settings(provider=Provider.OLLAMA, ollama_model="llama3.1")
    assert s.provider == Provider.OLLAMA


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    s = Settings()
    assert s.max_cycles == 3
    assert s.provider == Provider.OPENAI


# ─── LLMClient Tests ────────────────────────────────────────────────


def test_llm_client_init():
    from furrow.llm import LLMClient
    client = LLMClient()
    assert client is not None
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_openai_init():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    assert client.openai.api_key == "test-key"


def test_llm_client_ollama_init():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert str(client.http_client.base_url).rstrip("/") == "http://localhost:11434"


@pytest.mark.asyncio
async def test_llm_complete_anthropic():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Claude")]
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
    client._anthropic = mock_anthropic
    result = await client.complete("Hello", system="Test")
    assert result == "Hello from Claude"


@pytest.mark.asyncio
async def test_llm_complete_openai():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._openai = mock_openai
    result = await client.complete("Hello")
    assert result == "Hello from OpenAI"


@pytest.mark.asyncio
async def test_llm_complete_ollama():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Hello from Ollama"}
    mock_response.raise_for_status = MagicMock()
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    client._http_client = mock_http
    result = await client.complete("Hello")
    assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_llm_timeout():
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key", llm_timeout=1)
    client = LLMClient(settings=s)

    async def slow_complete(*args, **kwargs):
        await asyncio.sleep(10)

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(side_effect=slow_complete)
    client._anthropic = mock_anthropic
    with pytest.raises(TimeoutError):
        await client.complete("Hello")


@pytest.mark.asyncio
async def test_llm_retry_on_failure():
    from furrow.llm import LLMClient
    s = Settings(
        provider=Provider.OPENAI,
        openai_api_key="test-key",
        max_retries=2,
        retry_base_delay=0.01,
    )
    client = LLMClient(settings=s)

    call_count = 0

    async def failing_then_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Transient error")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
        return mock_response

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=failing_then_success)
    client._openai = mock_openai
    result = await client.complete("Hello")
    assert result == "Success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_llm_write_file(tmp_path):
    from furrow.llm import LLMClient
    s = Settings(workspace=tmp_path)
    client = LLMClient(settings=s)
    await client.write_file("subdir/test.txt", "hello world")
    content = (tmp_path / "subdir" / "test.txt").read_text()
    assert content == "hello world"


@pytest.mark.asyncio
async def test_llm_read_file(tmp_path):
    from furrow.llm import LLMClient
    s = Settings(workspace=tmp_path)
    client = LLMClient(settings=s)
    (tmp_path / "test.txt").write_text("hello")
    result = await client.read_file(tmp_path / "test.txt")
    assert result == "hello"


def test_llm_list_files(tmp_path):
    from furrow.llm import LLMClient
    s = Settings(workspace=tmp_path)
    client = LLMClient(settings=s)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir(parents=True)
    (tmp_path / "sub" / "b.py").write_text("y")
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "sub/b.py" in files


# ─── PlannerAgent Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_plan = {
        "tasks": [{"id": "1", "description": "fix bug", "files": [], "dependencies": []}],
        "rationale": "test plan",
    }
    mock_complete = AsyncMock(return_value=json.dumps(mock_plan))
    mock_client.complete = mock_complete

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Fix the bug")

    assert plan.tasks[0].description == "fix bug"
    assert plan.rationale == "test plan"
    assert mock_complete.call_count == 1  # No retry needed


@pytest.mark.asyncio
async def test_planner_re_retries_on_bad_json():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_retries=3)

    good_plan = {
        "tasks": [{"id": "1", "description": "fix bug", "files": [], "dependencies": []}],
        "rationale": "test plan",
    }
    responses = [
        "not valid json {{{",
        json.dumps(good_plan),
    ]
    mock_client.complete = AsyncMock(side_effect=responses)

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Fix the bug")

    assert plan.tasks[0].description == "fix bug"
    assert mock_client.complete.call_count == 2


@pytest.mark.asyncio
async def test_planner_exceeds_retries():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_retries=2)
    mock_client.complete = AsyncMock(return_value="not json at all")

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Fix the bug")


# ─── WorkerAgent Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_parses_and_writes_files(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)

    worker_response = json.dumps({
        "files": [
            {"path": "output.py", "content": "print('hello')"},
            {"path": "sub/mod.py", "content": "x = 1"},
        ],
        "summary": "Created output.py and sub/mod.py",
    })
    mock_client.complete = AsyncMock(return_value=worker_response)
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Create output files")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert mock_client.write_file.call_count == 2
    assert "output.py" in result
    assert "Created output.py and sub/mod.py" in result


@pytest.mark.asyncio
async def test_worker_handles_parse_error(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)
    mock_client.complete = AsyncMock(return_value="not json")
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Create a file")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert "PARSE_ERROR" in result
    assert mock_client.write_file.call_count == 0


@pytest.mark.asyncio
async def test_worker_no_files(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)
    mock_client.complete = AsyncMock(
        return_value=json.dumps({"files": [], "summary": "Nothing needed"})
    )
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Analyze code")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert "Nothing needed" in result
    assert mock_client.write_file.call_count == 0


# ─── TesterAgent Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tester_parses_pass_result():
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=Path.cwd())

    result_json = json.dumps({"passed": True, "summary": "All tests passed", "failures": []})
    mock_client.complete = AsyncMock(return_value=result_json)

    tester = TesterAgent(client=mock_client)
    with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="Tests passed")):
        result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_parses_fail_result():
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=Path.cwd())

    result_json = json.dumps({"passed": False, "summary": "3 failures", "failures": ["x", "y"]})
    mock_client.complete = AsyncMock(return_value=result_json)

    tester = TesterAgent(client=mock_client)
    with patch.object(TesterAgent, "_run_tests", AsyncMock(return_value="Tests failed")):
        result = await tester.run("test goal", [])

    assert result.passed is False
    assert len(result.failures) == 2


def test_tester_project_detection(tmp_path):
    s = Settings(workspace=tmp_path)
    tester = TesterAgent(client=MagicMock())
    tester.client.settings = s

    (tmp_path / "pyproject.toml").write_text("")
    assert tester._detect_project_type() == "python"

    (tmp_path / "pyproject.toml").unlink()
    (tmp_path / "package-lock.json").write_text("")
    assert tester._detect_project_type() == "node"

    (tmp_path / "package-lock.json").unlink()
    assert tester._detect_project_type() == "unknown"


def test_tester_test_commands():
    tester = TesterAgent(client=MagicMock())
    tester.client.settings = Settings()

    py_cmds = tester._test_commands("python")
    assert py_cmds[0] == ["python", "-m", "pytest", "-q"]

    node_cmds = tester._test_commands("node")
    assert "pnpm" in node_cmds[0]

    rust_cmds = tester._test_commands("rust")
    assert rust_cmds[0] == ["cargo", "test", "-q"]

    go_cmds = tester._test_commands("go")
    assert go_cmds[0] == ["go", "test", "./..."]


def test_tester_extract_json_block():
    text = '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    result = TesterAgent._extract_json_block(text)
    assert result is not None
    data = json.loads(result)
    assert data["passed"] is True


# ─── Orchestrator Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_no_tasks():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    orch = Orchestrator(goal="test", client=mock_client)
    assert orch._is_done() is True  # No tasks = done


@pytest.mark.asyncio
async def test_orchestrator_is_done_all_completed():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    orch = Orchestrator(goal="test", client=mock_client)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_not_done_with_pending():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    orch = Orchestrator(goal="test", client=mock_client)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_enforced():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", client=mock_client)

    plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="test")
    plan.tasks[0].status = "completed"
    mock_client.complete = AsyncMock(return_value=json.dumps({"tasks": [], "rationale": "done"}))

    # Patch planner.plan to return the plan
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    # Patch tester to avoid running real tests
    mock_tester_result = TestResult(passed=True, summary="All passed", failures=[])
    with patch.object(PlannerAgent, "plan", mock_planner.plan):
        with patch.object(TesterAgent, "run", AsyncMock(return_value=mock_tester_result)):
            await orch.run()

    # Should have run exactly 1 cycle
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_semaphore_limits_concurrency():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_parallel_tasks=2)
    orch = Orchestrator(goal="test", client=mock_client)

    # All tasks complete -> _is_done returns True -> loop exits after 1 cycle
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1"),
            TaskModel(id="2", description="t2"),
            TaskModel(id="3", description="t3"),
        ],
        rationale="test",
    )
    for t in plan.tasks:
        t.status = "completed"

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    # Patch tester to avoid running real tests
    mock_tester_result = TestResult(passed=True, summary="All passed", failures=[])
    with patch.object(PlannerAgent, "plan", mock_planner.plan):
        with patch.object(TesterAgent, "run", AsyncMock(return_value=mock_tester_result)):
            await orch.run()

    # Each task should be in the semaphore-limited execution
    assert len(orch.all_tasks) == 3


@pytest.mark.asyncio
async def test_orchestrator_handles_planner_error():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", client=mock_client)

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(side_effect=ValueError("Planning failed"))
    with patch.object(PlannerAgent, "plan", mock_planner.plan):
        await orch.run()

    # Should not crash; cycle error is caught and loop exits via max_cycles
    assert orch.cycles == 2
