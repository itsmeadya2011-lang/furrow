from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import (
    FileEdit,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
)
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_from_llm_output():
    llm_output = json.dumps(
        {
            "tasks": [
                {
                    "id": "1",
                    "description": "Create auth module",
                    "files": ["src/auth.py", "tests/test_auth.py"],
                    "dependencies": [],
                },
                {
                    "id": "2",
                    "description": "Add login endpoint",
                    "files": ["src/routes.py"],
                    "dependencies": ["1"],
                },
            ],
            "rationale": "Need auth before routes",
        }
    )
    data = json.loads(llm_output)
    plan = Plan(**data)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].files == ["src/auth.py", "tests/test_auth.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Need auth before routes"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_edit_new_file():
    edit = FileEdit(path="test.py", old_content=None, new_content="print('hello')")
    assert edit.path == "test.py"
    assert edit.old_content is None
    assert edit.new_content == "print('hello')"


def test_file_edit_replace():
    edit = FileEdit(path="test.py", old_content="old", new_content="new")
    assert edit.old_content == "old"
    assert edit.new_content == "new"


def test_file_edit_delete():
    edit = FileEdit(path="old.py", old_content="x", new_content=None)
    assert edit.new_content is None


def test_worker_result_defaults():
    result = WorkerResult(summary="Made changes")
    assert result.summary == "Made changes"
    assert result.files_changed == []
    assert result.edits == []


def test_worker_result_with_edits():
    result = WorkerResult(
        summary="Created file",
        files_changed=["main.py"],
        edits=[
            FileEdit(path="main.py", old_content=None, new_content="print('hi')"),
        ],
    )
    assert len(result.edits) == 1
    assert result.edits[0].path == "main.py"


def test_settings_provider_enum():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


# ---------------------------------------------------------------------------
# LLM Ollama provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_complete():
    settings = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434", model="llama3"
    )
    client = LLMClient(settings=settings)
    mock_response = {"response": "Hello from Ollama"}
    mock_http = MagicMock()
    mock_http.post = AsyncMock()
    mock_http.post.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value=mock_response),
    )
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http):
        result = await client._complete_ollama("hi", "", "llama3")
    assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_ollama_complete_with_system():
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)
    mock_response = {"response": "Got system prompt"}
    mock_http = MagicMock()
    mock_http.post = AsyncMock()
    mock_http.post.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value=mock_response),
    )
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http):
        result = await client._complete_ollama("prompt", "system prompt", "llama3")
    assert result == "Got system prompt"
    # Verify the payload includes both system and prompt
    call_args = mock_http.post.call_args
    payload = call_args.kwargs["json"]
    assert "system prompt" in payload["prompt"]
    assert payload["model"] == "llama3"
    assert payload["stream"] is False


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_writes_new_file(tmp_path):
    settings = Settings(provider=Provider.ANTHROPIC)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.write_file = AsyncMock()
    mock_client.read_file = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "summary": "Created config.py",
                "files_changed": ["config.py"],
                "edits": [
                    {
                        "path": str(tmp_path / "config.py"),
                        "old_content": None,
                        "new_content": "DEBUG = True",
                    },
                ],
            }
        )
    )

    task = TaskModel(id="1", description="Create config file")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert isinstance(result, WorkerResult)
    assert result.summary == "Created config.py"
    mock_client.write_file.assert_called_once_with(str(tmp_path / "config.py"), "DEBUG = True")


@pytest.mark.asyncio
async def test_worker_replaces_content(tmp_path):
    settings = Settings(provider=Provider.ANTHROPIC)
    target = tmp_path / "main.py"
    target.write_text("old line\n")
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.write_file = AsyncMock()

    async def mock_read_file(path):
        return target.read_text()

    mock_client.read_file = mock_read_file
    mock_client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "summary": "Fixed bug",
                "files_changed": [str(target)],
                "edits": [
                    {
                        "path": str(target),
                        "old_content": "old line\n",
                        "new_content": "new line\n",
                    },
                ],
            }
        )
    )

    task = TaskModel(id="1", description="Fix the bug")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result.summary == "Fixed bug"
    # The updated content should have "new line\n" replacing "old line\n"
    mock_client.write_file.assert_called_once_with(str(target), "new line\n")


@pytest.mark.asyncio
async def test_worker_handles_invalid_json():
    settings = Settings(provider=Provider.ANTHROPIC)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.write_file = AsyncMock()
    mock_client.read_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value="not valid json")

    task = TaskModel(id="1", description="Do something")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert isinstance(result, WorkerResult)
    assert "not valid json" in result.summary
    assert result.edits == []
    mock_client.write_file.assert_not_called()


# ---------------------------------------------------------------------------
# Orchestrator _is_done tests
# ---------------------------------------------------------------------------


def make_orchestrator(max_cycles=0):
    settings = Settings(provider=Provider.ANTHROPIC, max_cycles=max_cycles)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
    return orch


def test_is_done_no_plan():
    orch = make_orchestrator()
    assert orch._is_done() is True


def test_is_done_all_tasks_complete_tests_pass():
    orch = make_orchestrator()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_all_tasks_complete_but_tests_failed():
    orch = make_orchestrator()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
        ],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = TestResult(passed=False, summary="bad", failures=["something broke"])
    assert orch._is_done() is False


def test_is_done_some_tasks_failed():
    orch = make_orchestrator()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


def test_is_done_max_cycles_reached():
    orch = make_orchestrator(max_cycles=3)
    orch.cycles = 3
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1", status="completed")],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_max_cycles_hard_limit():
    """max_cycles is a hard stop — even failed tests won't extend past it."""
    orch = make_orchestrator(max_cycles=3)
    orch.cycles = 3
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1", status="completed")],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = TestResult(passed=False, summary="bad", failures=["broke"])
    assert orch._is_done() is True


def test_is_done_partial_tasks():
    orch = make_orchestrator()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ],
        rationale="done",
    )
    orch.current_plan = plan
    orch._test_result = None
    assert orch._is_done() is False
