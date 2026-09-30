import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ═══ Config Model Tests ═══

def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_with_values():
    t = TaskModel(
        id="2", description="do", files=["a.py"], dependencies=["1"],
        status="completed", result="done"
    )
    assert t.files == ["a.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_parse():  # keep original
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
    assert p.rationale == "nothing to do"


def test_test_result():  # keep original, just renamed
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_with_failures():
    t = TestResult(
        passed=False, summary="2 failures",
        failures=["test_a failed", "test_b failed"]
    )
    assert t.passed is False
    assert len(t.failures) == 2


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


# ═══ Orchestrator Tests ═══

def test_orchestrator_init():
    orch = Orchestrator(goal="test goal", client=MagicMock())
    assert orch.goal == "test goal"
    assert orch.tasks == []
    assert orch.cycles == 0
    assert orch.max_cycles == 0


def test_orchestrator_max_cycles_param():
    orch = Orchestrator(goal="test", client=MagicMock(), max_cycles=3)
    assert orch.max_cycles == 3


def test_orchestrator_is_done_empty():
    orch = Orchestrator(goal="test", client=MagicMock())
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_has_pending():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_has_failed():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="failed")]
    assert orch._is_done() is False


def test_orchestrator_get_tasks():
    orch = Orchestrator(goal="test", client=MagicMock())
    t = TaskModel(id="1", description="a")
    orch.tasks = [t]
    assert orch._get_tasks() == [t]


def test_orchestrator_incorporate_tasks_keeps_completed():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    plan_tasks = [TaskModel(id="2", description="b", status="pending")]
    orch._incorporate_tasks(plan_tasks)
    ids = {t.id for t in orch.tasks}
    assert "1" in ids  # completed task preserved
    assert "2" in ids  # new from plan


def test_orchestrator_get_ready_tasks_no_deps():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    ready = orch._get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "1"


def test_orchestrator_get_ready_tasks_deps_satisfied():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [
        TaskModel(id="1", description="a", status="pending", dependencies=["2"]),
        TaskModel(id="2", description="b", status="completed"),
    ]
    ready = orch._get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "1"


def test_orchestrator_get_ready_tasks_deps_unmet():
    orch = Orchestrator(goal="test", client=MagicMock())
    orch.tasks = [
        TaskModel(id="1", description="a", status="pending", dependencies=["2"]),
        TaskModel(id="2", description="b", status="pending"),
    ]
    ready = orch._get_ready_tasks()
    ready_ids = {t.id for t in ready}
    # Task 1 depends on task 2 which is still pending; it must not be ready.
    assert "1" not in ready_ids


def test_orchestrator_has_failed_dependency():
    orch = Orchestrator(goal="test", client=MagicMock())
    failed = TaskModel(id="2", description="b", status="failed")
    pending = TaskModel(id="1", description="a", status="pending", dependencies=["2"])
    orch.tasks = [pending, failed]
    assert orch._has_failed_dependency(pending) is True
    assert orch._has_failed_dependency(failed) is False


async def test_orchestrator_stops_at_max_cycles():
    """Orchestrator stops after max_cycles even if incomplete."""
    mock_client = AsyncMock()
    mock_client.settings = MagicMock()
    mock_client.settings.max_parallel_tasks = 5

    plan = Plan(tasks=[TaskModel(id="1", description="do stuff")], rationale="test")
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)

    orch = Orchestrator(goal="test", client=mock_client, max_cycles=2)
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(side_effect=Exception("boom"))
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["err"])
        )
        await orch.run()

    assert orch.cycles == 2


# ═══ PlannerAgent Tests ═══

def test_planner_prompt_has_json_shape():
    assert "tasks" in PLANNER_PROMPT
    assert "rationale" in PLANNER_PROMPT
    assert "id" in PLANNER_PROMPT


# ═══ WorkerAgent Tests ═══

def test_worker_prompt_has_operations():
    assert "operations" in WORKER_PROMPT
    assert '"write"' in WORKER_PROMPT
    assert '"edit"' in WORKER_PROMPT


def _make_mock_client(complete_return=None):
    """Helper to create a mock LLMClient for WorkerAgent tests."""
    mock = AsyncMock()
    mock.settings = MagicMock()
    mock.settings.worker_model = "test-model"
    if complete_return is not None:
        mock.complete = AsyncMock(return_value=complete_return)
    return mock


async def test_worker_handles_json_response():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Created test file",
        "operations": [
            {"action": "write", "path": "/tmp/furrow_test_file.txt", "content": "hello"}
        ]
    }))
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="Create file"), client=mock_client)
    result = await worker.run()

    assert "Created test file" in result
    assert "Wrote" in result
    mock_client.write_file.assert_awaited_once_with("/tmp/furrow_test_file.txt", "hello")


async def test_worker_handles_parse_error():
    mock_client = _make_mock_client("not valid json")
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=mock_client)
    result = await worker.run()

    assert "parse_error" in result


async def test_worker_edit_operation():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Edited file",
        "operations": [
            {"action": "edit", "path": "/tmp/furrow_edit.txt",
             "old_string": "old", "new_string": "new"}
        ]
    }))
    mock_client.read_file = AsyncMock(return_value="this is old text")
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="edit"), client=mock_client)
    result = await worker.run()

    assert "Edited file" in result
    mock_client.read_file.assert_awaited_once_with("/tmp/furrow_edit.txt")
    mock_client.write_file.assert_awaited_once_with("/tmp/furrow_edit.txt", "this is new text")


async def test_worker_read_operation():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Read context",
        "operations": [
            {"action": "read", "path": "/tmp/furrow_read.txt"}
        ]
    }))
    mock_client.read_file = AsyncMock(return_value="file contents")

    worker = WorkerAgent(task=TaskModel(id="1", description="read"), client=mock_client)
    result = await worker.run()

    assert "Read context" in result
    mock_client.read_file.assert_awaited_once_with("/tmp/furrow_read.txt")


async def test_worker_unknown_action():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Did stuff",
        "operations": [
            {"action": "fly", "path": "/tmp/some_file.txt"}
        ]
    }))
    mock_client.write_file = AsyncMock()
    mock_client.read_file = AsyncMock(return_value="content")

    worker = WorkerAgent(task=TaskModel(id="1", description="unknown action"), client=mock_client)
    result = await worker.run()

    assert "Unknown action" in result


async def test_worker_multiple_operations():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Multiple ops",
        "operations": [
            {"action": "write", "path": "/tmp/file_a.txt", "content": "AAA"},
            {"action": "write", "path": "/tmp/file_b.txt", "content": "BBB"},
        ]
    }))
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="multi"), client=mock_client)
    result = await worker.run()

    assert "Multiple ops" in result
    assert mock_client.write_file.await_count == 2
    mock_client.write_file.assert_any_await("/tmp/file_a.txt", "AAA")
    mock_client.write_file.assert_any_await("/tmp/file_b.txt", "BBB")


async def test_worker_empty_operations():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Nothing to do",
        "operations": []
    }))
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="noop"), client=mock_client)
    result = await worker.run()

    assert "Nothing to do" in result
    assert "Wrote" not in result
    mock_client.write_file.assert_not_awaited()


async def test_worker_edit_old_string_not_found():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Edit attempt",
        "operations": [
            {"action": "edit", "path": "/tmp/furrow_missing.txt",
             "old_string": "not found here", "new_string": "replaced"}
        ]
    }))
    mock_client.read_file = AsyncMock(return_value="nothing relevant")
    mock_client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="edit missing"), client=mock_client)
    result = await worker.run()

    assert "Edit attempt" in result
    assert "old_string not found" in result
    mock_client.write_file.assert_not_awaited()


async def test_worker_write_file_error():
    mock_client = _make_mock_client(json.dumps({
        "summary": "Write fail",
        "operations": [
            {"action": "write", "path": "/tmp/furrow_fail.txt", "content": "data"}
        ]
    }))
    mock_client.write_file = AsyncMock(side_effect=OSError("disk full"))

    worker = WorkerAgent(task=TaskModel(id="1", description="write"), client=mock_client)
    result = await worker.run()

    assert "Write fail" in result
    assert "Failed to write" in result


# ═══ TesterAgent Tests ═══

def test_tester_prompt_has_json_fields():
    assert "passed" in TESTER_PROMPT
    assert "summary" in TESTER_PROMPT
    assert "failures" in TESTER_PROMPT


async def test_tester_handles_json_response():
    mock_client = AsyncMock()
    mock_client.settings = MagicMock()
    mock_client.settings.tester_model = "test-model"
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": []
    }))

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output")

    result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.summary == "All tests passed"


async def test_tester_handles_json_failure():
    mock_client = AsyncMock()
    mock_client.settings = MagicMock()
    mock_client.settings.tester_model = "test-model"
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": False,
        "summary": "2 failures",
        "failures": ["test_a failed", "test_b failed"]
    }))

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="failures...")

    result = await tester.run("test goal", [])
    assert result.passed is False
    assert len(result.failures) == 2


async def test_tester_handles_parse_error_fallback():
    mock_client = AsyncMock()
    mock_client.settings = MagicMock()
    mock_client.settings.tester_model = "test-model"
    mock_client.complete = AsyncMock(return_value="The tests passed")

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output")

    result = await tester.run("test goal", [])
    assert result.passed is True
    assert "passed" in result.summary.lower()


async def test_tester_handles_test_runner_exception():
    mock_client = AsyncMock()
    mock_client.settings = MagicMock()
    mock_client.settings.tester_model = "test-model"
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "ok",
        "failures": []
    }))

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("runner explosion"))

    result = await tester.run("test goal", [])
    assert result.passed is False
    assert "runner explosion" in result.summary


# ═══ LLMClient Tests ═══

def test_llm_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


async def test_llm_complete_unsupported_provider():
    mock_settings = MagicMock()
    mock_settings.provider = "unsupported"
    mock_settings.model = "test-model"
    mock_settings.anthropic_api_key = None
    mock_settings.openai_api_key = None
    mock_settings.ollama_base_url = "http://localhost:11434"

    client = LLMClient(settings=mock_settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")


async def test_llm_read_and_write_file(tmp_path):
    mock_settings = MagicMock()
    client = LLMClient(settings=mock_settings)

    fpath = tmp_path / "test_write.txt"
    await client.write_file(str(fpath), "hello world")

    content = await client.read_file(str(fpath))
    assert content == "hello world"


def test_llm_list_files_existing_dir(tmp_path):
    (tmp_path / "a.py").write_text("content")
    (tmp_path / "b.py").write_text("content")

    mock_settings = MagicMock()
    client = LLMClient(settings=mock_settings)

    files = client.list_files(str(tmp_path))
    assert len(files) == 2
    assert "a.py" in files
    assert "b.py" in files


def test_llm_list_files_nonexistent_dir():
    mock_settings = MagicMock()
    client = LLMClient(settings=mock_settings)

    files = client.list_files("/nonexistent/path/that/does/not/exist")
    assert files == []
