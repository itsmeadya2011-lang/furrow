import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from furrow.agents.tester import _TestRunResult, TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_max_cycles_default():
    s = Settings()
    assert s.max_cycles == 25


def test_settings_max_cycles_zero_allowed():
    s = Settings(max_cycles=0)
    assert s.max_cycles == 0


def test_settings_max_cycles_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)


def test_settings_max_cycles_rejects_too_high():
    with pytest.raises(ValidationError):
        Settings(max_cycles=1001)


def test_settings_get_anthropic_key():
    s = Settings(anthropic_api_key=SecretStr("my-secret-key"))
    assert s.get_anthropic_key() == "my-secret-key"


def test_settings_get_anthropic_key_none():
    s = Settings()
    assert s.get_anthropic_key() is None


def test_settings_get_openai_key():
    s = Settings(openai_api_key=SecretStr("openai-secret"))
    assert s.get_openai_key() == "openai-secret"


def test_settings_get_openai_key_none():
    s = Settings()
    assert s.get_openai_key() is None


def test_settings_get_web_token():
    s = Settings(web_token=SecretStr("web-token-secret"))
    assert s.get_web_token() == "web-token-secret"


def test_settings_get_web_token_none():
    s = Settings()
    assert s.get_web_token() is None


def test_settings_model_dump_does_not_expose_secrets():
    s = Settings(
        anthropic_api_key=SecretStr("super-secret"),
        openai_api_key=SecretStr("openai-secret"),
        web_token=SecretStr("web-token"),
    )
    dumped = s.model_dump(mode="json")
    dumped_str = json.dumps(dumped)
    assert "super-secret" not in dumped_str
    assert "openai-secret" not in dumped_str
    assert "web-token" not in dumped_str


def test_llmclient_resolve_relative_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = Settings(workspace=workspace)
    client = LLMClient(settings=s)
    resolved = client._resolve_path("src/test.py")
    assert resolved == (workspace / "src/test.py").resolve()


def test_llmclient_resolve_absolute_path_within_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = Settings(workspace=workspace)
    client = LLMClient(settings=s)
    target = workspace / "src" / "test.py"
    resolved = client._resolve_path(str(target))
    assert resolved == target.resolve()


def test_llmclient_reject_relative_traversal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = Settings(workspace=workspace)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError):
        client._resolve_path("../etc/passwd")


def test_llmclient_reject_absolute_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    s = Settings(workspace=workspace)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError):
        client._resolve_path("/etc/passwd")


class FakeWorkerClient:
    def __init__(self):
        self.writes: list[tuple[str, str]] = []
        self.reads: list[str] = []
        self.settings = MagicMock()
        self.settings.worker_model = "test-model"
        self.read_file = AsyncMock(return_value="old content here")
        self.write_file = AsyncMock()
        self.complete = AsyncMock(return_value='{"summary": "done", "changes": []}')


@pytest.mark.asyncio
async def test_worker_structured_write():
    client = FakeWorkerClient()
    task = TaskModel(id="1", description="test", files=["file.txt"])
    worker = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(
        return_value='{"summary": "write file", "changes": [{"action": "write", "path": "file.txt", "content": "hello world"}]}'
    )
    result = await worker.run()
    assert len(client.writes) == 1
    assert client.writes[0][0] == "file.txt"
    assert client.writes[0][1] == "hello world"
    assert "write file" in result


@pytest.mark.asyncio
async def test_worker_structured_edit():
    client = FakeWorkerClient()
    task = TaskModel(id="1", description="test", files=["file.txt"])
    worker = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(
        return_value='{"summary": "edit file", "changes": [{"action": "edit", "path": "file.txt", "old_string": "old", "new_string": "new"}]}'
    )
    result = await worker.run()
    assert len(client.writes) == 1
    assert client.writes[0][0] == "file.txt"
    assert "edit file" in result


@pytest.mark.asyncio
async def test_worker_structured_no_op_raises():
    client = FakeWorkerClient()
    task = TaskModel(id="1", description="test", files=["file.txt"])
    worker = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(return_value='{"summary": "done", "changes": []}')
    with pytest.raises(RuntimeError):
        await worker.run()


@pytest.mark.asyncio
async def test_worker_plain_text_no_change_raises():
    client = FakeWorkerClient()
    task = TaskModel(id="1", description="test", files=["file.txt"])
    worker = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(return_value="No changes needed.")
    with pytest.raises(RuntimeError, match="LLM indicated no changes"):
        await worker.run()


@pytest.mark.asyncio
async def test_worker_partial_failure_raises_after_writes():
    client = FakeWorkerClient()
    task = TaskModel(id="1", description="test", files=["file1.txt", "file2.txt"])
    worker = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "summary": "partial",
                "changes": [
                    {"action": "write", "path": "file1.txt", "content": "ok"},
                    {"action": "write", "path": "file2.txt", "content": "fail"},
                ],
            }
        )
    )
    client.write_file = AsyncMock(side_effect=[None, RuntimeError("write failed")])
    with pytest.raises(RuntimeError, match="Partial failure"):
        await worker.run()
    assert len(client.writes) == 1
    assert client.writes[0][0] == "file1.txt"


def test_tester_parser_valid_json_codeblock():
    tester = TesterAgent(client=None)
    result = tester._parse_test_result(
        '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    )
    assert result["passed"] is True
    assert result["summary"] == "ok"
    assert result["failures"] == []


def test_tester_parser_valid_json_plain():
    tester = TesterAgent(client=None)
    result = tester._parse_test_result(
        '{"passed": false, "summary": "fail", "failures": ["err1"]}'
    )
    assert result["passed"] is False
    assert result["summary"] == "fail"
    assert result["failures"] == ["err1"]


def test_tester_parser_invalid_json_fallback():
    tester = TesterAgent(client=None)
    result = tester._parse_test_result("not json at all")
    assert result["passed"] is False
    assert result["summary"] == "not json at all"
    assert result["failures"] == ["not json at all"]


@pytest.mark.asyncio
async def test_tester_parser_non_bool_passed_coerced():
    tester = TesterAgent(client=None)
    result = tester._parse_test_result(
        '{"passed": "yes", "summary": "ok", "failures": []}'
    )
    assert result["passed"] is False


@pytest.mark.asyncio
async def test_tester_no_runner_result():
    tester = TesterAgent(client=None)
    mock_run = AsyncMock(
        return_value=_TestRunResult(
            output="No test runner found.",
            stdout="",
            stderr="",
            returncode=-1,
            runner_found=False,
            timed_out=False,
        )
    )
    with patch.object(tester, "_run_tests", mock_run):
        result = await tester.run("my goal", [])
        assert result.passed is False
        assert "No test runner" in result.summary


@pytest.mark.asyncio
async def test_orchestrator_task_registry():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 1
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test goal", client=fake_client)
    assert len(orc._tasks) == 0
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", files=[], dependencies=[])],
        rationale="test",
    )
    orc._merge_plan(plan)
    assert "1" in orc._tasks
    assert orc._tasks["1"].description == "do thing"
    assert orc._tasks["1"].status == "pending"


@pytest.mark.asyncio
async def test_orchestrator_task_registry_preserves_existing():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 1
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test goal", client=fake_client)
    orc._tasks["1"] = TaskModel(id="1", description="old", status="completed")
    plan = Plan(
        tasks=[TaskModel(id="1", description="updated", files=[], dependencies=[])],
        rationale="test",
    )
    orc._merge_plan(plan)
    assert "1" in orc._tasks
    assert orc._tasks["1"].description == "updated"
    assert orc._tasks["1"].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_new_task_forced_pending_even_if_planner_says_completed():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 1
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test goal", client=fake_client)
    plan = Plan(
        tasks=[TaskModel(id="1", description="new task", files=[], dependencies=[], status="completed", result="done")],
        rationale="test",
    )
    orc._merge_plan(plan)
    assert "1" in orc._tasks
    assert orc._tasks["1"].status == "pending"
    assert orc._tasks["1"].result is None


@pytest.mark.asyncio
async def test_orchestrator_duplicate_plan_ids_skipped():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 1
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test goal", client=fake_client)
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="first", files=[], dependencies=[]),
            TaskModel(id="1", description="second", files=[], dependencies=[]),
            TaskModel(id="2", description="third", files=[], dependencies=[]),
        ],
        rationale="test",
    )
    orc._merge_plan(plan)
    assert len(orc._tasks) == 2
    assert orc._tasks["1"].description == "first"
    assert orc._tasks["2"].description == "third"


@pytest.mark.asyncio
async def test_orchestrator_is_done_true():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 0
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test", client=fake_client)
    orc._last_test_passed = True
    orc._tasks["1"] = TaskModel(
        id="1", description="done", status="completed", result="ok"
    )
    assert orc._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_false_pending_task():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 0
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test", client=fake_client)
    orc._last_test_passed = True
    orc._tasks["1"] = TaskModel(id="1", description="pending", status="pending")
    assert orc._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_false_not_tested():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 0
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test", client=fake_client)
    orc._tasks["1"] = TaskModel(
        id="1", description="done", status="completed", result="ok"
    )
    assert orc._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_false_failed_task():
    fake_client = MagicMock()
    fake_client.settings.max_cycles = 0
    fake_client.settings.max_parallel_tasks = 5
    orc = Orchestrator(goal="test", client=fake_client)
    orc._last_test_passed = True
    orc._tasks["1"] = TaskModel(
        id="1", description="failed", status="failed", result="error"
    )
    assert orc._is_done() is False