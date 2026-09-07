from __future__ import annotations

import json

import pytest

from furrow.config import Plan, TaskModel, TestResult, configure_logging, get_logger


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_defaults():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_get_logger_returns_structlog_logger():
    logger = get_logger()
    assert logger is not None


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("DEBUG")  # second call should be a no-op
    logger = get_logger()
    assert logger is not None


# ---------------------------------------------------------------------------
# Orchestrator logic tests (no LLM calls — pure logic)
# ---------------------------------------------------------------------------

from furrow.core.orchestrator import Orchestrator


def make_task(status: str = "pending", result: str | None = None) -> TaskModel:
    return TaskModel(id="1", description="test task", status=status, result=result)


def test_is_done_no_plan():
    """With no plan, orchestrator should report done."""
    orch = Orchestrator(goal="test")
    orch.plan = None
    assert orch._is_done() is True


def test_is_done_empty_plan():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[], rationale="empty")
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = Orchestrator(goal="test")
    t = make_task(status="completed")
    orch.plan = Plan(tasks=[t], rationale="done")
    assert orch._is_done() is True


def test_is_done_has_failure():
    orch = Orchestrator(goal="test")
    t = make_task(status="failed")
    orch.plan = Plan(tasks=[t], rationale="failed")
    assert orch._is_done() is False


def test_is_done_pending_tasks():
    orch = Orchestrator(goal="test")
    t = make_task(status="pending")
    orch.plan = Plan(tasks=[t], rationale="in progress")
    assert orch._is_done() is False


def test_get_tasks_returns_plan_tasks():
    orch = Orchestrator(goal="test")
    t = make_task()
    orch.plan = Plan(tasks=[t], rationale="test")
    assert orch._get_tasks() == [t]


def test_get_tasks_none_when_no_plan():
    orch = Orchestrator(goal="test")
    orch.plan = None
    assert orch._get_tasks() == []


# ---------------------------------------------------------------------------
# Worker file-writing tests
# ---------------------------------------------------------------------------

from furrow.agents.worker import WorkerAgent


@pytest.mark.asyncio
async def test_worker_parses_and_writes_files(tmp_path):
    """Worker should parse JSON response and write files to disk."""
    task = TaskModel(id="t1", description="create a file")
    worker = WorkerAgent(task=task)

    # Override write_file to capture what would be written.
    written: dict[str, str] = {}

    async def fake_write(path, content):
        written[str(path)] = content

    worker.client.write_file = fake_write  # type: ignore[method-assign]

    response = json.dumps({
        "files": [
            {"path": "src/main.py", "content": "print('hello')"},
            {"path": "tests/test_main.py", "content": "def test_main(): pass"},
        ],
        "summary": "Created two files",
    })

    result = await worker._apply_result(response)
    assert "Created two files" in result
    assert "src/main.py" in written
    assert written["src/main.py"] == "print('hello')"
    assert "tests/test_main.py" in written


@pytest.mark.asyncio
async def test_worker_handles_invalid_json():
    task = TaskModel(id="t1", description="broken")
    worker = WorkerAgent(task=task)
    result = await worker._apply_result("not json at all")
    assert "Failed to parse" in result


@pytest.mark.asyncio
async def test_worker_handles_empty_files_list():
    task = TaskModel(id="t1", description="nothing")
    worker = WorkerAgent(task=task)
    response = json.dumps({"files": [], "summary": "No changes needed"})
    result = await worker._apply_result(response)
    assert "No changes needed" in result


@pytest.mark.asyncio
async def test_worker_strips_markdown_code_fences():
    """_parse_json should strip ```json ... ``` fences."""
    task = TaskModel(id="t1", description="fence test")
    worker = WorkerAgent(task=task)
    fenced = '```json\n{"files": [], "summary": "ok"}\n```'
    result = await worker._apply_result(fenced)
    assert "ok" in result


@pytest.mark.asyncio
async def test_worker_returns_summary_for_non_dict_json():
    """If LLM returns valid JSON that isn't an object, handle gracefully."""
    task = TaskModel(id="t1", description="non-dict")
    worker = WorkerAgent(task=task)
    result = await worker._apply_result("42")
    assert "not a JSON object" in result


# ---------------------------------------------------------------------------
# Tester tests
# ---------------------------------------------------------------------------

from furrow.agents.tester import TesterAgent


@pytest.mark.asyncio
async def test_tester_returns_runtime_error_when_no_runner():
    """When no test runner is found, _run_tests raises RuntimeError."""
    tester = TesterAgent()

    # Patch _run_tests to raise RuntimeError simulating no test runner.
    async def fake_run_tests():
        raise RuntimeError("No test runner found.")

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is False
    assert "No test runner found" in result.summary


@pytest.mark.asyncio
async def test_tester_parses_json_response():
    tester = TesterAgent()

    async def fake_run_tests():
        return "all passed"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    # Patch the LLM client complete method.
    async def fake_complete(prompt, system="", model=None):
        return json.dumps({"passed": True, "summary": "All tests passed", "failures": []})

    tester.client.complete = fake_complete  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_falls_back_on_invalid_json():
    tester = TesterAgent()

    async def fake_run_tests():
        return "test failed"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    async def fake_complete(prompt, system="", model=None):
        return "tests passed"

    tester.client.complete = fake_complete  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is True  # "passed" found in response
    assert "tests passed" in result.summary
