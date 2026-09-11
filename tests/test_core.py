from __future__ import annotations

import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_task_model_with_files_and_deps():
    t = TaskModel(
        id="2",
        description="implement auth",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["1"],
    )
    assert len(t.files) == 2
    assert t.dependencies == ["1"]


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


# Orchestrator tests
@pytest.mark.asyncio
async def test_orchestrator_is_done_no_plan():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is True  # No plan yet


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ],
        rationale="test",
    )
    orch._current_plan = plan
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="failed"),
        ],
        rationale="test",
    )
    orch._current_plan = plan
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_pending_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="pending"),
        ],
        rationale="test",
    )
    orch._current_plan = plan
    assert orch._is_done() is False


# Worker tests
@pytest.mark.asyncio
async def test_worker_summarize_changes():
    from furrow.agents.worker import WorkerAgent

    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = """FILE: src/main.py
```
print("hello")
```

FILE: tests/test_main.py
```
def test_main():
    pass
```
"""
    summary = worker._summarize_changes(response)
    assert "src/main.py" in summary
    assert "tests/test_main.py" in summary


@pytest.mark.asyncio
async def test_worker_summarize_no_files():
    from furrow.agents.worker import WorkerAgent

    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    summary = worker._summarize_changes("just some text")
    assert "no file changes" in summary.lower()


# LLMClient tests
def test_llm_client_init():
    from furrow.llm import LLMClient

    client = LLMClient()
    assert client.settings is not None


def test_llm_client_list_files(tmp_path):
    from furrow.llm import LLMClient

    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.txt").write_text("world")
    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "file1.txt" in files
    assert "file2.txt" in files


@pytest.mark.asyncio
async def test_llm_client_write_and_read_file(tmp_path):
    from furrow.llm import LLMClient

    client = LLMClient()
    test_file = tmp_path / "test.txt"
    await client.write_file(test_file, "hello world")
    content = await client.read_file(test_file)
    assert content == "hello world"


# Logging tests
def test_logging_config():
    from furrow.logging import configure_logging, get_logger

    configure_logging("DEBUG")
    logger = get_logger("test")
    assert logger is not None


# State persistence tests
@pytest.mark.asyncio
async def test_state_save_and_load(tmp_path):
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test goal")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task1", status="completed")],
        rationale="test",
    )
    orch._current_plan = plan
    orch.cycles = 3

    state_file = tmp_path / "state.json"
    orch.save_state(state_file)
    assert state_file.exists()

    orch2 = Orchestrator(goal="different")
    assert orch2.load_state(state_file) is True
    assert orch2.goal == "test goal"
    assert orch2.cycles == 3
    assert orch2._current_plan is not None
    assert len(orch2._current_plan.tasks) == 1


@pytest.mark.asyncio
async def test_state_load_nonexistent(tmp_path):
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch.load_state(tmp_path / "nonexistent.json") is False