import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_task_model_with_files_and_deps():
    task = TaskModel(
        id="1",
        description="test task",
        files=["src/a.py", "src/b.py"],
        dependencies=["2", "3"],
    )
    assert task.files == ["src/a.py", "src/b.py"]
    assert task.dependencies == ["2", "3"]


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
        TaskModel(id="3", description="task 3", dependencies=["1", "2"]),
    ]
    plan = Plan(tasks=tasks, rationale="sequential tasks")
    assert len(plan.tasks) == 3
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[2].dependencies == ["1", "2"]


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    settings = Settings(max_cycles=5, log_level="DEBUG")
    orchestrator = Orchestrator(goal="Test goal", settings=settings)
    
    assert orchestrator.goal == "Test goal"
    assert orchestrator.settings.max_cycles == 5
    assert orchestrator.cycles == 0
    assert orchestrator.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_state_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        settings = Settings(state_file=state_file)
        
        orchestrator = Orchestrator(goal="Test goal", settings=settings)
        orchestrator.cycles = 3
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed", result="Done"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        
        await orchestrator._save_state()
        
        # Verify file was created
        assert state_file.exists()
        
        # Load and verify
        with open(state_file) as f:
            state = json.load(f)
        
        assert state["goal"] == "Test goal"
        assert state["cycles"] == 3
        assert len(state["tasks"]) == 2
        assert state["tasks"][0]["status"] == "completed"
        assert state["tasks"][1]["status"] == "pending"


@pytest.mark.asyncio
async def test_orchestrator_load_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / ".furrow_state.json"
        settings = Settings(state_file=state_file)
        
        # Create state file
        state_data = {
            "goal": "Loaded goal",
            "cycles": 5,
            "tasks": [
                {"id": "1", "description": "Task 1", "files": [], "dependencies": [], "status": "completed", "result": "Done"},
                {"id": "2", "description": "Task 2", "files": [], "dependencies": [], "status": "failed", "result": "Error"},
            ],
            "timestamp": "2024-01-01T00:00:00",
        }
        with open(state_file, "w") as f:
            json.dump(state_data, f)
        
        orchestrator = Orchestrator(goal="Original goal", settings=settings)
        await orchestrator._load_state()
        
        assert orchestrator.goal == "Loaded goal"
        assert orchestrator.cycles == 5
        assert len(orchestrator.all_tasks) == 2
        assert orchestrator.all_tasks[0].status == "completed"
        assert orchestrator.all_tasks[1].status == "failed"


@pytest.mark.asyncio
async def test_orchestrator_get_runnable_tasks():
    settings = Settings()
    orchestrator = Orchestrator(goal="Test", settings=settings)
    
    # Pre-populate with completed task
    orchestrator.all_tasks = [
        TaskModel(id="1", description="Task 1", status="completed"),
    ]
    
    new_tasks = [
        TaskModel(id="1", description="Task 1"),  # Already completed
        TaskModel(id="2", description="Task 2", dependencies=["1"]),  # Dep on completed
        TaskModel(id="3", description="Task 3", dependencies=["4"]),  # Dep on missing
        TaskModel(id="4", description="Task 4", dependencies=[]),  # No deps
    ]
    
    runnable = orchestrator._get_runnable_tasks(new_tasks)
    
    # Should get task 2 (dep met) and task 4 (no deps)
    # Not task 1 (completed) or task 3 (dep not met)
    assert len(runnable) == 2
    assert runnable[0].id == "2"
    assert runnable[1].id == "4"


@pytest.mark.asyncio
async def test_orchestrator_is_done():
    settings = Settings()
    orchestrator = Orchestrator(goal="Test", settings=settings)
    
    # Empty tasks - not done
    assert orchestrator._is_done() is False
    
    # All completed - done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="Task 1", status="completed"),
        TaskModel(id="2", description="Task 2", status="completed"),
    ]
    assert orchestrator._is_done() is True
    
    # Has failed - not done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="Task 1", status="completed"),
        TaskModel(id="2", description="Task 2", status="failed"),
    ]
    assert orchestrator._is_done() is False
    
    # Has pending - not done
    orchestrator.all_tasks = [
        TaskModel(id="1", description="Task 1", status="completed"),
        TaskModel(id="2", description="Task 2", status="pending"),
    ]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_compact_history():
    settings = Settings()
    orchestrator = Orchestrator(goal="Test", settings=settings)
    
    # Create 25 completed tasks
    orchestrator.all_tasks = [
        TaskModel(id=str(i), description=f"Task {i}", status="completed")
        for i in range(25)
    ]
    
    orchestrator._compact_history()
    
    # Should have 1 summary + 10 recent = 11 tasks
    assert len(orchestrator.all_tasks) == 11
    assert orchestrator.all_tasks[0].id == "summary"
    assert "Previous 15 tasks completed" in orchestrator.all_tasks[0].description
    assert orchestrator.all_tasks[1].id == "15"
    assert orchestrator.all_tasks[-1].id == "24"


@pytest.mark.asyncio
async def test_orchestrator_compact_history_preserves_pending_failed():
    settings = Settings()
    orchestrator = Orchestrator(goal="Test", settings=settings)
    
    # Create 25 tasks: 20 completed, 3 pending, 2 failed
    orchestrator.all_tasks = [
        TaskModel(id=str(i), description=f"Task {i}", status="completed")
        for i in range(20)
    ] + [
        TaskModel(id="20", description="Pending 1", status="pending"),
        TaskModel(id="21", description="Pending 2", status="pending"),
        TaskModel(id="22", description="Failed 1", status="failed"),
    ]
    
    orchestrator._compact_history()
    
    # Should have 1 summary + 10 recent completed + 3 pending/failed = 14 tasks
    assert len(orchestrator.all_tasks) == 14
    assert orchestrator.all_tasks[0].id == "summary"
    # Last 10 completed
    assert orchestrator.all_tasks[1].id == "10"
    assert orchestrator.all_tasks[10].id == "19"
    # Then pending/failed
    assert orchestrator.all_tasks[11].id == "20"
    assert orchestrator.all_tasks[12].id == "21"
    assert orchestrator.all_tasks[13].id == "22"


@pytest.mark.asyncio
async def test_llm_client_read_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = LLMClient()
        test_file = Path(tmpdir) / "test.txt"
        test_content = "Hello, World!"
        
        await client.write_file(test_file, test_content)
        content = await client.read_file(test_file)
        
        assert content == test_content


@pytest.mark.asyncio
async def test_llm_client_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = LLMClient()
        test_dir = Path(tmpdir)
        (test_dir / "a.txt").write_text("a")
        (test_dir / "b.txt").write_text("b")
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "c.txt").write_text("c")
        
        files = client.list_files(test_dir)
        
        assert "a.txt" in files
        assert "b.txt" in files
        assert "subdir/c.txt" in files


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Settings.__fields__["provider"].default
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.workspace == Path.cwd()


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    
    settings = Settings()
    assert settings.max_cycles == 10
    assert settings.log_level == "DEBUG"


@pytest.mark.asyncio
async def test_worker_agent_returns_summary():
    """Test that worker agent returns a summary when JSON parsing fails."""
    mock_client = AsyncMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="Just a plain text response")
    mock_client.settings = type("S", (), {"worker_model": "test-model"})()
    
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel
    
    task = TaskModel(id="1", description="Test task", files=["test.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    
    result = await worker.run()
    
    assert result == "Just a plain text response"


@pytest.mark.asyncio
async def test_tester_agent_run_tests_no_runner():
    mock_client = AsyncMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "No test runner found", "failures": []}')
    mock_client.settings = type("S", (), {"tester_model": "test-model", "worker_model": "test-model"})()
    
    from furrow.agents.tester import TesterAgent
    from furrow.config import TaskModel
    
    tester = TesterAgent(client=mock_client)
    result = await tester.run("Test goal", [])
    
    assert result.passed is True
    assert "No test runner" in result.summary


def test_settings_state_file_default():
    settings = Settings()
    assert settings.state_file == Path.cwd() / ".furrow_state.json"


@pytest.mark.asyncio
async def test_orchestrator_update_task_in_history():
    settings = Settings()
    orchestrator = Orchestrator(goal="Test", settings=settings)
    
    # Add initial task
    orchestrator.all_tasks = [
        TaskModel(id="1", description="Task 1", status="pending"),
    ]
    
    # Update with completed
    updated_task = TaskModel(id="1", description="Task 1", status="completed", result="Done")
    orchestrator._update_task_in_history(updated_task)
    
    assert len(orchestrator.all_tasks) == 1
    assert orchestrator.all_tasks[0].status == "completed"
    assert orchestrator.all_tasks[0].result == "Done"
    
    # Add new task
    new_task = TaskModel(id="2", description="Task 2", status="completed", result="Also done")
    orchestrator._update_task_in_history(new_task)
    
    assert len(orchestrator.all_tasks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])