import pytest
import tempfile
from pathlib import Path
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.state import StateManager
from furrow.agents.worker import WorkerAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 10
    assert s.log_level == "INFO"
    assert s.json_logs is False


def test_state_manager_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = StateManager(workspace, "test goal")
        
        tasks = [
            TaskModel(id="1", description="task 1", status="completed", result="done"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        manager.save(tasks, 3, "updated goal")
        
        loaded = manager.load()
        assert loaded is not None
        assert loaded["goal"] == "updated goal"
        assert loaded["original_goal"] == "test goal"
        assert loaded["cycles"] == 3
        assert len(loaded["tasks"]) == 2
        
        restored = manager.restore_tasks(loaded)
        assert len(restored) == 2
        assert restored[0].id == "1"
        assert restored[0].status == "completed"
        assert restored[1].id == "2"
        assert restored[1].status == "pending"


def test_state_manager_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = StateManager(workspace, "test goal")
        
        tasks = [TaskModel(id="1", description="task 1")]
        manager.save(tasks, 1, "goal")
        assert manager.state_file.exists()
        
        manager.clear()
        assert not manager.state_file.exists()


def test_state_manager_load_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = StateManager(workspace, "test goal")
        loaded = manager.load()
        assert loaded is None


def test_worker_parse_file_changes():
    """Test that worker correctly parses file changes from response."""
    task = TaskModel(id="1", description="Create a file", files=["test.py"])
    worker = WorkerAgent(task=task)
    
    response = """```file:test.py
print("hello world")
```"""
    
    changes = worker._parse_file_changes(response)
    assert "test.py" in changes
    assert changes["test.py"] == 'print("hello world")'


def test_worker_parse_multiple_files():
    """Test parsing multiple file changes."""
    task = TaskModel(id="1", description="Create files", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task)
    
    response = """```file:a.py
content a
```
```file:b.py
content b
```"""
    
    changes = worker._parse_file_changes(response)
    assert len(changes) == 2
    assert changes["a.py"] == "content a"
    assert changes["b.py"] == "content b"


def test_worker_summarize_changes():
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task)
    
    changes = {"a.py": "content", "b.py": "content"}
    summary = worker._summarize_changes(changes, "response")
    assert "Modified 2 file(s)" in summary
    assert "a.py" in summary
    assert "b.py" in summary
    
    # Test empty changes returns truncated response
    summary = worker._summarize_changes({}, "some response text")
    assert summary == "some response text"
