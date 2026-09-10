import json
import pytest
from furrow.config import Plan, TaskModel, TestResult, Provider, Settings
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


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    assert task.files == ["a.py", "b.py"]


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", files=["src/x.py"]),
    ]
    p = Plan(tasks=tasks, rationale="two tasks")
    assert len(p.tasks) == 2
    assert p.tasks[1].files == ["src/x.py"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["error 1", "error 2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_worker_extract_tool_calls_json_block():
    """Test extracting tool calls from JSON code blocks."""
    worker = WorkerAgent(TaskModel(id="1", description="test"))
    
    response = '''
Here is my plan:
```json
{"tool": "read_file", "args": {"path": "test.py"}}
```
And another:
```json
{"tool": "write_file", "args": {"path": "out.py", "content": "print('hello')"}}
```
'''
    calls = worker._extract_tool_calls(response)
    assert len(calls) == 2
    assert calls[0]["tool"] == "read_file"
    assert calls[0]["args"]["path"] == "test.py"
    assert calls[1]["tool"] == "write_file"
    assert calls[1]["args"]["path"] == "out.py"


def test_worker_extract_tool_calls_inline():
    """Test extracting inline tool calls."""
    worker = WorkerAgent(TaskModel(id="1", description="test"))
    
    response = '{"tool": "read_file", "args": {"path": "test.py"}} some text {"tool": "list_files", "args": {"directory": "."}}'
    calls = worker._extract_tool_calls(response)
    assert len(calls) == 2
    assert calls[0]["tool"] == "read_file"
    assert calls[1]["tool"] == "list_files"


def test_worker_extract_tool_calls_mixed():
    """Test extracting mixed tool calls."""
    worker = WorkerAgent(TaskModel(id="1", description="test"))
    
    response = '''
First I'll read the file:
```json
{"tool": "read_file", "args": {"path": "main.py"}}
```
Then write: {"tool": "write_file", "args": {"path": "main.py", "content": "new"}}
'''
    calls = worker._extract_tool_calls(response)
    assert len(calls) == 2
    assert calls[0]["tool"] == "read_file"
    assert calls[1]["tool"] == "write_file"


def test_worker_no_tool_calls():
    """Test response with no tool calls returns empty list."""
    worker = WorkerAgent(TaskModel(id="1", description="test"))
    
    response = "I'm done with the task. Here's the summary."
    calls = worker._extract_tool_calls(response)
    assert calls == []


def test_plan_model_dump():
    """Test Plan model serialization."""
    p = Plan(
        tasks=[TaskModel(id="1", description="test", files=["a.py"])],
        rationale="test rationale"
    )
    data = p.model_dump()
    assert data["tasks"][0]["id"] == "1"
    assert data["tasks"][0]["files"] == ["a.py"]
    assert data["rationale"] == "test rationale"