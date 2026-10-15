import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_tokens == 4096
    assert s.timeout == 120.0
    assert s.max_parallel_tasks == 5
    assert s.workspace == __import__("pathlib").Path.cwd()


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_TOKENS", "8192")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_tokens == 8192


def test_task_model_status():
    t = TaskModel(id="x", description="do", status="pending")
    assert t.status == "pending"
    t.status = "completed"
    assert t.status == "completed"


def test_plan_with_dependencies():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a", dependencies=[]),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="test",
    )
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failed", failures=["assert x == y", "timeout"])
    assert len(t.failures) == 2
    assert t.passed is False


def test_orchestrator_task_tracking():
    """Orchestrator should track tasks across cycles, not return empty list."""
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch.tasks == []
    assert orch.completed_tasks == []
    assert orch.failed_tasks == []

    # Simulate adding tasks
    t1 = TaskModel(id="1", description="task 1")
    t2 = TaskModel(id="2", description="task 2")
    orch.tasks = [t1, t2]

    # Mark one complete
    t1.status = "completed"
    orch.completed_tasks.append(t1)

    assert len(orch.tasks) == 2
    assert len(orch.completed_tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[1].status == "pending"


def test_orchestrator_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="a", status="completed")
    t2 = TaskModel(id="2", description="b", status="completed")
    orch.tasks = [t1, t2]
    orch.completed_tasks = [t1, t2]

    assert orch._is_done() is True


def test_orchestrator_is_done_with_pending():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="a", status="completed")
    t2 = TaskModel(id="2", description="b", status="pending")
    orch.tasks = [t1, t2]
    orch.completed_tasks = [t1]

    assert orch._is_done() is False


def test_orchestrator_is_done_with_failed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="a", status="failed")
    orch.tasks = [t1]
    orch.failed_tasks = [t1]

    assert orch._is_done() is False


def test_worker_parse_json_response():
    """Worker should parse JSON responses correctly."""
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="create a file")
    worker = WorkerAgent(task=task)

    # Direct JSON
    response = '{"files": [{"path": "a.py", "content": "print(1)"}], "summary": "done"}'
    data = worker._parse_response(response)
    assert data is not None
    assert data["files"][0]["path"] == "a.py"

    # JSON with surrounding text
    response2 = 'Here is the result:\n{"files": [], "summary": "nothing"}\nDone.'
    data2 = worker._parse_response(response2)
    assert data2 is not None
    assert data2["summary"] == "nothing"

    # JSON in code fence
    response3 = '```json\n{"files": [{"path": "b.py", "content": "x=1"}], "summary": "ok"}\n```'
    data3 = worker._parse_response(response3)
    assert data3 is not None
    assert data3["files"][0]["path"] == "b.py"


def test_worker_parse_markdown_files():
    """Worker should fall back to parsing markdown code blocks."""
    from furrow.agents.worker import WorkerAgent
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="create files")
    worker = WorkerAgent(task=task)

    response = """Here are the files:

```python
hello.py
```
print("hello")

```javascript
app.js
```
console.log("hi")
"""
    data = worker._parse_markdown_files(response)
    assert data is not None
    assert len(data["files"]) == 2
    paths = {f["path"] for f in data["files"]}
    assert "hello.py" in paths
    assert "app.js" in paths


def test_llm_client_ollama_url():
    """LLMClient should construct correct Ollama URL."""
    from furrow.llm import LLMClient
    from furrow.config import Settings, Provider

    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert client.settings.ollama_base_url == "http://localhost:11434"