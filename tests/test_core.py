import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_dependencies():
    t = TaskModel(id="2", description="depends on 1", dependencies=["1"])
    assert t.dependencies == ["1"]


def test_task_status_default():
    t = TaskModel(id="3", description="new task")
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failures", failures=["fail1", "fail2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_orchestrator_is_done_empty():
    o = Orchestrator(goal="test")
    assert o._is_done() is False  # no tasks yet


def test_orchestrator_is_done_all_completed():
    o = Orchestrator(goal="test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_with_failure():
    o = Orchestrator(goal="test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False  # failures prevent completion


def test_orchestrator_is_done_pending():
    o = Orchestrator(goal="test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_topological_sort_simple():
    o = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="first"),
        TaskModel(id="2", description="second", dependencies=["1"]),
    ]
    ordered = o._topological_sort(tasks)
    assert ordered[0].id == "1"
    assert ordered[1].id == "2"


def test_topological_sort_independent():
    o = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    ordered = o._topological_sort(tasks)
    assert len(ordered) == 2


def test_group_by_level():
    o = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b", dependencies=["1"]),
        TaskModel(id="3", description="c", dependencies=["1"]),
    ]
    levels = o._group_by_level(tasks)
    assert len(levels) == 2
    assert len(levels[0]) == 1  # task 1 first
    assert len(levels[1]) == 2  # tasks 2,3 in parallel


def test_worker_code_block_parsing():
    """Test that the worker can parse fenced code blocks from LLM responses."""
    import asyncio
    from furrow.agents.worker import WorkerAgent

    async def test():
        # Create a mock client that returns a response with code blocks
        class MockClient:
            class settings:
                workspace = __import__("pathlib").Path(".")
                worker_model = "test-model"

            async def complete(self, prompt, model=None):
                return """Here's the change:

```src/new_file.py
def hello():
    print("hello world")
```

Done!"""

            async def write_file(self, path, content):
                pass

            def list_files(self, directory):
                return []

        task = TaskModel(id="1", description="create file", files=["src/new_file.py"])
        worker = WorkerAgent(task=task, client=MockClient())
        result = await worker.run()
        assert "Completed" in result

    asyncio.run(test())


def test_worker_no_code_blocks():
    """Test that the worker handles responses without code blocks gracefully."""
    import asyncio
    from furrow.agents.worker import WorkerAgent

    async def test():
        class MockClient:
            class settings:
                workspace = __import__("pathlib").Path(".")
                worker_model = "test-model"

            async def complete(self, prompt, model=None):
                return "I don't have any code to write."

            async def write_file(self, path, content):
                pass

            def list_files(self, directory):
                return []

        task = TaskModel(id="1", description="do something")
        worker = WorkerAgent(task=task, client=MockClient())
        result = await worker.run()
        assert "Completed" in result

    asyncio.run(test())


def test_llm_client_ollama_provider():
    """Test that Ollama provider is recognized."""
    from furrow.llm import LLMClient
    from furrow.config import Provider, Settings

    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OLLAMA