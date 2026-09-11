import pytest
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []
    o._current_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    assert len(o._get_tasks()) == 1
    assert o._get_tasks()[0].id == "1"


def test_orchestrator_is_done_empty():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_completed():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    o._current_plan.tasks[0].status = "completed"
    assert o._is_done() is True


def test_orchestrator_is_done_failed():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    o._current_plan.tasks[0].status = "failed"
    assert o._is_done() is False


def test_orchestrator_is_done_partial():
    o = Orchestrator(goal="test")
    o._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ],
        rationale="r",
    )
    o._current_plan.tasks[0].status = "completed"
    assert o._is_done() is False


def test_worker_file_parsing():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        import asyncio

        async def run_test():
            from furrow.config import Settings

            custom_settings = Settings(workspace=tmpdir)
            client = LLMClient(settings=custom_settings)
            worker = WorkerAgent(
                task=TaskModel(id="1", description="create a file"),
                client=client,
                workspace=tmpdir,
            )
            response = """Here is the file:
FILE: hello.py
```python
print("hello")
```
"""
            await worker._apply_changes(response)
            target = tmpdir + "/hello.py"
            with open(target) as f:
                content = f.read()
            assert 'print("hello")' in content

        asyncio.run(run_test())


def test_llm_client_complete_unsupported_provider():
    from furrow.config import Settings

    custom_settings = Settings(provider="unsupported")
    client = LLMClient(settings=custom_settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        import asyncio

        asyncio.run(client.complete("test"))

