from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class MockLLMClient:
    def __init__(self, response="", written_files=None, settings=None, fail_tests=False, fail_worker=False):
        self._response = response
        self.written_files = written_files if written_files is not None else []
        self.settings = settings or Settings()
        self.fail_tests = fail_tests
        self.fail_worker = fail_worker
        self.complete = AsyncMock(side_effect=self._mock_complete)
        self.write_file = AsyncMock(side_effect=self._mock_write_file)
    
    async def _mock_complete(self, prompt, system="", model=None):
        # Return appropriate response based on prompt type
        if "tester agent" in prompt.lower() or "test output" in prompt.lower():
            if self.fail_tests:
                return TestResult(passed=False, summary="Tests failed", failures=["test failure"]).model_dump_json()
            return TestResult(passed=True, summary="All tests passed", failures=[]).model_dump_json()
        elif "planning agent" in prompt.lower() or "break the user" in prompt.lower():
            return self._response
        elif "worker agent" in prompt.lower():
            if self.fail_worker:
                raise RuntimeError("Worker failed")
            return self._response
        return self._response
    
    async def _mock_write_file(self, path, content):
        self.written_files.append((str(path), content))


@pytest.mark.asyncio
async def test_orchestrator_task_tracking():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one", files=["src/a.py"]),
            TaskModel(id="2", description="task two", files=["src/b.py"]),
        ],
        rationale="test plan"
    )
    mock_client = MockLLMClient()
    mock_client.complete = AsyncMock(return_value=plan.model_dump_json())
    
    orchestrator = Orchestrator(goal="test goal", client=mock_client)
    await orchestrator._cycle()
    
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    plan = Plan(
        tasks=[TaskModel(id="1", description="task one", files=["src/a.py"])],
        rationale="test plan"
    )
    mock_client = MockLLMClient(response=plan.model_dump_json(), fail_tests=True, fail_worker=True)
    
    with patch('furrow.core.orchestrator.settings') as mock_settings:
        mock_settings.max_cycles = 2
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        await orchestrator.run()
    
    assert orchestrator.cycles == 2
    assert mock_client.complete.call_count >= 2


@pytest.mark.asyncio
async def test_orchestrator_is_done():
    orchestrator = Orchestrator(goal="test goal")
    
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="completed"),
    ]
    assert orchestrator._is_done() is True
    
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="failed"),
    ]
    assert orchestrator._is_done() is False
    
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="pending"),
        TaskModel(id="2", description="task two", status="completed"),
    ]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_worker_writes_files():
    written_files = []
    mock_client = MockLLMClient(written_files=written_files)
    mock_client.settings.worker_model = "test-model"
    
    response = "```python src/example.py\nprint('hello')\n```"
    mock_client.complete = AsyncMock(return_value=response)
    
    task = TaskModel(id="1", description="write hello", files=["src/example.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    
    assert len(written_files) == 1
    assert written_files[0][0] == "src/example.py"
    assert written_files[0][1] == "print('hello')\n"
    assert "Wrote 1 file(s)" in result


@pytest.mark.asyncio
async def test_worker_no_code_blocks():
    written_files = []
    mock_client = MockLLMClient(written_files=written_files)
    mock_client.settings.worker_model = "test-model"
    
    response = "I have analyzed the task and here is my response without code blocks."
    mock_client.complete = AsyncMock(return_value=response)
    
    task = TaskModel(id="1", description="analyze something", files=["src/example.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    
    assert result == response
    assert len(written_files) == 0
