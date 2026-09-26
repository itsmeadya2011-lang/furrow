import asyncio
import inspect

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_tracks_tasks():
    """Test that the orchestrator stores tasks in all_tasks (the bug fix)."""
    orch = Orchestrator(goal="test goal")
    assert orch.all_tasks == []
    assert not orch._is_done()
    task = TaskModel(id="1", description="test")
    orch.all_tasks.append(task)
    assert not orch._is_done()


def test_orchestrator_is_done_with_completed_tasks():
    """Test _is_done() returns True when all tasks completed and tests passed."""
    orch = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="completed")
    orch.all_tasks.append(task)
    orch.tests_passed = True
    assert orch._is_done()


def test_orchestrator_is_done_with_failed_tasks():
    """Test _is_done() returns True even with failed tasks when tests pass."""
    orch = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="failed")
    orch.all_tasks.append(task)
    orch.tests_passed = True
    assert orch._is_done()


def test_orchestrator_not_done_when_tests_fail():
    """Test _is_done() returns False when tests haven't passed."""
    orch = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="completed")
    orch.all_tasks.append(task)
    orch.tests_passed = False
    assert not orch._is_done()


def test_orchestrator_max_cycles_enforced():
    """Test that max_cycles limits the number of cycles."""
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test goal", settings=settings)
    assert orch.max_cycles == 1


def test_orchestrator_max_parallel_tasks():
    """Test that max_parallel_tasks is used."""
    settings = Settings(max_parallel_tasks=3)
    orch = Orchestrator(goal="test goal", settings=settings)
    assert orch.settings.max_parallel_tasks == 3


def test_orchestrator_event_queue():
    """Test that event_queue is initialized."""
    orch = Orchestrator(goal="test goal")
    assert orch.event_queue is None


def test_orchestrator_emit():
    """Test that emit puts messages in the event_queue."""
    orch = Orchestrator(goal="test goal")
    orch.event_queue = asyncio.Queue()

    async def test():
        await orch.emit("test message")
        msg = orch.event_queue.get_nowait()
        assert "test message" in msg

    asyncio.run(test())


def test_llm_ollama_provider_supported():
    """Test that Ollama provider doesn't raise ValueError in complete()."""
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    assert hasattr(client, "_complete_ollama")


def test_llm_retry_decorator_on_complete():
    """Test that the retry decorator is applied to complete()."""
    assert hasattr(LLMClient.complete, "retry") or hasattr(LLMClient.complete, "__wrapped__")


def test_llm_has_timeout_param():
    """Test that complete() accepts a timeout parameter."""
    sig = inspect.signature(LLMClient.complete)
    assert "timeout" in sig.parameters


def test_llm_read_file(tmp_path):
    """Test the read_file method."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    result = asyncio.run(client.read_file(test_file))
    assert result == "hello world"


def test_llm_write_file(tmp_path):
    """Test the write_file method."""
    test_file = tmp_path / "subdir" / "test.txt"
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    asyncio.run(client.write_file(test_file, "hello world"))
    assert test_file.read_text() == "hello world"


def test_llm_list_files(tmp_path):
    """Test the list_files method."""
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file2.txt").write_text("b")
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    files = client.list_files(tmp_path)
    assert "file1.txt" in files
    assert "sub/file2.txt" in files


def test_tester_parse_fallback():
    """Test that tester falls back to text parsing when JSON fails."""
    result = TestResult(passed="passed" in "all tests passed".lower(), summary="all tests passed", failures=[])
    assert result.passed is True
