import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from furrow import (
    LLMClient,
    Orchestrator,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    TesterAgent,
    WorkerAgent,
)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done():
    settings = Settings()
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)

    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True

    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=2)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)

    orch._cycle = AsyncMock()
    orch._is_done = MagicMock(return_value=False)

    asyncio.run(orch.run())

    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


def test_worker_writes_file():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value="file content")
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="do thing", files=["output.txt"])
    worker = WorkerAgent(task=task, client=client)

    result = asyncio.run(worker.run())

    assert result == "file content"
    client.write_file.assert_called_once_with("output.txt", "file content")


def test_tester_handles_no_runner():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "ok", "failures": []}'
    )

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")

    result = asyncio.run(tester.run("test goal", []))

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "ok"


def test_llm_ollama_provider():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ollama response"
    mock_ollama = AsyncMock()
    mock_ollama.chat.completions.create = AsyncMock(return_value=mock_response)
    client._ollama = mock_ollama

    result = asyncio.run(client.complete("test prompt"))

    mock_ollama.chat.completions.create.assert_called_once()
    assert result == "ollama response"


def test_worker_writes_multiple_files():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value=(
        "=== FILE: src/example.py ===\ndef hello():\n    return 'hi'\n"
        "=== FILE: tests/test_example.py ===\n"
        "def test_hello():\n    assert hello() == 'hi'\n"
    ))
    client.write_file = AsyncMock()

    task = TaskModel(
        id="1",
        description="create example module",
        files=["src/example.py", "tests/test_example.py"],
    )
    worker = WorkerAgent(task=task, client=client)

    asyncio.run(worker.run())

    assert client.write_file.call_count == 2
    written_files = {call.args[0] for call in client.write_file.call_args_list}
    assert "src/example.py" in written_files
    assert "tests/test_example.py" in written_files


def test_llm_edit_file():
    settings = Settings()
    client = LLMClient(settings=settings)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 1\n")
        filepath = f.name

    try:
        asyncio.run(client.edit_file(filepath, "return 1", "return 42"))
        content = Path(filepath).read_text()
        assert "return 42" in content
        assert "return 1" not in content
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_llm_edit_file_not_found():
    settings = Settings()
    client = LLMClient(settings=settings)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 1\n")
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="Could not find string"):
            asyncio.run(client.edit_file(filepath, "nonexistent", "replacement"))
    finally:
        Path(filepath).unlink(missing_ok=True)
