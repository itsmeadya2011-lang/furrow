import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import openai
import pytest
import pytest_asyncio

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import _is_retryable


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Orchestrator._is_done logic ---


def _make_orchestrator_with_tasks(tasks: list[TaskModel]) -> Orchestrator:
    """Create an Orchestrator with a pre-set plan containing the given tasks."""
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key")
    orch = Orchestrator(goal="test goal", client=client)
    orch.plan = Plan(tasks=tasks, rationale="test")
    return orch


def test_is_done_all_completed():
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch = _make_orchestrator_with_tasks(tasks)
    assert orch._is_done() is True


def test_is_done_with_failed_task():
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch = _make_orchestrator_with_tasks(tasks)
    assert orch._is_done() is False


def test_is_done_pending_tasks():
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch = _make_orchestrator_with_tasks(tasks)
    assert orch._is_done() is False


def test_is_done_no_plan():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key")
    orch = Orchestrator(goal="test goal", client=client)
    orch.plan = None
    assert orch._is_done() is True


def test_is_done_empty_tasks():
    orch = _make_orchestrator_with_tasks([])
    assert orch._is_done() is True


# --- Orchestrator max_cycles enforcement ---


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key", max_cycles=2)
    orch = Orchestrator(goal="test goal", client=client)

    orch._cycle = AsyncMock()
    orch._is_done = MagicMock(return_value=False)

    await orch.run()

    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_stops_when_is_done():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key", max_cycles=10)
    orch = Orchestrator(goal="test goal", client=client)

    orch._cycle = AsyncMock()
    orch._is_done = MagicMock(return_value=True)

    await orch.run()

    assert orch._cycle.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_unlimited_cycles_when_max_cycles_zero():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key", max_cycles=0)
    orch = Orchestrator(goal="test goal", client=client)

    call_count = 0
    max_calls = 3

    async def mock_cycle():
        nonlocal call_count
        call_count += 1

    def mock_is_done():
        return call_count >= max_calls

    orch._cycle = mock_cycle
    orch._is_done = mock_is_done

    with patch("furrow.core.orchestrator.console"):
        await orch.run()

    assert orch.cycles == max_calls


# --- Orchestrator semaphore ---


def test_semaphore_created_with_max_parallel_tasks():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key", max_parallel_tasks=3)
    orch = Orchestrator(goal="test goal", client=client)
    assert orch._semaphore._value == 3


def test_semaphore_default_value():
    client = MagicMock()
    client.settings = Settings(anthropic_api_key="test-key")
    orch = Orchestrator(goal="test goal", client=client)
    assert orch._semaphore._value == 5


# --- WorkerAgent._write_response_files ---


@pytest.mark.asyncio
async def test_write_response_files_parses_file_blocks():
    client = MagicMock()
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=client)

    response = """
Here's the code:

```python:src/foo.py
def hello():
    return "world"
```

And another file:

```txt:README.md
# Hello
```
"""
    count = await worker._write_response_files(response)

    assert count == 2
    assert client.write_file.call_count == 2

    first_call = client.write_file.call_args_list[0]
    assert first_call[0][0] == "src/foo.py"
    assert 'def hello():' in first_call[0][1]

    second_call = client.write_file.call_args_list[1]
    assert second_call[0][0] == "README.md"
    assert '# Hello' in second_call[0][1]


@pytest.mark.asyncio
async def test_write_response_files_skips_empty_filepath():
    client = MagicMock()
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=client)

    response = """
```
some code without path
```
"""
    count = await worker._write_response_files(response)
    assert count == 0
    client.write_file.assert_not_called()


@pytest.mark.asyncio
async def test_write_response_files_no_code_blocks():
    client = MagicMock()
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=client)

    count = await worker._write_response_files("Just a plain response.")
    assert count == 0
    client.write_file.assert_not_called()


# --- WorkerAgent._read_context_files ---


@pytest.mark.asyncio
async def test_read_context_files_reads_files():
    client = MagicMock()
    client.read_file = AsyncMock(side_effect=["content of a", "content of b"])
    task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker._read_context_files()

    assert "--- a.py ---" in result
    assert "content of a" in result
    assert "--- b.py ---" in result
    assert "content of b" in result


@pytest.mark.asyncio
async def test_read_context_files_empty_when_no_files():
    client = MagicMock()
    task = TaskModel(id="1", description="test", files=[])
    worker = WorkerAgent(task=task, client=client)

    result = await worker._read_context_files()
    assert result == ""


@pytest.mark.asyncio
async def test_read_context_files_skips_missing_files():
    client = MagicMock()
    client.read_file = AsyncMock(side_effect=[FileNotFoundError("missing"), "content of b"])
    task = TaskModel(id="1", description="test", files=["missing.py", "b.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker._read_context_files()

    assert "--- missing.py ---" not in result
    assert "--- b.py ---" in result
    assert "content of b" in result


# --- LLMClient retry predicate ---


def test_is_retryable_anthropic_429():
    exc = anthropic.APIStatusError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    exc.status_code = 429
    assert _is_retryable(exc) is True


def test_is_retryable_anthropic_529():
    exc = anthropic.APIStatusError(
        message="overloaded",
        response=MagicMock(),
        body=None,
    )
    exc.status_code = 529
    assert _is_retryable(exc) is True


def test_is_retryable_anthropic_401_not_retried():
    exc = anthropic.APIStatusError(
        message="unauthorized",
        response=MagicMock(),
        body=None,
    )
    exc.status_code = 401
    assert _is_retryable(exc) is False


def test_is_retryable_openai_rate_limit():
    exc = openai.RateLimitError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    assert _is_retryable(exc) is True


def test_is_retryable_openai_connection_error():
    exc = openai.APIConnectionError(request=MagicMock())
    assert _is_retryable(exc) is True


def test_is_retryable_os_error():
    exc = OSError("connection refused")
    assert _is_retryable(exc) is True


def test_is_retryable_generic_exception_not_retried():
    exc = ValueError("something wrong")
    assert _is_retryable(exc) is False


def test_is_retryable_with_cause_chain():
    cause = anthropic.APIStatusError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    cause.status_code = 429
    exc = RuntimeError("wrapped")
    exc.__cause__ = cause
    assert _is_retryable(exc) is True


# --- TesterAgent fallback ---


@pytest.mark.asyncio
async def test_tester_agent_json_parse_failure_defaults_to_failed():
    client = MagicMock()
    client.complete = AsyncMock(return_value="This is not valid JSON")
    client.settings = MagicMock()
    client.settings.tester_model = "test-model"

    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("some goal", [])

    assert result.passed is False
    assert result.summary == "This is not valid JSON"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_agent_invalid_json_structure_defaults_to_failed():
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"invalid": "structure"}')
    client.settings = MagicMock()
    client.settings.tester_model = "test-model"

    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("some goal", [])

    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_agent_valid_json_returns_result():
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "all good", "failures": []}'
    )
    client.settings = MagicMock()
    client.settings.tester_model = "test-model"

    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("some goal", [])

    assert result.passed is True
    assert result.summary == "all good"


# --- Settings validation ---


def test_settings_ollama_requires_valid_url():
    with pytest.raises(ValueError, match="ollama_base_url must be a valid URL"):
        Settings(provider=Provider.OLLAMA, ollama_base_url="ftp://localhost:11434")


def test_settings_max_parallel_tasks_must_be_positive():
    with pytest.raises(ValueError, match="max_parallel_tasks must be >= 1"):
        Settings(anthropic_api_key="test-key", max_parallel_tasks=0)


def test_settings_max_cycles_must_be_non_negative():
    with pytest.raises(ValueError, match="max_cycles must be >= 0"):
        Settings(anthropic_api_key="test-key", max_cycles=-1)


def test_settings_valid_anthropic_config():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    assert s.provider == Provider.ANTHROPIC


def test_settings_valid_ollama_config():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.provider == Provider.OLLAMA


# --- LLMClient provider key validation ---


def test_llm_client_anthropic_requires_key():
    from furrow.llm import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


def test_llm_client_openai_requires_key():
    from furrow.llm import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai
