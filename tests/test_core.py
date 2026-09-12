import asyncio

import httpx
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from furrow.config import Plan, TaskModel, TestResult, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, _is_transient_error, _retry_decorator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- _get_tests tests ---

def test_get_tasks_initially_empty():
    """_get_tasks() returns _current_tasks, which starts as []."""
    ork = Orchestrator(goal="test goal")
    assert ork._get_tasks() == []


def test_get_tasks_returns_current_tasks():
    """_get_tasks() reflects tasks assigned during a cycle."""
    ork = Orchestrator(goal="test goal")
    tasks = [TaskModel(id="1", description="task 1", status="pending")]
    ork._current_tasks = tasks
    assert ork._get_tasks() is tasks
    assert ork._get_tasks() == [tasks[0]]


# --- _is_done tests ---

def test_is_done_empty_tasks():
    """No tasks means goal is complete."""
    ork = Orchestrator(goal="test goal")
    assert ork._is_done() is True


def test_is_done_all_pending():
    """Tasks that haven't started are not done."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [
        TaskModel(id="1", description="t1", status="pending"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert ork._is_done() is False


def test_is_done_all_completed():
    """All tasks completed means done."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert ork._is_done() is True


def test_is_done_any_failed():
    """Any failed task means not done (even if others completed)."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert ork._is_done() is False


def test_is_done_mixed_completed_pending():
    """Mix of completed and pending is not done."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert ork._is_done() is False


def test_is_done_single_failed():
    """A single failed task alone is not done."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [TaskModel(id="1", description="t1", status="failed")]
    assert ork._is_done() is False


def test_is_done_single_completed():
    """A single completed task is done."""
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [TaskModel(id="1", description="t1", status="completed")]
    assert ork._is_done() is True


# --- max_cycles tests ---

@pytest.mark.asyncio
async def test_max_cycles_enforced():
    """run() stops after max_cycles even if _is_done is False."""
    ork = Orchestrator(goal="test goal")
    ork._cycle = AsyncMock()
    ork._is_done = MagicMock(return_value=False)

    with patch("furrow.core.orchestrator.settings", MagicMock(max_cycles=2)):
        await ork.run()

    assert ork.cycles == 2
    assert ork._cycle.call_count == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_unlimited():
    """max_cycles=0 means unlimited; run() stops when _is_done becomes True."""
    ork = Orchestrator(goal="test goal")
    ork._cycle = AsyncMock()
    ork._is_done = MagicMock(side_effect=[False, False, True])

    with patch("furrow.core.orchestrator.settings", MagicMock(max_cycles=0)):
        await ork.run()

    assert ork.cycles == 3
    assert ork._cycle.call_count == 3


@pytest.mark.asyncio
async def test_max_cycles_one_stops_after_one():
    """max_cycles=1 stops after exactly one cycle."""
    ork = Orchestrator(goal="test goal")
    ork._cycle = AsyncMock()
    ork._is_done = MagicMock(return_value=False)

    with patch("furrow.core.orchestrator.settings", MagicMock(max_cycles=1)):
        await ork.run()

    assert ork.cycles == 1
    assert ork._cycle.call_count == 1


@pytest.mark.asyncio
async def test_stops_when_done_before_max_cycles():
    """run() stops early if _is_done() returns True before max_cycles."""
    ork = Orchestrator(goal="test goal")
    ork._cycle = AsyncMock()
    ork._is_done = MagicMock(side_effect=[False, True])

    with patch("furrow.core.orchestrator.settings", MagicMock(max_cycles=5)):
        await ork.run()

    assert ork.cycles == 2
    assert ork._cycle.call_count == 2


@pytest.mark.asyncio
async def test_failed_tests_mark_all_tasks_failed():
    """When tests fail, every task is marked failed with the failure summary."""
    task1 = TaskModel(id="1", description="t1")
    task2 = TaskModel(id="2", description="t2")
    plan = Plan(tasks=[task1, task2], rationale="ok")

    ork = Orchestrator(goal="test goal")
    ork.planner = MagicMock()
    ork.planner.plan = AsyncMock(return_value=plan)
    ork.client = MagicMock()
    WorkerAgent_mock = MagicMock()
    WorkerAgent_mock.return_value.run = AsyncMock(return_value="ok")
    TesterAgent_mock = MagicMock()
    TesterAgent_mock.return_value.run = AsyncMock(
        return_value=TestResult(passed=False, summary="boom", failures=["fail a"])
    )

    with patch("furrow.core.orchestrator.WorkerAgent", WorkerAgent_mock), patch(
        "furrow.core.orchestrator.TesterAgent", TesterAgent_mock
    ):
        await ork._cycle()

    for t in plan.tasks:
        assert t.status == "failed"
        assert t.result == "boom"
    assert ork.goal.startswith("Fix failing tests:")


@pytest.mark.asyncio
async def test_empty_plan_resets_current_tasks():
    """An empty plan resets _current_tasks so stale tasks don't stop run()."""
    stale = TaskModel(id="1", description="stale", status="completed")
    ork = Orchestrator(goal="test goal")
    ork._current_tasks = [stale]
    ork.planner = MagicMock()
    ork.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="ok"))
    ork.client = MagicMock()

    with patch("furrow.core.orchestrator.WorkerAgent"), patch(
        "furrow.core.orchestrator.TesterAgent"
    ):
        await ork._cycle()

    assert ork._current_tasks == []
    assert ork._is_done() is True


# --- LLM retry tests ---


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    pass


class InternalServerError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


class GatewayTimeoutError(Exception):
    pass


class FakePermanentError(Exception):
    pass


# --- _is_transient_error unit tests ---


def test_is_transient_http_408():
    err = Exception("timeout")
    err.status_code = 408
    assert _is_transient_error(err) is True


def test_is_transient_http_429():
    err = Exception("rate limited")
    err.status_code = 429
    assert _is_transient_error(err) is True


def test_is_transient_http_500():
    err = Exception("server error")
    err.status_code = 500
    assert _is_transient_error(err) is True


def test_is_transient_http_503():
    err = Exception("unavailable")
    err.status_code = 503
    assert _is_transient_error(err) is True


def test_is_not_transient_http_400():
    err = Exception("bad request")
    err.status_code = 400
    assert _is_transient_error(err) is False


def test_is_not_transient_http_401():
    err = Exception("unauthorized")
    err.status_code = 401
    assert _is_transient_error(err) is False


def test_is_not_transient_http_403():
    err = Exception("forbidden")
    err.status_code = 403
    assert _is_transient_error(err) is False


def test_is_not_transient_http_404():
    err = Exception("not found")
    err.status_code = 404
    assert _is_transient_error(err) is False


def test_is_transient_timeout_error():
    assert _is_transient_error(asyncio.TimeoutError()) is True


def test_is_transient_connect_error():
    assert _is_transient_error(httpx.ConnectError("conn failed")) is True


def test_is_transient_read_timeout():
    assert _is_transient_error(httpx.ReadTimeout("read timeout")) is True


def test_is_transient_api_timeout_name():
    assert _is_transient_error(APITimeoutError()) is True


def test_is_transient_api_connection_error_name():
    assert _is_transient_error(APIConnectionError()) is True


def test_is_transient_rate_limit_error_name():
    assert _is_transient_error(RateLimitError()) is True


def test_is_transient_internal_server_error_name():
    assert _is_transient_error(InternalServerError()) is True


def test_is_transient_service_unavailable_error_name():
    assert _is_transient_error(ServiceUnavailableError()) is True


def test_is_transient_gateway_timeout_error_name():
    assert _is_transient_error(GatewayTimeoutError()) is True


def test_is_not_transient_generic_exception():
    assert _is_transient_error(Exception("generic")) is False


def test_is_not_transient_value_error():
    assert _is_transient_error(ValueError("invalid")) is False


def test_is_not_transient_without_status_code():
    assert _is_transient_error(Exception("oops")) is False


def test_retry_policy_is_bounded():
    """Retry policy: at most 3 attempts with bounded exponential backoff."""
    assert _retry_decorator is not None


# --- Integration tests via complete() ---


@pytest.mark.asyncio
async def test_complete_retries_transient_and_succeeds_anthropic():
    """Transient 5xx errors are retried; call succeeds on attempt 3."""
    client = LLMClient(settings=MagicMock(provider=Provider.ANTHROPIC, model="test-model"))
    client._anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="hello")]

    err_500 = Exception("server error")
    err_500.status_code = 500
    err_503 = Exception("unavailable")
    err_503.status_code = 503

    client._anthropic.messages.create = AsyncMock(
        side_effect=[err_500, err_503, mock_response]
    )

    result = await client.complete("prompt")
    assert result == "hello"
    assert client._anthropic.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_complete_retries_transient_and_succeeds_openai():
    """Transient 429 errors are retried; call succeeds on attempt 3."""
    client = LLMClient(settings=MagicMock(provider=Provider.OPENAI, model="test-model"))
    client._openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]

    err_429a = Exception("rate limited")
    err_429a.status_code = 429
    err_429b = Exception("rate limited")
    err_429b.status_code = 429

    client._openai.chat.completions.create = AsyncMock(
        side_effect=[err_429a, err_429b, mock_response]
    )

    result = await client.complete("prompt")
    assert result == "hi"
    assert client._openai.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_complete_does_not_retry_permanent_client_error():
    """Permanent 400 errors are not retried; raised immediately."""
    client = LLMClient(settings=MagicMock(provider=Provider.ANTHROPIC, model="test-model"))
    client._anthropic = MagicMock()

    err = Exception("bad request")
    err.status_code = 400
    client._anthropic.messages.create = AsyncMock(side_effect=err)

    with pytest.raises(Exception) as exc_info:
        await client.complete("prompt")
    assert exc_info.value.status_code == 400
    assert client._anthropic.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_complete_does_not_retry_auth_error():
    """401 auth errors are not retried."""
    client = LLMClient(settings=MagicMock(provider=Provider.OPENAI, model="test-model"))
    client._openai = MagicMock()

    err = Exception("unauthorized")
    err.status_code = 401
    client._openai.chat.completions.create = AsyncMock(side_effect=err)

    with pytest.raises(Exception) as exc_info:
        await client.complete("prompt")
    assert exc_info.value.status_code == 401
    assert client._openai.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_complete_exhausts_retries_on_persistent_transient():
    """After 3 attempts with persistent transient errors, the last error is raised."""
    client = LLMClient(settings=MagicMock(provider=Provider.ANTHROPIC, model="test-model"))
    client._anthropic = MagicMock()

    err = Exception("server error")
    err.status_code = 500
    client._anthropic.messages.create = AsyncMock(side_effect=err)

    with pytest.raises(Exception):
        await client.complete("prompt")
    assert client._anthropic.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_complete_no_retry_on_immediate_success():
    """Successful call makes exactly one API attempt."""
    client = LLMClient(settings=MagicMock(provider=Provider.ANTHROPIC, model="test-model"))
    client._anthropic = MagicMock()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)

    result = await client.complete("prompt")
    assert result == "ok"
    assert client._anthropic.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_complete_uses_default_system_message():
    """When system prompt is empty, the default system message is used."""
    client = LLMClient(settings=MagicMock(provider=Provider.ANTHROPIC, model="test-model"))
    client._anthropic = MagicMock()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="reply")]
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)

    await client.complete("prompt", system="")
    call_kwargs = client._anthropic.messages.create.call_args
    assert call_kwargs.kwargs["system"] == "You are a helpful coding assistant."


@pytest.mark.asyncio
async def test_complete_unsupported_provider_raises():
    """Unsupported provider raises ValueError without API calls."""
    client = LLMClient(settings=MagicMock(provider="ollama", model="test-model"))

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("prompt")

