"""Comprehensive tests for the Furrow package.

Covers config models/settings, the LLM client (provider dispatch, Ollama
HTTP calls, retry behaviour, file helpers), the orchestrator (task
lifecycle, completion detection, events, state persistence, parallelism
limits), the agents (planner, worker, tester) and the web server
endpoints. All tests are self-contained: no real API calls are made and
no real files are written except state files under tmp_path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_fixed

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture(autouse=True)
def _fast_retry():
    """Make tenacity retries instant so retry tests run fast."""
    old_wait = LLMClient.complete.retry.wait
    LLMClient.complete.retry.wait = wait_fixed(0)
    yield
    LLMClient.complete.retry.wait = old_wait


def make_settings(**overrides) -> Settings:
    kwargs = {"anthropic_api_key": "test-key", "openai_api_key": "test-key"}
    kwargs.update(overrides)
    return Settings(**kwargs)


def make_mock_client(tmp_path: Path, **overrides) -> MagicMock:
    """A MagicMock shaped like an LLMClient with real Settings."""
    kwargs = {"state_file": tmp_path / ".furrow-state.json"}
    kwargs.update(overrides)
    client = MagicMock()
    client.settings = Settings(**kwargs)
    client.complete = AsyncMock()
    return client


def mock_llm_client(response: str) -> MagicMock:
    client = MagicMock()
    client.settings = Settings()
    client.complete = AsyncMock(return_value=response)
    return client


def make_event_recorder() -> tuple[list[tuple[str, dict]], callable]:
    """Return (events, on_event) where on_event appends (event, data) tuples."""
    events: list[tuple[str, dict]] = []

    def on_event(event: str, data: dict) -> None:
        events.append((event, data))

    return events, on_event


# ---------------------------------------------------------------------------
# Config / models
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-3-5-sonnet-20241022"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_retries == 3
    assert s.max_tokens == 8192
    assert s.ollama_model == "qwen2.5-coder:latest"


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_state_file_default():
    s = Settings()
    assert str(s.state_file).endswith(".furrow-state.json")


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def test_repr():
    client = LLMClient(settings=make_settings())
    assert repr(client) == "LLMClient(provider='anthropic', model='claude-3-5-sonnet-20241022')"


async def test_complete_dispatches_anthropic():
    client = LLMClient(settings=make_settings())
    with patch.object(LLMClient, "_complete_anthropic", new=AsyncMock(return_value="result")) as m:
        out = await client.complete("hi")
    assert out == "result"
    m.assert_awaited_once_with("hi", "", "claude-3-5-sonnet-20241022")


async def test_complete_dispatches_openai():
    client = LLMClient(settings=make_settings(provider=Provider.OPENAI))
    with patch.object(LLMClient, "_complete_openai", new=AsyncMock(return_value="result")) as m:
        out = await client.complete("hi")
    assert out == "result"
    m.assert_awaited_once_with("hi", "", "claude-3-5-sonnet-20241022")


async def test_complete_dispatches_ollama():
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))
    with patch.object(LLMClient, "_complete_ollama", new=AsyncMock(return_value="result")) as m:
        out = await client.complete("hi")
    assert out == "result"
    m.assert_awaited_once_with("hi", "", "qwen2.5-coder:latest")


async def test_complete_unknown_provider_raises():
    s = make_settings()
    s.provider = "bad"  # type: ignore[assignment]
    client = LLMClient(settings=s)
    with pytest.raises(ValueError):
        await client.complete("hi")


def test_resolve_model_ollama():
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))
    assert client._resolve_model(None) == "qwen2.5-coder:latest"


def test_resolve_model_default():
    client = LLMClient(settings=make_settings())
    assert client._resolve_model(None) == "claude-3-5-sonnet-20241022"


def test_resolve_model_override():
    client = LLMClient(settings=make_settings())
    assert client._resolve_model("custom-model") == "custom-model"


async def test_complete_ollama_makes_request():
    client = LLMClient(
        settings=make_settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    )
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": "ollama says hi"}}]}
    response.raise_for_status = MagicMock()
    http_instance = MagicMock()
    http_instance.post = AsyncMock(return_value=response)
    with patch("furrow.llm.httpx.AsyncClient") as client_cls:
        client_cls.return_value.__aenter__.return_value = http_instance
        out = await client.complete("hi")
    assert out == "ollama says hi"
    http_instance.post.assert_awaited_once()
    args, kwargs = http_instance.post.call_args
    assert args[0] == "http://localhost:11434/chat/completions"
    payload = kwargs["json"]
    assert payload["model"] == "qwen2.5-coder:latest"
    assert payload["max_tokens"] == 8192
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


async def test_retry_succeeds_on_second_attempt():
    client = LLMClient(settings=make_settings())
    m = AsyncMock(side_effect=[Exception("transient failure"), "ok"])
    with patch.object(LLMClient, "_complete_anthropic", new=m):
        out = await client.complete("hi")
    assert out == "ok"
    assert m.await_count == 2


async def test_retry_exhausts_and_reraises():
    client = LLMClient(settings=make_settings())
    m = AsyncMock(side_effect=Exception("always fails"))
    with patch.object(LLMClient, "_complete_anthropic", new=m):
        with pytest.raises(Exception, match="always fails"):
            await client.complete("hi")
    assert m.await_count == 3  # stop_after_attempt(settings.max_retries)


async def test_read_file_encoding(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("héllo", encoding="utf-8")
    client = LLMClient(settings=make_settings())
    with patch("furrow.llm.aiofiles.open") as m:
        m.return_value.__aenter__.return_value.read.return_value = "content"
        out = await client.read_file(f)
    assert out == "content"
    m.assert_called_once_with(f, "r", encoding="utf-8")


async def test_write_file_encoding(tmp_path):
    target = tmp_path / "sub" / "b.txt"
    client = LLMClient(settings=make_settings())
    with patch("furrow.llm.aiofiles.open") as m:
        await client.write_file(target, "data")
    m.assert_called_once_with(target, "w", encoding="utf-8")
    assert target.parent.exists()  # parent directories are created


def test_list_files_missing_dir(tmp_path):
    client = LLMClient(settings=make_settings())
    assert client.list_files(tmp_path / "does-not-exist") == []


def test_list_files_permission_error(tmp_path):
    d = tmp_path / "restricted"
    d.mkdir()
    (d / "f.txt").write_text("x")
    client = LLMClient(settings=make_settings())
    with patch.object(Path, "rglob", side_effect=PermissionError):
        assert client.list_files(d) == []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_get_tasks_empty(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    assert orch._get_tasks() == []


def test_get_tasks_returns_plan_tasks(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    orch.plan = plan
    assert orch._get_tasks() == plan.tasks


def test_is_done_no_tasks(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    assert orch._is_done() is True


def test_is_done_all_completed(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    orch.plan = Plan(
        tasks=[TaskModel(id="1", description="a", status="completed")], rationale="r"
    )
    assert orch._is_done() is True


def test_is_done_failed_persistent(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a", status="failed")], rationale="r")
    assert orch._is_done() is False  # first failure is recorded as retryable
    assert orch._is_done() is True  # same task failing again means retries exhausted


def test_is_done_failed_new(tmp_path):
    orch = Orchestrator(goal="g", client=make_mock_client(tmp_path))
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a", status="failed")], rationale="r")
    assert orch._is_done() is False
    orch.plan = Plan(tasks=[TaskModel(id="2", description="b", status="failed")], rationale="r")
    assert orch._is_done() is False  # a new failing task means keep going


async def test_max_cycles_stops(tmp_path):
    client = make_mock_client(tmp_path)
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    events, on_event = make_event_recorder()
    with (
        patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner,
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
        patch("furrow.core.orchestrator.TesterAgent") as MockTester,
    ):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="failing", failures=["err"])
        )
        orch = Orchestrator(goal="g", client=client, on_event=on_event)
        await orch.run(max_cycles=1)
    assert ("stopped", {"cycles": 1, "reason": "max_cycles"}) in events
    assert ("done", {"cycles": 1, "reason": "complete"}) not in events


async def test_on_event_callbacks(tmp_path):
    client = make_mock_client(tmp_path)
    plan = Plan(
        tasks=[TaskModel(id="1", description="a", files=["x.py"])], rationale="r"
    )
    events, on_event = make_event_recorder()
    with (
        patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner,
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
        patch("furrow.core.orchestrator.TesterAgent") as MockTester,
    ):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="did it")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all good")
        )
        orch = Orchestrator(goal="g", client=client, on_event=on_event)
        await orch.run()
    names = [name for name, _data in events]
    for expected in (
        "cycle_start",
        "plan_ready",
        "task_start",
        "task_completed",
        "testing",
        "tests_passed",
        "cycle_end",
        "done",
    ):
        assert expected in names
    assert ("task_start", {"task_id": "1", "description": "a", "files": ["x.py"]}) in events
    assert ("task_completed", {"task_id": "1", "result": "did it"}) in events
    assert ("done", {"cycles": 1, "reason": "complete"}) in events


def test_state_save_and_load(tmp_path):
    state_file = tmp_path / "state.json"
    client = make_mock_client(tmp_path)
    client.settings.state_file = state_file
    orch = Orchestrator(goal="original goal", client=client)
    orch.cycles = 2
    orch._failed_task_ids = {"t1"}
    orch.plan = Plan(
        tasks=[TaskModel(id="t1", description="a", status="completed", result="ok")],
        rationale="r",
    )
    orch._save_state()

    data = json.loads(state_file.read_text())
    assert data["goal"] == "original goal"
    assert data["cycles"] == 2
    assert data["failed_task_ids"] == ["t1"]
    assert data["tasks"][0]["status"] == "completed"

    client2 = make_mock_client(tmp_path)
    client2.settings.state_file = state_file
    orch2 = Orchestrator(goal="ignored", client=client2)
    assert orch2.goal == "original goal"
    assert orch2.cycles == 2
    assert orch2._failed_task_ids == {"t1"}


async def test_semaphore_limits_parallelism(tmp_path):
    client = make_mock_client(tmp_path, max_parallel_tasks=2)
    plan = Plan(tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(5)], rationale="r")
    state = {"current": 0, "peak": 0}

    async def fake_run():
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.05)
        state["current"] -= 1
        return "ok"

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner,
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
        patch("furrow.core.orchestrator.TesterAgent") as MockTester,
    ):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(side_effect=fake_run)
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok")
        )
        orch = Orchestrator(goal="g", client=client)
        await orch.run()
    assert state["peak"] <= 2
    assert state["peak"] == 2  # the semaphore actually allows up to the limit


async def test_run_emits_done_event(tmp_path):
    client = make_mock_client(tmp_path)
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    events, on_event = make_event_recorder()
    with (
        patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner,
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
        patch("furrow.core.orchestrator.TesterAgent") as MockTester,
    ):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok")
        )
        orch = Orchestrator(goal="g", client=client, on_event=on_event)
        await orch.run()
    assert ("done", {"cycles": 1, "reason": "complete"}) in events


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


async def test_planner_plan_parses_json():
    payload = json.dumps({"tasks": [{"id": "1", "description": "do"}], "rationale": "r"})
    client = mock_llm_client(payload)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("goal")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "r"
    client.complete.assert_awaited_once()
    prompt = client.complete.call_args.args[0]
    assert "goal" in prompt.lower()
    assert client.complete.call_args.kwargs["model"] == Settings().planner_model


async def test_planner_plan_raises_on_invalid_json():
    client = mock_llm_client("this is not json")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("goal")


async def test_worker_run_calls_complete():
    client = mock_llm_client("result text")
    task = TaskModel(id="1", description="implement login", files=["auth.py"])
    worker = WorkerAgent(task=task, client=client)
    out = await worker.run()
    assert out == "result text"
    client.complete.assert_awaited_once()
    prompt = client.complete.call_args.args[0]
    assert "implement login" in prompt
    assert "auth.py" in prompt
    assert client.complete.call_args.kwargs["model"] == Settings().worker_model


async def test_tester_run_json_response():
    client = mock_llm_client(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="")):
        tester = TesterAgent(client=client)
        result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


async def test_tester_run_fallback_response():
    client = mock_llm_client("Tests passed: 3/3")
    with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="")):
        tester = TesterAgent(client=client)
        result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == "Tests passed: 3/3"


async def test_tester_run_tests_exception():
    client = mock_llm_client("{}")
    with patch.object(
        TesterAgent, "_run_tests", new=AsyncMock(side_effect=RuntimeError("no runner"))
    ):
        tester = TesterAgent(client=client)
        result = await tester.run("goal", [])
    assert result.passed is False
    assert "no runner" in result.summary
    assert result.failures == ["no runner"]


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from furrow.web.server import app

    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_returns_html():
    from fastapi.testclient import TestClient

    from furrow.web.server import app

    with TestClient(app) as test_client:
        response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html>" in response.text
