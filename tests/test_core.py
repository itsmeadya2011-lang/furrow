from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import State
from furrow.llm import LLMClient

# --- Original sanity checks (kept) ---


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Orchestrator._is_done ---


def test_is_done_no_plan(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    assert orch._is_done() is True


def test_is_done_all_completed_tests_pass(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    t = TaskModel(id="1", description="d")
    t.status = "completed"
    orch._plan = Plan(tasks=[t], rationale="r")
    orch._last_test_passed = True
    assert orch._is_done() is True


def test_is_done_tests_fail(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    t = TaskModel(id="1", description="d")
    t.status = "completed"
    orch._plan = Plan(tasks=[t], rationale="r")
    orch._last_test_passed = False
    assert orch._is_done() is False


def test_is_done_task_failed(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    t = TaskModel(id="1", description="d")
    t.status = "failed"
    orch._plan = Plan(tasks=[t], rationale="r")
    orch._last_test_passed = True
    assert orch._is_done() is False


def test_is_done_empty_plan(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    orch._plan = Plan(tasks=[], rationale="done")
    assert orch._is_done() is True


# --- max_cycles ---


@pytest.mark.asyncio
async def test_max_cycles_stops_loop(tmp_path):
    orch = Orchestrator(goal="x", state_path=tmp_path / "s.json")
    orch.settings = Settings(max_cycles=1)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="nope"))
    await orch.run()
    assert orch.cycles == 1


# --- State persistence ---


def test_state_save_load(tmp_path):
    state = State(path=tmp_path / "state.json")
    state.goal = "build feature"
    state.cycles = 3
    state.tasks = [TaskModel(id="1", description="do thing", status="completed")]
    state.last_test_result = TestResult(passed=True, summary="ok")
    state.save()
    assert (tmp_path / "state.json").exists()

    loaded = State(path=tmp_path / "state.json")
    loaded.load()
    assert loaded.goal == "build feature"
    assert loaded.cycles == 3
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].status == "completed"
    assert loaded.last_test_result is not None
    assert loaded.last_test_result.passed is True


def test_state_load_missing_file(tmp_path):
    state = State(path=tmp_path / "nope.json")
    state.load()
    assert state.goal == ""
    assert state.cycles == 0
    assert state.tasks == []
    assert state.last_test_result is None


def test_state_load_corrupt(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json {")
    state = State(path=p)
    state.load()
    assert state.goal == ""
    assert state.cycles == 0


def test_state_update_persists(tmp_path):
    state = State(path=tmp_path / "state.json")
    state.update(goal="g", cycles=2)
    loaded = State(path=tmp_path / "state.json")
    loaded.load()
    assert loaded.goal == "g"
    assert loaded.cycles == 2


# --- LLMClient ---


@pytest.mark.asyncio
async def test_complete_no_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=s)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            await client.complete("hi")


@pytest.mark.asyncio
async def test_complete_ollama_routes_and_parses():
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    resp = MagicMock()
    resp.json = MagicMock(return_value={"response": "ollama reply"})
    resp.raise_for_status = MagicMock()
    instance = AsyncMock()
    instance.post = AsyncMock(return_value=resp)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=instance)
    cm.__aexit__ = AsyncMock()

    with patch("furrow.llm.httpx.AsyncClient", return_value=cm):
        result = await client.complete("hello")
    assert result == "ollama reply"
    instance.post.assert_called_once()
    _, kwargs = instance.post.call_args
    assert kwargs["json"]["model"] == "llama3"
    assert kwargs["json"]["system"]
    assert kwargs["json"]["stream"] is False


@pytest.mark.asyncio
async def test_complete_retry_on_timeout():
    with patch.dict("os.environ", {}, clear=True):
        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=s)

    calls = {"n": 0}

    async def fail_then_succeed(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("boom")
        return "ok"

    with patch.object(client, "_complete_anthropic", fail_then_succeed):
        result = await client.complete("hi")
    assert result == "ok"
    assert calls["n"] == 2
