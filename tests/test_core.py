from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeClient:
    """In-memory stand-in for LLMClient returning scripted JSON responses."""

    def __init__(
        self,
        settings: Settings | None = None,
        plans=None,
        plan: Plan | None = None,
        plan_error: str | None = None,
        worker_responses=None,
        test_results=None,
        tester_raw: str | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings()
        self._plans = list(plans or [])
        self._plan = plan
        self._plan_error = plan_error
        self._worker_responses = list(worker_responses or [])
        self._test_results = list(test_results or [])
        self._tester_raw = tester_raw
        self.writes: dict[str, str] = {}
        self.reads: dict[str, str] = {}
        self.complete_calls = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls += 1
        first = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
        if first.startswith("You are a planning agent"):
            if self._plan_error is not None:
                return self._plan_error
            if self._plans:
                p = self._plans.pop(0)
            elif self._plan is not None:
                p = self._plan
            else:
                p = Plan(tasks=[], rationale="empty")
            return json.dumps(p.model_dump())
        if first.startswith("You are a worker agent"):
            return self._worker_responses.pop(0) if self._worker_responses else ""
        if first.startswith("You are a tester agent"):
            if self._tester_raw is not None:
                return self._tester_raw
            tr = (
                self._test_results.pop(0)
                if self._test_results
                else TestResult(passed=True, summary="ok")
            )
            return json.dumps(tr.model_dump())
        return ""

    async def read_file(self, path) -> str:
        return self.reads.get(str(path), "")

    async def write_file(self, path, content: str) -> None:
        self.writes[str(path)] = content

    def list_files(self, directory: str) -> list[str]:
        return []


def _ok_run_tests(output: str):
    async def _f(self) -> str:
        return output

    return _f


def _raising_run_tests(exc: Exception):
    async def _f(self):
        raise exc

    return _f


# ---------------------------------------------------------------------------
# Config / model tests
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_ollama_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.ollama_model == "llama3.1"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_cycles == 0


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_success():
    plan = Plan(tasks=[TaskModel(id="1", description="x", files=["a.py"])], rationale="r")
    client = FakeClient(plan=plan)
    result = await PlannerAgent(client=client).plan("do x")
    assert result.rationale == "r"
    assert result.tasks[0].description == "x"


@pytest.mark.asyncio
async def test_planner_parse_error_raises():
    client = FakeClient(plan_error="not json {")
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("do x")


# ---------------------------------------------------------------------------
# Worker tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_writes_files():
    payload = '{"summary":"added feature","changes":[{"path":"src/foo.py","content":"print(1)"}]}'
    client = FakeClient(worker_responses=[payload])
    task = TaskModel(id="1", description="add foo", files=["src/foo.py"])
    result = await WorkerAgent(task=task, client=client).run()
    assert "Wrote: src/foo.py" in result
    assert client.writes["src/foo.py"] == "print(1)"


@pytest.mark.asyncio
async def test_worker_handles_bad_json():
    client = FakeClient(worker_responses=["totally not json"])
    task = TaskModel(id="1", description="x", files=[])
    result = await WorkerAgent(task=task, client=client).run()
    assert "Could not parse structured response" in result
    assert client.writes == {}


@pytest.mark.asyncio
async def test_worker_runs_commands(monkeypatch):
    import furrow.agents.worker as worker_mod

    payload = '{"summary":"setup","changes":[],"commands":["echo hello"]}'
    client = FakeClient(worker_responses=[payload])
    calls: list = []

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"hello", b"")

        async def wait(self):
            return 0

        def kill(self):
            pass

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return FakeProc()

    monkeypatch.setattr(worker_mod.asyncio, "create_subprocess_exec", fake_exec)

    task = TaskModel(id="1", description="x", files=[])
    result = await WorkerAgent(task=task, client=client).run()
    assert "Ran:" in result
    assert calls and calls[0][0] == "echo"


# ---------------------------------------------------------------------------
# Tester tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_no_runner_returns_result(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("ran pytest, all green"))
    client = FakeClient(test_results=[TestResult(passed=True, summary="green")])
    res = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="x")])
    assert res.passed is True
    assert res.summary == "green"


@pytest.mark.asyncio
async def test_tester_handles_run_exception(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _raising_run_tests(RuntimeError("boom")))
    client = FakeClient()
    res = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="x")])
    assert res.passed is False
    assert "boom" in res.summary
    assert "boom" in res.failures[0]


@pytest.mark.asyncio
async def test_tester_non_json_fallback(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("pass"))
    client = FakeClient(tester_raw="all tests passed")
    res = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="x")])
    assert res.passed is True
    assert res.summary == "all tests passed"


# ---------------------------------------------------------------------------
# LLM client tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_complete_dispatches_anthropic(monkeypatch):
    client = LLMClient(settings=Settings())
    called: list = []

    async def fake(self, prompt, system, model):
        called.append(model)
        return "anthropic-result"

    monkeypatch.setattr(LLMClient, "_create_anthropic", fake)
    res = await client.complete("hi")
    assert res == "anthropic-result"
    assert called and len(called) == 1


@pytest.mark.asyncio
async def test_llm_complete_dispatches_ollama(monkeypatch):
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    called: list = []

    async def fake(self, prompt, system, model):
        called.append(model)
        return "ollama-result"

    monkeypatch.setattr(LLMClient, "_create_ollama", fake)
    res = await client.complete("hi")
    assert res == "ollama-result"
    assert called and len(called) == 1
    assert settings.provider == Provider.OLLAMA


@pytest.mark.asyncio
async def test_llm_retries_on_api_error(monkeypatch):
    import anthropic

    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("tenacity.nap.sleep", _noop_sleep)

    client = LLMClient(settings=Settings())
    attempts = {"n": 0}

    class _Resp:
        content = [type("c", (), {"text": "ok"})()]

    class _Messages:
        async def create(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise anthropic.APIError("transient")
            return _Resp()

    class _Anthropic:
        def __init__(self):
            self.messages = _Messages()

    client._anthropic = _Anthropic()
    res = await client._create_anthropic("p", "sys", None)
    assert res == "ok"
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_completes_when_tests_pass(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="task1", files=[])], rationale="plan")
    client = FakeClient(
        plan=plan,
        worker_responses=['{"summary":"done","changes":[],"commands":[]}'],
        test_results=[TestResult(passed=True, summary="all good")],
    )
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("tests passed"))
    orch = Orchestrator(goal="build x", client=client, settings=client.settings)
    await orch.run()
    assert orch.cycles == 1
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch.test_result.passed is True
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_retries_on_failure_then_passes(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="t", files=[])], rationale="p")
    client = FakeClient(
        plan=plan,
        worker_responses=['{"summary":"d","changes":[],"commands":[]}"']
        * 2,
        test_results=[TestResult(passed=False, summary="fail", failures=["x broken"])],
    )
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("fail"))
    orch = Orchestrator(goal="build x", client=client, settings=client.settings)
    await orch.run()
    assert orch.cycles == 2
    assert orch.tasks[0].status == "completed"
    assert orch.test_result.passed is True
    assert "Fix failing tests" in orch.goal


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="t", files=[])], rationale="p")
    client = FakeClient(
        plan=plan,
        worker_responses=['{"summary":"d","changes":[],"commands":[]}"']
        * 2,
        test_results=[TestResult(passed=False, summary="fail", failures=["x"])],
    )
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("fail"))
    orch = Orchestrator(
        goal="x", client=client, settings=Settings(max_cycles=1)
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_empty_plan_stops(monkeypatch):
    plan = Plan(tasks=[], rationale="nothing to do")
    client = FakeClient(plan=plan, worker_responses=[], test_results=[])
    orch = Orchestrator(goal="noop", client=client, settings=client.settings)
    await orch.run()
    assert orch.cycles == 1
    assert orch.tasks == []
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_stops_after_empty_plan_on_retry(monkeypatch):
    plan_with_task = Plan(tasks=[TaskModel(id="1", description="t", files=[])], rationale="p")
    empty_plan = Plan(tasks=[], rationale="stuck")
    client = FakeClient(
        plans=[plan_with_task, empty_plan],
        worker_responses=['{"summary":"d","changes":[],"commands":[]}'],
        test_results=[TestResult(passed=False, summary="fail", failures=["x broken"])],
    )
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("fail"))
    orch = Orchestrator(goal="build x", client=client, settings=client.settings)
    await orch.run()
    assert orch.cycles == 2
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_calls_output_callback(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="t", files=[])], rationale="p")
    client = FakeClient(
        plan=plan,
        worker_responses=['{"summary":"d","changes":[],"commands":[]}'],
        test_results=[TestResult(passed=True, summary="ok")],
    )
    monkeypatch.setattr(TesterAgent, "_run_tests", _ok_run_tests("pass"))
    received: list = []
    orch = Orchestrator(
        goal="x",
        client=client,
        settings=client.settings,
        output_callback=received.append,
    )
    await orch.run()
    assert any("Cycle" in m for m in received)
    assert any("Tests PASSED" in m for m in received)
    assert any("Goal complete" in m for m in received)
