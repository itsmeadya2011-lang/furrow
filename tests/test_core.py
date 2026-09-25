from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

TesterAgent.__test__ = False  # type: ignore[attr-defined]
TestResult.__test__ = False  # type: ignore[attr-defined]


class MockSettings:
    def __init__(self, **overrides) -> None:
        self.provider = Provider.ANTHROPIC
        self.model = "test-model"
        self.planner_model = "planner-model"
        self.worker_model = "worker-model"
        self.tester_model = "tester-model"
        self.ollama_base_url = "http://localhost:11434"
        self.anthropic_api_key = "sk-test-anthropic"
        self.openai_api_key = "sk-test-openai"
        self.max_parallel_tasks = 5
        self.max_cycles = 0
        self.workspace = Path.cwd()
        self.log_level = "INFO"
        for key, value in overrides.items():
            setattr(self, key, value)


class MockClient:
    def __init__(self, settings: MockSettings | None = None, responses: list[str] | None = None) -> None:
        self.settings = settings or MockSettings()
        self.writes: dict[str, str] = {}
        self.deletes: list[str] = []
        self.complete_calls: list[dict] = []
        self._responses: deque = deque(responses or [])

    def add_response(self, response: str) -> None:
        self._responses.append(response)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "model": model})
        if not self._responses:
            return ""
        return self._responses.popleft()

    async def read_file(self, path: str) -> str:
        return self.writes.get(str(path), "")

    async def write_file(self, path: str, content: str) -> None:
        self.writes[str(path)] = content

    def list_files(self, directory: str) -> list[str]:
        return list(self.writes.keys())


async def _fake_run_tests(self) -> str:
    return "fake test output\n"


def _make_plan_json(tasks: list[dict] | None = None, rationale: str = "ok") -> str:
    return json.dumps({"tasks": tasks or [], "rationale": rationale})


def _make_worker_json(summary: str, changes: list[dict]) -> str:
    return json.dumps({"summary": summary, "changes": changes})


def _make_tester_json(passed: bool, summary: str, failures: list[str] | None = None) -> str:
    return json.dumps({"passed": passed, "summary": summary, "failures": failures or []})


# --- PlannerAgent tests ---


async def test_plan_parse():
    plan_json = _make_plan_json(
        tasks=[
            {
                "id": "1",
                "description": "do thing",
                "files": ["src/app.py"],
                "dependencies": [],
            }
        ],
        rationale="ok",
    )
    client = MockClient(responses=[plan_json])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("build thing")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"
    assert client.complete_calls[0]["model"] == client.settings.planner_model


async def test_plan_no_tasks():
    plan_json = _make_plan_json(tasks=[], rationale="Goal already complete, nothing to do.")
    client = MockClient(responses=[plan_json])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("complete the goal")

    assert isinstance(plan, Plan)
    assert plan.tasks == []
    assert plan.rationale == "Goal already complete, nothing to do."


async def test_plan_parse_invalid_json_raises():
    client = MockClient(responses=["not valid json {"])
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError):
        await planner.plan("x")


# --- WorkerAgent tests ---


async def test_worker_applies_changes():
    changes = [
        {
            "file": "src/new_module.py",
            "content": "def new_feature():\n    return 42\n",
            "action": "create",
        },
        {
            "file": "src/app.py",
            "content": "# updated app\n",
            "action": "edit",
        },
    ]
    worker_json = _make_worker_json("Implemented feature", changes)
    client = MockClient(responses=[worker_json])
    client.writes["src/app.py"] = "# original app\n"
    task = TaskModel(id="1", description="Add a feature", files=["src/app.py"])

    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert summary == "Implemented feature"
    assert client.writes["src/new_module.py"] == "def new_feature():\n    return 42\n"
    assert client.writes["src/app.py"] == "# updated app\n"
    assert len(client.complete_calls) == 1


async def test_worker_raw_fallback():
    client = MockClient(responses=["Sorry, I cannot help with that."])
    task = TaskModel(id="1", description="Do something", files=["src/app.py"])

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "Sorry, I cannot help with that."
    assert client.writes == {}


# --- TesterAgent tests ---


async def test_tester_parse_passed(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    tester_json = _make_tester_json(passed=True, summary="All tests passed")
    client = MockClient(responses=[tester_json])

    tester = TesterAgent(client=client)
    result = await tester.run("build feature", [])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_parse_failed(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    tester_json = _make_tester_json(
        passed=False, summary="2 tests failed", failures=["test_foo failed", "test_bar failed"]
    )
    client = MockClient(responses=[tester_json])

    tester = TesterAgent(client=client)
    result = await tester.run("build feature", [])

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.summary == "2 tests failed"
    assert result.failures == ["test_foo failed", "test_bar failed"]


async def test_tester_fallback(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    client = MockClient(responses=["All tests passed. Looks good to me."])

    tester = TesterAgent(client=client)
    result = await tester.run("build feature", [])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed. Looks good to me."
    assert result.failures == []


# --- Config / LLM tests ---


def test_settings_ollama_provider():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")

    assert settings.provider == Provider.OLLAMA
    assert settings.ollama_base_url == "http://localhost:11434"


async def test_llm_client_ollama_supported():
    settings = Settings(provider=Provider.OLLAMA, model="llama3", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    client._httpx_client = _FakeOllamaHTTPXClient({"response": "ollama reply"})

    result = await client.complete("hello ollama")

    assert result == "ollama reply"
    assert client._httpx_client.last_post["path"] == "/api/generate"
    assert client._httpx_client.last_post["json"]["model"] == "llama3"
    assert client._httpx_client.last_post["json"]["prompt"] == "hello ollama"


class _FakeOllamaResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeOllamaHTTPXClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_post: dict | None = None

    async def post(self, path: str, json: dict | None = None) -> _FakeOllamaResponse:
        self.last_post = {"path": path, "json": json}
        return _FakeOllamaResponse(self.payload)


# --- Orchestrator tests (validate the halt bug fix + loop improvements) ---


async def _run_orchestrator(goal, client, on_event):
    orch = Orchestrator(goal=goal, client=client, on_event=on_event)
    await orch.run()
    return orch


async def test_orchestrator_complete_first_cycle(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    plan_json = _make_plan_json(
        tasks=[
            {"id": "1", "description": "Add login feature", "files": ["src/auth.py"], "dependencies": []}
        ],
        rationale="Add login",
    )
    worker_json = _make_worker_json(
        "Implemented login", [{"file": "src/auth.py", "content": "# auth\n", "action": "create"}]
    )
    tester_json = _make_tester_json(passed=True, summary="All tests passed")

    client = MockClient(responses=[plan_json, worker_json, tester_json])
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    orch = await _run_orchestrator("Implement auth feature - goal complete", client, on_event)

    types = [e["type"] for e in events]
    assert orch.cycles == 1
    assert types[-1] == "done"
    done_event = events[-1]
    assert "goal complete" in done_event["summary"]
    assert done_event["cycles"] == 1


async def test_orchestrator_continues_on_failure(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)

    plan_json = _make_plan_json(
        tasks=[{"id": "1", "description": "Add feature", "files": ["src/app.py"], "dependencies": []}],
        rationale="plan",
    )
    worker_json = _make_worker_json("done", [{"file": "src/app.py", "content": "x", "action": "create"}])
    fail_json = _make_tester_json(
        passed=False, summary="fail", failures=["test_login_failed", "test_auth_broken"]
    )
    plan_fix_json = _make_plan_json(
        tasks=[{"id": "2", "description": "Fix login", "files": ["src/auth.py"], "dependencies": []}],
        rationale="fix",
    )

    client = MockClient(responses=[plan_json, worker_json, fail_json, plan_fix_json, worker_json, _make_tester_json(True, "fixed")])
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    orch = await _run_orchestrator("Implement auth", client, on_event)

    types = [e["type"] for e in events]
    assert orch.cycles == 2
    assert types.count("cycle_start") == 2
    assert types[-1] == "done"
    reformed = [e for e in events if e["type"] == "max_cycles_reformed"]
    assert len(reformed) == 1
    assert reformed[0]["goal"].startswith("Fix failing tests:")
    assert "test_login_failed" in reformed[0]["goal"]


async def test_orchestrator_max_cycles(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)

    settings = MockSettings(max_cycles=2)
    plan_json = _make_plan_json(
        tasks=[{"id": "1", "description": "Add feature", "files": ["src/app.py"], "dependencies": []}],
        rationale="plan",
    )
    worker_json = _make_worker_json("done", [{"file": "src/app.py", "content": "x", "action": "create"}])
    fail_json = _make_tester_json(passed=False, summary="fail", failures=["broken"])

    client = MockClient(
        settings=settings,
        responses=[plan_json, worker_json, fail_json, plan_json, worker_json, fail_json],
    )
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    orch = await _run_orchestrator("Do something", client, on_event)

    types = [e["type"] for e in events]
    assert "max_cycles_reached" in types
    assert types.count("cycle_start") == 2
    assert types[-1] == "max_cycles_reached"
    assert orch.cycles == 3


async def test_orchestrator_no_tasks(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    plan_json = _make_plan_json(tasks=[], rationale="Goal already complete, nothing to do.")
    client = MockClient(responses=[plan_json])
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    orch = await _run_orchestrator("Implement auth - goal complete", client, on_event)

    types = [e["type"] for e in events]
    assert orch.cycles == 1
    assert "no_tasks" in types
    assert "done" in types
    assert types[-1] == "done"


async def test_orchestrator_emits_events(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    plan_json = _make_plan_json(
        tasks=[{"id": "1", "description": "Add login feature", "files": ["src/auth.py"], "dependencies": []}],
        rationale="Add login",
    )
    worker_json = _make_worker_json(
        "Implemented login", [{"file": "src/auth.py", "content": "# auth\n", "action": "create"}]
    )
    tester_json = _make_tester_json(passed=True, summary="All tests passed")

    client = MockClient(responses=[plan_json, worker_json, tester_json])
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    await _run_orchestrator("Implement auth - goal complete", client, on_event)

    types = [e["type"] for e in events]
    assert types == ["goal", "cycle_start", "plan", "task_completed", "test_result", "done"]


async def test_orchestrator_stops_on_error(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)
    client = MockClient(responses=["not valid json {"])
    events: list[dict] = []

    async def on_event(event: dict):
        events.append(event)

    orch = await _run_orchestrator("Do something", client, on_event)

    types = [e["type"] for e in events]
    assert orch.cycles == 1
    assert "error" in types
    assert types[-1] == "error"
    error_event = events[-1]
    assert "Failed to parse plan" in error_event["message"]


# --- Config model test (kept from original) ---


def test_test_result():
    t = TestResult(passed=False, summary="2 failed", failures=["a", "b"])
    assert t.passed is False
    assert len(t.failures) == 2
