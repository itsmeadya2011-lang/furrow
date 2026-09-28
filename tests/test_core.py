from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import Completion, LLMClient, ToolCall

# These library classes start with "Test" and would otherwise be picked up by
# pytest's default collection. Mark them as non-collectable to avoid warnings.
TesterAgent.__test__ = False  # type: ignore[attr-defined]
TestResult.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Stand-in for LLMClient that simulates tool calls without network access."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.write_path: str = "out.txt"
        self.write_content: str = "hello"
        self.complete_return: str = '{"passed": true, "summary": "ok", "failures": []}'
        self.complete_calls = 0
        self._tool_round = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls += 1
        return self.complete_return

    async def complete_with_tools(
        self, messages: list[dict], system: str = "", model: str | None = None
    ) -> Completion:
        if self._tool_round == 0:
            self._tool_round += 1
            return Completion(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": self.write_path, "content": self.write_content},
                    )
                ],
            )
        return Completion(text=f"Done: wrote {self.write_path}", tool_calls=[])

    async def execute_tool(self, tool: ToolCall) -> str:
        if tool.name == "write_file":
            p = Path(tool.arguments["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(tool.arguments["content"])
            return f"Wrote {tool.arguments['path']}"
        if tool.name == "read_file":
            return Path(tool.arguments["path"]).read_text()
        if tool.name == "list_files":
            return json.dumps([])
        return f"Unknown tool: {tool.name}"


class FakePlanner:
    def __init__(self, client=None) -> None:
        self.client = client

    async def plan(self, goal: str) -> Plan:
        return Plan(
            tasks=[TaskModel(id="1", description="d", files=[], dependencies=[], status="pending")],
            rationale="r",
        )


class FakeTester:
    def __init__(self, client=None, passed: bool = True) -> None:
        self.client = client
        self.passed = passed

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        for t in tasks:
            t.status = "completed"
            t.result = "ok"
        if self.passed:
            return TestResult(passed=True, summary="ok", failures=[])
        return TestResult(passed=False, summary="bad", failures=["f1"])


# ---------------------------------------------------------------------------
# Models / config
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=False, summary="bad")
    assert t.failures == []


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"


def test_taskmodel_requires_id():
    with pytest.raises(ValidationError):
        TaskModel(description="d")  # missing id


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


async def test_planner_parse_success():
    fake = FakeLLMClient()
    fake.complete_return = (
        '{"tasks":[{"id":"1","description":"x","files":[],"dependencies":[]}],"rationale":"r"}'
    )
    plan = await PlannerAgent(client=fake).plan("do something")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "x"
    assert plan.rationale == "r"
    assert fake.complete_calls == 1


async def test_planner_parse_invalid():
    fake = FakeLLMClient()
    fake.complete_return = "not json at all"
    with pytest.raises(ValueError):
        await PlannerAgent(client=fake).plan("g")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def test_worker_writes_file(tmp_path):
    target = tmp_path / "out.txt"
    fake = FakeLLMClient()
    fake.write_path = str(target)
    fake.write_content = "hello world"
    task = TaskModel(id="1", description="write a file", files=[str(target)], dependencies=[])
    result = await WorkerAgent(task=task, client=fake).run()
    assert target.read_text() == "hello world"
    assert "wrote" in result.lower()


async def test_worker_no_tools_returns_text():
    fake = FakeLLMClient()
    fake._tool_round = 1  # return text on the first call instead of a tool call
    fake.write_path = "irrelevant"
    task = TaskModel(id="1", description="summarize", files=[], dependencies=[])
    result = await WorkerAgent(task=task, client=fake).run()
    assert "wrote" in result.lower() and result


async def test_worker_max_rounds_no_summary(tmp_path):
    class AlwaysToolClient(FakeLLMClient):
        async def complete_with_tools(
            self, messages: list[dict], system: str = "", model: str | None = None
        ) -> Completion:
            return Completion(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": self.write_path, "content": self.write_content},
                    )
                ],
            )

    fake = AlwaysToolClient()
    fake.write_path = str(tmp_path / "unused")
    task = TaskModel(id="1", description="loop", files=[], dependencies=[])
    result = await WorkerAgent(task=task, client=fake).run()
    assert result == "Reached maximum tool-call rounds without a final summary."


# ---------------------------------------------------------------------------
# LLM client tools (no network)
# ---------------------------------------------------------------------------


async def test_execute_tool_write_read_replace_list(tmp_path):
    client = LLMClient(settings=Settings())
    p = tmp_path / "file.txt"
    res = await client.execute_tool(
        ToolCall(id="1", name="write_file", arguments={"path": str(p), "content": "abc"})
    )
    assert "Wrote" in res and p.read_text() == "abc"

    res2 = await client.execute_tool(
        ToolCall(id="2", name="read_file", arguments={"path": str(p)})
    )
    assert res2 == "abc"

    res3 = await client.execute_tool(
        ToolCall(
            id="3",
            name="replace_file",
            arguments={"path": str(p), "old_string": "abc", "new_string": "xyz"},
        )
    )
    assert "Replaced" in res3 and p.read_text() == "xyz"

    res4 = await client.execute_tool(
        ToolCall(id="4", name="list_files", arguments={"directory": str(tmp_path)})
    )
    assert json.loads(res4) == ["file.txt"]


async def test_execute_tool_unknown_and_missing_old(tmp_path):
    client = LLMClient(settings=Settings())
    p = tmp_path / "f.txt"
    res = await client.execute_tool(ToolCall(id="x", name="nope", arguments={}))
    assert res == "Unknown tool: nope"

    p.write_text("abc")
    res2 = await client.execute_tool(
        ToolCall(
            id="y",
            name="replace_file",
            arguments={"path": str(p), "old_string": "zzz", "new_string": "q"},
        )
    )
    assert "not found" in res2


async def test_read_write_file_helpers(tmp_path):
    client = LLMClient(settings=Settings())
    p = tmp_path / "deep" / "nested" / "a.txt"
    await client.write_file(p, "contents")
    assert p.read_text() == "contents"
    assert await client.read_file(p) == "contents"


def test_list_files_helper(tmp_path):
    client = LLMClient(settings=Settings())
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    files = client.list_files(tmp_path)
    assert sorted(files) == ["a.txt", "b.txt"]


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


async def test_tester_parses_json_result():
    fake = FakeLLMClient()
    fake.complete_return = '{"passed": false, "summary": "failing", "failures": ["case1"]}'
    t = TesterAgent(client=fake)

    async def _fake_run_tests():
        return "error output here"

    t._run_tests = _fake_run_tests
    result = await t.run("g", [])
    assert result.passed is False
    assert "case1" in result.failures
    assert result.summary == "failing"
    assert fake.complete_calls == 1


async def test_tester_fallback_on_non_json():
    fake = FakeLLMClient()
    fake.complete_return = "tests passed, nothing to fix"
    t = TesterAgent(client=fake)

    async def _fake():
        return "ok"

    t._run_tests = _fake
    result = await t.run("g", [])
    assert result.passed is True


async def test_tester_handles_test_exception():
    fake = FakeLLMClient()
    t = TesterAgent(client=fake)

    async def _boom():
        raise RuntimeError("kaboom")

    t._run_tests = _boom
    result = await t.run("g", [])
    assert result.passed is False
    assert "kaboom" in result.summary
    assert "kaboom" in result.failures
    assert fake.complete_calls == 0  # complete not called when tests raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator(goal="g", client=FakeLLMClient(), settings=Settings())
    assert orch._is_done() is True


def test_orchestrator_is_done_scenarios():
    orch = Orchestrator(goal="g", client=FakeLLMClient(), settings=Settings())

    orch.tasks = [TaskModel(id="1", description="d", status="completed")]
    orch.last_test_result = TestResult(passed=True, summary="s", failures=[])
    assert orch._is_done() is True

    # any failed -> not done
    orch.tasks = [TaskModel(id="1", description="d", status="failed")]
    orch.last_test_result = TestResult(passed=False, summary="s", failures=["f"])
    assert orch._is_done() is False

    # incomplete -> not done
    orch.tasks = [TaskModel(id="1", description="d", status="pending")]
    orch.last_test_result = TestResult(passed=True, summary="s", failures=[])
    assert orch._is_done() is False

    # completed but tests failed -> not done
    orch.tasks = [TaskModel(id="1", description="d", status="completed")]
    orch.last_test_result = TestResult(passed=False, summary="s", failures=["f"])
    assert orch._is_done() is False


def test_orchestrator_get_tasks_returns_self_tasks():
    orch = Orchestrator(goal="g", client=FakeLLMClient(), settings=Settings())
    tasks = [TaskModel(id="1", description="d")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


async def test_orchestrator_completes_when_passed(tmp_path):
    fake = FakeLLMClient()
    fake.write_path = str(tmp_path / "o.txt")
    orch = Orchestrator(goal="g", client=fake, settings=Settings(max_cycles=10))
    orch.planner = FakePlanner(client=fake)
    orch.tester = FakeTester(client=fake, passed=True)
    await orch.run()
    assert orch.cycles == 1
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed
    assert all(t.status == "completed" for t in orch.tasks)


async def test_orchestrator_runs_until_max_cycles(tmp_path):
    fake = FakeLLMClient()
    fake.write_path = str(tmp_path / "o.txt")
    orch = Orchestrator(goal="g", client=fake, settings=Settings(max_cycles=3))
    orch.planner = FakePlanner(client=fake)
    orch.tester = FakeTester(client=fake, passed=False)
    await orch.run()
    assert orch.cycles == 3
    assert orch.last_test_result is not None
    assert not orch.last_test_result.passed


async def test_orchestrator_max_parallel_tasks(tmp_path):
    fake = FakeLLMClient()
    fake.write_path = str(tmp_path / "o.txt")

    class Planner2:
        def __init__(self, client=None):
            self.client = client

        async def plan(self, goal: str) -> Plan:
            return Plan(
                tasks=[
                    TaskModel(id="1", description="d1", files=[], dependencies=[], status="pending"),
                    TaskModel(id="2", description="d2", files=[], dependencies=[], status="pending"),
                ],
                rationale="r",
            )

    orch = Orchestrator(
        goal="g", client=fake, settings=Settings(max_cycles=10, max_parallel_tasks=2)
    )
    orch.planner = Planner2(client=fake)
    orch.tester = FakeTester(client=fake, passed=True)
    await orch.run()
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------


def test_web_index():
    from furrow.web.server import app

    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Furrow" in r.text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_start_help():
    from click.testing import CliRunner

    from furrow.cli.main import main

    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
