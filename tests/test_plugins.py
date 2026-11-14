from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerResult
from furrow.config import Plan, RunResult, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore
from furrow.llm import LLMResponse, ToolCall
from furrow.plugins import Plugin, PluginManager


class RecordingPlugin(Plugin):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.events: list[tuple] = []

    async def pre_cycle(self, orchestrator, cycle_number):
        self.events.append(("pre_cycle", cycle_number))

    async def post_cycle(self, orchestrator, cycle_number, result):
        self.events.append(("post_cycle", cycle_number, getattr(result, "rationale", None)))

    async def pre_task_execution(self, task):
        self.events.append(("pre_task", task.id, task.status))

    async def post_task_execution(self, task, result):
        self.events.append(("post_task", task.id, result.success))


class CustomToolPlugin(Plugin):
    def get_worker_tools(self):
        return [
            {
                "name": "echo_tool",
                "description": "echo back a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ]

    async def execute_worker_tool(self, name, arguments, worker):
        if name == "echo_tool":
            return f"echo:{arguments.get('value', '')}"
        return None


class DetectPlugin(Plugin):
    async def detect_test_command(self, tester, workspace):
        return ["custom-test", "--ok"]


class RunTestsPlugin(Plugin):
    async def run_tests(self, tester, goal, tasks):
        return TestResult(passed=True, summary="plugin tests ok", failures=[])


class FakeWorker:
    def __init__(
        self,
        task,
        client=None,
        settings=None,
        workspace=None,
        goal="",
        plugin_manager=None,
    ):
        self.task = task

    async def run(self):
        return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])


class FakePlanner:
    async def plan(self, goal, context=""):
        return Plan(tasks=[TaskModel(id="t1", description="do")], rationale="r")


class FakeTester:
    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="ok", failures=[])


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, messages, tools=None, system="", model=None):
        return self.responses.pop(0)

    async def complete(self, prompt, system="", model=None):
        return ""

    async def read_file(self, path):
        return Path(path).read_text()

    async def write_file(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def list_files(self, directory):
        return [str(p) for p in Path(directory).rglob("*") if p.is_file()]


def test_settings_accepts_plugins_list():
    s = Settings(plugins=["pkg.mod.PluginA", "other.PluginB"], _env_file=None)
    assert s.plugins == ["pkg.mod.PluginA", "other.PluginB"]


@pytest.mark.asyncio
async def test_plugin_manager_loads_from_dotted_paths(tmp_path, monkeypatch):
    mod_dir = tmp_path / "plugpkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "p.py").write_text(
        textwrap.dedent(
            """
            from furrow.plugins import Plugin
            class MyPlugin(Plugin):
                pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = Settings(plugins=["plugpkg.p.MyPlugin"], _env_file=None)
    pm = PluginManager.from_settings(settings)
    assert [p.__class__.__name__ for p in pm.plugins] == ["MyPlugin"]


@pytest.mark.asyncio
async def test_plugin_manager_loads_entry_points(monkeypatch):
    class EPPlugin(Plugin):
        pass

    class FakeEP:
        name = "ep"

        def load(self):
            return EPPlugin

    class FakeEPS:
        def select(self, group=None):
            assert group == "furrow.plugins"
            return [FakeEP()]

    monkeypatch.setattr("furrow.plugins.metadata.entry_points", lambda: FakeEPS())
    pm = PluginManager.from_settings(Settings(_env_file=None))
    assert any(isinstance(p, EPPlugin) for p in pm.plugins)


@pytest.mark.asyncio
async def test_orchestrator_cycle_and_task_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    plugin = RecordingPlugin()
    pm = PluginManager(settings=Settings(workspace=tmp_path, _env_file=None), plugins=[plugin])
    settings = Settings(workspace=tmp_path, max_cycles=1, _env_file=None)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator("goal", settings=settings, state_store=store, plugin_manager=pm)
    orch.planner = FakePlanner()
    orch.tester = FakeTester()
    result = await orch.run()
    assert isinstance(result, RunResult)
    assert ("pre_cycle", 1) in plugin.events
    assert ("post_cycle", 1, "r") in plugin.events
    assert ("pre_task", "t1", "running") in plugin.events
    assert ("post_task", "t1", True) in plugin.events


@pytest.mark.asyncio
async def test_worker_custom_tool_hook(tmp_path):
    settings = Settings(workspace=tmp_path, max_worker_iterations=5, _env_file=None)
    plugin = CustomToolPlugin()
    pm = PluginManager(settings=settings, plugins=[plugin])
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="echo_tool", arguments={"value": "hi"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    task = TaskModel(id="1", description="x")
    from furrow.agents.worker import WorkerAgent

    agent = WorkerAgent(
        task=task, client=client, settings=settings, workspace=tmp_path, plugin_manager=pm
    )
    res = await agent.run()
    assert res.success is True
    assert res.iterations == 2
    # Verify the plugin tool result was returned into conversation
    # Re-run with capturing client to inspect messages passed to chat
    class CapturingClient(FakeLLMClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.calls = []

        async def chat(self, messages, tools=None, system="", model=None):
            self.calls.append({"messages": messages, "tools": tools})
            return await super().chat(messages, tools=tools, system=system, model=model)

    responses2 = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="echo_tool", arguments={"value": "hi"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    c2 = CapturingClient(responses2)
    agent2 = WorkerAgent(
        task=task, client=c2, settings=settings, workspace=tmp_path, plugin_manager=pm
    )
    res2 = await agent2.run()
    assert res2.success is True
    assert any(t.get("name") == "echo_tool" for t in (c2.calls[0].get("tools") or []))
    second_msgs = c2.calls[1]["messages"]
    found = False
    for m in second_msgs:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result" and b.get("tool_use_id") == "1":
                    assert b.get("content") == "echo:hi"
                    found = True
    assert found


@pytest.mark.asyncio
async def test_tester_custom_command_and_runner_plugins(tmp_path):
    settings = Settings(workspace=tmp_path, _env_file=None)
    # detect override
    pm1 = PluginManager(settings=settings, plugins=[DetectPlugin()])
    tester = TesterAgent(
        client=FakeLLMClient([]),
        settings=settings,
        workspace=tmp_path,
        plugin_manager=pm1,
    )
    assert await tester.detect_command() == ["custom-test", "--ok"]
    # full run override
    pm2 = PluginManager(settings=settings, plugins=[RunTestsPlugin()])
    tester2 = TesterAgent(
        client=FakeLLMClient([]),
        settings=settings,
        workspace=tmp_path,
        plugin_manager=pm2,
    )
    res = await tester2.run("goal", [])
    assert res.passed is True
    assert res.summary == "plugin tests ok"
