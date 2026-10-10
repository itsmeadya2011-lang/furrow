import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator


def _plan(tasks, rationale="ok"):
    return {"tasks": [{"id": tid, "description": desc, "files": [], "dependencies": deps} for tid, desc, deps in tasks], "rationale": rationale}


def _result(passed, failures=None, summary="ok"):
    return {"passed": passed, "summary": summary, "failures": failures or []}


class FakeClient:
    def __init__(self, plans, test_results, workspace, fail_descs=()):
        self.settings = SimpleNamespace(
            workspace=Path(workspace),
            planner_model="p",
            worker_model="w",
            tester_model="t",
        )
        self.plans = plans
        self.test_results = test_results
        self.fail_descs = set(fail_descs)
        self.plan_calls = 0
        self.test_calls = 0
        self.completion_order = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.written = {}

    async def complete(self, prompt, system="", model=None):
        if "planning agent" in prompt:
            plan = self.plans[min(self.plan_calls, len(self.plans) - 1)]
            self.plan_calls += 1
            return json.dumps(plan)
        if "tester agent" in prompt:
            tr = self.test_results[min(self.test_calls, len(self.test_results) - 1)]
            self.test_calls += 1
            return json.dumps(tr)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        await asyncio.sleep(0.01)
        self.concurrent -= 1
        desc = prompt.split("Task: ")[1].split("\n")[0] if "Task: " in prompt else ""
        self.completion_order.append(desc)
        if desc in self.fail_descs:
            raise RuntimeError("boom")
        return json.dumps({"summary": "did " + desc, "edits": []})

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(path)] = content


def _make_orchestrator(tmp_path, plans, test_results, fail_descs=(), max_cycles=0, max_parallel_tasks=5):
    client = FakeClient(plans, test_results, tmp_path, fail_descs)
    settings = Settings(workspace=tmp_path, max_cycles=max_cycles, max_parallel_tasks=max_parallel_tasks)
    events = []
    orch = Orchestrator(goal="do thing", client=client, settings=settings, on_event=events.append)
    return orch, client, events


async def test_orchestrator_runs_tasks_and_stops_on_passing_tests(tmp_path):
    plan = _plan([("1", "t1", []), ("2", "t2", [])])
    orch, client, events = _make_orchestrator(tmp_path, [plan], [_result(True)])
    await orch.run()
    assert orch.cycles == 1
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)
    assert client.completion_order == ["t1", "t2"]
    assert any(e.get("reason") == "tests passing" for e in events if e.get("type") == "done")


async def test_orchestrator_respects_max_cycles(tmp_path):
    plan = _plan([("1", "t1", [])])
    orch, _client, events = _make_orchestrator(
        tmp_path, [plan], [_result(False, failures=["f1"])], max_cycles=3
    )
    await orch.run()
    assert orch.cycles == 3
    assert any("max_cycles" in e.get("reason", "") for e in events if e.get("type") == "done")


async def test_orchestrator_preserves_goal(tmp_path):
    plan = _plan([("1", "t1", [])])
    orch, _client, _events = _make_orchestrator(
        tmp_path, [plan], [_result(False, failures=["f1"])], max_cycles=2
    )
    await orch.run()
    assert orch.goal == "do thing"
    assert "Fix failing tests" in orch.focus


async def test_orchestrator_dependency_order(tmp_path):
    plan = _plan([("a", "a", []), ("b", "b", ["a"])])
    orch, client, _events = _make_orchestrator(tmp_path, [plan], [_result(True)])
    await orch.run()
    assert client.completion_order == ["a", "b"]


async def test_orchestrator_skips_tasks_of_failed_dependencies(tmp_path):
    plan = _plan([("a", "a", []), ("b", "b", ["a"])])
    orch, _client, _events = _make_orchestrator(tmp_path, [plan], [_result(True)], fail_descs=["a"])
    await orch.run()
    by_id = {t.id: t for t in orch.tasks}
    assert by_id["a"].status == "failed"
    assert by_id["b"].status == "skipped"


async def test_orchestrator_respects_max_parallel(tmp_path):
    plan = _plan([("1", "t1", []), ("2", "t2", []), ("3", "t3", []), ("4", "t4", [])])
    orch, client, _events = _make_orchestrator(tmp_path, [plan], [_result(True)], max_parallel_tasks=2)
    await orch.run()
    assert client.max_concurrent <= 2


async def test_orchestrator_emits_events(tmp_path):
    plan = _plan([("1", "t1", []), ("2", "t2", [])])
    orch, _client, events = _make_orchestrator(tmp_path, [plan], [_result(True)])
    await orch.run()
    types = {e.get("type") for e in events}
    assert {"cycle", "plan", "task_start", "task_done", "tests", "done"} <= types
    task_dones = [e for e in events if e.get("type") == "task_done"]
    assert task_dones and all({"task_id", "status", "result"} <= set(e) for e in task_dones)
    tests_events = [e for e in events if e.get("type") == "tests"]
    assert tests_events and "passed" in tests_events[0]["result"]


async def test_orchestrator_empty_plan_halt(tmp_path):
    orch, _client, events = _make_orchestrator(tmp_path, [{"tasks": [], "rationale": "done"}], [_result(True)])
    await orch.run()
    assert orch.cycles == 1
    assert any(e.get("reason") == "plan empty" for e in events if e.get("type") == "done")