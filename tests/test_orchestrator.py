from __future__ import annotations

import json

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from tests.conftest import EventCollector, FakeLLMClient, make_plan_json, make_test_json

ONE_TASK_PLAN = make_plan_json(
    tasks=[{"id": "1", "description": "do thing", "files": [], "dependencies": []}]
)


class FailingWorkerClient(FakeLLMClient):
    async def complete(self, prompt, system="", model=None):
        if "worker agent" in prompt:
            raise RuntimeError("worker crashed")
        return await super().complete(prompt, system=system, model=model)


async def test_single_cycle_when_tests_pass(tmp_path):
    settings = Settings(workspace=tmp_path)
    fake = FakeLLMClient(
        plans=[
            make_plan_json(
                tasks=[
                    {"id": "1", "description": "first", "files": ["a.py"], "dependencies": []},
                    {"id": "2", "description": "second", "files": ["b.py"], "dependencies": []},
                ]
            )
        ],
        tests=[make_test_json(passed=True, summary="all good")],
        settings=settings,
    )
    collector = EventCollector()
    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=collector)
    await orch.run()

    assert [e for e in collector.events if e["event"] == "done"] == [
        {"event": "done", "cycles": 1, "reason": "complete"}
    ]
    assert orch.cycles == 1
    assert [t["status"] for t in orch.state()["tasks"]] == ["completed", "completed"]
    for name in ("start", "cycle", "plan", "task_start", "task_result", "tests", "cycle_end", "done"):
        assert name in collector.names()
    assert (tmp_path / ".furrow" / "state.json").exists()


async def test_failing_tests_trigger_second_cycle_with_goal_preserved(tmp_path):
    settings = Settings(workspace=tmp_path)
    one_task_plan = make_plan_json(
        tasks=[{"id": "1", "description": "do thing", "files": [], "dependencies": []}]
    )
    fake = FakeLLMClient(
        plans=[one_task_plan, one_task_plan],
        tests=[
            make_test_json(passed=False, summary="1 failed", failures=["test_x failed"]),
            make_test_json(passed=True, summary="ok"),
        ],
        settings=settings,
    )
    collector = EventCollector()
    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=collector)
    await orch.run()

    assert orch.cycles == 2
    assert [e for e in collector.events if e["event"] == "done"] == [
        {"event": "done", "cycles": 2, "reason": "complete"}
    ]
    plan_prompts = [p for p in fake.prompts if "planning agent" in p]
    assert len(plan_prompts) == 2
    assert "Original goal: Build feature" in plan_prompts[1]
    assert "Fix these failing tests" in plan_prompts[1]
    assert "test_x failed" in plan_prompts[1]


async def test_max_cycles_enforced(tmp_path):
    settings = Settings(max_cycles=2, workspace=tmp_path)
    fake = FakeLLMClient(
        plans=[ONE_TASK_PLAN],
        tests=[make_test_json(passed=False, summary="1 failed", failures=["test_x failed"])],
        settings=settings,
    )
    collector = EventCollector()
    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=collector)
    await orch.run()

    assert orch.cycles == 2
    assert [e for e in collector.events if e["event"] == "done"] == [
        {"event": "done", "cycles": 2, "reason": "max_cycles"}
    ]


async def test_no_tasks_planned_stops(tmp_path):
    settings = Settings(workspace=tmp_path)
    fake = FakeLLMClient(
        plans=[json.dumps({"tasks": [], "rationale": "nothing left"})],
        settings=settings,
    )
    collector = EventCollector()
    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=collector)
    await orch.run()

    assert orch.cycles == 1
    assert [e for e in collector.events if e["event"] == "done"] == [
        {"event": "done", "cycles": 1, "reason": "complete"}
    ]


async def test_stop_requested(tmp_path):
    settings = Settings(workspace=tmp_path)
    fake = FakeLLMClient(
        plans=[ONE_TASK_PLAN],
        tests=[make_test_json(passed=False, summary="1 failed", failures=["test_x failed"])],
        settings=settings,
    )
    events: list[dict] = []

    def on_event(event):
        events.append(event)
        if event["event"] == "cycle_end":
            orch.stop()

    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=on_event)
    await orch.run()

    assert [e for e in events if e["event"] == "done"] == [
        {"event": "done", "cycles": 1, "reason": "stopped"}
    ]


async def test_worker_failure_marks_task_failed(tmp_path):
    settings = Settings(workspace=tmp_path)
    fake = FailingWorkerClient(
        plans=[ONE_TASK_PLAN],
        tests=[make_test_json(passed=True, summary="ok")],
        settings=settings,
    )
    collector = EventCollector()
    orch = Orchestrator(goal="Build feature", client=fake, settings=settings, on_event=collector)
    await orch.run()

    assert orch.state()["tasks"][0]["status"] == "failed"
    assert [e for e in collector.events if e["event"] == "done"] == [
        {"event": "done", "cycles": 1, "reason": "complete"}
    ]
    task_results = [e for e in collector.events if e["event"] == "task_result" and e.get("task_id") == "1"]
    assert task_results
    assert task_results[0]["status"] == "failed"
    assert "worker crashed" in task_results[0]["result"]