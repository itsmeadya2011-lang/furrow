from __future__ import annotations

import asyncio
import json

import pytest

from furrow.core.orchestrator import Orchestrator

ONE_TASK_PLAN = {
    "tasks": [
        {"id": "1", "description": "do the thing", "files": [], "dependencies": []}
    ],
    "rationale": "first slice",
}
TWO_TASK_PLAN = {
    "tasks": [
        {"id": "1", "description": "task one", "files": [], "dependencies": []},
        {"id": "2", "description": "task two", "files": [], "dependencies": []},
    ],
    "rationale": "two parallel tasks",
}
THREE_TASK_PLAN = {
    "tasks": [
        {"id": "1", "description": "a"},
        {"id": "2", "description": "b"},
        {"id": "3", "description": "c"},
    ],
    "rationale": "three tasks",
}
EMPTY_PLAN = {"tasks": [], "rationale": "Goal already complete"}
PASSED_RESULT = {"passed": True, "summary": "all green", "failures": []}
FAILED_RESULT = {"passed": False, "summary": "1 failure", "failures": ["test1 failed"]}

FINISH_MSG = json.dumps({"action": "finish", "summary": "done"})


@pytest.fixture(autouse=True)
def _no_real_tests(monkeypatch):
    async def _fake(self):
        return "exit code: 0\nPASSED"

    monkeypatch.setattr("furrow.agents.tester.TesterAgent._run_tests", _fake)


def _find_events(events: list[dict], etype: str) -> list[dict]:
    return [e for e in events if e.get("type") == etype]


# --------------------------------------------------------------------------- #
# Single cycle, 2 tasks, tests pass -> "complete"
# --------------------------------------------------------------------------- #
async def test_run_complete_single_cycle_two_tasks(tmp_path, make_llm, recorded_bus):
    fake = make_llm(
        workspace=tmp_path,
        responses=[TWO_TASK_PLAN, FINISH_MSG, FINISH_MSG, PASSED_RESULT],
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal="Build a thing", client=fake, event_bus=bus)

    await orch.run()

    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert types[1] == "cycle.start"
    assert types[2] == "plan"
    assert types[-1] == "halt"

    plan_ev = _find_events(events, "plan")[0]
    assert len(plan_ev["plan"]["tasks"]) == 2

    task_starts = _find_events(events, "task.start")
    task_completes = _find_events(events, "task.complete")
    assert len(task_starts) == 2
    assert len(task_completes) == 2

    test_result_ev = _find_events(events, "test.result")[0]
    assert test_result_ev["result"]["passed"] is True

    halt_ev = _find_events(events, "halt")[0]
    assert halt_ev["reason"] == "complete"
    assert orch.cycles == 1


# --------------------------------------------------------------------------- #
# max_cycles=2 halts after 2 cycles with "max_cycles"
# --------------------------------------------------------------------------- #
async def test_max_cycles_halts_after_two_cycles(tmp_path, make_llm, recorded_bus):
    fake = make_llm(
        workspace=tmp_path,
        responses=[
            ONE_TASK_PLAN, FINISH_MSG, FAILED_RESULT,  # cycle 1 (tests fail)
            ONE_TASK_PLAN, FINISH_MSG, FAILED_RESULT,  # cycle 2
        ],
        max_cycles=2,
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal="Build a thing", client=fake, event_bus=bus)

    await orch.run()

    halts = _find_events(events, "halt")
    assert len(halts) == 1
    assert halts[0]["reason"] == "max_cycles"
    assert halts[0]["cycles"] == 2
    assert orch.cycles == 2
    assert len(_find_events(events, "cycle.start")) == 2


# --------------------------------------------------------------------------- #
# max_parallel_tasks=1 limits concurrency
# --------------------------------------------------------------------------- #
_CONCURRENCY_STATE: dict[str, int] = {"current": 0, "max": 0}


class _ConcurrencyFakeWorker:
    def __init__(self, task, client):
        self.task = task
        self.client = client

    async def run(self):
        _CONCURRENCY_STATE["current"] += 1
        _CONCURRENCY_STATE["max"] = max(
            _CONCURRENCY_STATE["max"], _CONCURRENCY_STATE["current"]
        )
        await asyncio.sleep(0.05)
        _CONCURRENCY_STATE["current"] -= 1
        return "done"


@pytest.mark.parametrize("max_parallel,expected_max", [(1, 1), (5, 3)])
async def test_max_parallel_limits_concurrency(
    tmp_path, monkeypatch, make_llm, recorded_bus, max_parallel, expected_max
):
    _CONCURRENCY_STATE["current"] = 0
    _CONCURRENCY_STATE["max"] = 0
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _ConcurrencyFakeWorker)

    fake = make_llm(
        workspace=tmp_path,
        responses=[THREE_TASK_PLAN, PASSED_RESULT],
        max_parallel_tasks=max_parallel,
        max_cycles=1,
    )
    bus, _events = recorded_bus
    orch = Orchestrator(goal="Build a thing", client=fake, event_bus=bus)

    await orch.run()

    assert orch.cycles == 1
    assert _CONCURRENCY_STATE["max"] == expected_max


# --------------------------------------------------------------------------- #
# Planner error emits error event and halts with "planner_error"
# --------------------------------------------------------------------------- #
async def test_planner_error_halts_planner_error(tmp_path, make_llm, recorded_bus):
    fake = make_llm(
        workspace=tmp_path,
        responses=[ValueError("bad json"), ValueError("bad json")],
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal="Build a thing", client=fake, event_bus=bus)

    await orch.run()  # must not raise

    error_ev = _find_events(events, "error")
    assert len(error_ev) == 1
    assert error_ev[0]["stage"] == "planner"
    halt_ev = _find_events(events, "halt")
    assert halt_ev[0]["reason"] == "planner_error"


# --------------------------------------------------------------------------- #
# Test failure rewrites goal preserving original_goal
# --------------------------------------------------------------------------- #
async def test_test_failure_rewrites_goal_preserving_original(tmp_path, make_llm, recorded_bus):
    original = "Implement feature X"
    fake = make_llm(
        workspace=tmp_path,
        responses=[
            ONE_TASK_PLAN, FINISH_MSG, FAILED_RESULT,  # cycle 1
            ONE_TASK_PLAN, FINISH_MSG, PASSED_RESULT,  # cycle 2
        ],
        max_cycles=2,
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal=original, client=fake, event_bus=bus)

    await orch.run()

    assert orch.original_goal == original
    assert original in orch.goal
    assert "Fix these failing tests" in orch.goal
    assert "test1 failed" in orch.goal

    # The planner on cycle 2 receives the rewritten goal in its prompt.
    planner_calls = [c for c in fake.calls if c[0] == "complete_json"]
    c2_prompt = planner_calls[2][1]  # 1st=planner c1, 2nd=tester c1, 3rd=planner c2
    assert "Fix these failing tests" in c2_prompt
    assert "test1 failed" in c2_prompt

    assert _find_events(events, "halt")[0]["reason"] == "max_cycles"


# --------------------------------------------------------------------------- #
# Empty plan halts with "no_tasks"
# --------------------------------------------------------------------------- #
async def test_empty_plan_halts_no_tasks(tmp_path, make_llm, recorded_bus):
    fake = make_llm(
        workspace=tmp_path,
        responses=[EMPTY_PLAN],
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal="Already done", client=fake, event_bus=bus)

    await orch.run()

    plan_ev = _find_events(events, "plan")[0]
    assert len(plan_ev["plan"]["tasks"]) == 0
    halt_ev = _find_events(events, "halt")
    assert len(halt_ev) == 1
    assert halt_ev[0]["reason"] == "no_tasks"
    assert orch.cycles == 1


# --------------------------------------------------------------------------- #
# History is passed to planner on subsequent cycles
# --------------------------------------------------------------------------- #
async def test_history_passed_to_planner_on_subsequent_cycles(tmp_path, make_llm, recorded_bus):
    fake = make_llm(
        workspace=tmp_path,
        responses=[
            ONE_TASK_PLAN, FINISH_MSG, FAILED_RESULT,  # cycle 1
            EMPTY_PLAN,  # cycle 2 -> planner returns empty -> no_tasks
        ],
        max_cycles=2,
    )
    bus, events = recorded_bus
    orch = Orchestrator(goal="Build a thing", client=fake, event_bus=bus)

    await orch.run()

    planner_calls = [c for c in fake.calls if c[0] == "complete_json"]
    # calls: planner c1, tester c1, planner c2
    assert len(planner_calls) >= 3
    c1_prompt = planner_calls[0][1]
    c2_prompt = planner_calls[2][1]
    assert "Previous cycle history" not in c1_prompt
    assert "Previous cycle history" in c2_prompt
    assert _find_events(events, "halt")[0]["reason"] == "no_tasks"
    assert orch.cycles == 2
