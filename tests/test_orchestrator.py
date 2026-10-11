import asyncio
import json

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeClient:
    def __init__(self, responses=None):
        self.settings = Settings(max_parallel_tasks=5, max_cycles=0)
        self._responses = responses or {}

    async def complete(self, prompt, system="", model=None):
        # Determine which agent is calling based on model or prompt content
        prompt_str = prompt or ""
        # Check prompt markers as specified
        if "planning agent" in prompt_str:
            return '```json\n{"tasks": [{"id": "1", "description": "do", "files": [], "dependencies": []}], "rationale": "r"}\n```'
        if "tester agent" in prompt_str:
            return json.dumps({"passed": True, "summary": "ok", "failures": []})
        # Default for worker
        return "done"

def _make_fake_client(responder):
    class Fake:
        def __init__(self):
            self.settings = Settings(max_parallel_tasks=5, max_cycles=0)

        async def complete(self, prompt, system="", model=None):
            return responder(prompt, system, model)
    return Fake()

async def test_orchestrator_halts_when_complete():
    def responder(prompt, system="", model=None):
        p = prompt or ""
        if "planning agent" in p:
            return '```json\n{"tasks": [{"id": "1", "description": "do", "files": [], "dependencies": []}], "rationale": "r"}\n```'
        if "tester agent" in p:
            return json.dumps({"passed": True, "summary": "ok", "failures": []})
        return "ok"

    events = []
    client = _make_fake_client(responder)
    orch = Orchestrator(goal="x", client=client, settings=Settings(max_cycles=5), on_event=lambda e: events.append(e))
    await orch.run()
    event_types = [e["type"] for e in events]
    assert "cycle" in event_types
    assert "plan" in event_types
    assert "task" in event_types
    assert "test" in event_types
    assert "done" in event_types
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["reason"] == "goal complete"

async def test_orchestrator_max_cycles():
    def responder(prompt, system="", model=None):
        p = prompt or ""
        if "planning agent" in p:
            return '```json\n{"tasks": [{"id": "1", "description": "do", "files": [], "dependencies": []}], "rationale": "r"}\n```'
        if "tester agent" in p:
            return json.dumps({"passed": False, "summary": "fail", "failures": ["f1"]})
        return "ok"

    events = []
    client = _make_fake_client(responder)
    orch = Orchestrator(goal="x", client=client, settings=Settings(max_cycles=2), on_event=lambda e: events.append(e))
    await orch.run()
    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["reason"] == "max cycles reached"
    assert orch.cycles == 2

async def test_orchestrator_dependency_order():
    order = []
    def responder(prompt, system="", model=None):
        p = prompt or ""
        if "planning agent" in p:
            return '```json\n{"tasks": [{"id": "1", "description": "a", "files": [], "dependencies": []}, {"id": "2", "description": "b", "files": [], "dependencies": ["1"]}], "rationale": "r"}\n```'
        if "tester agent" in p:
            return json.dumps({"passed": True, "summary": "ok", "failures": []})
        # worker - record task id based on prompt content
        if '"id": "1"' in p or "Task: a" in p or "id 1" in p:
            order.append("1")
        elif '"id": "2"' in p or "Task: b" in p:
            order.append("2")
        else:
            # try to infer from task id in plan structure by matching
            order.append("w")
        return "ok"

    client = _make_fake_client(responder)
    orch = Orchestrator(goal="x", client=client, settings=Settings(max_cycles=5))
    await orch.run()
    assert order[0] == "1"
    assert order[1] == "2"

async def test_orchestrator_semaphore():
    current = 0
    peak = 0
    lock = asyncio.Lock()

    def responder(prompt, system="", model=None):
        p = prompt or ""
        if "planning agent" in p:
            return '```json\n{"tasks": [{"id": "1", "description": "a", "files": [], "dependencies": []}, {"id": "2", "description": "b", "files": [], "dependencies": []}, {"id": "3", "description": "c", "files": [], "dependencies": []}], "rationale": "r"}\n```'
        if "tester agent" in p:
            return json.dumps({"passed": True, "summary": "ok", "failures": []})
        return "ok"

    client = _make_fake_client(responder)
    # Patch worker behavior
    from furrow.agents import worker as worker_mod

    original = worker_mod.WorkerAgent.run

    async def wrapped_run(self):
        nonlocal current, peak
        async with lock:
            current += 1
            if current > peak:
                peak = current
        try:
            await asyncio.sleep(0.001)
            return "ok"
        finally:
            async with lock:
                current -= 1

    worker_mod.WorkerAgent.run = wrapped_run
    try:
        orch = Orchestrator(goal="x", client=client, settings=Settings(max_parallel_tasks=1, max_cycles=5))
        await orch.run()
    finally:
        worker_mod.WorkerAgent.run = original

    assert peak == 1

async def test_orchestrator_failed_dependency():
    def responder(prompt, system="", model=None):
        p = prompt or ""
        if "planning agent" in p:
            return '```json\n{"tasks": [{"id": "1", "description": "a", "files": [], "dependencies": []}, {"id": "2", "description": "b", "files": [], "dependencies": ["1"]}], "rationale": "r"}\n```'
        if "tester agent" in p:
            return json.dumps({"passed": True, "summary": "ok", "failures": []})
        return "ok"

    client = _make_fake_client(responder)
    from furrow.agents import worker as worker_mod

    original = worker_mod.WorkerAgent.run

    async def wrapped_run(self):
        # Fail task 1
        if getattr(self.task, "id", None) == "1":
            raise Exception("boom")
        return "ok"

    worker_mod.WorkerAgent.run = wrapped_run
    try:
        orch = Orchestrator(goal="x", client=client, settings=Settings(max_cycles=5))
        await orch.run()
    finally:
        worker_mod.WorkerAgent.run = original

    # Find task 2 in tasks
    task2 = next(t for t in orch.tasks if t.id == "2")
    assert task2.status == "failed"
    assert task2.result == "Dependency failed or circular dependency"
