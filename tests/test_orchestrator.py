import asyncio
import json

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


PLAN = json.dumps(
    {
        "tasks": [{"id": "1", "description": "do it", "files": ["a.py"], "dependencies": []}],
        "rationale": "r",
    }
)
PLAN2 = json.dumps(
    {
        "tasks": [{"id": "2", "description": "fix it", "files": ["a.py"], "dependencies": []}],
        "rationale": "r",
    }
)
PASSED = json.dumps({"passed": True, "summary": "all good", "failures": []})
FAILED = json.dumps({"passed": False, "summary": "broken", "failures": ["boom in a.py"]})


class FakeClient(LLMClient):
    def __init__(self, responses, settings):
        super().__init__(settings=settings)
        self._responses = list(responses)
        self.calls = []
        self.written = {}

    async def complete(self, prompt, system="", model=None):
        self.calls.append((prompt, model))
        if self._responses:
            return self._responses.pop(0)
        return PASSED

    async def write_file(self, path, content):
        self.written[str(path)] = content

    async def read_file(self, path):
        return self.written.get(str(path), "")


def _run(responses, max_cycles=10, max_parallel=2):
    settings = Settings(max_cycles=max_cycles, max_parallel_tasks=max_parallel, provider="anthropic")
    client = FakeClient(responses, settings)
    orch = Orchestrator(goal="build the thing", client=client, settings=settings)
    asyncio.run(orch.run())
    return orch


def test_success_single_cycle():
    orch = _run([PLAN, "implemented a.py", PASSED])
    assert orch.cycles == 1
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"


def test_fail_then_pass():
    orch = _run([PLAN, "w1", FAILED, PLAN2, "w2", PASSED], max_cycles=5)
    assert orch.cycles == 2
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)


def test_max_cycles_honored():
    orch = _run([PLAN, "w", FAILED], max_cycles=1)
    assert orch.cycles == 1
    assert orch.tasks[0].status == "failed"


def test_empty_plan_stops():
    orch = _run([json.dumps({"tasks": [], "rationale": "done"})])
    assert orch.cycles == 1
    assert orch.tasks == []
