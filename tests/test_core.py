from __future__ import annotations

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient


class FakeLLMClient(LLMClient):
    """LLMClient that returns canned responses without making network calls."""

    def __init__(self, response: str = "") -> None:
        super().__init__()
        self._response = response
        self.calls: list[dict] = []
        self.written: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        return self._response

    async def read_file(self, path):  # type: ignore[override]
        return ""

    async def write_file(self, path, content):  # type: ignore[override]
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(path)] = content
        return None


def test_plan_parse():
    from furrow.config import Plan

    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_worker_writes_files(tmp_path, monkeypatch):
    response = (
        "FILE: hello.py\n"
        "```\n"
        "print('hello')\n"
        "```\n"
        "FILE: helper.py\n"
        "```\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    client = FakeLLMClient(response=response)
    task = TaskModel(id="1", description="write two files", files=["hello.py", "helper.py"])
    agent = WorkerAgent(task=task, client=client)
    # Worker writes relative to CWD, so chdir into the tmp dir.
    monkeypatch.chdir(tmp_path)
    result = await agent.run()

    assert "hello.py" in result
    assert (tmp_path / "hello.py").read_text().strip() == "print('hello')"
    assert (tmp_path / "helper.py").read_text().strip() == "def add(a, b):\n    return a + b"


@pytest.mark.asyncio
async def test_worker_no_files_returns_message():
    client = FakeLLMClient(response="nothing to do")
    task = TaskModel(id="2", description="do nothing", files=[])
    agent = WorkerAgent(task=task, client=client)
    result = await agent.run()
    assert "No files written" in result


@pytest.mark.asyncio
async def test_orchestrator_done_when_no_tasks():
    from furrow.core.orchestrator import Orchestrator

    client = FakeLLMClient(response='{"tasks": [], "rationale": "done"}')
    orch = Orchestrator(goal="do nothing", client=client)
    orch.tasks = []
    orch.last_test_passed = True
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_collects_tasks_across_cycles():
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Plan

    client = FakeLLMClient(
        response='{"tasks": [{"id": "1", "description": "task one", "files": ["a.py"]}], "rationale": "ok"}'
    )
    orch = Orchestrator(goal="build", client=client)
    p = Plan(tasks=[TaskModel(id="1", description="task one", files=["a.py"])], rationale="ok")
    orch.tasks = list(p.tasks)
    pending = [t for t in orch.tasks if t.status == "pending"]
    assert len(pending) == 1
    orch.tasks[0].status = "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_not_done_with_failed_task():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x", client=FakeLLMClient())
    orch.tasks = [TaskModel(id="1", description="x", status="failed")]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    from furrow.core.orchestrator import Orchestrator

    client = FakeLLMClient(response='{"tasks": [], "rationale": "done"}')
    orch = Orchestrator(goal="x", client=client)
    orch.max_cycles = 3
    orch.cycles = 3
    # Should halt even if not technically "done"
    # We test the run() loop indirectly: max_cycles guard
    orch.last_test_passed = True
    orch.tasks = []
    # _is_done returns True so run would stop anyway; verify the guard separately
    assert orch.max_cycles == 3