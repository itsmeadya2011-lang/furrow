from __future__ import annotations

import pytest
from types import SimpleNamespace

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel

FAKE_SETTINGS = SimpleNamespace(planner_model="p", worker_model="w", tester_model="t")


class FakeClient:
    """Minimal stand-in for LLMClient that records calls and returns canned text."""

    def __init__(self, response: str = "", written: list | None = None) -> None:
        self.response = response
        self.written = written if written is not None else []
        self.settings = FAKE_SETTINGS

    async def complete(self, prompt: str, system: str = "", model=None) -> str:
        return self.response

    async def write_file(self, path: str, content: str) -> None:
        self.written.append((path, content))


async def test_worker_writes_file_blocks():
    client = FakeClient(
        response=(
            "Did the thing\n"
            "FILE: a.py\n```python\nprint(1)\n```\n"
            "FILE: b.py\n```\nprint(2)\n```"
        )
    )
    task = TaskModel(id="1", description="make files")
    result = await WorkerAgent(task=task, client=client).run()

    assert "Did the thing" in result
    assert "Wrote files:" in result
    paths = {path for path, _ in client.written}
    assert paths == {"a.py", "b.py"}
    contents = dict(client.written)
    assert contents["a.py"] == "print(1)\n"
    assert contents["b.py"] == "print(2)\n"


async def test_worker_no_file_blocks():
    client = FakeClient(response="Just a summary, nothing to write.")
    task = TaskModel(id="2", description="think")
    result = await WorkerAgent(task=task, client=client).run()

    assert result == "Just a summary, nothing to write."
    assert client.written == []


async def test_planner_parses_valid_json():
    client = FakeClient(
        response='{"tasks":[{"id":"1","description":"x"}],"rationale":"r"}'
    )
    plan = await PlannerAgent(client=client).plan("goal")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "r"


async def test_planner_raises_on_bad_json():
    client = FakeClient(response="not json at all")
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("goal")


async def test_tester_passes_json():
    client = FakeClient(response='{"passed": true, "summary": "ok", "failures": []}')
    tester = TesterAgent(client=client)

    async def fake_run_tests() -> str:
        return "STDOUT:\n"

    tester._run_tests = fake_run_tests
    result = await tester.run("goal", [TaskModel(id="1", description="x")])
    assert isinstance(result, TestResult)
    assert result.passed is True


async def test_tester_fallback_parse():
    client = FakeClient(response="everything passed! no json here")
    tester = TesterAgent(client=client)

    async def fake_run_tests() -> str:
        return "STDOUT:\nsome output"

    tester._run_tests = fake_run_tests
    result = await tester.run("goal", [TaskModel(id="1", description="x")])
    assert result.passed is True
    assert result.summary == "everything passed! no json here"


def test_llm_ollama_base_url():
    client = __import__("furrow.llm", fromlist=["LLMClient"]).LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    )
    assert str(client.ollama.base_url) == "http://localhost:11434/v1"


async def test_llm_unsupported_provider_raises():
    LLMClient = __import__("furrow.llm", fromlist=["LLMClient"]).LLMClient
    client = LLMClient(settings=Settings())
    client.settings.provider = "unsupported"
    with pytest.raises(ValueError):
        await client.complete("hi")
