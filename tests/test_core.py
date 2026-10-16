import json

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


class FakeClient:
    """Test double duck-typing LLMClient (no real API calls or network)."""

    def __init__(self, settings=None, complete_return="", read_return="", read_side_effect=None):
        self.settings = settings or Settings(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            planner_model="claude-3-5-haiku-20241022",
            worker_model="claude-3-5-sonnet-20241022",
            tester_model="claude-3-5-sonnet-20241022",
            max_parallel_tasks=5,
            max_cycles=0,
        )
        self.complete_return = complete_return
        self.read_return = read_return
        self.read_side_effect = None
        self.complete_calls = []
        self.read_calls = []
        self.written = {}

    async def complete(self, prompt, system="", model=None):
        self.complete_calls.append((prompt, system, model))
        return self.complete_return

    async def read_file(self, path):
        self.read_calls.append(path)
        if self.read_side_effect is not None:
            raise self.read_side_effect
        return self.read_return

    async def write_file(self, path, content):
        self.written[path] = content


def make_orchestrator(client=None):
    return Orchestrator(goal="implement thing", client=client or FakeClient())


# ---- Orchestrator logic (no LLM needed) ----


def test_is_done_no_tasks():
    o = make_orchestrator()
    o.tasks = []
    assert o._is_done() is False


def test_is_done_all_completed():
    o = make_orchestrator()
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_is_done_any_failed():
    o = make_orchestrator()
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_is_done_mixed_pending_completed():
    o = make_orchestrator()
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_merge_tasks_adds_new_tasks():
    o = make_orchestrator()
    t = TaskModel(id="1", description="do thing", files=["src/a.py"], dependencies=[])
    o._merge_tasks([t])
    assert o._get_tasks() == [t]


def test_merge_tasks_preserves_completed():
    o = make_orchestrator()
    prev = TaskModel(id="1", description="old", status="completed", result="already done")
    o.tasks = [prev]
    newly = TaskModel(id="1", description="new plan", status="pending")
    o._merge_tasks([newly])
    tasks = o._get_tasks()
    assert len(tasks) == 1
    assert tasks[0] is prev
    assert tasks[0].status == "completed"
    assert tasks[0].result == "already done"
    assert tasks[0].description == "old"


def test_merge_tasks_readds_pending():
    o = make_orchestrator()
    prev = TaskModel(id="1", description="old", status="pending")
    o.tasks = [prev]
    newly = TaskModel(id="1", description="new plan", status="pending")
    o._merge_tasks([newly])
    tasks = o._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].description == "new plan"
    assert tasks[0].status == "pending"


def test_get_tasks_returns_self_tasks():
    o = make_orchestrator()
    o.tasks = [TaskModel(id="1", description="a")]
    assert o._get_tasks() is o.tasks


# ---- WorkerAgent (with FakeClient) ----


def _worker_task(files=None, description="implement foo"):
    return TaskModel(id="1", description=description, files=files or [], dependencies=[])


async def test_worker_writes_files_and_returns_summary():
    client = FakeClient(
        complete_return=json.dumps(
            {"files": {"src/foo.py": "print(1)"}, "summary": "created foo"}
        )
    )
    worker = WorkerAgent(task=_worker_task(files=["src/foo.py"]), client=client)
    result = await worker.run()
    assert client.written["src/foo.py"] == "print(1)"
    assert result == "created foo"


async def test_worker_reads_existing_file_content_in_prompt():
    client = FakeClient(read_return="existing", complete_return=json.dumps({"files": {}, "summary": "ok"}))
    worker = WorkerAgent(task=_worker_task(files=["src/foo.py"]), client=client)
    await worker.run()
    prompt = client.complete_calls[0][0]
    assert "existing" in prompt


async def test_worker_missing_file_treated_as_empty():
    client = FakeClient(
        read_side_effect=FileNotFoundError("no such file"),
        complete_return=json.dumps({"files": {"src/foo.py": "print(1)"}, "summary": "ok"}),
    )
    worker = WorkerAgent(task=_worker_task(files=["src/foo.py"]), client=client)
    result = await worker.run()
    assert result == "ok"
    assert client.written["src/foo.py"] == "print(1)"


async def test_worker_invalid_json_raises_valueerror():
    client = FakeClient(complete_return="not json")
    worker = WorkerAgent(task=_worker_task(files=["src/foo.py"]), client=client)
    with pytest.raises(ValueError):
        await worker.run()


async def test_worker_empty_files_list():
    client = FakeClient(complete_return=json.dumps({"files": {}, "summary": "nothing to do"}))
    worker = WorkerAgent(task=_worker_task(files=[]), client=client)
    result = await worker.run()
    assert result == "nothing to do"


# ---- Config/models ----


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---- LLMClient dispatch (no network) ----


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "ollama reply"}}


class FakeHttpxClient:
    def __init__(self):
        self.post_calls = []

    async def post(self, url, json=None, timeout=None):
        self.post_calls.append((url, json, timeout))
        return FakeResponse()


async def test_llm_complete_ollama_uses_chat_endpoint():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    fake = FakeHttpxClient()
    client._ollama = fake
    result = await client.complete("hi")
    assert result == "ollama reply"
    assert len(fake.post_calls) == 1
    url, payload, _timeout = fake.post_calls[0]
    assert url == "/api/chat"
    assert payload["stream"] is False
