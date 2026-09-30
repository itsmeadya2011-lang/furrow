import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import FileEdit, Plan, TaskModel, TestResult, WorkerResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_edit_and_worker_result():
    edit = FileEdit(path="src/foo.py", content="print('hello')")
    assert edit.path == "src/foo.py"
    wr = WorkerResult(summary="Added foo", files_changed=["src/foo.py"], edits=[edit])
    assert wr.files_changed == ["src/foo.py"]
    assert wr.edits[0].content == "print('hello')"


def test_worker_result_defaults():
    wr = WorkerResult(summary="ok")
    assert wr.files_changed == []
    assert wr.edits == []


class FakeSettings:
    def __init__(self, max_cycles=0, max_parallel_tasks=5, workspace=None):
        self.max_cycles = max_cycles
        self.max_parallel_tasks = max_parallel_tasks
        self.planner_model = "planner"
        self.worker_model = "worker"
        self.tester_model = "tester"
        self.workspace = workspace or Path.cwd()


def make_orchestrator(max_cycles=0, max_parallel_tasks=5):
    client = MagicMock()
    client.settings = FakeSettings(
        max_cycles=max_cycles, max_parallel_tasks=max_parallel_tasks
    )
    return Orchestrator(goal="test", client=client)


def test_orchestrator_is_done_with_completed_tasks():
    orch = make_orchestrator(max_cycles=0)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    orch = make_orchestrator(max_cycles=0)
    orch.tasks = [TaskModel(id="1", description="a", status="failed")]
    orch._test_result = TestResult(passed=False, summary="err", failures=["err"])
    assert orch._is_done() is False


def test_orchestrator_is_done_with_pending_tasks():
    orch = make_orchestrator(max_cycles=0)
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


def test_orchestrator_is_done_no_tasks():
    orch = make_orchestrator(max_cycles=0)
    orch._test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_orchestrator_is_done_before_test_result():
    orch = make_orchestrator(max_cycles=0)
    orch._test_result = None
    assert orch._is_done() is False


def test_orchestrator_is_done_max_cycles_reached():
    orch = make_orchestrator(max_cycles=2)
    orch.cycles = 2
    orch._test_result = TestResult(passed=False, summary="err", failures=["e"])
    assert orch._is_done() is True


def test_orchestrator_get_tasks_persists():
    orch = make_orchestrator(max_cycles=0)
    t = TaskModel(id="1", description="a")
    orch.tasks = [t]
    assert orch._get_tasks() == [t]


class FakeClient:
    def __init__(self, workspace, response, file_contents=None):
        from furrow.config import Settings

        self.settings = Settings(workspace=workspace)
        self.response = response
        self.written = {}
        self.read_files = []
        self.complete = AsyncMock(return_value=response)
        self._file_contents = file_contents or {}

    async def read_file(self, path):
        self.read_files.append(str(path))
        p = Path(path)
        if p.exists():
            return p.read_text()
        return self._file_contents.get(str(p), "")

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(path)] = content


async def test_worker_parses_structured_response_and_writes_files(tmp_path):
    response_json = json.dumps(
        {
            "summary": "Created foo.py",
            "files_changed": ["foo.py"],
            "edits": [{"path": "foo.py", "content": "print('hello')\n"}],
        }
    )
    fake = FakeClient(workspace=tmp_path, response=response_json)
    task = TaskModel(id="1", description="create foo.py", files=["foo.py"])
    worker = WorkerAgent(task=task, client=fake)
    result = await worker.run()

    assert "foo.py" in result
    written_path = tmp_path / "foo.py"
    assert written_path.read_text() == "print('hello')\n"


async def test_worker_reads_existing_file_context(tmp_path):
    existing = tmp_path / "foo.py"
    existing.write_text("# old\n")
    response_json = json.dumps(
        {
            "summary": "Updated foo.py",
            "files_changed": ["foo.py"],
            "edits": [{"path": "foo.py", "content": "# new\n"}],
        }
    )
    fake = FakeClient(workspace=tmp_path, response=response_json)
    task = TaskModel(id="1", description="update foo.py", files=["foo.py"])
    worker = WorkerAgent(task=task, client=fake)
    result = await worker.run()

    assert result  # non-empty summary
    assert str(tmp_path / "foo.py") in fake.read_files
    assert (tmp_path / "foo.py").read_text() == "# new\n"


async def test_worker_handles_json_parse_error(tmp_path):
    fake = FakeClient(workspace=tmp_path, response="not valid json {{{")
    task = TaskModel(id="1", description="do something", files=[])
    worker = WorkerAgent(task=task, client=fake)
    result = await worker.run()

    assert "JSON parse failed" in result or "raw response" in result


async def test_llm_ollama_provider_dispatches_and_parses():
    from furrow.config import Provider, Settings

    real = Settings(provider=Provider.OLLAMA, ollama_base_url="http://test:11434")
    client = LLMClient(settings=real)

    response_body = {"response": "ollama result"}

    class FakeResponse:
        def __init__(self):
            self.status_code = 200
            self._json = response_body
            self.text = ""

        def json(self):
            return self._json

    class FakeAsyncHTTPClient:
        def __init__(self, *args, **kwargs):
            self.post = AsyncMock(return_value=FakeResponse())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    with patch("furrow.llm.httpx.AsyncClient", FakeAsyncHTTPClient):
        result = await client.complete("hi", system="sys", model="llama2")

    assert result == "ollama result"


async def test_llm_ollama_provider_raises_on_non_200(monkeypatch):
    from furrow.config import Provider, Settings

    real = Settings(provider=Provider.OLLAMA, ollama_base_url="http://test:11434")
    client = LLMClient(settings=real)

    class FakeBadResponse:
        status_code = 500
        text = "internal"

        def json(self):
            return {}

    class FakeAsyncHTTPClient:
        def __init__(self, *args, **kwargs):
            self.post = AsyncMock(return_value=FakeBadResponse())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    with patch("furrow.llm.httpx.AsyncClient", FakeAsyncHTTPClient), pytest.raises(ValueError):
        await client.complete("hi", system="sys", model="llama2")
