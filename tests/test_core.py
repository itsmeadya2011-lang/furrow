import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class MockLLMClient:
    """A test-double for LLMClient that returns canned responses and performs real file I/O."""

    def __init__(self, responses=None, max_cycles=0):
        self.settings = MagicMock()
        self.settings.max_cycles = max_cycles
        self.settings.worker_model = "mock-model"
        self.settings.planner_model = "mock-planner"
        self.settings.tester_model = "mock-tester"
        self._responses = list(responses or [])
        self._response_index = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self._responses:
            resp = self._responses[min(self._response_index, len(self._responses) - 1)]
            self._response_index += 1
            return resp
        return "{}"

    async def read_file(self, path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory) -> list[str]:
        return []


def make_orchestrator(client=None, max_cycles=0, goal="test goal"):
    client = client or MockLLMClient(max_cycles=max_cycles)
    return Orchestrator(goal=goal, client=client)


def make_worker(task=None, client=None, responses=None):
    task = task or TaskModel(id="t1", description="do a thing", files=[])
    client = client or MockLLMClient(responses=responses)
    return WorkerAgent(task=task, client=client)


# --- Orchestrator._topological_sort ---


def test_topological_sort_orders_dependencies_before_dependents():
    orch = make_orchestrator()
    t_a = TaskModel(id="a", description="a", dependencies=[])
    t_b = TaskModel(id="b", description="b", dependencies=["a"])
    t_c = TaskModel(id="c", description="c", dependencies=["a", "b"])
    ordered = orch._topological_sort([t_c, t_a, t_b])
    ids = [t.id for t in ordered]
    assert ids.index("a") < ids.index("b")
    assert ids.index("b") < ids.index("c")
    assert ids.index("a") < ids.index("c")


def test_topological_sort_handles_cycle():
    orch = make_orchestrator()
    t_x = TaskModel(id="x", description="x", dependencies=["y"])
    t_y = TaskModel(id="y", description="y", dependencies=["x"])
    ordered = orch._topological_sort([t_x, t_y])
    assert len(ordered) == 2
    assert {t.id for t in ordered} == {"x", "y"}


def test_topological_sort_independent_tasks_preserved():
    orch = make_orchestrator()
    t1 = TaskModel(id="1", description="1", dependencies=[])
    t2 = TaskModel(id="2", description="2", dependencies=[])
    t3 = TaskModel(id="3", description="3", dependencies=["1"])
    ordered = orch._topological_sort([t3, t2, t1])
    ids = [t.id for t in ordered]
    assert ids.index("1") < ids.index("3")
    assert set(ids) == {"1", "2", "3"}


# --- Orchestrator max_cycles ---


async def test_orchestrator_max_cycles_stops_loop():
    client = MockLLMClient(max_cycles=3)
    orch = Orchestrator(goal="g", client=client)
    orch._cycle = AsyncMock()
    await orch.run()
    assert orch.cycles == 3


async def test_orchestrator_max_cycles_zero_runs_at_least_once():
    client = MockLLMClient(max_cycles=0)
    orch = Orchestrator(goal="g", client=client)
    orch._cycle = AsyncMock()
    # _is_done is False when no tasks/tests passed, so we must force break via patch
    with patch.object(orch, "_is_done", return_value=True):
        await orch.run()
    assert orch.cycles == 1


# --- Orchestrator._is_done ---


def test_is_done_all_completed_and_tests_passed():
    orch = make_orchestrator(max_cycles=0)
    orch._tasks = [TaskModel(id="t1", description="d", status="completed")]
    orch._tests_passed = True
    assert orch._is_done() is True


def test_is_done_tasks_not_completed():
    orch = make_orchestrator(max_cycles=0)
    orch._tasks = [TaskModel(id="t1", description="d", status="pending")]
    orch._tests_passed = True
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = make_orchestrator(max_cycles=0)
    orch._tasks = [TaskModel(id="t1", description="d", status="completed")]
    orch._tests_passed = False
    assert orch._is_done() is False


def test_is_done_no_tasks_returns_false():
    orch = make_orchestrator(max_cycles=0)
    orch._tasks = []
    orch._tests_passed = True
    assert orch._is_done() is False


def test_is_done_failed_task_present():
    orch = make_orchestrator(max_cycles=0)
    orch._tasks = [
        TaskModel(id="t1", description="d", status="completed"),
        TaskModel(id="t2", description="d", status="failed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


def test_is_done_max_cycles_reached_overrides():
    orch = make_orchestrator(max_cycles=2)
    orch._tasks = []
    orch._tests_passed = False
    orch.cycles = 2
    assert orch._is_done() is True


# --- WorkerAgent._parse_response ---


def test_parse_response_valid_json():
    worker = make_worker()
    resp = '{"operations": [{"action": "create", "path": "x.py", "content": "print(1)"}], "summary": "created x.py"}'
    result = worker._parse_response(resp)
    assert result["operations"][0]["action"] == "create"
    assert result["summary"] == "created x.py"


def test_parse_response_invalid_json():
    worker = make_worker()
    resp = "this is not json at all"
    with pytest.raises(ValueError):
        worker._parse_response(resp)


def test_parse_response_json_embedded_in_markdown():
    worker = make_worker()
    resp = "```json\n{\"operations\": [], \"summary\": \"ok\"}\n```"
    result = worker._parse_response(resp)
    assert result["summary"] == "ok"
    assert result["operations"] == []


def test_parse_response_json_with_intro_text():
    worker = make_worker()
    resp = "Here is my plan:\n{\"operations\": [{\"action\": \"update\", \"path\": \"a.py\", \"content\": \"hi\"}], \"summary\": \"done\"}"
    result = worker._parse_response(resp)
    assert result["operations"][0]["action"] == "update"
    assert result["summary"] == "done"


def test_parse_response_nested_json():
    worker = make_worker()
    resp = '{"operations": [], "summary": "ok", "data": {"nested": {"deep": true}}}'
    result = worker._parse_response(resp)
    assert result["data"]["nested"]["deep"] is True


# --- WorkerAgent.execute_operations ---


async def test_execute_operations_create_update_delete(tmp_path):
    client = MockLLMClient()
    worker = make_worker(client=client)

    create_path = tmp_path / "created.txt"
    update_path = tmp_path / "updated.txt"
    update_path.write_text("original")
    delete_path = tmp_path / "deleted.txt"
    delete_path.write_text("to be deleted")

    ops = [
        {"action": "create", "path": str(create_path), "content": "hello"},
        {"action": "update", "path": str(update_path), "content": "updated"},
        {"action": "delete", "path": str(delete_path)},
    ]

    changed = await worker.execute_operations(ops)

    assert str(create_path) in changed
    assert str(update_path) in changed
    assert str(delete_path) in changed
    assert create_path.read_text() == "hello"
    assert update_path.read_text() == "updated"
    assert not delete_path.exists()


async def test_execute_operations_no_path_skipped():
    client = MockLLMClient()
    worker = make_worker(client=client)
    ops = [{"action": "create", "content": "x"}]
    changed = await worker.execute_operations(ops)
    assert changed == []


async def test_execute_operations_delete_nonexistent(tmp_path):
    client = MockLLMClient()
    worker = make_worker(client=client)
    missing = tmp_path / "nope.txt"
    ops = [{"action": "delete", "path": str(missing)}]
    changed = await worker.execute_operations(ops)
    assert changed == []


async def test_execute_operations_creates_parent_dirs(tmp_path):
    client = MockLLMClient()
    worker = make_worker(client=client)
    nested = tmp_path / "sub" / "deep" / "file.txt"
    ops = [{"action": "create", "path": str(nested), "content": "nested content"}]
    changed = await worker.execute_operations(ops)
    assert changed == [str(nested)]
    assert nested.read_text() == "nested content"


# --- WorkerAgent._read_files ---


async def test_read_files_existing_and_missing(tmp_path):
    client = MockLLMClient()
    worker = make_worker(client=client)

    existing = tmp_path / "exists.txt"
    existing.write_text("hello world")
    missing = tmp_path / "missing.txt"

    files = [str(existing), str(missing)]
    contents = await worker._read_files(files)

    assert contents[str(existing)] == "hello world"
    assert contents[str(missing)] == "(file does not exist yet)"


async def test_read_files_all_exist(tmp_path):
    client = MockLLMClient()
    worker = make_worker(client=client)

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("alpha")
    f2.write_text("beta")

    contents = await worker._read_files([str(f1), str(f2)])

    assert contents == {str(f1): "alpha", str(f2): "beta"}


async def test_read_files_empty_list():
    client = MockLLMClient()
    worker = make_worker(client=client)
    contents = await worker._read_files([])
    assert contents == {}
