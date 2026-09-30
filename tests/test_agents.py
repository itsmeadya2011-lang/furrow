"""Comprehensive tests for the Furrow agent classes and LLMClient file helpers.

These tests cover:
  * ``LLMClient`` file-management methods (read/write/edit/exists/list).
  * ``WorkerAgent`` prompt construction, edit application, and error handling.
  * ``PlannerAgent`` JSON-plan parsing.
  * ``TesterAgent`` test-result parsing.

File-operation tests are run against the real ``LLMClient`` using pytest's
``tmp_path`` fixture so they exercise the real disk-based implementation.
Agent tests use a lightweight in-memory ``FakeLLMClient`` so no API key or
network access is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_fake_settings() -> MagicMock:
    """Return a MagicMock standing in for :class:`Settings`.

    Only the attributes actually touched by the agents are required: the model
    name used by each agent and the provider.  These are set explicitly so the
    tests never depend on environment variables or the global settings singleton.
    """
    settings = MagicMock()
    settings.provider = Provider.ANTHROPIC
    settings.worker_model = "test-model"
    settings.planner_model = "test-model"
    settings.tester_model = "test-model"
    return settings


class FakeLLMClient:
    """In-memory fake of :class:`LLMClient` used by the agent tests.

    File operations are backed by an in-memory dict so the tests never touch
    real disk, while ``complete`` is an :class:`AsyncMock` that returns a
    pre-set response string.  The ``settings`` attribute exposes the model
    names that ``WorkerAgent`` / ``PlannerAgent`` / ``TesterAgent`` read.
    """

    def __init__(self, response_text: str = "", files: dict[str, str] | None = None) -> None:
        self.files: dict[str, str] = dict(files or {})
        self.complete = AsyncMock(return_value=response_text)
        self.settings = _make_fake_settings()

    async def file_exists(self, path: str) -> bool:
        return str(path) in self.files

    async def read_file(self, path: str) -> str:
        return self.files[str(path)]

    async def write_file(self, path: str, content: str) -> None:
        self.files[str(path)] = content

    async def edit_file(self, path: str, old_string: str, new_string: str) -> bool:
        key = str(path)
        if key not in self.files:
            return False
        content = self.files[key]
        if old_string not in content:
            return False
        self.files[key] = content.replace(old_string, new_string, 1)
        return True


def _make_task(description: str = "test task", files: list[str] | None = None) -> TaskModel:
    """Convenience factory for a minimal :class:`TaskModel`."""
    return TaskModel(id="1", description=description, files=files or [])


@pytest.fixture
def llm_client() -> LLMClient:
    """A real :class:`LLMClient` constructed with explicit (default) settings.

    Using an explicit ``Settings`` instance keeps the file-operation tests
    independent of the global settings singleton while avoiding any network or
    API-key access since only the file helpers are exercised.
    """
    return LLMClient(settings=Settings())


# ---------------------------------------------------------------------------
# LLMClient file-operation tests
# ---------------------------------------------------------------------------


class TestLLMClientFiles:
    """Tests for ``LLMClient`` file-helper methods (no API key required)."""

    async def test_file_exists_true(self, tmp_path, llm_client):
        """``file_exists`` returns ``True`` for a file that exists on disk."""
        f = tmp_path / "exists.txt"
        f.write_text("hello")
        assert await llm_client.file_exists(f) is True

    async def test_file_exists_false(self, tmp_path, llm_client):
        """``file_exists`` returns ``False`` for a path that does not exist."""
        assert await llm_client.file_exists(tmp_path / "nope.txt") is False

    async def test_read_file(self, tmp_path, llm_client):
        """``read_file`` returns the full contents of an existing file."""
        f = tmp_path / "a.txt"
        f.write_text("file contents here")
        assert await llm_client.read_file(f) == "file contents here"

    async def test_write_file(self, tmp_path, llm_client):
        """``write_file`` writes content to a new file."""
        f = tmp_path / "out.txt"
        await llm_client.write_file(f, "written content")
        assert f.read_text() == "written content"

    async def test_write_file_creates_parent_dirs(self, tmp_path, llm_client):
        """``write_file`` creates missing parent directories for nested paths."""
        f = tmp_path / "nested" / "deep" / "file.txt"
        await llm_client.write_file(f, "nested content")
        assert f.exists()
        assert f.read_text() == "nested content"

    async def test_edit_file_replaces(self, tmp_path, llm_client):
        """``edit_file`` replaces the first occurrence of ``old_string``.

        It returns ``True`` when the replacement succeeds and the file is
        updated on disk.
        """
        f = tmp_path / "edit.txt"
        await llm_client.write_file(f, "the old value here")
        result = await llm_client.edit_file(f, "old", "new")
        assert result is True
        assert f.read_text() == "the new value here"

    async def test_edit_file_not_found(self, tmp_path, llm_client):
        """``edit_file`` returns ``False`` when ``old_string`` is absent.

        The original file content must remain unchanged in that case.
        """
        f = tmp_path / "edit.txt"
        await llm_client.write_file(f, "the old value here")
        result = await llm_client.edit_file(f, "nonexistent", "replacement")
        assert result is False
        assert f.read_text() == "the old value here"

    def test_list_files(self, tmp_path, llm_client):
        """``list_files`` returns the relative names of files in a directory."""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        files = llm_client.list_files(tmp_path)
        assert "a.txt" in files
        assert "b.txt" in files

    def test_list_files_nonexistent_dir(self, tmp_path, llm_client):
        """``list_files`` returns an empty list for a non-existent directory."""
        assert llm_client.list_files(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------


class TestWorkerAgent:
    """Tests for ``WorkerAgent.run`` prompt construction and edit execution."""

    async def test_worker_reads_existing_files(self):
        """Files listed on the task are read and embedded in the LLM prompt.

        The fake LLM's ``complete`` is an ``AsyncMock``; we inspect the prompt
        it received via ``call_args`` to confirm existing file contents were
        included as context.
        """
        file_content = "def hello():\n    return 'hi'\n"
        fake = FakeLLMClient(
            response_text=json.dumps({"edits": [], "summary": "ok"}),
            files={"app/main.py": file_content},
        )
        task = _make_task(description="test task", files=["app/main.py"])
        worker = WorkerAgent(task=task, client=fake)
        await worker.run()

        prompt = fake.complete.call_args.args[0]
        assert file_content in prompt
        assert "app/main.py" in prompt

    async def test_worker_writes_new_file(self):
        """A ``"write"`` edit from the LLM creates the file via the client."""
        edit_json = json.dumps(
            {
                "edits": [
                    {"action": "write", "file": "new_file.py", "content": "print('hello')\n"}
                ],
                "summary": "Created new file",
            }
        )
        fake = FakeLLMClient(response_text=edit_json, files={})
        task = _make_task(description="create a file", files=["new_file.py"])
        worker = WorkerAgent(task=task, client=fake)
        await worker.run()
        assert fake.files["new_file.py"] == "print('hello')\n"

    async def test_worker_edits_existing_file(self):
        """An ``"edit"`` edit replaces ``old_string`` with ``new_string``."""
        edit_json = json.dumps(
            {
                "edits": [
                    {
                        "action": "edit",
                        "file": "existing.py",
                        "old_string": "old",
                        "new_string": "new",
                    }
                ],
                "summary": "Edited existing file",
            }
        )
        fake = FakeLLMClient(
            response_text=edit_json,
            files={"existing.py": "the old value"},
        )
        task = _make_task(description="edit a file", files=["existing.py"])
        worker = WorkerAgent(task=task, client=fake)
        await worker.run()
        assert fake.files["existing.py"] == "the new value"

    async def test_worker_handles_json_parse_failure(self):
        """Non-JSON LLM output is returned verbatim without raising."""
        fake = FakeLLMClient(response_text="this is not valid JSON")
        task = _make_task(description="broken", files=[])
        worker = WorkerAgent(task=task, client=fake)
        result = await worker.run()
        assert result == "this is not valid JSON"

    async def test_worker_handles_unknown_action(self):
        """An unrecognized edit action is logged but does not crash the worker."""
        edit_json = json.dumps(
            {
                "edits": [{"action": "delete", "file": "somefile.py"}],
                "summary": "Done",
            }
        )
        fake = FakeLLMClient(response_text=edit_json, files={})
        task = _make_task(description="unknown action", files=[])
        worker = WorkerAgent(task=task, client=fake)
        result = await worker.run()
        assert "Done" in result


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    """Tests for ``PlannerAgent.plan`` JSON parsing into a :class:`Plan`."""

    async def test_planner_parses_valid_json(self):
        """A valid plan JSON is parsed into a ``Plan`` with tasks and rationale."""
        plan_json = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "task one", "files": ["f1.py"], "dependencies": []},
                    {
                        "id": "2",
                        "description": "task two",
                        "files": ["f2.py"],
                        "dependencies": ["1"],
                    },
                ],
                "rationale": "split into two tasks",
            }
        )
        fake = FakeLLMClient(response_text=plan_json)
        planner = PlannerAgent(client=fake)
        plan = await planner.plan("build something")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].description == "task one"
        assert plan.tasks[0].files == ["f1.py"]
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "split into two tasks"

    async def test_planner_raises_on_invalid_json(self):
        """Non-JSON output causes ``plan()`` to raise ``ValueError``."""
        fake = FakeLLMClient(response_text="not json")
        planner = PlannerAgent(client=fake)
        with pytest.raises(ValueError):
            await planner.plan("build something")


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------


class TestTesterAgent:
    """Tests for ``TesterAgent.run`` result parsing."""

    async def test_tester_parses_passed(self):
        """A ``"passed"`` result JSON yields a ``TestResult`` with ``passed=True``."""
        result_json = json.dumps(
            {
                "passed": True,
                "summary": "All tests passed",
                "failures": [],
            }
        )
        fake = FakeLLMClient(response_text=result_json)
        tester = TesterAgent(client=fake)
        with patch.object(
            TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="test output"
        ):
            result = await tester.run("the goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    async def test_tester_parses_failed(self):
        """A ``"failed"`` result JSON yields ``passed=False`` with failures populated."""
        result_json = json.dumps(
            {
                "passed": False,
                "summary": "2 tests failed",
                "failures": ["test_a failed", "test_b failed"],
            }
        )
        fake = FakeLLMClient(response_text=result_json)
        tester = TesterAgent(client=fake)
        with patch.object(
            TesterAgent, "_run_tests", new_callable=AsyncMock, return_value="failures"
        ):
            result = await tester.run("the goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.summary == "2 tests failed"
        assert result.failures == ["test_a failed", "test_b failed"]
